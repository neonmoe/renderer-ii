use alloc::rc::Rc;
use core::ops::Range;

use arrayvec::ArrayVec;
use ash::vk;

use crate::arena::buffers::ForBuffers;
use crate::arena::images::ForImages;
use crate::arena::{VulkanArena, VulkanArenaError};
use crate::physical_device::TEXTURE_FORMATS;
use crate::uploader::Uploader;
use crate::vulkan_raii::{Device, ImageView};

pub mod ntex;
pub mod pbr_defaults;

pub struct ImageData<'a> {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    /// Note: not necessarily actual pixels, just the pixel data.
    pub pixels: &'a [u8],
    /// The ranges from `pixels` that represent different mip levels.
    pub mip_ranges: ArrayVec<Range<usize>, 16>,
}

impl ImageData<'_> {
    pub fn get_create_info(&self, kind: TextureKind) -> vk::ImageCreateInfo<'static> {
        let ImageData { width, height, format, mip_ranges, .. } = self;
        let format = kind.convert_format(*format);
        let extent = vk::Extent3D::default().width(*width).height(*height).depth(1);
        vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(extent)
            .mip_levels(mip_ranges.len() as u32)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .samples(vk::SampleCountFlags::TYPE_1)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
    }
}

#[derive(Clone, Copy)]
pub enum TextureKind {
    SrgbColor,
    LinearColor,
    NormalMap,
}

impl TextureKind {
    pub fn convert_format(self, format: vk::Format) -> vk::Format {
        match self {
            TextureKind::SrgbColor => to_srgb(format),
            TextureKind::NormalMap | TextureKind::LinearColor => format,
        }
    }
}

pub fn create_pixel(
    device: &Device,
    staging_arena: &mut VulkanArena<ForBuffers>,
    uploader: &mut Uploader,
    arena: &mut VulkanArena<ForImages>,
    pixels: [u8; 4],
    kind: TextureKind,
    debug_identifier: &str,
) -> Result<ImageView, VulkanArenaError> {
    let pixel_mip_range = 0..pixels.len();
    let image_data = ImageData {
        width: 1,
        height: 1,
        format: vk::Format::R8G8B8A8_UNORM,
        pixels: &pixels,
        mip_ranges: ArrayVec::from_iter([pixel_mip_range]),
    };
    load_image(device, staging_arena, uploader, arena, &image_data, kind, debug_identifier)
}

#[profiling::function]
pub fn load_image(
    device: &Device,
    staging_arena: &mut VulkanArena<ForBuffers>,
    uploader: &mut Uploader,
    arena: &mut VulkanArena<ForImages>,
    image_data: &ImageData,
    kind: TextureKind,
    debug_identifier: &str,
) -> Result<ImageView, VulkanArenaError> {
    let ImageData { width, height, format, .. } = *image_data;
    let pixels = &image_data.pixels;
    let mip_ranges = &image_data.mip_ranges;

    let format = kind.convert_format(format);
    if cfg!(debug_assertions) {
        debug_assert!(TEXTURE_FORMATS.contains(&format));
    }

    let extent = vk::Extent3D::default().width(width).height(height).depth(1);
    let &mut Uploader { graphics_queue_family, transfer_queue_family, .. } = uploader;

    let staging_buffer = {
        profiling::scope!("allocate and create staging buffer");
        let buffer_size = pixels.len() as vk::DeviceSize;
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        staging_arena.create_buffer(buffer_create_info, pixels, None, None, format_args!("staging buffer for {debug_identifier}"))?
    };

    let image_allocation = {
        profiling::scope!("allocate gpu texture");
        let image_info = image_data.get_create_info(kind);
        arena.create_image(image_info, format_args!("{debug_identifier}"))?
    };

    uploader.start_upload(
        staging_buffer,
        format_args!("{debug_identifier}"),
        |device, staging_buffer, command_buffer| {
            let mut current_mip_level_extent = extent;
            for (mip_level, mip_range) in mip_ranges.iter().enumerate() {
                profiling::scope!("vk::cmd_copy_buffer_to_image");
                let subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(mip_level as u32)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);
                let layout_to_transfer_dst = [vk::ImageMemoryBarrier2::default()
                    .image(image_allocation.inner)
                    .subresource_range(subresource_range)
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COPY)];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&layout_to_transfer_dst);
                unsafe { device.cmd_pipeline_barrier2(command_buffer, &dep_info) };

                let subresource_layers_dst = vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(mip_level as u32)
                    .base_array_layer(0)
                    .layer_count(1);
                // NOTE: Only works for square block sizes. May cause issues down the line.
                let texel_size = (mip_range.len() as f32).sqrt() as u32;
                let image_copy_region = vk::BufferImageCopy::default()
                    .buffer_offset(mip_range.start as vk::DeviceSize)
                    .buffer_row_length(current_mip_level_extent.width.max(texel_size))
                    .buffer_image_height(current_mip_level_extent.height.max(texel_size))
                    .image_subresource(subresource_layers_dst)
                    .image_extent(current_mip_level_extent);
                unsafe {
                    device.cmd_copy_buffer_to_image(
                        command_buffer,
                        staging_buffer.inner,
                        image_allocation.inner,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[image_copy_region],
                    );
                }

                let release_to_graphics_queue = [vk::ImageMemoryBarrier2::default()
                    .image(image_allocation.inner)
                    .subresource_range(subresource_range)
                    .src_queue_family_index(transfer_queue_family)
                    .dst_queue_family_index(graphics_queue_family)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&release_to_graphics_queue);
                unsafe { device.cmd_pipeline_barrier2(command_buffer, &dep_info) };

                current_mip_level_extent.width = (current_mip_level_extent.width / 2).max(1);
                current_mip_level_extent.height = (current_mip_level_extent.height / 2).max(1);
                current_mip_level_extent.depth = (current_mip_level_extent.depth / 2).max(1);
            }
        },
        |device, command_buffer| {
            for mip_level in 0..mip_ranges.len() {
                profiling::scope!("record transfer->shader barrier");
                let subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(mip_level as u32)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1);
                let layout_to_shader_and_acquire_from_transfer_queue = [vk::ImageMemoryBarrier2::default()
                    .image(image_allocation.inner)
                    .subresource_range(subresource_range)
                    .src_queue_family_index(transfer_queue_family)
                    .dst_queue_family_index(graphics_queue_family)
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&layout_to_shader_and_acquire_from_transfer_queue);
                unsafe { device.cmd_pipeline_barrier2(command_buffer, &dep_info) };
            }
        },
    );

    let image_view = {
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(mip_ranges.len() as u32)
            .base_array_layer(0)
            .layer_count(1);
        let image_view_create_info = vk::ImageViewCreateInfo::default()
            .image(image_allocation.inner)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(subresource_range);

        let image_view =
            unsafe { device.create_image_view(&image_view_create_info, None) }.expect("vulkan image view creation should not fail");
        ImageView { inner: image_view, device: device.clone(), image: Rc::new(image_allocation.into()) }
    };
    crate::name_vulkan_object(device, image_view.inner, format_args!("{debug_identifier}"));

    Ok(image_view)
}

/// If the `format` is a 'UNORM' format, and has an SRGB variant,
/// return that.
///
/// Does not work for the 3D ASTC ones.
fn to_srgb(format: vk::Format) -> vk::Format {
    match format {
        vk::Format::R8_UNORM => vk::Format::R8_SRGB,
        vk::Format::R8G8_UNORM => vk::Format::R8G8_SRGB,
        vk::Format::R8G8B8_UNORM => vk::Format::R8G8B8_SRGB,
        vk::Format::R8G8B8A8_UNORM => vk::Format::R8G8B8A8_SRGB,
        vk::Format::B8G8R8_UNORM => vk::Format::B8G8R8_SRGB,
        vk::Format::B8G8R8A8_UNORM => vk::Format::B8G8R8A8_SRGB,
        vk::Format::A8B8G8R8_UNORM_PACK32 => vk::Format::A8B8G8R8_SRGB_PACK32,
        vk::Format::BC2_UNORM_BLOCK => vk::Format::BC2_SRGB_BLOCK,
        vk::Format::BC3_UNORM_BLOCK => vk::Format::BC3_SRGB_BLOCK,
        vk::Format::BC7_UNORM_BLOCK => vk::Format::BC7_SRGB_BLOCK,
        vk::Format::BC1_RGB_UNORM_BLOCK => vk::Format::BC1_RGB_SRGB_BLOCK,
        vk::Format::BC1_RGBA_UNORM_BLOCK => vk::Format::BC1_RGBA_SRGB_BLOCK,
        vk::Format::ETC2_R8G8B8_UNORM_BLOCK => vk::Format::ETC2_R8G8B8_SRGB_BLOCK,
        vk::Format::ETC2_R8G8B8A1_UNORM_BLOCK => vk::Format::ETC2_R8G8B8A1_SRGB_BLOCK,
        vk::Format::ETC2_R8G8B8A8_UNORM_BLOCK => vk::Format::ETC2_R8G8B8A8_SRGB_BLOCK,
        vk::Format::PVRTC1_2BPP_UNORM_BLOCK_IMG => vk::Format::PVRTC1_2BPP_SRGB_BLOCK_IMG,
        vk::Format::PVRTC1_4BPP_UNORM_BLOCK_IMG => vk::Format::PVRTC1_4BPP_SRGB_BLOCK_IMG,
        vk::Format::PVRTC2_2BPP_UNORM_BLOCK_IMG => vk::Format::PVRTC2_2BPP_SRGB_BLOCK_IMG,
        vk::Format::PVRTC2_4BPP_UNORM_BLOCK_IMG => vk::Format::PVRTC2_4BPP_SRGB_BLOCK_IMG,
        vk::Format::ASTC_4X4_UNORM_BLOCK => vk::Format::ASTC_4X4_SRGB_BLOCK,
        vk::Format::ASTC_5X4_UNORM_BLOCK => vk::Format::ASTC_5X4_SRGB_BLOCK,
        vk::Format::ASTC_5X5_UNORM_BLOCK => vk::Format::ASTC_5X5_SRGB_BLOCK,
        vk::Format::ASTC_6X5_UNORM_BLOCK => vk::Format::ASTC_6X5_SRGB_BLOCK,
        vk::Format::ASTC_6X6_UNORM_BLOCK => vk::Format::ASTC_6X6_SRGB_BLOCK,
        vk::Format::ASTC_8X5_UNORM_BLOCK => vk::Format::ASTC_8X5_SRGB_BLOCK,
        vk::Format::ASTC_8X6_UNORM_BLOCK => vk::Format::ASTC_8X6_SRGB_BLOCK,
        vk::Format::ASTC_8X8_UNORM_BLOCK => vk::Format::ASTC_8X8_SRGB_BLOCK,
        vk::Format::ASTC_10X5_UNORM_BLOCK => vk::Format::ASTC_10X5_SRGB_BLOCK,
        vk::Format::ASTC_10X6_UNORM_BLOCK => vk::Format::ASTC_10X6_SRGB_BLOCK,
        vk::Format::ASTC_10X8_UNORM_BLOCK => vk::Format::ASTC_10X8_SRGB_BLOCK,
        vk::Format::ASTC_10X10_UNORM_BLOCK => vk::Format::ASTC_10X10_SRGB_BLOCK,
        vk::Format::ASTC_12X10_UNORM_BLOCK => vk::Format::ASTC_12X10_SRGB_BLOCK,
        vk::Format::ASTC_12X12_UNORM_BLOCK => vk::Format::ASTC_12X12_SRGB_BLOCK,
        f => f,
    }
}
