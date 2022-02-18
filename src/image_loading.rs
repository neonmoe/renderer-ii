use crate::vulkan_raii::ImageView;
use crate::{debug_utils, Error, FrameIndex, Gpu, VulkanArena};
use ash::version::DeviceV1_0;
use ash::vk;
use std::rc::Rc;

mod ktx;

#[derive(Clone, Copy)]
pub enum TextureKind {
    SrgbColor,
    LinearColor,
    NormalMap,
}

impl TextureKind {
    fn convert_format(self, format: vk::Format) -> vk::Format {
        match self {
            TextureKind::SrgbColor => to_srgb(format),
            TextureKind::NormalMap => to_snorm(format),
            TextureKind::LinearColor => format,
        }
    }
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

/// If the `format` is a 'UNORM' format, return the 'SNORM' variant if
/// there is one.
fn to_snorm(format: vk::Format) -> vk::Format {
    match format {
        vk::Format::R8_UNORM => vk::Format::R8_SNORM,
        vk::Format::R8G8_UNORM => vk::Format::R8G8_SNORM,
        vk::Format::R8G8B8_UNORM => vk::Format::R8G8B8_SNORM,
        vk::Format::R8G8B8A8_UNORM => vk::Format::R8G8B8A8_SNORM,
        vk::Format::B8G8R8_UNORM => vk::Format::B8G8R8_SNORM,
        vk::Format::B8G8R8A8_UNORM => vk::Format::B8G8R8A8_SNORM,
        vk::Format::A8B8G8R8_UNORM_PACK32 => vk::Format::A8B8G8R8_SNORM_PACK32,
        vk::Format::A2R10G10B10_UNORM_PACK32 => vk::Format::A2R10G10B10_SNORM_PACK32,
        vk::Format::A2B10G10R10_UNORM_PACK32 => vk::Format::A2B10G10R10_SNORM_PACK32,
        vk::Format::R16_UNORM => vk::Format::R16_SNORM,
        vk::Format::R16G16_UNORM => vk::Format::R16G16_SNORM,
        vk::Format::R16G16B16_UNORM => vk::Format::R16G16B16_SNORM,
        vk::Format::R16G16B16A16_UNORM => vk::Format::R16G16B16A16_SNORM,
        vk::Format::BC4_UNORM_BLOCK => vk::Format::BC4_SNORM_BLOCK,
        vk::Format::BC5_UNORM_BLOCK => vk::Format::BC5_SNORM_BLOCK,
        vk::Format::EAC_R11_UNORM_BLOCK => vk::Format::EAC_R11_SNORM_BLOCK,
        vk::Format::EAC_R11G11_UNORM_BLOCK => vk::Format::EAC_R11G11_SNORM_BLOCK,
        f => f,
    }
}

/// Creates an ImageView that consists of a single pixel with the
/// given color.
///
/// The pixel channels are laid out as: [r, g, b, a].
pub fn create_pixel(
    gpu: &Gpu,
    arena: &VulkanArena,
    temp_arenas: &[VulkanArena],
    frame_index: FrameIndex,
    pixels: [u8; 4],
    kind: TextureKind,
) -> Result<ImageView, Error> {
    let format = kind.convert_format(vk::Format::R8G8B8A8_UNORM);
    let extent = *vk::Extent3D::builder().width(1).height(1).depth(1);

    let temp_arena = frame_index.get_arena(temp_arenas);
    let staging_buffer = {
        profiling::scope!("allocate and create 1px staging buffer");
        let buffer_size = pixels.len() as vk::DeviceSize;
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        temp_arena.create_buffer(*buffer_create_info)?
    };

    {
        profiling::scope!("write to 1px staging buffer");
        unsafe { staging_buffer.write(pixels.as_ptr(), 0, vk::WHOLE_SIZE) }?;
    }

    let image_allocation = {
        profiling::scope!("allocate 1px gpu texture");
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(extent)
            .mip_levels(1)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);
        arena.create_image(*image_info)?
    };

    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
        .build();

    gpu.run_command_buffer(frame_index, vk::PipelineStageFlags::FRAGMENT_SHADER, |command_buffer| {
        profiling::scope!("record vkCmdCopyBufferToImage for 1px image");
        let barrier_from_undefined_to_transfer_dst = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image_allocation.inner)
            .subresource_range(subresource_range)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .build();
        unsafe {
            gpu.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_from_undefined_to_transfer_dst],
            );
        }
        let subresource_layers_dst = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .base_array_layer(0)
            .layer_count(1)
            .build();
        let image_copy_region = vk::BufferImageCopy::builder()
            .buffer_offset(0)
            .buffer_row_length(1)
            .buffer_image_height(1)
            .image_subresource(subresource_layers_dst)
            .image_extent(extent)
            .build();
        unsafe {
            gpu.device.cmd_copy_buffer_to_image(
                command_buffer,
                staging_buffer.buffer.inner,
                image_allocation.inner,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[image_copy_region],
            );
        }
        let barrier_from_transfer_dst_to_shader = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image_allocation.inner)
            .subresource_range(subresource_range)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .build();
        unsafe {
            gpu.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_from_transfer_dst_to_shader],
            );
        }
    })?;

    let image_view = {
        let image_view_create_info = vk::ImageViewCreateInfo::builder()
            .image(image_allocation.inner)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(subresource_range);

        let image_view = unsafe { gpu.device.create_image_view(&image_view_create_info, None) }.map_err(Error::VulkanImageViewCreation)?;
        ImageView {
            inner: image_view,
            device: gpu.device.clone(),
            image: Rc::new(image_allocation.into()),
        }
    };

    gpu.add_temp_buffer(frame_index, staging_buffer.buffer);

    Ok(image_view)
}

/// Loads a ktx into (width, height, format, pixel bytes). Note that
/// the pixel bytes do not map to a linear rgba array: the data is
/// compressed. The format reflects this.
#[profiling::function]
pub fn load_ktx(
    gpu: &Gpu,
    arena: &VulkanArena,
    temp_arenas: &[VulkanArena],
    frame_index: FrameIndex,
    bytes: &[u8],
    kind: TextureKind,
    debug_identifier: &str,
) -> Result<ImageView, Error> {
    let ktx::KtxData {
        width,
        height,
        format,
        pixels,
        mip_ranges,
    } = ktx::decode(bytes)?;
    let format = kind.convert_format(format);
    let extent = *vk::Extent3D::builder().width(width).height(height).depth(1);

    let temp_arena = frame_index.get_arena(temp_arenas);
    let staging_buffer = {
        profiling::scope!("allocate and create staging buffer");
        let buffer_size = pixels.len() as vk::DeviceSize;
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        temp_arena.create_buffer(*buffer_create_info)?
    };
    debug_utils::name_vulkan_object(
        &gpu.device,
        staging_buffer.buffer.inner,
        format_args!("staging buffer: {}", debug_identifier),
    );

    {
        profiling::scope!("write to staging buffer");
        unsafe { staging_buffer.write(pixels.as_ptr(), 0, vk::WHOLE_SIZE) }?;
    }

    let image_allocation = {
        profiling::scope!("allocate gpu texture");
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(extent)
            .mip_levels(mip_ranges.len() as u32)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);
        arena.create_image(*image_info)?
    };
    debug_utils::name_vulkan_object(&gpu.device, image_allocation.inner, format_args!("{}", debug_identifier));

    gpu.run_command_buffer(frame_index, vk::PipelineStageFlags::FRAGMENT_SHADER, |command_buffer| {
        let mut current_mip_level_extent = extent;
        for (mip_level, mip_range) in mip_ranges.iter().enumerate() {
            profiling::scope!(&format!("record vkCmdCopyBufferToImage for mip #{}", mip_level));
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(mip_level as u32)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build();
            let barrier_from_undefined_to_transfer_dst = vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image_allocation.inner)
                .subresource_range(subresource_range)
                .src_access_mask(vk::AccessFlags::empty())
                .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .build();
            unsafe {
                gpu.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_from_undefined_to_transfer_dst],
                );
            }
            let subresource_layers_dst = vk::ImageSubresourceLayers::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(mip_level as u32)
                .base_array_layer(0)
                .layer_count(1)
                .build();
            // NOTE: Only works for square block sizes. May cause issues down the line.
            let texel_size = (mip_range.len() as f32).sqrt() as u32;
            let image_copy_region = vk::BufferImageCopy::builder()
                .buffer_offset(mip_range.start as vk::DeviceSize)
                .buffer_row_length(current_mip_level_extent.width.max(texel_size))
                .buffer_image_height(current_mip_level_extent.height.max(texel_size))
                .image_subresource(subresource_layers_dst)
                .image_extent(current_mip_level_extent)
                .build();
            unsafe {
                gpu.device.cmd_copy_buffer_to_image(
                    command_buffer,
                    staging_buffer.buffer.inner,
                    image_allocation.inner,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[image_copy_region],
                );
            }
            current_mip_level_extent.width = (current_mip_level_extent.width / 2).max(1);
            current_mip_level_extent.height = (current_mip_level_extent.height / 2).max(1);
            current_mip_level_extent.depth = (current_mip_level_extent.depth / 2).max(1);
            let subresource_range = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(mip_level as u32)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1)
                .build();
            let barrier_from_transfer_dst_to_shader = vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image_allocation.inner)
                .subresource_range(subresource_range)
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .build();
            unsafe {
                gpu.device.cmd_pipeline_barrier(
                    command_buffer,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier_from_transfer_dst_to_shader],
                );
            }
        }
    })?;

    let image_view = {
        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(mip_ranges.len() as u32)
            .base_array_layer(0)
            .layer_count(1)
            .build();
        let image_view_create_info = vk::ImageViewCreateInfo::builder()
            .image(image_allocation.inner)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(subresource_range);

        let image_view = unsafe { gpu.device.create_image_view(&image_view_create_info, None) }.map_err(Error::VulkanImageViewCreation)?;
        ImageView {
            inner: image_view,
            device: gpu.device.clone(),
            image: Rc::new(image_allocation.into()),
        }
    };
    debug_utils::name_vulkan_object(&gpu.device, image_view.inner, format_args!("{}", debug_identifier));

    gpu.add_temp_buffer(frame_index, staging_buffer.buffer);

    Ok(image_view)
}
