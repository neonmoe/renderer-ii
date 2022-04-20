use crate::arena::VulkanArenaError;
use crate::uploader::UploadError;
use crate::vulkan_raii::{Device, ImageView};
use crate::{debug_utils, ForImages, Uploader, VulkanArena};
use crate::{VulkanArenaMeasurementError, VulkanArenaMeasurer};
use ash::vk;
use std::rc::Rc;

mod ntex;

#[derive(thiserror::Error, Debug)]
pub enum ImageLoadingError {
    #[error("failed to decode ntex file")]
    Ntex(#[source] ntex::NtexDecodeError),
    #[error("failed to create staging buffer for the image")]
    StagingBufferCreation(#[source] VulkanArenaError),
    #[error("failed to create the image")]
    ImageCreation(#[source] VulkanArenaError),
    #[error("failed to start upload for image")]
    StartTextureUpload(#[source] UploadError),
    #[error("failed to create the image view")]
    ImageViewCreation(#[source] vk::Result),
}

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

const WHITE: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF];
const BLACK: [u8; 4] = [0, 0, 0, 0xFF];
const NORMAL_Z: [u8; 4] = [0, 0, 0xFF, 0];
const M_AND_R: [u8; 4] = [0, 0x88, 0, 0];

pub struct PbrDefaults {
    pub base_color: ImageView,
    pub metallic_roughness: ImageView,
    pub normal: ImageView,
    pub occlusion: ImageView,
    pub emissive: ImageView,
}

impl PbrDefaults {
    pub fn new(device: &Device, uploader: &mut Uploader, arena: &mut VulkanArena<ForImages>) -> Result<PbrDefaults, ImageLoadingError> {
        profiling::scope!("pbr default textures creation");

        let mut create_pixel = |color, kind, name| create_pixel(device, uploader, arena, color, kind, name);
        let base_color = create_pixel(WHITE, TextureKind::SrgbColor, "default pbr base color")?;
        let metallic_roughness = create_pixel(M_AND_R, TextureKind::LinearColor, "default pbr metallic/roughness")?;
        let normal = create_pixel(NORMAL_Z, TextureKind::NormalMap, "default pbr normals")?;
        let occlusion = create_pixel(WHITE, TextureKind::LinearColor, "default pbr occlusion")?;
        let emissive = create_pixel(BLACK, TextureKind::SrgbColor, "default pbr emissive")?;

        Ok(PbrDefaults {
            base_color,
            metallic_roughness,
            normal,
            occlusion,
            emissive,
        })
    }

    pub fn measure(measurer: &mut VulkanArenaMeasurer<ForImages>) -> Result<(), VulkanArenaMeasurementError> {
        for kind in [
            TextureKind::SrgbColor,   // Base color
            TextureKind::LinearColor, // Metallic/roughness
            TextureKind::NormalMap,   // Normals
            TextureKind::LinearColor, // Ambient occlusion
            TextureKind::SrgbColor,   // Emissive
        ] {
            let format = kind.convert_format(vk::Format::R8G8B8A8_UNORM);
            let extent = *vk::Extent3D::builder().width(1).height(1).depth(1);
            let image_create_info = get_image_create_info(extent, 1, format);
            measurer.add_image(image_create_info)?;
        }
        Ok(())
    }
}

fn get_image_create_info(extent: vk::Extent3D, mip_levels: u32, format: vk::Format) -> vk::ImageCreateInfo {
    vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(extent)
        .mip_levels(mip_levels)
        .array_layers(1)
        .format(format)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .samples(vk::SampleCountFlags::TYPE_1)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .build()
}

fn create_pixel(
    device: &Device,
    uploader: &mut Uploader,
    arena: &mut VulkanArena<ForImages>,
    pixels: [u8; 4],
    kind: TextureKind,
    debug_identifier: &str,
) -> Result<ImageView, ImageLoadingError> {
    let format = kind.convert_format(vk::Format::R8G8B8A8_UNORM);
    let extent = *vk::Extent3D::builder().width(1).height(1).depth(1);
    let &mut Uploader {
        graphics_queue_family,
        transfer_queue_family,
        ..
    } = uploader;

    let staging_buffer = {
        profiling::scope!("allocate and create 1px staging buffer");
        let buffer_size = pixels.len() as vk::DeviceSize;
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        uploader
            .staging_arena
            .create_buffer(
                *buffer_create_info,
                &pixels,
                None,
                format_args!("staging buffer for {}", debug_identifier),
            )
            .map_err(ImageLoadingError::StagingBufferCreation)?
    };

    let image_allocation = {
        profiling::scope!("allocate 1px gpu texture");
        let image_info = get_image_create_info(extent, 1, format);
        arena
            .create_image(image_info, format_args!("{}", debug_identifier))
            .map_err(ImageLoadingError::ImageCreation)?
    };

    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
        .build();

    uploader
        .start_upload(
            staging_buffer,
            format_args!("{}", debug_identifier),
            |device, staging_buffer, command_buffer| {
                profiling::scope!("vk::cmd_copy_buffer_to_image");
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
                    device.cmd_pipeline_barrier(
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
                    device.cmd_copy_buffer_to_image(
                        command_buffer,
                        staging_buffer.inner,
                        image_allocation.inner,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &[image_copy_region],
                    );
                }
            },
            |device, command_buffer| {
                profiling::scope!("record transfer->shader barrier");
                let barrier_from_transfer_dst_to_shader = vk::ImageMemoryBarrier::builder()
                    .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .src_queue_family_index(transfer_queue_family)
                    .dst_queue_family_index(graphics_queue_family)
                    .image(image_allocation.inner)
                    .subresource_range(subresource_range)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ)
                    .build();
                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[barrier_from_transfer_dst_to_shader],
                    );
                }
            },
        )
        .map_err(ImageLoadingError::StartTextureUpload)?;

    let image_view = {
        let image_view_create_info = vk::ImageViewCreateInfo::builder()
            .image(image_allocation.inner)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(subresource_range);

        let image_view =
            unsafe { device.create_image_view(&image_view_create_info, None) }.map_err(ImageLoadingError::ImageViewCreation)?;
        ImageView {
            inner: image_view,
            device: device.clone(),
            image: Rc::new(image_allocation.into()),
        }
    };
    debug_utils::name_vulkan_object(device, image_view.inner, format_args!("{}", debug_identifier));

    Ok(image_view)
}

pub fn get_ntex_create_info(bytes: &[u8], kind: TextureKind) -> Result<vk::ImageCreateInfo, ImageLoadingError> {
    let ntex::NtexData {
        width,
        height,
        format,
        mip_ranges,
        ..
    } = ntex::decode(bytes).map_err(ImageLoadingError::Ntex)?;
    let format = kind.convert_format(format);
    let extent = *vk::Extent3D::builder().width(width).height(height).depth(1);
    Ok(get_image_create_info(extent, mip_ranges.len() as u32, format))
}

#[profiling::function]
pub fn load_ntex(
    device: &Device,
    uploader: &mut Uploader,
    arena: &mut VulkanArena<ForImages>,
    bytes: &[u8],
    kind: TextureKind,
    debug_identifier: &str,
) -> Result<ImageView, ImageLoadingError> {
    let ntex::NtexData {
        width,
        height,
        format,
        pixels,
        mip_ranges,
    } = ntex::decode(bytes).map_err(ImageLoadingError::Ntex)?;
    let format = kind.convert_format(format);
    let extent = *vk::Extent3D::builder().width(width).height(height).depth(1);
    let &mut Uploader {
        graphics_queue_family,
        transfer_queue_family,
        ..
    } = uploader;

    let staging_buffer = {
        profiling::scope!("allocate and create staging buffer");
        let buffer_size = pixels.len() as vk::DeviceSize;
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        uploader
            .staging_arena
            .create_buffer(
                *buffer_create_info,
                pixels,
                None,
                format_args!("staging buffer for {}", debug_identifier),
            )
            .map_err(ImageLoadingError::StagingBufferCreation)?
    };

    let image_allocation = {
        profiling::scope!("allocate gpu texture");
        let image_info = get_image_create_info(extent, mip_ranges.len() as u32, format);
        arena
            .create_image(image_info, format_args!("{}", debug_identifier))
            .map_err(ImageLoadingError::ImageCreation)?
    };

    uploader
        .start_upload(
            staging_buffer,
            format_args!("{}", debug_identifier),
            |device, staging_buffer, command_buffer| {
                let mut current_mip_level_extent = extent;
                for (mip_level, mip_range) in mip_ranges.iter().enumerate() {
                    profiling::scope!("vk::cmd_copy_buffer_to_image");
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
                        device.cmd_pipeline_barrier(
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
                        device.cmd_copy_buffer_to_image(
                            command_buffer,
                            staging_buffer.inner,
                            image_allocation.inner,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &[image_copy_region],
                        );
                    }
                    current_mip_level_extent.width = (current_mip_level_extent.width / 2).max(1);
                    current_mip_level_extent.height = (current_mip_level_extent.height / 2).max(1);
                    current_mip_level_extent.depth = (current_mip_level_extent.depth / 2).max(1);
                }
            },
            |device, command_buffer| {
                for mip_level in 0..mip_ranges.len() {
                    profiling::scope!("record transfer->shader barrier");
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
                        .src_queue_family_index(transfer_queue_family)
                        .dst_queue_family_index(graphics_queue_family)
                        .image(image_allocation.inner)
                        .subresource_range(subresource_range)
                        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .build();
                    unsafe {
                        device.cmd_pipeline_barrier(
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
            },
        )
        .map_err(ImageLoadingError::StartTextureUpload)?;

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

        let image_view =
            unsafe { device.create_image_view(&image_view_create_info, None) }.map_err(ImageLoadingError::ImageViewCreation)?;
        ImageView {
            inner: image_view,
            device: device.clone(),
            image: Rc::new(image_allocation.into()),
        }
    };
    debug_utils::name_vulkan_object(device, image_view.inner, format_args!("{}", debug_identifier));

    Ok(image_view)
}
