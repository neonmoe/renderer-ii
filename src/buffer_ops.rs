use crate::{Error, FrameIndex, Gpu};
use ash::version::DeviceV1_0;
use ash::vk;
use ash::Device;
use std::ops::Range;
use std::{mem, ptr};

#[profiling::function]
pub(crate) fn copy_to_allocation<T>(
    src: &[T],
    allocator: &vk_mem::Allocator,
    allocation: &vk_mem::Allocation,
    alloc_info: &vk_mem::AllocationInfo,
) -> Result<(), Error> {
    let dst_ptr = alloc_info.get_mapped_data();
    let size = src.len() * mem::size_of::<T>();
    let src_ptr = src.as_ptr() as *const u8;
    unsafe { ptr::copy_nonoverlapping(src_ptr, dst_ptr, size) };
    allocator
        .flush_allocation(allocation, 0, vk::WHOLE_SIZE as usize)
        .map_err(Error::VmaFlushAllocation)?;
    Ok(())
}

#[profiling::function]
pub(crate) fn start_buffer_upload(
    gpu: &Gpu,
    frame_index: FrameIndex,
    buffer: vk::Buffer,
    buffer_size: vk::DeviceSize,
) -> Result<(vk::Fence, vk::Buffer, vk_mem::Allocation), Error> {
    let (upload_fence, (buffer, allocation)) = gpu.run_command_buffer(frame_index, vk::PipelineStageFlags::VERTEX_INPUT, |cb| {
        queue_buffer_copy(&gpu.device, cb, &gpu.allocator, buffer, buffer_size)
    })?;
    Ok((upload_fence, buffer, allocation))
}

#[profiling::function]
pub(crate) fn start_image_upload(
    gpu: &Gpu,
    frame_index: FrameIndex,
    pixels: vk::Buffer,
    mipmap_ranges: Option<&[Range<usize>]>,
    extent: vk::Extent3D,
    format: vk::Format,
) -> Result<(vk::Fence, vk::Image, vk_mem::Allocation, u32), Error> {
    let (upload_fence, (image, allocation, mip_levels)) =
        gpu.run_command_buffer(frame_index, vk::PipelineStageFlags::FRAGMENT_SHADER, |cb| {
            if let Some(mipmap_ranges) = mipmap_ranges {
                queue_image_copy_with_explicit_mipmaps(&gpu.device, cb, &gpu.allocator, pixels, mipmap_ranges, extent, format)
            } else {
                queue_image_copy_with_generated_mipmaps(&gpu.device, cb, &gpu.allocator, pixels, extent, format)
            }
        })?;

    Ok((upload_fence, image, allocation, mip_levels))
}

#[profiling::function]
fn queue_buffer_copy(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    allocator: &vk_mem::Allocator,
    src: vk::Buffer,
    buffer_size: vk::DeviceSize,
) -> Result<(vk::Buffer, vk_mem::Allocation), Error> {
    let (buffer, allocation, _) = {
        profiling::scope!("allocate gpu buffer");
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(buffer_size)
            .usage(
                // Just prepare for everything for simplicity.
                vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::VERTEX_BUFFER
                    | vk::BufferUsageFlags::INDEX_BUFFER
                    | vk::BufferUsageFlags::UNIFORM_BUFFER,
            )
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ..Default::default()
        };
        allocator
            .create_buffer(&buffer_create_info, &allocation_create_info)
            .map_err(Error::VmaBufferAllocation)?
    };

    let copy_regions = [vk::BufferCopy::builder().size(buffer_size).build()];
    unsafe { device.cmd_copy_buffer(command_buffer, src, buffer, &copy_regions) };
    Ok((buffer, allocation))
}

#[profiling::function]
fn queue_image_copy_with_generated_mipmaps(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    allocator: &vk_mem::Allocator,
    pixel_buffer: vk::Buffer,
    mut extent: vk::Extent3D,
    format: vk::Format,
) -> Result<(vk::Image, vk_mem::Allocation, u32), Error> {
    let mip_levels = ((extent.width.max(extent.height) as f32).log2()) as u32 + 1;
    let (image, allocation, _) = {
        profiling::scope!("allocate gpu texture");
        let image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(extent)
            .mip_levels(mip_levels)
            .array_layers(1)
            .format(format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(vk::SampleCountFlags::TYPE_1);
        let allocation_info = vk_mem::AllocationCreateInfo {
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ..Default::default()
        };
        allocator
            .create_image(&image_info, &allocation_info)
            .map_err(Error::VmaImageAllocation)?
    };

    for mip_level in 0..mip_levels {
        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(mip_level)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
            .build();

        let barrier_from_undefined_to_transfer_dst = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
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
            .mip_level(mip_level)
            .base_array_layer(0)
            .layer_count(1)
            .build();

        if mip_level == 0 {
            let image_copy_region = vk::BufferImageCopy::builder()
                .buffer_offset(0)
                .buffer_row_length(extent.width)
                .buffer_image_height(extent.height)
                .image_subresource(subresource_layers_dst)
                .image_extent(extent)
                .build();
            unsafe {
                device.cmd_copy_buffer_to_image(
                    command_buffer,
                    pixel_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[image_copy_region],
                );
            }
        } else {
            let subresource_layers_src = vk::ImageSubresourceLayers::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(mip_level - 1)
                .base_array_layer(0)
                .layer_count(1)
                .build();

            let src_offsets = [
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: extent.width as i32,
                    y: extent.height as i32,
                    z: extent.depth as i32,
                },
            ];

            extent.width = (extent.width / 2).max(1);
            extent.height = (extent.height / 2).max(1);
            extent.depth = (extent.depth / 2).max(1);

            let dst_offsets = [
                vk::Offset3D::default(),
                vk::Offset3D {
                    x: extent.width as i32,
                    y: extent.height as i32,
                    z: extent.depth as i32,
                },
            ];

            let blit_previous_mip_to_current = vk::ImageBlit::builder()
                .src_subresource(subresource_layers_src)
                .src_offsets(src_offsets)
                .dst_subresource(subresource_layers_dst)
                .dst_offsets(dst_offsets)
                .build();
            let blits = [blit_previous_mip_to_current];
            unsafe {
                device.cmd_blit_image(
                    command_buffer,
                    image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &blits,
                    vk::Filter::LINEAR,
                );
            }
        }

        // Prepare to be a source of the blit operation on the next iteration
        let barrier_from_transfer_dst_to_transfer_src = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(subresource_range)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .build();
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier_from_transfer_dst_to_transfer_src],
            );
        }
    }

    for mip_level in 0..mip_levels {
        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(mip_level)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
            .build();
        let barrier_from_transfer_src_to_shader = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .subresource_range(subresource_range)
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
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
                &[barrier_from_transfer_src_to_shader],
            );
        }
    }

    Ok((image, allocation, mip_levels))
}

#[profiling::function]
fn queue_image_copy_with_explicit_mipmaps(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    allocator: &vk_mem::Allocator,
    pixel_buffer: vk::Buffer,
    mip_ranges: &[Range<usize>],
    mut extent: vk::Extent3D,
    format: vk::Format,
) -> Result<(vk::Image, vk_mem::Allocation, u32), Error> {
    let (image, allocation, _) = {
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
        let allocation_info = vk_mem::AllocationCreateInfo {
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ..Default::default()
        };
        allocator
            .create_image(&image_info, &allocation_info)
            .map_err(Error::VmaImageAllocation)?
    };

    for (mip_level, mip_range) in mip_ranges.iter().enumerate() {
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
            .image(image)
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
            .buffer_row_length(extent.width.max(texel_size))
            .buffer_image_height(extent.height.max(texel_size))
            .image_subresource(subresource_layers_dst)
            .image_extent(extent)
            .build();
        unsafe {
            device.cmd_copy_buffer_to_image(
                command_buffer,
                pixel_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[image_copy_region],
            );
        }

        extent.width = (extent.width / 2).max(1);
        extent.height = (extent.height / 2).max(1);
        extent.depth = (extent.depth / 2).max(1);

        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(mip_level as u32)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
            .build();
        let barrier_from_transfer_src_to_shader = vk::ImageMemoryBarrier::builder()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
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
                &[barrier_from_transfer_src_to_shader],
            );
        }
    }

    Ok((image, allocation, mip_ranges.len() as u32))
}
