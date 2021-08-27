use crate::{Error, FrameIndex, Gpu};
use ash::version::DeviceV1_0;
use ash::vk;
use ash::Device;
use std::{mem, ptr};

#[profiling::function]
pub(crate) fn copy_to_allocation<T>(
    src: &[T],
    gpu: &Gpu,
    allocation: &vk_mem::Allocation,
    alloc_info: &vk_mem::AllocationInfo,
) -> Result<(), Error> {
    let dst_ptr = alloc_info.get_mapped_data();
    let size = src.len() * mem::size_of::<T>();
    let src_ptr = src.as_ptr() as *const u8;
    unsafe { ptr::copy_nonoverlapping(src_ptr, dst_ptr, size) };
    gpu.allocator
        .flush_allocation(allocation, 0, vk::WHOLE_SIZE as usize)
        .map_err(Error::VmaFlushAllocation)?;
    Ok(())
}

#[profiling::function]
pub(crate) fn start_buffer_upload(
    gpu: &Gpu,
    frame_index: FrameIndex,
    pool: vk_mem::AllocatorPool,
    buffer: vk::Buffer,
    buffer_size: vk::DeviceSize,
) -> Result<(vk::Buffer, vk_mem::Allocation), Error> {
    let (buffer, allocation) =
        gpu.run_command_buffer(frame_index, vk::PipelineStageFlags::VERTEX_INPUT, |cb| {
            queue_buffer_copy(&gpu.device, cb, &gpu.allocator, pool, buffer, buffer_size)
        })?;
    Ok((buffer, allocation))
}

#[profiling::function]
pub(crate) fn start_image_upload(
    gpu: &Gpu,
    frame_index: FrameIndex,
    pool: vk_mem::AllocatorPool,
    pixels: vk::Buffer,
    extent: vk::Extent3D,
    format: vk::Format,
) -> Result<(vk::Image, vk_mem::Allocation), Error> {
    let (image, allocation) =
        gpu.run_command_buffer(frame_index, vk::PipelineStageFlags::FRAGMENT_SHADER, |cb| {
            queue_image_copy(
                &gpu.device,
                cb,
                &gpu.allocator,
                pool,
                pixels,
                extent,
                format,
            )
        })?;

    Ok((image, allocation))
}

#[profiling::function]
fn queue_buffer_copy(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    allocator: &vk_mem::Allocator,
    pool: vk_mem::AllocatorPool,
    src: vk::Buffer,
    buffer_size: vk::DeviceSize,
) -> Result<(vk::Buffer, vk_mem::Allocation), Error> {
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
        pool: Some(pool),
        ..Default::default()
    };
    let (buffer, allocation, _) = {
        profiling::scope!("allocate vulkan buffer");
        allocator
            .create_buffer(&buffer_create_info, &allocation_create_info)
            .map_err(Error::VmaBufferAllocation)?
    };

    let copy_regions = [vk::BufferCopy::builder().size(buffer_size).build()];
    unsafe { device.cmd_copy_buffer(command_buffer, src, buffer, &copy_regions) };
    Ok((buffer, allocation))
}

#[profiling::function]
fn queue_image_copy(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    allocator: &vk_mem::Allocator,
    pool: vk_mem::AllocatorPool,
    pixel_buffer: vk::Buffer,
    extent: vk::Extent3D,
    format: vk::Format,
) -> Result<(vk::Image, vk_mem::Allocation), Error> {
    let image_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::TYPE_2D)
        .extent(extent)
        .mip_levels(1)
        .array_layers(1)
        .format(format)
        .tiling(vk::ImageTiling::OPTIMAL)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(vk::SampleCountFlags::TYPE_1);
    let allocation_info = vk_mem::AllocationCreateInfo {
        pool: Some(pool),
        ..Default::default()
    };
    let (image, allocation, _) = allocator
        .create_image(&image_info, &allocation_info)
        .map_err(Error::VmaImageAllocation)?;

    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1)
        .build();
    let barrier_from_undefined_to_transfer = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .build();
    let barrier_from_transfer_to_shader = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource_range)
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .build();

    let subresource_layers = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1)
        .build();
    let image_copy_region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(extent.width)
        .buffer_image_height(extent.height)
        .image_subresource(subresource_layers)
        .image_extent(extent)
        .build();

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier_from_undefined_to_transfer],
        );
        device.cmd_copy_buffer_to_image(
            command_buffer,
            pixel_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[image_copy_region],
        );
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[barrier_from_transfer_to_shader],
        );
    }

    Ok((image, allocation))
}
