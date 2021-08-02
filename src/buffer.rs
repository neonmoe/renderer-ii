use crate::{Error, Gpu};
use ash::version::DeviceV1_0;
use ash::vk;
use ash::Device;
use std::{mem, ptr};

/// Contains the fence for the upload queue submit, and everything
/// that needs to be cleaned up after it's done.
pub(crate) struct BufferUpload {
    pub finished_upload: vk::Fence,
    pub upload_cmdbuf: vk::CommandBuffer,
    pub staging_buffer: Option<vk::Buffer>,
    pub staging_allocation: Option<vk_mem::Allocation>,
}

pub struct Buffer<'a> {
    /// Held by [Buffer] to be able to destroy resources on drop.
    pub gpu: &'a Gpu<'a>,

    buffer: vk::Buffer,
    buffer_size: vk::DeviceSize,
    allocation: vk_mem::Allocation,
    alloc_info: vk_mem::AllocationInfo,
    pub(crate) editable: bool,
}

impl Drop for Buffer<'_> {
    fn drop(&mut self) {
        let _ = self
            .gpu
            .allocator
            .destroy_buffer(self.buffer, &self.allocation);
    }
}

impl Buffer<'_> {
    /// Creates a new buffer. Ensure that the vertices match the
    /// pipeline. If not `editable`, call [Gpu::wait_buffer_uploads]
    /// after your buffer creation code, before they're rendered.
    ///
    /// Currently the buffers are always created as INDEX | VERTEX |
    /// UNIFORM buffers.
    pub fn new<'a, T>(gpu: &'a Gpu<'_>, data: &[T], editable: bool) -> Result<Buffer<'a>, Error> {
        let buffer_size = (data.len() * mem::size_of::<T>()) as vk::DeviceSize;
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            pool: Some(gpu.staging_cpu_buffer_pool.clone()),
            ..Default::default()
        };
        let (vk_buffer, allocation, alloc_info) = gpu
            .allocator
            .create_buffer(&buffer_create_info, &allocation_create_info)
            .map_err(Error::VmaBufferAllocation)?;
        let buffer_ptr = alloc_info.get_mapped_data();
        copy_buffer_data(data, buffer_ptr);
        gpu.allocator
            .flush_allocation(&allocation, 0, vk::WHOLE_SIZE as usize)
            .map_err(Error::VmaFlushAllocation)?;
        let mut buffer = Buffer {
            gpu,
            buffer: vk_buffer,
            buffer_size,
            allocation,
            alloc_info,
            editable,
        };

        if !editable {
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(gpu.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffers = unsafe {
                gpu.device
                    .allocate_command_buffers(&command_buffer_allocate_info)
                    .map_err(Error::VulkanCommandBuffersAllocation)
            }?;
            let upload_cmdbuf = command_buffers[0];

            let finished_upload = unsafe {
                gpu.device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .map_err(Error::VulkanFenceCreation)
            }?;

            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe {
                gpu.device
                    .begin_command_buffer(upload_cmdbuf, &command_buffer_begin_info)
                    .map_err(Error::VulkanBeginCommandBuffer)
            }?;
            let (gpu_buffer, gpu_allocation, gpu_alloc_info) = copy_vk_buffer(
                &gpu.device,
                upload_cmdbuf,
                &gpu.allocator,
                gpu.main_gpu_buffer_pool.clone(),
                buffer.buffer,
                buffer_size,
            )?;
            unsafe { gpu.device.end_command_buffer(upload_cmdbuf) }
                .map_err(Error::VulkanEndCommandBuffer)?;

            let command_buffers = [upload_cmdbuf];
            let submit_infos = [vk::SubmitInfo::builder()
                .command_buffers(&command_buffers)
                .build()];
            unsafe {
                gpu.device
                    .queue_submit(gpu.graphics_queue, &submit_infos, finished_upload)
                    .map_err(Error::VulkanQueueSubmit)
            }?;

            let buffer_upload = BufferUpload {
                finished_upload,
                upload_cmdbuf,
                staging_buffer: Some(buffer.buffer),
                staging_allocation: Some(buffer.allocation),
            };
            gpu.add_buffer_upload(buffer_upload);

            buffer.buffer = gpu_buffer;
            buffer.allocation = gpu_allocation;
            buffer.alloc_info = gpu_alloc_info;
        }

        Ok(buffer)
    }

    pub fn update_data<T>(&self, gpu: &Gpu, new_data: &[T]) -> Result<(), Error> {
        if !self.editable {
            Err(Error::BufferNotEditable)
        } else {
            copy_buffer_data(new_data, self.alloc_info.get_mapped_data());
            gpu.allocator
                .flush_allocation(&self.allocation, 0, vk::WHOLE_SIZE as usize)
                .map_err(Error::VmaFlushAllocation)?;
            Ok(())
        }
    }

    /// Returns a [vk::Buffer] which contains the contents of this
    /// [Buffer].
    ///
    /// If self is editable, a temporary copy will be
    /// returned. [Gpu::wait_buffer_uploads] needs to be called before
    /// the returned buffer can be used for rendering. If not called
    /// manually, it's called right before rendering.
    pub fn buffer(&self, frame_index: u32) -> Result<vk::Buffer, Error> {
        if self.editable {
            let pool_index = frame_index as usize % self.gpu.temp_gpu_buffer_pools.len();
            let temp_pool = self.gpu.temp_gpu_buffer_pools[pool_index].clone();
            let (temp_buffer, temp_alloc, _, upload_cmdbuf, finished_upload) =
                start_buffer_upload(&self.gpu, temp_pool.clone(), self.buffer, self.buffer_size)?;
            self.gpu.add_buffer_upload(BufferUpload {
                finished_upload,
                upload_cmdbuf,
                staging_buffer: None,
                staging_allocation: None,
            });
            self.gpu
                .add_temporary_buffer(frame_index, temp_buffer, temp_alloc);
            Ok(temp_buffer)
        } else {
            Ok(self.buffer)
        }
    }
}

fn copy_buffer_data<T>(data: &[T], dst_ptr: *mut u8) {
    let size = data.len() * mem::size_of::<T>();
    let data_ptr = data.as_ptr() as *const u8;
    unsafe { ptr::copy_nonoverlapping(data_ptr, dst_ptr, size) };
}

fn start_buffer_upload(
    gpu: &Gpu,
    pool: vk_mem::AllocatorPool,
    buffer: vk::Buffer,
    buffer_size: vk::DeviceSize,
) -> Result<
    (
        vk::Buffer,
        vk_mem::Allocation,
        vk_mem::AllocationInfo,
        vk::CommandBuffer,
        vk::Fence,
    ),
    Error,
> {
    let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(gpu.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let command_buffers = unsafe {
        gpu.device
            .allocate_command_buffers(&command_buffer_allocate_info)
            .map_err(Error::VulkanCommandBuffersAllocation)
    }?;
    let upload_cmdbuf = command_buffers[0];

    let finished_upload = unsafe {
        gpu.device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .map_err(Error::VulkanFenceCreation)
    }?;

    let command_buffer_begin_info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        gpu.device
            .begin_command_buffer(upload_cmdbuf, &command_buffer_begin_info)
            .map_err(Error::VulkanBeginCommandBuffer)
    }?;
    let (buffer, allocation, alloc_info) = copy_vk_buffer(
        &gpu.device,
        upload_cmdbuf,
        &gpu.allocator,
        pool,
        buffer,
        buffer_size,
    )?;
    unsafe { gpu.device.end_command_buffer(upload_cmdbuf) }
        .map_err(Error::VulkanEndCommandBuffer)?;

    let command_buffers = [upload_cmdbuf];
    let submit_infos = [vk::SubmitInfo::builder()
        .command_buffers(&command_buffers)
        .build()];
    unsafe {
        gpu.device
            .queue_submit(gpu.graphics_queue, &submit_infos, finished_upload)
            .map_err(Error::VulkanQueueSubmit)
    }?;

    Ok((
        buffer,
        allocation,
        alloc_info,
        upload_cmdbuf,
        finished_upload,
    ))
}

fn copy_vk_buffer(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    allocator: &vk_mem::Allocator,
    pool: vk_mem::AllocatorPool,
    src: vk::Buffer,
    buffer_size: vk::DeviceSize,
) -> Result<(vk::Buffer, vk_mem::Allocation, vk_mem::AllocationInfo), Error> {
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
    let (buffer, allocation, alloc_info) = allocator
        .create_buffer(&buffer_create_info, &allocation_create_info)
        .map_err(Error::VmaBufferAllocation)?;

    let copy_regions = [vk::BufferCopy::builder().size(buffer_size).build()];
    unsafe { device.cmd_copy_buffer(command_buffer, src, buffer, &copy_regions) };
    Ok((buffer, allocation, alloc_info))
}
