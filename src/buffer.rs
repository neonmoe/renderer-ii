use crate::buffer_ops::{self, BufferUpload};
use crate::{Error, FrameIndex, Gpu};
use ash::vk;
use std::hash::{Hash, Hasher};
use std::mem;

pub struct Buffer<'a> {
    /// Held by [Buffer] to be able to destroy resources on drop.
    pub gpu: &'a Gpu<'a>,

    buffer: vk::Buffer,
    buffer_size: vk::DeviceSize,
    allocation: vk_mem::Allocation,
    alloc_info: vk_mem::AllocationInfo,
    pub(crate) editable: bool,
}

impl PartialEq for Buffer<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.buffer == other.buffer
    }
}

impl Eq for Buffer<'_> {}

impl Hash for Buffer<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.buffer.hash(state);
    }
}

impl Drop for Buffer<'_> {
    #[profiling::function]
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
    #[profiling::function]
    pub fn new<'a, T>(
        gpu: &'a Gpu<'_>,
        frame_index: FrameIndex,
        data: &[T],
        editable: bool,
    ) -> Result<Buffer<'a>, Error> {
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
        buffer_ops::copy_to_allocation(data, gpu, &allocation, &alloc_info)?;
        let mut buffer = Buffer {
            gpu,
            buffer: vk_buffer,
            buffer_size,
            allocation,
            alloc_info,
            editable,
        };

        if !editable {
            let pool = gpu.main_gpu_buffer_pool.clone();
            let (
                gpu_buffer,
                gpu_allocation,
                gpu_alloc_info,
                upload_cmdbuf,
                finished_upload,
                wait_stage,
            ) = buffer_ops::start_buffer_upload(gpu, pool, vk_buffer, buffer_size)?;

            gpu.add_buffer_upload(
                frame_index,
                BufferUpload {
                    finished_upload,
                    wait_stage,
                    upload_cmdbuf,
                    staging_buffer: Some(buffer.buffer),
                    staging_allocation: Some(buffer.allocation),
                },
            );

            buffer.buffer = gpu_buffer;
            buffer.allocation = gpu_allocation;
            buffer.alloc_info = gpu_alloc_info;
        }

        Ok(buffer)
    }

    #[profiling::function]
    pub fn update_data<T>(&self, gpu: &Gpu, new_data: &[T]) -> Result<(), Error> {
        if !self.editable {
            Err(Error::BufferNotEditable)
        } else {
            buffer_ops::copy_to_allocation(new_data, gpu, &self.allocation, &self.alloc_info)
        }
    }

    /// Returns a [vk::Buffer] which contains the contents of this
    /// [Buffer].
    ///
    /// If self is editable, a temporary copy will be
    /// returned. [Gpu::wait_buffer_uploads] needs to be called before
    /// the returned buffer can be used for rendering. If not called
    /// manually, it's called right before rendering.
    #[profiling::function]
    pub fn buffer(&self, frame_index: FrameIndex) -> Result<vk::Buffer, Error> {
        if self.editable {
            let pool_index = self.gpu.frame_mod(frame_index);
            let temp_pool = self.gpu.temp_gpu_buffer_pools[pool_index].clone();
            let (temp_buffer, temp_alloc, _, upload_cmdbuf, finished_upload, wait_stage) =
                buffer_ops::start_buffer_upload(
                    &self.gpu,
                    temp_pool,
                    self.buffer,
                    self.buffer_size,
                )?;
            self.gpu.add_buffer_upload(
                frame_index,
                BufferUpload {
                    finished_upload,
                    wait_stage,
                    upload_cmdbuf,
                    staging_buffer: None,
                    staging_allocation: None,
                },
            );
            self.gpu
                .add_temporary_buffer(frame_index, temp_buffer, temp_alloc);
            Ok(temp_buffer)
        } else {
            Ok(self.buffer)
        }
    }
}
