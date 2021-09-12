use crate::buffer_ops;
use crate::{Error, FrameIndex, Gpu};
use ash::vk;
use std::hash::{Hash, Hasher};
use std::mem;

pub struct Buffer<'a> {
    /// Held by [Buffer] to be able to destroy resources on drop.
    pub gpu: &'a Gpu<'a>,

    pub buffer: vk::Buffer,
    allocation: vk_mem::Allocation,
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
        let _ = self.gpu.allocator.destroy_buffer(self.buffer, &self.allocation);
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
    pub fn new<'a, T>(gpu: &'a Gpu<'_>, frame_index: FrameIndex, data: &[T]) -> Result<Buffer<'a>, Error> {
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
        let (staging_buffer, staging_allocation, staging_alloc_info) = gpu
            .allocator
            .create_buffer(&buffer_create_info, &allocation_create_info)
            .map_err(Error::VmaBufferAllocation)?;
        buffer_ops::copy_to_allocation(data, gpu, &staging_allocation, &staging_alloc_info)?;

        let pool = gpu.main_gpu_buffer_pool.clone();
        let (buffer, allocation) = match buffer_ops::start_buffer_upload(gpu, frame_index, pool, staging_buffer, buffer_size) {
            Ok(result) => result,
            Err(err) => {
                let _ = gpu.allocator.destroy_buffer(staging_buffer, &staging_allocation);
                return Err(err);
            }
        };

        gpu.add_temporary_buffer(frame_index, staging_buffer, staging_allocation);

        Ok(Buffer { gpu, buffer, allocation })
    }
}
