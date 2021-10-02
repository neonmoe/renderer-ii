use crate::buffer_ops;
use crate::resources::AllocatedBuffer;
use crate::{Error, FrameIndex, Gpu};
use ash::vk;
use std::hash::{Hash, Hasher};
use std::mem;

pub struct Buffer {
    pub(crate) buffer: vk::Buffer,
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.buffer == other.buffer
    }
}

impl Eq for Buffer {}

impl Hash for Buffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.buffer.hash(state);
    }
}

impl Buffer {
    /// Creates a new buffer. Ensure that the vertices match the
    /// pipeline. If not `editable`, call [Gpu::wait_buffer_uploads]
    /// after your buffer creation code, before they're rendered.
    ///
    /// Currently the buffers are always created as INDEX | VERTEX |
    /// UNIFORM buffers.
    #[profiling::function]
    pub fn new<T>(gpu: &Gpu, frame_index: FrameIndex, data: &[T]) -> Result<Buffer, Error> {
        let buffer_size = (data.len() * mem::size_of::<T>()) as vk::DeviceSize;
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE,
            ..Default::default()
        };
        let (staging_buffer, staging_allocation, staging_alloc_info) = gpu
            .allocator
            .create_buffer(&buffer_create_info, &allocation_create_info)
            .map_err(Error::VmaBufferAllocation)?;
        buffer_ops::copy_to_allocation(data, &gpu.allocator, &staging_allocation, &staging_alloc_info)?;

        let (upload_fence, buffer, allocation) = match buffer_ops::start_buffer_upload(gpu, frame_index, staging_buffer, buffer_size) {
            Ok(result) => result,
            Err(err) => {
                let _ = gpu.allocator.destroy_buffer(staging_buffer, &staging_allocation);
                return Err(err);
            }
        };

        gpu.resources.add_buffer(
            upload_fence,
            Some(AllocatedBuffer(staging_buffer, staging_allocation)),
            AllocatedBuffer(buffer, allocation),
        );

        Ok(Buffer { buffer })
    }
}
