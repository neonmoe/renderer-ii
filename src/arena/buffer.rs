use super::*;
use ash::version::DeviceV1_0;
use ash::vk;
use ash::Device;
use std::ptr;

/// A Vulkan buffer allocated from an [Arena].
#[derive(PartialEq, Eq, Hash)]
pub struct BufferAllocation {
    pub buffer: vk::Buffer,
    /// Offset from the backing memory.
    pub(crate) offset: vk::DeviceSize,
    pub(crate) size: vk::DeviceSize,
}

pub struct WritableBufferAllocation<'a> {
    pub(crate) device: &'a Device,
    pub(crate) memory: vk::DeviceMemory,
    pub(crate) memory_size: vk::DeviceSize,
    pub(crate) non_coherent_atom_size: vk::DeviceSize,
    pub(crate) buffer_allocation: BufferAllocation,
}

impl WritableBufferAllocation<'_> {
    /// Write `size` bytes from `src` into the buffer, `offset` bytes
    /// after the start of the buffer. Both `offset` and `size` are
    /// clamped to the buffer's bounds, so if `offset == 0` and `size
    /// == VK_WHOLE_SIZE`, the entire buffer will be written, reading
    /// from `src`.
    pub unsafe fn write(&self, src: *const u8, offset: vk::DeviceSize, size: vk::DeviceSize) -> Result<(), Error> {
        let offset = (self.buffer_allocation.offset + offset).min(self.buffer_allocation.offset + self.buffer_allocation.size);
        let aligned_offset = align_down(offset, self.non_coherent_atom_size);
        let size = size.min(self.buffer_allocation.size);
        let aligned_size = align_up(size, self.non_coherent_atom_size).min(self.memory_size - aligned_offset);

        // NOTE: The mapped range is aligned to satisfy vkMappedMemoryRange requirements later.
        let dst = self
            .device
            .map_memory(self.memory, aligned_offset, aligned_size, vk::MemoryMapFlags::empty())
            .map_err(Error::VulkanMapMemory)? as *mut u8;
        let dst = dst.offset((offset - aligned_offset) as isize);
        ptr::copy_nonoverlapping(src, dst, size as usize);

        // NOTE: VkMappedMemoryRanges have notable alignment requirements, and they have been taken into account.
        let ranges = [vk::MappedMemoryRange::builder()
            .memory(self.memory)
            .offset(aligned_offset)
            .size(aligned_size)
            .build()];
        self.device.flush_mapped_memory_ranges(&ranges).map_err(Error::VulkanFlushMapped)?;
        self.device.unmap_memory(self.memory);

        Ok(())
    }
}
