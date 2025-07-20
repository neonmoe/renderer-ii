use core::fmt::Arguments;
use core::ptr;
use core::sync::atomic::Ordering;

use ash::vk;

use crate::arena::{ArenaType, VulkanArena, VulkanArenaError};
use crate::uploader::Uploader;
use crate::vulkan_raii::Buffer;

#[derive(Clone, Copy)]
pub struct BufferUsage {
    pub access_mask: vk::AccessFlags2,
    pub stage_mask: vk::PipelineStageFlags2,
}

impl BufferUsage {
    pub const VERTEX: BufferUsage =
        BufferUsage { access_mask: vk::AccessFlags2::VERTEX_ATTRIBUTE_READ, stage_mask: vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT };
    pub const INDEX: BufferUsage =
        BufferUsage { access_mask: vk::AccessFlags2::INDEX_READ, stage_mask: vk::PipelineStageFlags2::INDEX_INPUT };
    pub const UNIFORM: BufferUsage =
        BufferUsage { access_mask: vk::AccessFlags2::UNIFORM_READ, stage_mask: vk::PipelineStageFlags2::ALL_GRAPHICS };
    pub const INDIRECT_DRAW: BufferUsage =
        BufferUsage { access_mask: vk::AccessFlags2::INDIRECT_COMMAND_READ, stage_mask: vk::PipelineStageFlags2::DRAW_INDIRECT };
    pub const COPY_SRC: BufferUsage =
        BufferUsage { access_mask: vk::AccessFlags2::TRANSFER_READ, stage_mask: vk::PipelineStageFlags2::COPY };
}

pub struct MappedBuffer {
    pub buffer: Buffer,
    /// Safety: as long as the `buffer` exists, the arena can't be reset, and the device memory is
    /// not destroyed, thus there can't be any other pointers pointing at the same memory, and the
    /// memory stays alive.
    ptr: *mut [u8],
}

impl MappedBuffer {
    pub fn data_mut(&mut self) -> &mut [u8] {
        // Safety: as long as this mutable borrow exists, self.buffer can't be dropped, and thus the
        // backing memory is exclusively ours and is not deallocated.
        unsafe { &mut *self.ptr }
    }

    /// Returns the internal mapped pointer and the buffer separately. Nothing
    /// happens to the required preconditions however, so the caller must keep
    /// Buffer alive at least as long as the pointer.
    pub unsafe fn split(self) -> (Buffer, *mut [u8]) {
        (self.buffer, self.ptr)
    }
}

pub struct ForBuffers;

impl ArenaType for ForBuffers {
    const MAPPABLE: bool = true;
}

impl VulkanArena<ForBuffers> {
    /// Stores the buffer reference in this arena, to be destroyed when the
    /// arena is reset. Ideal for temporary arenas whose buffers just have to
    /// live until the arena is reset.
    pub fn add_buffer(&mut self, buffer: Buffer) {
        // TODO: This should not exist, temporary buffers should be stored in per-frame-storage, which should be cleaned when they're no longer in use, only after which clearing should be possible
        self.pinned_buffers.push(buffer);
    }

    /// Creates an empty buffer with the given create info, returns a
    /// [`MappedBuffer`] which allows writing to the backing data. Errors if
    /// called on a non-mapped arena.
    #[profiling::function]
    pub fn create_staging_buffer(
        &mut self,
        buffer_create_info: vk::BufferCreateInfo,
        name: Arguments,
    ) -> Result<MappedBuffer, VulkanArenaError> {
        if self.mapped_memory_ptr.is_null() {
            return Err(VulkanArenaError::NotWritable);
        }
        let buffer = self.create_empty_buffer(buffer_create_info, name)?;
        let ptr = unsafe { self.mapped_memory_ptr.offset(buffer.offset as isize) };
        let len = buffer.size as usize;
        Ok(MappedBuffer { buffer, ptr: ptr::slice_from_raw_parts_mut(ptr, len) })
    }

    #[profiling::function]
    pub fn create_buffer(
        &mut self,
        buffer_create_info: vk::BufferCreateInfo,
        usage: BufferUsage,
        src: &[u8],
        staging_arena: Option<&mut VulkanArena<ForBuffers>>,
        uploader: Option<&mut Uploader>,
        name: Arguments,
    ) -> Result<Buffer, VulkanArenaError> {
        if self.mapped_memory_ptr.is_null() {
            if let (Some(staging_arena), Some(uploader)) = (staging_arena, uploader) {
                let dst_buffer = staging_arena.create_empty_buffer(buffer_create_info, format_args!("{name}"))?;
                let staging_create_info = vk::BufferCreateInfo::default()
                    .size(buffer_create_info.size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE);
                let mut staging_buffer =
                    staging_arena.create_staging_buffer(staging_create_info, format_args!("staging buffer for {name}"))?;
                let staging_data = staging_buffer.data_mut();
                {
                    profiling::scope!("copying data to vulkan staging buffer");
                    staging_data.copy_from_slice(src);
                }
                uploader.copy_buffer(usage, staging_buffer.buffer, &dst_buffer, name);
                Ok(dst_buffer)
            } else {
                return Err(VulkanArenaError::NotWritable);
            }
        } else {
            let mut mapped_buffer = self.create_staging_buffer(buffer_create_info, name)?;
            let buffer_data = mapped_buffer.data_mut();
            {
                profiling::scope!("copying data to vulkan buffer");
                buffer_data.copy_from_slice(src);
            }
            Ok(mapped_buffer.buffer)
        }
    }

    pub fn create_empty_buffer(&mut self, buffer_create_info: vk::BufferCreateInfo, name: Arguments) -> Result<Buffer, VulkanArenaError> {
        profiling::scope!("vulkan buffer creation");
        let buffer = unsafe { self.device.create_buffer(&buffer_create_info, None) }
            .expect("system should have enough memory to allocate vulkan buffers");
        let buffer_memory_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let alignment = buffer_memory_requirements.alignment;

        let offset = self.offset.next_multiple_of(alignment);
        let required_size = buffer_memory_requirements.size;

        if self.total_size - offset < required_size {
            unsafe { self.device.destroy_buffer(buffer, None) };
            return Err(VulkanArenaError::OutOfMemory {
                identifier: self.debug_identifier.clone(),
                used: crate::Bytes(offset),
                total: crate::Bytes(self.total_size),
                required: crate::Bytes(required_size),
            });
        }

        match unsafe { self.device.bind_buffer_memory(buffer, self.memory.inner, offset) } {
            Ok(()) => {}
            Err(err) => {
                unsafe { self.device.destroy_buffer(buffer, None) };
                panic!("vulkan buffer memory binding should not fail: {err}");
            }
        }

        crate::name_vulkan_object(&self.device, buffer, name);

        let new_offset = offset + required_size;
        if self.device_local {
            crate::vram_usage::IN_USE.fetch_add(new_offset - self.offset, Ordering::Relaxed);
        }
        self.offset = new_offset;

        Ok(Buffer { inner: buffer, device: self.device.clone(), memory: self.memory.clone(), offset, size: buffer_create_info.size })
    }
}
