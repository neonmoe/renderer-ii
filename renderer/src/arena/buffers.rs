use core::fmt::Arguments;
use core::sync::atomic::Ordering;

use ash::vk;

use crate::arena::{ArenaType, VulkanArena, VulkanArenaError};
use crate::uploader::Uploader;
use crate::vulkan_raii::Buffer;

pub struct MappedBuffer {
    pub buffer: Buffer,
    /// Safety: as long as the `buffer` exists, the arena can't be reset, and the device memory is
    /// not destroyed, thus there can't be any other pointers pointing at the same memory, and the
    /// memory stays alive.
    ptr: *mut u8,
    len: usize,
}

impl MappedBuffer {
    pub fn data_mut(&mut self) -> &mut [u8] {
        // Safety: as long as this mutable borrow exists, self.buffer can't be dropped, and thus the
        // backing memory is exclusively ours and is not deallocated.
        unsafe { core::slice::from_raw_parts_mut(self.ptr, self.len) }
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

    /// Creates an empty buffer with the given create info, returns a MappedBuffer which allows
    /// writing to the backing data. Errors if called on a non-mapped arena.
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
        Ok(MappedBuffer { buffer, ptr, len })
    }

    /// Copies from the `src` to the `dst` using the `uploader`.
    #[profiling::function]
    pub fn copy_buffer(&mut self, src: Buffer, dst: &Buffer, uploader: &mut Uploader, name: Arguments) {
        let &mut Uploader {
            graphics_queue_family,
            transfer_queue_family,
            ..
        } = uploader;
        let buffer_memory_barrier = vk::BufferMemoryBarrier2::default().buffer(dst.inner).offset(0).size(vk::WHOLE_SIZE);
        uploader.start_upload(
            src,
            name,
            |device, staging_buffer, command_buffer| {
                let (src_buf, dst_buf) = (staging_buffer.inner, dst.inner);
                let copy_regions = [vk::BufferCopy::default().size(dst.size)];
                unsafe { device.cmd_copy_buffer(command_buffer, src_buf, dst_buf, &copy_regions) };
                let release_from_transfer_to_graphics = [buffer_memory_barrier
                    .src_queue_family_index(transfer_queue_family)
                    .dst_queue_family_index(graphics_queue_family)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)];
                let dep_info = vk::DependencyInfo::default().buffer_memory_barriers(&release_from_transfer_to_graphics);
                unsafe { device.sync2.cmd_pipeline_barrier2(command_buffer, &dep_info) };
            },
            |device, command_buffer| {
                let barrier_from_transfer_to_graphics = [buffer_memory_barrier
                    .src_queue_family_index(transfer_queue_family)
                    .dst_queue_family_index(graphics_queue_family)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags2::VERTEX_ATTRIBUTE_READ) // TODO: expose as an arg?
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_INPUT)];
                let dep_info = vk::DependencyInfo::default().buffer_memory_barriers(&barrier_from_transfer_to_graphics);
                unsafe { device.sync2.cmd_pipeline_barrier2(command_buffer, &dep_info) };
            },
        );
    }

    #[profiling::function]
    pub fn create_buffer(
        &mut self,
        buffer_create_info: vk::BufferCreateInfo,
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
                staging_data.copy_from_slice(src);
                self.copy_buffer(staging_buffer.buffer, &dst_buffer, uploader, name);
                Ok(dst_buffer)
            } else {
                return Err(VulkanArenaError::NotWritable);
            }
        } else {
            let mut mapped_buffer = self.create_staging_buffer(buffer_create_info, name)?;
            let buffer_data = mapped_buffer.data_mut();
            buffer_data.copy_from_slice(src);
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

        Ok(Buffer {
            inner: buffer,
            device: self.device.clone(),
            memory: self.memory.clone(),
            offset,
            size: buffer_create_info.size,
        })
    }
}
