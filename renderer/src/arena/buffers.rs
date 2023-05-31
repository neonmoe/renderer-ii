use crate::arena::{ArenaType, VulkanArena, VulkanArenaError};
use crate::uploader::Uploader;
use crate::vulkan_raii::Buffer;
use ash::vk;
use core::fmt::Arguments;
use core::ptr;
use core::sync::atomic::Ordering;

pub struct ForBuffers;
impl ArenaType for ForBuffers {
    const MAPPABLE: bool = true;
}

impl VulkanArena<ForBuffers> {
    /// Stores the buffer reference in this arena, to be destroyed when the
    /// arena is reset. Ideal for temporary arenas whose buffers just have to
    /// live until the arena is reset.
    pub fn add_buffer(&mut self, buffer: Buffer) {
        self.pinned_buffers.push(buffer);
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
        profiling::scope!("vulkan buffer creation");
        let buffer = unsafe { self.device.create_buffer(&buffer_create_info, None) }.map_err(VulkanArenaError::BufferCreation)?;
        let buffer_memory_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let alignment = buffer_memory_requirements.alignment;

        let offset = self.offset.next_multiple_of(alignment);
        let required_size = buffer_memory_requirements.size;

        if self.total_size - offset < required_size {
            unsafe { self.device.destroy_buffer(buffer, None) };
            return Err(VulkanArenaError::OutOfMemory {
                identifier: self.debug_identifier.clone(),
                used: offset,
                total: self.total_size,
                required: required_size,
            });
        }

        match unsafe { self.device.bind_buffer_memory(buffer, self.memory.inner, offset) }.map_err(VulkanArenaError::BufferBinding) {
            Ok(_) => {}
            Err(err) => {
                unsafe { self.device.destroy_buffer(buffer, None) };
                return Err(err);
            }
        }
        crate::name_vulkan_object(&self.device, buffer, name);

        if self.mapped_memory_ptr.is_null() {
            if let (Some(uploader), Some(staging_arena)) = (uploader, staging_arena) {
                profiling::scope!("staging buffer creation");
                let staging_info = vk::BufferCreateInfo::default()
                    .size(buffer_create_info.size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE);
                let staging_buffer =
                    staging_arena.create_buffer(staging_info, src, None, None, format_args!("staging buffer for {name}"))?;
                let &mut Uploader {
                    graphics_queue_family,
                    transfer_queue_family,
                    ..
                } = uploader;

                uploader
                    .start_upload(
                        staging_buffer,
                        name,
                        |device, staging_buffer, command_buffer| {
                            profiling::scope!("record buffer copy cmd from staging");
                            let barrier_from_graphics_to_transfer = [vk::BufferMemoryBarrier2::default()
                                .buffer(buffer)
                                .offset(0)
                                .size(vk::WHOLE_SIZE)
                                .src_queue_family_index(graphics_queue_family)
                                .dst_queue_family_index(transfer_queue_family)
                                .src_access_mask(vk::AccessFlags2::NONE)
                                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                                .dst_stage_mask(vk::PipelineStageFlags2::COPY)];
                            let dep_info = vk::DependencyInfo::default().buffer_memory_barriers(&barrier_from_graphics_to_transfer);
                            unsafe { device.sync2.cmd_pipeline_barrier2(command_buffer, &dep_info) };
                            let (src, dst) = (staging_buffer.inner, buffer);
                            let copy_regions = [vk::BufferCopy::default().size(buffer_create_info.size)];
                            unsafe { device.cmd_copy_buffer(command_buffer, src, dst, &copy_regions) };
                        },
                        |device, command_buffer| {
                            profiling::scope!("vk::cmd_pipeline_barrier");
                            let barrier_from_transfer_to_graphics = [vk::BufferMemoryBarrier2::default()
                                .buffer(buffer)
                                .offset(0)
                                .size(vk::WHOLE_SIZE)
                                .src_queue_family_index(transfer_queue_family)
                                .dst_queue_family_index(graphics_queue_family)
                                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                                .dst_access_mask(vk::AccessFlags2::VERTEX_ATTRIBUTE_READ)
                                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                                .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_INPUT)];
                            let dep_info = vk::DependencyInfo::default().buffer_memory_barriers(&barrier_from_transfer_to_graphics);
                            unsafe { device.sync2.cmd_pipeline_barrier2(command_buffer, &dep_info) };
                        },
                    )
                    .map_err(VulkanArenaError::Upload)?;
            } else {
                return Err(VulkanArenaError::NotWritable);
            }
        } else {
            profiling::scope!("writing buffer data");
            let dst = unsafe { self.mapped_memory_ptr.offset(offset as isize) };
            unsafe { ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        }

        let new_offset = offset + required_size;
        if self.device_local {
            crate::vram_usage::IN_USE.fetch_add(new_offset - self.offset, Ordering::Relaxed);
        }
        self.offset = new_offset;

        Ok(Buffer {
            inner: buffer,
            device: self.device.clone(),
            memory: self.memory.clone(),
            size: src.len() as vk::DeviceSize,
        })
    }
}
