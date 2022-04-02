//! An arena allocator for managing GPU memory.
use crate::debug_utils;
use crate::error::Error;
use crate::vulkan_raii::{Buffer, Device, DeviceMemory, Image};
use crate::Uploader;
use ash::vk;
use ash::Instance;
use std::fmt::Arguments;
use std::ptr;
use std::rc::Rc;

pub struct VulkanArena {
    device: Rc<Device>,
    memory: Rc<DeviceMemory>,
    mapped_memory_ptr: *mut u8,
    total_size: vk::DeviceSize,
    /// The location where the available memory starts. Gets set to 0
    /// when the arena is reset, bumped when allocating.
    ///
    /// NOTE: Cells are not Sync, so all accesses to the offset are
    /// single-threaded. This is why it's safe to get() and replace()
    /// instead of using atomic operations. At least currently, Arenas
    /// are per-thread. A possible multi-threaded arena should use an
    /// atomic int here.
    offset: vk::DeviceSize,
    buffer_image_granularity: vk::DeviceSize,
    previous_allocation_was_image: bool,
    pinned_buffers: Vec<Buffer>,
    debug_identifier: String,
}

impl Drop for VulkanArena {
    fn drop(&mut self) {
        if !self.mapped_memory_ptr.is_null() {
            unsafe { self.device.unmap_memory(self.memory.inner) };
        }
    }
}

impl VulkanArena {
    pub fn new(
        instance: &Instance,
        device: &Rc<Device>,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
        optimal_flags: vk::MemoryPropertyFlags,
        fallback_flags: vk::MemoryPropertyFlags,
        debug_identifier_args: Arguments,
    ) -> Result<VulkanArena, Error> {
        let debug_identifier = format!("{}", debug_identifier_args);
        let (memory_type_index, memory_flags) = get_memory_type_index(instance, physical_device, optimal_flags, fallback_flags, size)
            .ok_or_else(|| Error::VulkanNoMatchingHeap(debug_identifier.clone(), fallback_flags))?;
        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(memory_type_index);
        let memory = unsafe { device.allocate_memory(&alloc_info, None) }
            .map_err(|err| Error::VulkanAllocate(err, debug_identifier.clone(), size))?;
        debug_utils::name_vulkan_object(device, memory, debug_identifier_args);

        let mapped_memory_ptr = if memory_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT) {
            unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) }.map_err(Error::VulkanMapMemory)? as *mut u8
        } else {
            ptr::null_mut()
        };

        let physical_device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let buffer_image_granularity = physical_device_properties.limits.buffer_image_granularity;

        Ok(VulkanArena {
            device: device.clone(),
            memory: Rc::new(DeviceMemory {
                inner: memory,
                device: device.clone(),
            }),
            mapped_memory_ptr,
            total_size: size,
            offset: 0,
            buffer_image_granularity,
            previous_allocation_was_image: false,
            pinned_buffers: Vec::new(),
            debug_identifier,
        })
    }

    #[profiling::function]
    pub fn create_buffer(
        &mut self,
        buffer_create_info: vk::BufferCreateInfo,
        src: &[u8],
        uploader: Option<&mut Uploader>,
        name: Arguments,
    ) -> Result<Buffer, Error> {
        let buffer = unsafe { self.device.create_buffer(&buffer_create_info, None) }.map_err(Error::VulkanBufferCreation)?;
        let buffer_memory_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let alignment = buffer_memory_requirements.alignment;
        let alignment = if self.previous_allocation_was_image {
            alignment.max(self.buffer_image_granularity)
        } else {
            alignment
        };

        let offset = align_up(self.offset, alignment);
        let size = buffer_memory_requirements.size;
        debug_assert_eq!(size, align_up(src.len() as vk::DeviceSize, alignment));

        if self.total_size - offset < size {
            unsafe { self.device.destroy_buffer(buffer, None) };
            return Err(Error::ArenaOutOfMemory {
                identifier: self.debug_identifier.clone(),
                used: offset,
                total: self.total_size,
                required: size,
            });
        }

        match unsafe { self.device.bind_buffer_memory(buffer, self.memory.inner, offset) }.map_err(Error::VulkanBufferBinding) {
            Ok(_) => {}
            Err(err) => {
                unsafe { self.device.destroy_buffer(buffer, None) };
                return Err(err);
            }
        }
        debug_utils::name_vulkan_object(&self.device, buffer, name);

        if self.mapped_memory_ptr.is_null() {
            if let Some(uploader) = uploader {
                let staging_info = vk::BufferCreateInfo::builder()
                    .size(size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build();
                let staging_buffer =
                    uploader
                        .staging_arena
                        .create_buffer(staging_info, src, None, format_args!("staging buffer for {}", name))?;
                let &mut Uploader {
                    graphics_queue_family,
                    transfer_queue_family,
                    ..
                } = uploader;

                uploader.start_upload(
                    staging_buffer,
                    name,
                    |device, staging_buffer, command_buffer| {
                        profiling::scope!("queue buffer copy from staging");
                        let barrier_from_graphics_to_transfer = vk::BufferMemoryBarrier::builder()
                            .buffer(buffer)
                            .offset(0)
                            .size(vk::WHOLE_SIZE)
                            .src_queue_family_index(graphics_queue_family)
                            .dst_queue_family_index(transfer_queue_family)
                            .src_access_mask(vk::AccessFlags::NONE)
                            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .build();
                        unsafe {
                            device.cmd_pipeline_barrier(
                                command_buffer,
                                vk::PipelineStageFlags::TOP_OF_PIPE,
                                vk::PipelineStageFlags::TRANSFER,
                                vk::DependencyFlags::empty(),
                                &[],
                                &[barrier_from_graphics_to_transfer],
                                &[],
                            );
                        }
                        let (src, dst) = (staging_buffer.inner, buffer);
                        let copy_regions = [vk::BufferCopy::builder().size(size).build()];
                        unsafe { device.cmd_copy_buffer(command_buffer, src, dst, &copy_regions) };
                    },
                    |device, command_buffer| {
                        profiling::scope!("vkCmdPipelineBarrier");
                        let barrier_from_transfer_to_graphics = vk::BufferMemoryBarrier::builder()
                            .buffer(buffer)
                            .offset(0)
                            .size(vk::WHOLE_SIZE)
                            .src_queue_family_index(transfer_queue_family)
                            .dst_queue_family_index(graphics_queue_family)
                            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                            .dst_access_mask(vk::AccessFlags::VERTEX_ATTRIBUTE_READ)
                            .build();
                        unsafe {
                            device.cmd_pipeline_barrier(
                                command_buffer,
                                vk::PipelineStageFlags::TRANSFER,
                                vk::PipelineStageFlags::VERTEX_INPUT,
                                vk::DependencyFlags::empty(),
                                &[],
                                &[barrier_from_transfer_to_graphics],
                                &[],
                            );
                        }
                    },
                )?;
            } else {
                return Err(Error::ArenaNotWritable);
            }
        } else {
            let dst = unsafe { self.mapped_memory_ptr.offset(offset as isize) };
            unsafe { ptr::copy_nonoverlapping(src.as_ptr(), dst, src.len()) };
        }

        self.offset = offset + size;
        self.previous_allocation_was_image = false;

        Ok(Buffer {
            inner: buffer,
            device: self.device.clone(),
            memory: self.memory.clone(),
            size,
        })
    }

    pub fn create_image(&mut self, image_create_info: vk::ImageCreateInfo, name: Arguments) -> Result<Image, Error> {
        let image = unsafe { self.device.create_image(&image_create_info, None) }.map_err(Error::VulkanImageCreation)?;
        let image_memory_requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let alignment = image_memory_requirements.alignment.max(self.buffer_image_granularity);

        let offset = align_up(self.offset, alignment);
        let size = image_memory_requirements.size;

        if self.total_size - offset < size {
            unsafe { self.device.destroy_image(image, None) };
            return Err(Error::ArenaOutOfMemory {
                identifier: self.debug_identifier.clone(),
                used: offset,
                total: self.total_size,
                required: size,
            });
        }

        match unsafe { self.device.bind_image_memory(image, self.memory.inner, offset) }.map_err(Error::VulkanImageBinding) {
            Ok(_) => {}
            Err(err) => {
                unsafe { self.device.destroy_image(image, None) };
                return Err(err);
            }
        }

        debug_utils::name_vulkan_object(&self.device, image, name);

        self.offset = offset + size;
        self.previous_allocation_was_image = true;
        Ok(Image {
            inner: image,
            device: self.device.clone(),
            memory: self.memory.clone(),
        })
    }

    /// Stores the buffer in this arena, to be destroyed when the
    /// arena is reset. Ideal for temporary arenas whose buffers just
    /// have to live "long enough."
    pub fn add_buffer(&mut self, buffer: Buffer) {
        self.pinned_buffers.push(buffer);
    }

    /// Attempts to reset the arena, marking all graphics memory owned
    /// by it as usable again. If some of the memory allocated from
    /// this arena is still in use, Err is returned and the arena is
    /// not reset.
    pub fn reset(&mut self) -> Result<(), Error> {
        if Rc::strong_count(&self.memory) > 1 + self.pinned_buffers.len() {
            Err(Error::ArenaNotResettable)
        } else {
            self.pinned_buffers.clear();
            self.offset = 0;
            Ok(())
        }
    }
}

fn get_memory_type_index(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    optimal_flags: vk::MemoryPropertyFlags,
    fallback_flags: vk::MemoryPropertyFlags,
    size: vk::DeviceSize,
) -> Option<(u32, vk::MemoryPropertyFlags)> {
    // TODO(low): Use VK_EXT_memory_budget to pick a heap that can fit the size, it's already enabled (if available)
    let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let types = &memory_properties.memory_types[..memory_properties.memory_type_count as usize];
    let heaps = &memory_properties.memory_heaps[..memory_properties.memory_heap_count as usize];
    for (i, memory_type) in types.iter().enumerate() {
        let heap_index = memory_type.heap_index as usize;
        let flags = memory_type.property_flags;
        if flags.contains(optimal_flags) && heaps[heap_index].size >= size {
            return Some((i as u32, flags));
        }
    }
    for (i, memory_type) in types.iter().enumerate() {
        let heap_index = memory_type.heap_index as usize;
        let flags = memory_type.property_flags;
        if flags.contains(fallback_flags) && heaps[heap_index].size >= size {
            return Some((i as u32, flags));
        }
    }
    None
}

/// Returns `value` or the nearest integer greater than `value` which
/// is divisible by `align_to`.
#[allow(dead_code)]
fn align_up(value: vk::DeviceSize, align_to: vk::DeviceSize) -> vk::DeviceSize {
    if value % align_to == 0 {
        value
    } else {
        value + align_to - (value % align_to)
    }
}

/// Returns `value` or the nearest integer less than `value` which
/// is divisible by `align_to`.
#[allow(dead_code)]
fn align_down(value: vk::DeviceSize, align_to: vk::DeviceSize) -> vk::DeviceSize {
    if value % align_to == 0 {
        value
    } else {
        value - (value % align_to)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn align_up_works() {
        assert_eq!(16, super::align_up(15, 8));
        assert_eq!(16, super::align_up(9, 8));
        assert_eq!(64, super::align_up(9, 64));
        assert_eq!(64 * 3, super::align_up(64 * 3 - 1, 64));
        assert_eq!(64 * 3, super::align_up(64 * 3 - 32, 64));
        assert_eq!(64 * 3, super::align_up(64 * 3 - 63, 64));
        assert_eq!(64 * 3, super::align_up(64 * 3, 64));
        assert_eq!(0, super::align_up(0, 64));
    }

    #[test]
    fn align_down_works() {
        assert_eq!(8, super::align_down(15, 8));
        assert_eq!(8, super::align_down(9, 8));
        assert_eq!(0, super::align_down(9, 64));
        assert_eq!(64 * 2, super::align_down(64 * 3 - 1, 64));
        assert_eq!(64 * 2, super::align_down(64 * 3 - 32, 64));
        assert_eq!(64 * 2, super::align_down(64 * 3 - 63, 64));
        assert_eq!(64 * 3, super::align_down(64 * 3, 64));
        assert_eq!(0, super::align_down(0, 64));
    }
}
