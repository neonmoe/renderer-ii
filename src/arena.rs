//! An arena allocator for managing GPU memory.
use crate::debug_utils;
use crate::error::Error;
use crate::vulkan_raii::{Buffer, Device, DeviceMemory, Image};
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk;
use ash::Instance;
use std::cell::Cell;
use std::ptr;
use std::rc::Rc;

/// A Vulkan buffer allocated from an [Arena].
#[derive(PartialEq, Eq, Hash)]
pub struct BufferAllocation {
    pub buffer: Buffer,
    offset: vk::DeviceSize,
    size: vk::DeviceSize,
    non_coherent_atom_size: vk::DeviceSize,
    backing_memory_size: vk::DeviceSize,
}

impl BufferAllocation {
    /// Write `size` bytes from `src` into the buffer, `offset` bytes
    /// after the start of the buffer. Both `offset` and `size` are
    /// clamped to the buffer's bounds, so if `offset == 0` and `size
    /// == VK_WHOLE_SIZE`, the entire buffer will be written, reading
    /// from `src`.
    pub unsafe fn write(&self, src: *const u8, offset: vk::DeviceSize, size: vk::DeviceSize) -> Result<(), Error> {
        let offset = (self.offset + offset).min(self.offset + self.size);
        let aligned_offset = align_down(offset, self.non_coherent_atom_size);
        let size = size.min(self.size);
        let aligned_size = align_up(size, self.non_coherent_atom_size).min(self.backing_memory_size - aligned_offset);

        // NOTE: The mapped range is aligned to satisfy vkMappedMemoryRange requirements later.
        let dst = self
            .buffer
            .device
            .map_memory(self.buffer.memory.inner, aligned_offset, aligned_size, vk::MemoryMapFlags::empty())
            .map_err(Error::VulkanMapMemory)? as *mut u8;
        let dst = dst.offset((offset - aligned_offset) as isize);
        ptr::copy_nonoverlapping(src, dst, size as usize);

        // NOTE: VkMappedMemoryRanges have notable alignment requirements, and they have been taken into account.
        let ranges = [vk::MappedMemoryRange::builder()
            .memory(self.buffer.memory.inner)
            .offset(aligned_offset)
            .size(aligned_size)
            .build()];
        self.buffer
            .device
            .flush_mapped_memory_ranges(&ranges)
            .map_err(Error::VulkanFlushMapped)?;
        self.buffer.device.unmap_memory(self.buffer.memory.inner);

        Ok(())
    }
}

pub struct VulkanArena {
    device: Rc<Device>,
    memory: Rc<DeviceMemory>,
    total_size: vk::DeviceSize,
    /// The location where the available memory starts. Gets set to 0
    /// when the arena is reset, bumped when allocating.
    ///
    /// NOTE: Cells are not Sync, so all accesses to the offset are
    /// single-threaded. This is why it's safe to get() and replace()
    /// instead of using atomic operations. At least currently, Arenas
    /// are per-thread. A possible multi-threaded arena should use an
    /// atomic int here.
    offset: Cell<vk::DeviceSize>,
    non_coherent_atom_size: vk::DeviceSize,
    buffer_image_granularity: vk::DeviceSize,
    previous_allocation_was_image: Cell<bool>,
    debug_identifier: &'static str,
}

impl VulkanArena {
    pub fn new(
        instance: &Instance,
        device: &Rc<Device>,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
        optimal_flags: vk::MemoryPropertyFlags,
        fallback_flags: vk::MemoryPropertyFlags,
        debug_identifier: &'static str,
    ) -> Result<VulkanArena, Error> {
        let memory_type_index = get_memory_type_index(instance, physical_device, optimal_flags, fallback_flags, size)
            .ok_or(Error::VulkanNoMatchingHeap(debug_identifier, fallback_flags))?;
        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(memory_type_index);
        let memory =
            unsafe { device.allocate_memory(&alloc_info, None) }.map_err(|err| Error::VulkanAllocate(err, debug_identifier, size))?;
        debug_utils::name_vulkan_object(device, memory, format_args!("{}", debug_identifier));

        let physical_device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let buffer_image_granularity = physical_device_properties.limits.buffer_image_granularity;
        let non_coherent_atom_size = physical_device_properties.limits.non_coherent_atom_size;

        Ok(VulkanArena {
            device: device.clone(),
            memory: Rc::new(DeviceMemory {
                inner: memory,
                device: device.clone(),
            }),
            total_size: size,
            offset: Cell::new(0),
            non_coherent_atom_size,
            buffer_image_granularity,
            previous_allocation_was_image: Cell::new(false),
            debug_identifier,
        })
    }

    pub fn create_buffer(&self, buffer_create_info: vk::BufferCreateInfo) -> Result<BufferAllocation, Error> {
        let buffer = unsafe { self.device.create_buffer(&buffer_create_info, None) }.map_err(Error::VulkanBufferCreation)?;
        let buffer_memory_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let alignment = buffer_memory_requirements.alignment;
        let alignment = if self.previous_allocation_was_image.get() {
            alignment.max(self.buffer_image_granularity)
        } else {
            alignment
        };

        let offset = align_up(self.offset.get(), alignment);
        let size = buffer_memory_requirements.size;

        if self.total_size - offset < size {
            unsafe { self.device.destroy_buffer(buffer, None) };
            return Err(Error::ArenaOutOfMemory {
                identifier: self.debug_identifier,
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

        self.offset.set(offset + size);
        self.previous_allocation_was_image.set(false);
        Ok(BufferAllocation {
            buffer: Buffer {
                inner: buffer,
                device: self.device.clone(),
                memory: self.memory.clone(),
            },
            offset,
            size,
            non_coherent_atom_size: self.non_coherent_atom_size,
            backing_memory_size: self.total_size,
        })
    }

    pub fn create_image(&self, image_create_info: vk::ImageCreateInfo) -> Result<Image, Error> {
        let image = unsafe { self.device.create_image(&image_create_info, None) }.map_err(Error::VulkanImageCreation)?;
        let image_memory_requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let alignment = image_memory_requirements.alignment.max(self.buffer_image_granularity);

        let offset = align_up(self.offset.get(), alignment);
        let size = image_memory_requirements.size;

        if self.total_size - offset < size {
            unsafe { self.device.destroy_image(image, None) };
            return Err(Error::ArenaOutOfMemory {
                identifier: self.debug_identifier,
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

        self.offset.set(offset + size);
        self.previous_allocation_was_image.set(true);
        Ok(Image {
            inner: image,
            device: self.device.clone(),
            memory: self.memory.clone(),
        })
    }

    /// Attempts to reset the arena, marking all graphics memory owned
    /// by it as usable again. If some of the memory allocated from
    /// this arena is still in use, Err is returned and the arena is
    /// not reset.
    pub fn reset(&self) -> Result<(), Error> {
        if Rc::strong_count(&self.memory) > 1 {
            Err(Error::ArenaNotResettable)
        } else {
            self.offset.set(0);
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
) -> Option<u32> {
    // TODO: Use VK_EXT_memory_budget to pick a heap that can fit the size, it's already enabled (if available)
    let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let types = &memory_properties.memory_types[..memory_properties.memory_type_count as usize];
    let heaps = &memory_properties.memory_heaps[..memory_properties.memory_heap_count as usize];
    for (i, memory_type) in types.iter().enumerate() {
        let heap_index = memory_type.heap_index as usize;
        if memory_type.property_flags.contains(optimal_flags) && heaps[heap_index].size >= size {
            return Some(i as u32);
        }
    }
    for (i, memory_type) in types.iter().enumerate() {
        let heap_index = memory_type.heap_index as usize;
        if memory_type.property_flags.contains(fallback_flags) && heaps[heap_index].size >= size {
            return Some(i as u32);
        }
    }
    None
}

/// Returns `value` or the nearest integer greater than `value` which
/// is divisible by `align_to`.
fn align_up(value: vk::DeviceSize, align_to: vk::DeviceSize) -> vk::DeviceSize {
    if value % align_to == 0 {
        value
    } else {
        value + align_to - (value % align_to)
    }
}

/// Returns `value` or the nearest integer less than `value` which
/// is divisible by `align_to`.
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
