//! An arena allocator for managing GPU memory.
use crate::error::Error;
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk;
use ash::{Device, Instance};
use std::cell::Cell;
use typed_arena::Arena;

mod buffer;
mod simple_wrappers;

pub use buffer::{BufferAllocation, WritableBufferAllocation};
pub use simple_wrappers::{ImageAllocation, ImageView};

/// An arena of Vulkan resources.
///
/// One VulkanArena maps to one vkDeviceMemory, and that memory is
/// allocated linearly to resources created. All resources created by
/// a VulkanArena are only released when it is dropped.
///
/// The lifetimes of the Vulkan objects are handled statically: all
/// the resources rely on:
///
/// - the backing memory,
/// - the device,
/// - any "parent" objects (like vkImage for vkImageViews)
///
/// Since the [Arena] owns both the memory and the device, those two
/// points are handled by the [Arena] always owning the resource,
/// since it only lends the resources. The parent-relations are
/// handled in the [Drop] implementation of the VulkanArena.
pub struct VulkanArena<'device> {
    pub device: &'device Device,
    memory: vk::DeviceMemory,
    flags: vk::MemoryPropertyFlags,
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
    // Vulkan resource holders:
    buffers: Arena<BufferAllocation>,
    images: Arena<ImageAllocation>,
    image_views: Arena<ImageView>,
}

impl Drop for VulkanArena<'_> {
    fn drop(&mut self) {
        self.reset();
        unsafe { self.device.free_memory(self.memory, None) };
    }
}

impl VulkanArena<'_> {
    pub fn new<'device>(
        instance: &Instance,
        device: &'device Device,
        physical_device: vk::PhysicalDevice,
        size: vk::DeviceSize,
        optimal_flags: vk::MemoryPropertyFlags,
        fallback_flags: vk::MemoryPropertyFlags,
        debug_identifier: &'static str,
    ) -> Result<VulkanArena<'device>, Error> {
        let (memory_type_index, flags) = get_memory_type_index(instance, physical_device, optimal_flags, fallback_flags, size)
            .ok_or(Error::VulkanNoMatchingHeap(debug_identifier, fallback_flags))?;
        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(memory_type_index);
        let memory =
            unsafe { device.allocate_memory(&alloc_info, None) }.map_err(|err| Error::VulkanAllocate(err, debug_identifier, size))?;

        let physical_device_properties = unsafe { instance.get_physical_device_properties(physical_device) };
        let buffer_image_granularity = physical_device_properties.limits.buffer_image_granularity;
        let non_coherent_atom_size = physical_device_properties.limits.non_coherent_atom_size;

        let arena = VulkanArena {
            device,
            memory,
            flags,
            total_size: size,
            offset: Cell::new(0),
            non_coherent_atom_size,
            buffer_image_granularity,
            previous_allocation_was_image: Cell::new(false),
            debug_identifier,
            buffers: Arena::new(),
            images: Arena::new(),
            image_views: Arena::new(),
        };

        #[cfg(feature = "vulkan-validation")]
        if flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            arena.fill_uninitialized_memory(arena.total_size);
        }

        Ok(arena)
    }

    pub fn create_buffer<F: FnOnce(&mut WritableBufferAllocation<'_>) -> Result<(), Error>>(
        &self,
        buffer_create_info: vk::BufferCreateInfo,
        write_alloc: F,
    ) -> Result<&BufferAllocation, Error> {
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

        match unsafe { self.device.bind_buffer_memory(buffer, self.memory, offset) }.map_err(Error::VulkanBufferBinding) {
            Ok(_) => {}
            Err(err) => {
                unsafe { self.device.destroy_buffer(buffer, None) };
                return Err(err);
            }
        }

        self.offset.set(offset + size);
        self.previous_allocation_was_image.set(false);
        let mut writable = WritableBufferAllocation {
            device: self.device,
            memory: self.memory,
            memory_size: self.total_size,
            non_coherent_atom_size: self.non_coherent_atom_size,
            buffer_allocation: BufferAllocation { buffer, offset, size },
        };
        write_alloc(&mut writable)?;
        Ok(self.buffers.alloc(writable.buffer_allocation))
    }

    pub fn create_image(&self, image_create_info: vk::ImageCreateInfo) -> Result<&ImageAllocation, Error> {
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

        match unsafe { self.device.bind_image_memory(image, self.memory, offset) }.map_err(Error::VulkanImageBinding) {
            Ok(_) => {}
            Err(err) => {
                unsafe { self.device.destroy_image(image, None) };
                return Err(err);
            }
        }

        self.offset.set(offset + size);
        self.previous_allocation_was_image.set(true);
        Ok(self.images.alloc(ImageAllocation { image }))
    }

    /// Creates an image view that lives as long as the Arena (or
    /// until [Arena::reset], if it is ever called), meant for
    /// creating image views from images allocated by this Arena.
    pub fn create_image_view(&self, create_info: &vk::ImageViewCreateInfo) -> Result<&ImageView, Error> {
        let image_view = unsafe { self.device.create_image_view(create_info, None) }.map_err(Error::VulkanImageViewCreation)?;
        Ok(self.image_views.alloc(ImageView { image_view }))
    }

    pub fn reset(&mut self) {
        for allocation in self.buffers.iter_mut() {
            unsafe { self.device.destroy_buffer(allocation.buffer, None) };
        }
        self.buffers = Arena::new();
        for allocation in self.images.iter_mut() {
            unsafe { self.device.destroy_image(allocation.image, None) };
        }
        self.images = Arena::new();
        for image_view in self.image_views.iter_mut() {
            unsafe { self.device.destroy_image_view(image_view.image_view, None) };
        }
        self.image_views = Arena::new();
        if self.flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            self.fill_uninitialized_memory(self.offset.get());
        }
        self.offset.set(0);
    }

    #[cfg(feature = "vulkan-validation")]
    fn fill_uninitialized_memory(&self, up_to: vk::DeviceSize) {
        if let Ok(dst) = unsafe { self.device.map_memory(self.memory, 0, self.total_size, vk::MemoryMapFlags::empty()) } {
            let uninitialized_pattern: u64 = 0xF000DEADF000DEAD;
            for i in 0..up_to as isize / std::mem::size_of::<u64>() as isize {
                unsafe { std::ptr::write((dst as *mut u64).offset(i), uninitialized_pattern) };
            }

            let ranges = [vk::MappedMemoryRange::builder()
                .memory(self.memory)
                .offset(0)
                .size(self.total_size)
                .build()];
            let _ = unsafe { self.device.flush_mapped_memory_ranges(&ranges) };
            unsafe { self.device.unmap_memory(self.memory) };
        }
    }

    #[cfg(not(feature = "vulkan-validation"))]
    fn fill_uninitialized_memory(&self, _: vk::DeviceSize) {}
}

fn get_memory_type_index(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    optimal_flags: vk::MemoryPropertyFlags,
    fallback_flags: vk::MemoryPropertyFlags,
    size: vk::DeviceSize,
) -> Option<(u32, vk::MemoryPropertyFlags)> {
    // TODO: Use VK_EXT_memory_budget to pick a heap that can fit the size, it's already enabled (if available)
    let memory_properties = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let types = &memory_properties.memory_types[..memory_properties.memory_type_count as usize];
    let heaps = &memory_properties.memory_heaps[..memory_properties.memory_heap_count as usize];
    for (i, memory_type) in types.iter().enumerate() {
        let heap_index = memory_type.heap_index as usize;
        if memory_type.property_flags.contains(optimal_flags) && heaps[heap_index].size >= size {
            return Some((i as u32, optimal_flags));
        }
    }
    for (i, memory_type) in types.iter().enumerate() {
        let heap_index = memory_type.heap_index as usize;
        if memory_type.property_flags.contains(fallback_flags) && heaps[heap_index].size >= size {
            return Some((i as u32, optimal_flags));
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
