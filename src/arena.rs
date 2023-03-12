//! An arena allocator for managing GPU memory.
use crate::debug_utils;
use crate::uploader::UploadError;
use crate::vulkan_raii::{Buffer, Device, DeviceMemory, Image};
use crate::{display_utils, PhysicalDevice, Uploader};
use alloc::rc::Rc;
use ash::vk;
use ash::Instance;
use core::fmt::{Arguments, Debug};
use core::marker::PhantomData;
use core::ptr;
use core::sync::atomic::Ordering;

#[derive(thiserror::Error, Debug)]
pub enum VulkanArenaError {
    #[error("no heap with the required flags (id: {0}, flags: {1:?})")]
    MissingMemoryType(String, vk::MemoryPropertyFlags),
    #[error("no heap with specified flags has enough memory (id: {0}, flags: {1:?}, size: {2})")]
    HeapsOutOfMemory(String, vk::MemoryPropertyFlags, display_utils::Bytes),
    #[error("vulkan memory allocation failed (id: {1}, size: {2})")]
    Allocate(#[source] vk::Result, String, display_utils::Bytes),
    #[error("mapping vulkan memory failed (id: {1}, size: {2})")]
    Map(#[source] vk::Result, String, display_utils::Bytes),
    #[error("tried to reset arena while some resources allocated from it are still in use ({0} refs)")]
    NotResettable(usize),
    #[error("arena {identifier} ({used}/{total} bytes used) cannot fit {required} bytes")]
    OutOfMemory {
        identifier: String,
        used: vk::DeviceSize,
        total: vk::DeviceSize,
        required: vk::DeviceSize,
    },
    #[error("tried to write to arena without HOST_VISIBLE | HOST_COHERENT without providing an uploader and/or staging memory")]
    NotWritable,
    #[error("failed to start upload for transferring the staging memory to device local memory")]
    Upload(#[source] UploadError),
    #[error("failed to create buffer {0:?} (probably out of host or device memory)")]
    BufferCreation(#[source] vk::Result),
    #[error("failed to bind buffer to arena memory (probably out of host or device memory)")]
    BufferBinding(#[source] vk::Result),
    #[error("failed to create image {0:?} (probably out of host or device memory)")]
    ImageCreation(#[source] vk::Result),
    #[error("failed to bind image to arena memory (probably out of host or device memory)")]
    ImageBinding(#[source] vk::Result),
}

pub trait ArenaType {
    const MAPPABLE: bool;
}

pub struct ForBuffers;
impl ArenaType for ForBuffers {
    const MAPPABLE: bool = true;
}

pub struct ForImages;
impl ArenaType for ForImages {
    const MAPPABLE: bool = false;
}

pub struct VulkanArena<T: ArenaType> {
    device: Device,
    memory: Rc<DeviceMemory>,
    mapped_memory_ptr: *mut u8,
    total_size: vk::DeviceSize,
    offset: vk::DeviceSize,
    pinned_buffers: Vec<Buffer>,
    device_local: bool,
    debug_identifier: String,
    _arena_type_marker: PhantomData<T>,
}

impl<T: ArenaType> Drop for VulkanArena<T> {
    fn drop(&mut self) {
        if !self.mapped_memory_ptr.is_null() {
            unsafe { self.device.unmap_memory(self.memory.inner) };
        }
        if self.device_local {
            // IN_USE will be subtracted by the total size by DeviceMemory.
            crate::vram_usage::IN_USE.fetch_add(self.total_size - self.offset, Ordering::Relaxed);
        }
    }
}

impl<T: ArenaType> VulkanArena<T> {
    pub fn new(
        instance: &Instance,
        device: &Device,
        physical_device: &PhysicalDevice,
        size: vk::DeviceSize,
        memory_properties: MemoryProps,
        debug_identifier_args: Arguments,
    ) -> Result<VulkanArena<T>, VulkanArenaError> {
        profiling::scope!("gpu memory arena creation");
        let debug_identifier = format!("{}", debug_identifier_args);
        let (memory_type_index, memory_flags) =
            get_memory_type_index(instance, physical_device, memory_properties, size, &debug_identifier)?;
        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size)
            .memory_type_index(memory_type_index);
        let memory = {
            profiling::scope!("vk::allocate_memory");
            log::trace!("vk::allocate_memory({} bytes, index {})", size, memory_type_index);
            unsafe { device.allocate_memory(&alloc_info, None) }
                .map_err(|err| VulkanArenaError::Allocate(err, debug_identifier.clone(), display_utils::Bytes(size)))?
        };
        debug_utils::name_vulkan_object(device, memory, debug_identifier_args);

        let mapped_memory_ptr = if T::MAPPABLE
            && memory_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT)
        {
            unsafe { device.map_memory(memory, 0, size, vk::MemoryMapFlags::empty()) }
                .map_err(|err| VulkanArenaError::Map(err, debug_identifier.clone(), display_utils::Bytes(size)))? as *mut u8
        } else {
            ptr::null_mut()
        };

        let device_local = memory_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
        let device_local_size = if device_local { size } else { 0 };
        let memory = Rc::new(DeviceMemory::new(memory, device.clone(), device_local_size));
        // IN_USE gets bumped by DeviceMemory::new, subtract it back down because none of it is actually in use.
        crate::vram_usage::IN_USE.fetch_sub(device_local_size, core::sync::atomic::Ordering::Relaxed);

        Ok(VulkanArena {
            device: device.clone(),
            memory,
            mapped_memory_ptr,
            total_size: size,
            offset: 0,
            pinned_buffers: Vec::new(),
            device_local,
            debug_identifier,
            _arena_type_marker: PhantomData {},
        })
    }

    pub fn memory_in_use(&self) -> vk::DeviceSize {
        self.offset
    }

    /// Attempts to reset the arena, marking all graphics memory owned by it as
    /// usable again. If some of the memory allocated from this arena is still
    /// in use, Err is returned and the arena is not reset.
    pub fn reset(&mut self) -> Result<(), VulkanArenaError> {
        let total_refs = Rc::strong_count(&self.memory);
        let internal_refs = 1 + self.pinned_buffers.len();
        if total_refs > internal_refs {
            Err(VulkanArenaError::NotResettable(total_refs - internal_refs))
        } else {
            self.pinned_buffers.clear();
            if self.device_local {
                // Everything created from this arena no longer exists, so none of the memory is in use anymore.
                crate::vram_usage::IN_USE.fetch_sub(self.offset, Ordering::Relaxed);
            }
            self.offset = 0;
            Ok(())
        }
    }
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

        let offset = align_up(self.offset, alignment);
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
        debug_utils::name_vulkan_object(&self.device, buffer, name);

        if self.mapped_memory_ptr.is_null() {
            if let (Some(uploader), Some(staging_arena)) = (uploader, staging_arena) {
                profiling::scope!("staging buffer creation");
                let staging_info = vk::BufferCreateInfo::builder()
                    .size(buffer_create_info.size)
                    .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE)
                    .build();
                let staging_buffer =
                    staging_arena.create_buffer(staging_info, src, None, None, format_args!("staging buffer for {}", name))?;
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
                            let barrier_from_graphics_to_transfer = [vk::BufferMemoryBarrier2::builder()
                                .buffer(buffer)
                                .offset(0)
                                .size(vk::WHOLE_SIZE)
                                .src_queue_family_index(graphics_queue_family)
                                .dst_queue_family_index(transfer_queue_family)
                                .src_access_mask(vk::AccessFlags2::NONE)
                                .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                                .src_stage_mask(vk::PipelineStageFlags2::NONE)
                                .dst_stage_mask(vk::PipelineStageFlags2::COPY)
                                .build()];
                            let dep_info = vk::DependencyInfo::builder().buffer_memory_barriers(&barrier_from_graphics_to_transfer);
                            unsafe { device.cmd_pipeline_barrier2(command_buffer, &dep_info) };
                            let (src, dst) = (staging_buffer.inner, buffer);
                            let copy_regions = [vk::BufferCopy::builder().size(buffer_create_info.size).build()];
                            unsafe { device.cmd_copy_buffer(command_buffer, src, dst, &copy_regions) };
                        },
                        |device, command_buffer| {
                            profiling::scope!("vk::cmd_pipeline_barrier");
                            let barrier_from_transfer_to_graphics = [vk::BufferMemoryBarrier2::builder()
                                .buffer(buffer)
                                .offset(0)
                                .size(vk::WHOLE_SIZE)
                                .src_queue_family_index(transfer_queue_family)
                                .dst_queue_family_index(graphics_queue_family)
                                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                                .dst_access_mask(vk::AccessFlags2::VERTEX_ATTRIBUTE_READ)
                                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                                .dst_stage_mask(vk::PipelineStageFlags2::VERTEX_INPUT)
                                .build()];
                            let dep_info = vk::DependencyInfo::builder().buffer_memory_barriers(&barrier_from_transfer_to_graphics);
                            unsafe { device.cmd_pipeline_barrier2(command_buffer, &dep_info) };
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

impl VulkanArena<ForImages> {
    pub fn create_image(&mut self, image_create_info: vk::ImageCreateInfo, name: Arguments) -> Result<Image, VulkanArenaError> {
        profiling::scope!("vulkan image creation");
        let image = unsafe { self.device.create_image(&image_create_info, None) }.map_err(VulkanArenaError::ImageCreation)?;
        let image_memory_requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let alignment = image_memory_requirements.alignment;

        let offset = align_up(self.offset, alignment);
        let size = image_memory_requirements.size;

        if self.total_size - offset < size {
            unsafe { self.device.destroy_image(image, None) };
            return Err(VulkanArenaError::OutOfMemory {
                identifier: self.debug_identifier.clone(),
                used: offset,
                total: self.total_size,
                required: size,
            });
        }

        match unsafe { self.device.bind_image_memory(image, self.memory.inner, offset) }.map_err(VulkanArenaError::ImageBinding) {
            Ok(_) => {}
            Err(err) => {
                unsafe { self.device.destroy_image(image, None) };
                return Err(err);
            }
        }

        debug_utils::name_vulkan_object(&self.device, image, name);

        let new_offset = offset + size;
        if self.device_local {
            crate::vram_usage::IN_USE.fetch_add(new_offset - self.offset, Ordering::Relaxed);
        }
        self.offset = new_offset;

        Ok(Image {
            inner: image,
            device: self.device.clone(),
            memory: self.memory.clone(),
        })
    }
}

fn get_memory_type_index(
    instance: &Instance,
    physical_device: &PhysicalDevice,
    flags: MemoryProps,
    size: vk::DeviceSize,
    debug_identifier: &str,
) -> Result<(u32, vk::MemoryPropertyFlags), VulkanArenaError> {
    let budget_supported = physical_device.extension_supported("VK_EXT_memory_budget");
    let mut memory_properties = vk::PhysicalDeviceMemoryProperties2::builder();
    let mut budget_props = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
    if budget_supported {
        memory_properties = memory_properties.push_next(&mut budget_props);
    }
    let mut memory_properties = memory_properties.build();
    unsafe { instance.get_physical_device_memory_properties2(physical_device.inner, &mut memory_properties) };

    let props = memory_properties.memory_properties;
    let types = &props.memory_types[..props.memory_type_count as usize];
    let heaps = &props.memory_heaps[..props.memory_heap_count as usize];
    let mut valid_type_found = false;
    for (i, memory_type) in types.iter().enumerate() {
        let heap_index = memory_type.heap_index as usize;
        let prop_flags = memory_type.property_flags;
        let budget = if budget_supported {
            budget_props.heap_budget[heap_index]
        } else {
            heaps[heap_index].size
        };
        if prop_flags.contains(flags.optimal) && !prop_flags.intersects(flags.unwanted) {
            valid_type_found = true;
            if budget >= size {
                return Ok((i as u32, prop_flags));
            }
        }
    }
    for (i, memory_type) in types.iter().enumerate() {
        let heap_index = memory_type.heap_index as usize;
        let budget = if budget_supported {
            budget_props.heap_budget[heap_index]
        } else {
            heaps[heap_index].size
        };
        if memory_type.property_flags.contains(flags.fallback) {
            valid_type_found = true;
            if budget >= size {
                return Ok((i as u32, memory_type.property_flags));
            }
        }
    }

    if valid_type_found {
        Err(VulkanArenaError::HeapsOutOfMemory(
            debug_identifier.to_string(),
            flags.fallback,
            display_utils::Bytes(size),
        ))
    } else {
        Err(VulkanArenaError::MissingMemoryType(debug_identifier.to_string(), flags.fallback))
    }
}

/// A heap with the `optimal` and without the `unwanted` flags is used if found,
/// otherwise the first one that has the `fallback` flags is used.
pub struct MemoryProps {
    pub optimal: vk::MemoryPropertyFlags,
    pub unwanted: vk::MemoryPropertyFlags,
    pub fallback: vk::MemoryPropertyFlags,
}
impl MemoryProps {
    pub fn for_framebuffers() -> MemoryProps {
        MemoryProps {
            optimal: vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::LAZILY_ALLOCATED,
            unwanted: vk::MemoryPropertyFlags::empty(),
            fallback: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        }
    }

    pub fn for_textures() -> MemoryProps {
        MemoryProps {
            optimal: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            unwanted: vk::MemoryPropertyFlags::empty(),
            fallback: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        }
    }

    pub fn for_buffers() -> MemoryProps {
        MemoryProps {
            optimal: vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            unwanted: vk::MemoryPropertyFlags::empty(),
            fallback: vk::MemoryPropertyFlags::DEVICE_LOCAL,
        }
    }

    pub fn for_staging() -> MemoryProps {
        MemoryProps {
            optimal: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            unwanted: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            fallback: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        }
    }
}

/// Returns `value` or the nearest integer greater than `value` which
/// is divisible by `align_to`.
pub(crate) fn align_up(value: vk::DeviceSize, align_to: vk::DeviceSize) -> vk::DeviceSize {
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
