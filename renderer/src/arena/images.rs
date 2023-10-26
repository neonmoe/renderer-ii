use core::fmt::Arguments;
use core::sync::atomic::Ordering;

use ash::vk;

use crate::arena::{ArenaType, VulkanArena, VulkanArenaError};
use crate::vulkan_raii::Image;

pub struct ForImages;

impl ArenaType for ForImages {
    const MAPPABLE: bool = false;
}

impl VulkanArena<ForImages> {
    pub fn create_image(&mut self, image_create_info: vk::ImageCreateInfo, name: Arguments) -> Result<Image, VulkanArenaError> {
        profiling::scope!("vulkan image creation");
        let image = unsafe { self.device.create_image(&image_create_info, None) }.map_err(VulkanArenaError::ImageCreation)?;
        let image_memory_requirements = unsafe { self.device.get_image_memory_requirements(image) };
        let alignment = image_memory_requirements.alignment;

        let offset = self.offset.next_multiple_of(alignment);
        let size = image_memory_requirements.size;

        if self.total_size - offset < size {
            unsafe { self.device.destroy_image(image, None) };
            return Err(VulkanArenaError::OutOfMemory {
                identifier: self.debug_identifier.clone(),
                used: crate::Bytes(offset),
                total: crate::Bytes(self.total_size),
                required: crate::Bytes(size),
            });
        }

        match unsafe { self.device.bind_image_memory(image, self.memory.inner, offset) }.map_err(VulkanArenaError::ImageBinding) {
            Ok(()) => {}
            Err(err) => {
                unsafe { self.device.destroy_image(image, None) };
                return Err(err);
            }
        }

        crate::name_vulkan_object(&self.device, image, name);

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
