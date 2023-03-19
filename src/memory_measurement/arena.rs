use crate::arena::{self, ArenaType, ForBuffers, ForImages};
use crate::debug_utils;
use crate::vulkan_raii::Device;
use ash::vk;
use core::marker::PhantomData;

#[derive(thiserror::Error, Debug)]
pub enum VulkanArenaMeasurementError {
    #[error("failed to create a buffer for measuring its memory requirements")]
    BufferCreation(#[source] vk::Result),
    #[error("failed to create an image for measuring its memory requirements")]
    ImageCreation(#[source] vk::Result),
}

pub struct VulkanArenaMeasurer<T: ArenaType> {
    pub measured_size: vk::DeviceSize,
    device: Device,
    _arena_type_marker: PhantomData<T>,
}

impl<T: ArenaType> VulkanArenaMeasurer<T> {
    pub fn new(device: &Device) -> VulkanArenaMeasurer<T> {
        VulkanArenaMeasurer {
            measured_size: 0,
            device: device.clone(),
            _arena_type_marker: PhantomData {},
        }
    }
}

impl VulkanArenaMeasurer<ForBuffers> {
    pub fn add_buffer(&mut self, buffer_create_info: vk::BufferCreateInfo) -> Result<(), VulkanArenaMeasurementError> {
        profiling::scope!("query buffer memory requirements");
        let buffer =
            unsafe { self.device.create_buffer(&buffer_create_info, None) }.map_err(VulkanArenaMeasurementError::BufferCreation)?;
        debug_utils::name_vulkan_object(&self.device, buffer, format_args!("memory requirement querying temp buffer"));
        let buffer_memory_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        unsafe { self.device.destroy_buffer(buffer, None) };
        let alignment = buffer_memory_requirements.alignment;
        let offset = arena::align_up(self.measured_size, alignment);
        let size = buffer_memory_requirements.size;
        self.measured_size = offset + size;
        Ok(())
    }
}

impl VulkanArenaMeasurer<ForImages> {
    pub fn add_image(&mut self, image_create_info: vk::ImageCreateInfo) -> Result<(), VulkanArenaMeasurementError> {
        profiling::scope!("query image memory requirements");
        let image = unsafe { self.device.create_image(&image_create_info, None) }.map_err(VulkanArenaMeasurementError::ImageCreation)?;
        debug_utils::name_vulkan_object(&self.device, image, format_args!("memory requirement querying temp image"));
        let image_memory_requirements = unsafe { self.device.get_image_memory_requirements(image) };
        unsafe { self.device.destroy_image(image, None) };
        let alignment = image_memory_requirements.alignment;
        let offset = arena::align_up(self.measured_size, alignment);
        let size = image_memory_requirements.size;
        self.measured_size = offset + size;
        Ok(())
    }
}
