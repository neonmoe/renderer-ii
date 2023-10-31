use core::marker::PhantomData;

use ash::vk;

use crate::arena::buffers::ForBuffers;
use crate::arena::images::ForImages;
use crate::arena::ArenaType;
use crate::vulkan_raii::Device;

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
    pub fn add_buffer(&mut self, buffer_create_info: vk::BufferCreateInfo) {
        profiling::scope!("query buffer memory requirements");
        let buffer = unsafe { self.device.create_buffer(&buffer_create_info, None) }.expect("vulkan buffer creation should not fail");
        crate::name_vulkan_object(&self.device, buffer, format_args!("memory requirement querying temp buffer"));
        let buffer_memory_requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        unsafe { self.device.destroy_buffer(buffer, None) };
        let alignment = buffer_memory_requirements.alignment;
        let offset = self.measured_size.next_multiple_of(alignment);
        let size = buffer_memory_requirements.size;
        self.measured_size = offset + size;
    }
}

impl VulkanArenaMeasurer<ForImages> {
    pub fn add_image(&mut self, image_create_info: vk::ImageCreateInfo) {
        profiling::scope!("query image memory requirements");
        let image = unsafe { self.device.create_image(&image_create_info, None) }.expect("vulkan image creation should not fail");
        crate::name_vulkan_object(&self.device, image, format_args!("memory requirement querying temp image"));
        let image_memory_requirements = unsafe { self.device.get_image_memory_requirements(image) };
        unsafe { self.device.destroy_image(image, None) };
        let alignment = image_memory_requirements.alignment;
        let offset = self.measured_size.next_multiple_of(alignment);
        let size = image_memory_requirements.size;
        self.measured_size = offset + size;
    }
}
