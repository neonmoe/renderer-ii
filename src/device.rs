use crate::vulkan_raii::Device;
use crate::{debug_utils, Error, PhysicalDevice};
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::{vk, Instance};

pub fn create_device(instance: &Instance, physical_device: &PhysicalDevice) -> Result<Device, Error> {
    // Just to have an array to point at for the queue priorities.
    let ones = [1.0, 1.0, 1.0];
    let queue_family_indices = [
        physical_device.graphics_family_index,
        physical_device.surface_family_index,
        physical_device.transfer_family_index,
    ];
    let queue_create_infos = create_device_queue_create_infos(&queue_family_indices, &ones);
    let mut extensions = vec![cstr!("VK_KHR_swapchain").as_ptr()];
    log::debug!("Device extension: VK_KHR_swapchain");
    if physical_device.extension_supported("VK_EXT_memory_budget") {
        extensions.push(cstr!("VK_EXT_memory_budget").as_ptr());
        log::debug!("Device extension (optional): VK_EXT_memory_budget");
    }

    let mut physical_device_descriptor_indexing_features =
        vk::PhysicalDeviceDescriptorIndexingFeatures::builder().descriptor_binding_partially_bound(true);
    let device_create_info = vk::DeviceCreateInfo::builder()
        .push_next(&mut physical_device_descriptor_indexing_features)
        .queue_create_infos(&queue_create_infos)
        .enabled_features(&physical_device.features)
        .enabled_extension_names(&extensions);
    let device = unsafe {
        instance
            .create_device(physical_device.inner, &device_create_info, None)
            .map_err(Error::VulkanDeviceCreation)
    }?;

    let mut queues = [vk::Queue::default(); 3];
    get_device_queues(&device, &queue_family_indices, &mut queues);
    let [graphics_queue, surface_queue, transfer_queue] = queues;
    debug_utils::name_vulkan_object(&device, graphics_queue, format_args!("graphics"));
    debug_utils::name_vulkan_object(&device, surface_queue, format_args!("present"));
    debug_utils::name_vulkan_object(&device, transfer_queue, format_args!("transfer"));

    Ok(Device {
        inner: device,
        graphics_queue,
        surface_queue,
        transfer_queue,
    })
}

fn create_device_queue_create_infos(queue_family_indices: &[u32], ones: &[f32]) -> Vec<vk::DeviceQueueCreateInfo> {
    let mut results: Vec<vk::DeviceQueueCreateInfo> = Vec::with_capacity(queue_family_indices.len());
    'queue_families: for &queue_family_index in queue_family_indices {
        for create_info in &results {
            if create_info.queue_family_index == queue_family_index {
                continue 'queue_families;
            }
        }
        let count = queue_family_indices.iter().filter(|index| **index == queue_family_index).count();
        let create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .queue_priorities(&ones[..count])
            .build();
        results.push(create_info);
    }
    results
}

fn get_device_queues<const N: usize>(device: &ash::Device, family_indices: &[u32; N], queues: &mut [vk::Queue; N]) {
    let mut picks = Vec::with_capacity(N);
    for (&queue_family_index, queue) in family_indices.iter().zip(queues.iter_mut()) {
        let queue_index = picks.iter().filter(|index| **index == queue_family_index).count() as u32;
        *queue = unsafe { device.get_device_queue(queue_family_index, queue_index) };
        picks.push(queue_family_index);
    }
}
