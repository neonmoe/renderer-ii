use crate::physical_device::QueueFamily;
use crate::vulkan_raii::Device;
use crate::{debug_utils, physical_device_features, Error, PhysicalDevice};
use ash::{vk, Instance};

pub fn create_device(instance: &Instance, physical_device: &PhysicalDevice) -> Result<Device, Error> {
    profiling::scope!("vulkan device creation");

    // Just to have an array to point at for the queue priorities.
    let ones = [1.0, 1.0, 1.0];
    let queue_families = [
        physical_device.graphics_queue_family,
        physical_device.transfer_queue_family,
        physical_device.surface_queue_family,
    ];
    let queue_create_infos = create_device_queue_create_infos(&queue_families, &ones);
    let mut extensions = vec![cstr!("VK_KHR_swapchain").as_ptr()];
    log::debug!("Device extension: VK_KHR_swapchain");
    if physical_device.extension_supported("VK_EXT_memory_budget") {
        extensions.push(cstr!("VK_EXT_memory_budget").as_ptr());
        log::debug!("Device extension (optional): VK_EXT_memory_budget");
    }

    // Features in core Vulkan, provided by the target api version, which this crate uses:
    // - VK_KHR_dynamic_rendering (Vulkan 1.3) (unused until the validation layers catch up)
    // - VK_EXT_pipeline_creation_cache_control (Vulkan 1.3)

    let device_create_info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_create_infos)
        .enabled_extension_names(&extensions);
    let device = physical_device_features::create_device_with_feature_requirements(instance, physical_device.inner, device_create_info)?;

    let mut queues = [vk::Queue::default(); 3];
    get_device_queues(&device, &queue_families, &mut queues);
    let [graphics_queue, transfer_queue, surface_queue] = queues;
    if graphics_queue == surface_queue && graphics_queue == transfer_queue {
        debug_utils::name_vulkan_object(&device, graphics_queue, format_args!("graphics+surface+transfer"));
    } else if graphics_queue == surface_queue {
        debug_utils::name_vulkan_object(&device, graphics_queue, format_args!("graphics+surface"));
        debug_utils::name_vulkan_object(&device, transfer_queue, format_args!("transfer"));
    } else if surface_queue == transfer_queue {
        debug_utils::name_vulkan_object(&device, graphics_queue, format_args!("graphics"));
        debug_utils::name_vulkan_object(&device, surface_queue, format_args!("surface+transfer"));
    } else {
        debug_utils::name_vulkan_object(&device, graphics_queue, format_args!("graphics"));
        debug_utils::name_vulkan_object(&device, surface_queue, format_args!("surface"));
        debug_utils::name_vulkan_object(&device, transfer_queue, format_args!("transfer"));
    }

    let device = Device {
        inner: device,
        graphics_queue,
        surface_queue,
        transfer_queue,
    };
    debug_utils::name_vulkan_object(&device, device.inner.handle(), format_args!("{}", physical_device.name));
    debug_utils::name_vulkan_object(&device, physical_device.inner, format_args!("{}", physical_device.name));

    Ok(device)
}

fn create_device_queue_create_infos(queue_families: &[QueueFamily], ones: &[f32]) -> Vec<vk::DeviceQueueCreateInfo> {
    let mut results: Vec<vk::DeviceQueueCreateInfo> = Vec::with_capacity(queue_families.len());
    'queue_families: for &queue_family in queue_families {
        for create_info in &results {
            if create_info.queue_family_index == queue_family.index {
                continue 'queue_families;
            }
        }
        let count = queue_families.iter().filter(|qf| qf.index == queue_family.index).count();
        let count = count.min(queue_family.max_count);
        let create_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family.index)
            .queue_priorities(&ones[..count])
            .build();
        results.push(create_info);
    }
    results
}

fn get_device_queues<const N: usize>(device: &ash::Device, queue_families: &[QueueFamily; N], queues: &mut [vk::Queue; N]) {
    let mut picks = Vec::with_capacity(N);
    for (&queue_family, queue) in queue_families.iter().zip(queues.iter_mut()) {
        let queue_index = picks.iter().filter(|index| **index == queue_family.index).count() as u32;
        let queue_index = queue_index.min((queue_family.max_count - 1) as u32);
        *queue = unsafe { device.get_device_queue(queue_family.index, queue_index) };
        picks.push(queue_family.index);
    }
}
