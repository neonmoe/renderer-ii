use alloc::boxed::Box;
use core::ffi::c_char;

use arrayvec::ArrayVec;
use ash::extensions::khr;
use ash::{vk, Entry, Instance};

use crate::physical_device::{physical_device_features, PhysicalDevice, QueueFamily};
use crate::vulkan_raii::Device;

impl PhysicalDevice {
    /// Creates a new `VkDevice`. It only needs to be destroyed if creating a new one.
    pub fn create_device(&self, entry: &Entry, instance: &Instance) -> Result<Device, vk::Result> {
        profiling::scope!("vulkan device creation");

        // Just to have an array to point at for the queue priorities.
        let ones = [1.0, 1.0, 1.0];
        let queue_families = [self.graphics_queue_family, self.transfer_queue_family, self.surface_queue_family];
        let queue_create_infos = create_device_queue_create_infos(&queue_families, &ones);

        let mut extensions: ArrayVec<*const c_char, { physical_device_features::TOTAL_DEVICE_EXTENSIONS }> = ArrayVec::new();
        for name in physical_device_features::REQUIRED_DEVICE_EXTENSIONS {
            extensions.push(name.as_ptr());
            log::debug!("Device extension: {}", name.to_str().unwrap());
        }
        for name in physical_device_features::OPTIONAL_DEVICE_EXTENSIONS {
            let name_str = name.to_str().unwrap();
            if self.extension_supported(name_str) {
                extensions.push(name.as_ptr());
                log::debug!("Device extension (optional): {}", name_str);
            }
        }

        let device_create_info =
            vk::DeviceCreateInfo::default().queue_create_infos(&queue_create_infos).enabled_extension_names(&extensions);
        let device = physical_device_features::create_with_features(instance, self.inner, device_create_info)?;

        let mut queues = [vk::Queue::default(); 3];
        get_device_queues(&device, &queue_families, &mut queues);
        let [graphics_queue, transfer_queue, surface_queue] = queues;
        if graphics_queue == surface_queue && graphics_queue == transfer_queue {
            crate::name_vulkan_object(&device, graphics_queue, format_args!("graphics+surface+transfer"));
        } else if graphics_queue == surface_queue {
            crate::name_vulkan_object(&device, graphics_queue, format_args!("graphics+surface"));
            crate::name_vulkan_object(&device, transfer_queue, format_args!("transfer"));
        } else if surface_queue == transfer_queue {
            crate::name_vulkan_object(&device, graphics_queue, format_args!("graphics"));
            crate::name_vulkan_object(&device, surface_queue, format_args!("surface+transfer"));
        } else {
            crate::name_vulkan_object(&device, graphics_queue, format_args!("graphics"));
            crate::name_vulkan_object(&device, surface_queue, format_args!("surface"));
            crate::name_vulkan_object(&device, transfer_queue, format_args!("transfer"));
        }

        let device = Device {
            sync2: khr::Synchronization2::new(instance, &device),
            surface: khr::Surface::new(entry, instance),
            swapchain: khr::Swapchain::new(instance, &device),
            dynamic_rendering: khr::DynamicRendering::new(instance, &device),
            inner: Box::leak(Box::new(device)),
            graphics_queue,
            surface_queue,
            transfer_queue,
        };

        crate::name_vulkan_object(&device, device.inner.handle(), format_args!("{}", self.name));
        // These two do not seem to interact with validation layers / RenderDoc very well.
        //crate::name_vulkan_object(&device, physical_device.inner, format_args!("{}", physical_device.name));
        //crate::name_vulkan_object(&device, instance.handle(), format_args!("{}", env!("CARGO_PKG_NAME")));

        Ok(device)
    }
}

fn create_device_queue_create_infos<'a, const N: usize>(
    queue_families: &[QueueFamily; N],
    ones: &'a [f32],
) -> ArrayVec<vk::DeviceQueueCreateInfo<'a>, N> {
    let mut results = ArrayVec::<vk::DeviceQueueCreateInfo, N>::new();
    'queue_families: for &queue_family in queue_families {
        for create_info in &results {
            if create_info.queue_family_index == queue_family.index {
                continue 'queue_families;
            }
        }
        let count = queue_families.iter().filter(|qf| qf.index == queue_family.index).count();
        let count = count.min(queue_family.max_count);
        let create_info = vk::DeviceQueueCreateInfo::default().queue_family_index(queue_family.index).queue_priorities(&ones[..count]);
        results.push(create_info);
    }
    results
}

fn get_device_queues<const N: usize>(device: &ash::Device, queue_families: &[QueueFamily; N], queues: &mut [vk::Queue; N]) {
    let mut picks = ArrayVec::<u32, N>::new();
    for (&queue_family, queue) in queue_families.iter().zip(queues.iter_mut()) {
        let queue_index = picks.iter().filter(|index| **index == queue_family.index).count() as u32;
        let queue_index = queue_index.min((queue_family.max_count - 1) as u32);
        *queue = unsafe { device.get_device_queue(queue_family.index, queue_index) };
        picks.push(queue_family.index);
    }
}
