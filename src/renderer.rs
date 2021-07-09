use crate::{Error, Foundation};
use ash::extensions::khr;
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::{vk, Device};

pub struct Renderer<'a> {
    /// Held by Renderer to ensure that the Devices are dropped before
    /// the Instance.
    _foundation: &'a Foundation,
    /// A list of all viable PhysicalDevices, can be used to populate
    /// e.g. a GPU selection list.
    #[allow(dead_code)]
    physical_devices: Vec<vk::PhysicalDevice>,
    #[allow(dead_code)]
    device: Device,
    #[allow(dead_code)]
    queue: vk::Queue,
}

impl Renderer<'_> {
    pub fn new(foundation: &Foundation) -> Result<Renderer<'_>, Error> {
        let surface_ext = khr::Surface::new(&foundation.entry, &foundation.instance);
        let queue_family_supports_surface = |pd: vk::PhysicalDevice, index: u32| {
            let support = unsafe {
                surface_ext.get_physical_device_surface_support(pd, index, foundation.surface)
            };
            if let Ok(true) = support {
                true
            } else {
                false
            }
        };

        let all_physical_devices = unsafe { foundation.instance.enumerate_physical_devices() }
            .map_err(|err| Error::VulkanEnumeratePhysicalDevices(err))?;
        let physical_devices_and_indices = all_physical_devices
            .into_iter()
            .filter_map(|physical_device| {
                let queue_families = unsafe {
                    foundation
                        .instance
                        .get_physical_device_queue_family_properties(physical_device)
                };
                let filter_queue_family = |(i, props): (usize, &vk::QueueFamilyProperties)| {
                    let (index, properties) = (i as u32, props);
                    if !properties.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        None
                    } else if !queue_family_supports_surface(physical_device, index) {
                        None
                    } else {
                        Some(index)
                    }
                };
                let graphics_capable_queue_family_index = queue_families
                    .iter()
                    .enumerate()
                    .find_map(filter_queue_family)?;
                Some((physical_device, graphics_capable_queue_family_index as u32))
            })
            .collect::<Vec<(vk::PhysicalDevice, u32)>>();
        let (physical_device, graphics_family_index) = physical_devices_and_indices
            .iter()
            .next()
            .map(|(pd, idx)| (*pd, *idx))
            .ok_or(Error::VulkanPhysicalDeviceMissing)?;
        let physical_devices = physical_devices_and_indices
            .into_iter()
            .map(|(pd, _)| pd)
            .collect();

        let queue_priorities = &[1.0];
        let queue_create_infos = &[vk::DeviceQueueCreateInfo {
            queue_family_index: graphics_family_index,
            queue_count: queue_priorities.len() as u32,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        }];
        let physical_device_features = &[vk::PhysicalDeviceFeatures {
            ..Default::default()
        }];
        let device_create_info = vk::DeviceCreateInfo {
            p_queue_create_infos: queue_create_infos.as_ptr(),
            queue_create_info_count: queue_create_infos.len() as u32,
            p_enabled_features: physical_device_features.as_ptr(),
            ..Default::default()
        };

        // TODO: Can use allocator
        let allocation_callbacks = None;
        let device = unsafe {
            foundation
                .instance
                .create_device(physical_device, &device_create_info, allocation_callbacks)
                .map_err(|err| Error::VulkanDeviceCreation(err))
        }?;
        let queue = unsafe { device.get_device_queue(graphics_family_index, 0) };

        // Next up: setup swapchain?

        Ok(Renderer {
            _foundation: foundation,
            physical_devices,
            device,
            queue,
        })
    }
}
