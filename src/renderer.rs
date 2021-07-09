use crate::{Error, Foundation};
use ash::extensions::khr;
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::{vk, Device};

pub struct Renderer<'a> {
    /// Held by Renderer to ensure that the Devices are dropped before
    /// the Instance.
    _foundation: &'a Foundation,
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
        let physical_devices = all_physical_devices
            .into_iter()
            .filter_map(|physical_device| {
                let queue_families = unsafe {
                    foundation
                        .instance
                        .get_physical_device_queue_family_properties(physical_device)
                };
                let mut graphics_queue_family = None;
                let mut surface_queue_family = None;
                for (index, queue_family) in queue_families.into_iter().enumerate() {
                    if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        graphics_queue_family = Some(index as u32);
                    }
                    if queue_family_supports_surface(physical_device, index as u32) {
                        surface_queue_family = Some(index as u32);
                    }
                    if graphics_queue_family == surface_queue_family {
                        // If there's a queue which supports both, prefer that one.
                        break;
                    }
                }
                if let (Some(graphics_queue_family), Some(surface_queue_family)) =
                    (graphics_queue_family, surface_queue_family)
                {
                    Some((physical_device, graphics_queue_family, surface_queue_family))
                } else {
                    None
                }
            })
            .collect::<Vec<(vk::PhysicalDevice, u32, u32)>>();
        log::debug!("Suitable physical devices: {:#?}", physical_devices);

        // Just pick the first one for now. Not sure what kind of a
        // situation would yield multiple physical devices capable of
        // presenting to the surface.
        let (physical_device, graphics_family_index, surface_family_index) = physical_devices
            .into_iter()
            .next()
            .ok_or(Error::VulkanPhysicalDeviceMissing)?;

        let queue_priorities = &[1.0];
        // TODO: always create two infos or prefer to make one with queue_count if they're the same?
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
            device,
            queue,
        })
    }
}
