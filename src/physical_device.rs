use crate::{physical_device_features, Error};
use ash::extensions::khr;
use ash::vk;
use ash::{Entry, Instance};
use std::ffi::CStr;

/// A unique id for every distinct GPU.
pub struct GpuId([u8; 16]);

#[derive(Clone, Copy, PartialEq)]
pub struct QueueFamily {
    pub index: u32,
    pub max_count: usize,
}

pub struct PhysicalDevice {
    pub name: String,
    pub uuid: GpuId,
    pub inner: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub graphics_queue_family: QueueFamily,
    pub surface_queue_family: QueueFamily,
    pub transfer_queue_family: QueueFamily,
    pub extensions: Vec<String>,
}

impl PhysicalDevice {
    pub fn extension_supported(&self, extension: &str) -> bool {
        self.extensions.iter().any(|e| e == extension)
    }
}

/// Returns a list of GPUs that have the required capabilities. They
/// are also ordered by performance, based on a heuristic (mostly:
/// discrete gpu > integrated gpu > virtual gpu > cpu).
pub fn get_physical_devices(entry: &Entry, instance: &Instance, surface: vk::SurfaceKHR) -> Result<Vec<PhysicalDevice>, Error> {
    let surface_ext = khr::Surface::new(entry, instance);
    let mut capable_physical_devices = unsafe { instance.enumerate_physical_devices() }
        .map_err(Error::VulkanEnumeratePhysicalDevices)?
        .into_iter()
        .filter_map(|physical_device| filter_capable_device(instance, &surface_ext, surface, physical_device))
        .collect::<Vec<_>>();
    capable_physical_devices.sort_by(|a, b| {
        let type_score = |properties: vk::PhysicalDeviceProperties| match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 30,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 20,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 10,
            vk::PhysicalDeviceType::CPU => 0,
            _ => 0,
        };
        let queue_score = |graphics_queue, surface_queue| {
            if graphics_queue == surface_queue {
                1
            } else {
                0
            }
        };
        let a_score = type_score(a.properties) + queue_score(a.graphics_queue_family.index, a.surface_queue_family.index);
        let b_score = type_score(b.properties) + queue_score(b.graphics_queue_family.index, b.surface_queue_family.index);
        b_score.cmp(&a_score)
    });
    Ok(capable_physical_devices)
}

fn filter_capable_device(
    instance: &Instance,
    surface_ext: &khr::Surface,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> Option<PhysicalDevice> {
    let extensions = get_extensions(instance, physical_device);
    if extensions.iter().all(|s| s != "VK_KHR_swapchain") {
        return None;
    }

    if !physical_device_features::has_required_features(instance, physical_device) {
        return None;
    }

    let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let mut graphics_queue_family = None;
    let mut surface_queue_family = None;
    let mut transfer_queue_family = None;
    for (index, queue_family_properties) in queue_families.into_iter().enumerate() {
        let queue_family = QueueFamily {
            index: index as u32,
            max_count: queue_family_properties.queue_count as usize,
        };
        let surface_support = unsafe { surface_ext.get_physical_device_surface_support(physical_device, index as u32, surface) };
        if graphics_queue_family.is_none() || surface_queue_family.is_none() || graphics_queue_family != surface_queue_family {
            if queue_family_properties.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                graphics_queue_family = Some(queue_family);
            }
            if surface_support == Ok(true) {
                surface_queue_family = Some(queue_family);
            }
        }
        // Prefer transfer-only queues, as they can probably do
        // transfers without disturbing rendering.
        let transfer_only = queue_family_properties.queue_flags.contains(vk::QueueFlags::TRANSFER)
            && !queue_family_properties.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            && !queue_family_properties.queue_flags.contains(vk::QueueFlags::COMPUTE);
        if transfer_queue_family.is_none() || transfer_only {
            transfer_queue_family = Some(queue_family);
        }
    }

    let properties = unsafe { instance.get_physical_device_properties(physical_device) };
    let format_properties = unsafe { instance.get_physical_device_format_properties(physical_device, vk::Format::R8G8B8A8_SRGB) };
    match format_properties.optimal_tiling_features {
        features if !features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR) => {
            log::warn!(
                "physical device '{}' does not have SAMPLED_IMAGE_FILTER_LINEAR for optimal tiling 32-bit srgb images",
                get_device_name(&properties)
            );
            return None;
        }
        _ => {}
    }

    if let (Some(graphics_queue_family), Some(surface_queue_family), Some(transfer_queue_family)) =
        (graphics_queue_family, surface_queue_family, transfer_queue_family)
    {
        let name = get_device_name(&properties);
        let pd_type = match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => " (Discrete GPU)",
            vk::PhysicalDeviceType::INTEGRATED_GPU => " (Integrated GPU)",
            vk::PhysicalDeviceType::VIRTUAL_GPU => " (vGPU)",
            vk::PhysicalDeviceType::CPU => " (CPU)",
            _ => "",
        };
        let name = format!("{}{}", name, pd_type);
        let uuid = GpuId(properties.pipeline_cache_uuid);

        Some(PhysicalDevice {
            name,
            uuid,
            inner: physical_device,
            properties,
            graphics_queue_family,
            surface_queue_family,
            transfer_queue_family,
            extensions,
        })
    } else {
        None
    }
}

#[profiling::function]
fn get_extensions(instance: &Instance, physical_device: vk::PhysicalDevice) -> Vec<String> {
    match unsafe { instance.enumerate_device_extension_properties(physical_device) } {
        Ok(extensions) => extensions
            .iter()
            .map(|extension_properties| {
                let extension_name_slice = &extension_properties.extension_name[..];
                unsafe { CStr::from_ptr(extension_name_slice.as_ptr()) }
                    .to_string_lossy()
                    .to_string()
            })
            .collect::<Vec<String>>(),
        Err(_) => Vec::with_capacity(0),
    }
}

fn get_device_name(properties: &vk::PhysicalDeviceProperties) -> std::borrow::Cow<'_, str> {
    unsafe { CStr::from_ptr((&properties.device_name[..]).as_ptr()) }.to_string_lossy()
}
