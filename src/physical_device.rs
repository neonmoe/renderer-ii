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
    pub swapchain_format: vk::Format,
    pub swapchain_color_space: vk::ColorSpaceKHR,
    pub depth_format: vk::Format,
    pub extensions: Vec<String>,
}

impl PhysicalDevice {
    pub fn extension_supported(&self, extension: &str) -> bool {
        self.extensions.iter().any(|e| e == extension)
    }

    /// Returns how many bytes of memorythis process is using from the physical
    /// device's memory heaps.
    pub fn get_memory_usage(&self, instance: &Instance) -> Option<u64> {
        profiling::scope!("query used vram");
        if self.extension_supported("VK_EXT_memory_budget") {
            let mut memory_budget_properties = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
            let mut memory_properties = vk::PhysicalDeviceMemoryProperties2::builder()
                .push_next(&mut memory_budget_properties)
                .build();
            unsafe { instance.get_physical_device_memory_properties2(self.inner, &mut memory_properties) };
            Some(memory_budget_properties.heap_usage.iter().sum())
        } else {
            None
        }
    }
}

/// Returns a list of GPUs that have the required capabilities. They
/// are also ordered by performance, based on a heuristic (mostly:
/// discrete gpu > integrated gpu > virtual gpu > cpu).
pub fn get_physical_devices(entry: &Entry, instance: &Instance, surface: vk::SurfaceKHR) -> Result<Vec<PhysicalDevice>, Error> {
    profiling::scope!("physical device enumeration");
    let surface_ext = khr::Surface::new(entry, instance);
    let physical_devices = {
        profiling::scope!("vk::enumerate_physical_devices");
        unsafe { instance.enumerate_physical_devices() }.map_err(Error::VulkanEnumeratePhysicalDevices)?
    };
    let mut capable_physical_devices = physical_devices
        .into_iter()
        .filter_map(|physical_device| filter_capable_device(instance, &surface_ext, surface, physical_device))
        .collect::<Result<Vec<_>, Error>>()?;
    capable_physical_devices.sort_by(|a, b| {
        let type_score = |properties: &vk::PhysicalDeviceProperties| match properties.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => 30,
            vk::PhysicalDeviceType::INTEGRATED_GPU => 20,
            vk::PhysicalDeviceType::VIRTUAL_GPU => 10,
            vk::PhysicalDeviceType::CPU => 0,
            _ => 0,
        };
        let queue_score = |gfx, surf| if gfx == surf { 1 } else { 0 };
        let a_score = type_score(&a.properties) + queue_score(a.graphics_queue_family.index, a.surface_queue_family.index);
        let b_score = type_score(&b.properties) + queue_score(b.graphics_queue_family.index, b.surface_queue_family.index);
        b_score.cmp(&a_score)
    });
    Ok(capable_physical_devices)
}

fn filter_capable_device(
    instance: &Instance,
    surface_ext: &khr::Surface,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> Option<Result<PhysicalDevice, Error>> {
    profiling::scope!("physical device capability checks");

    let properties = unsafe {
        profiling::scope!("vk::get_physical_device_properties");
        instance.get_physical_device_properties(physical_device)
    };
    if properties.api_version < vk::API_VERSION_1_3 {
        return None;
    }

    let extensions = get_extensions(instance, physical_device);
    if extensions.iter().all(|s| s != "VK_KHR_swapchain") {
        return None;
    }

    if !physical_device_features::has_required_features(instance, physical_device) {
        return None;
    }

    let queue_families = unsafe {
        profiling::scope!("vk::get_physical_device_queue_family_properties");
        instance.get_physical_device_queue_family_properties(physical_device)
    };
    let mut graphics_queue_family = None;
    let mut surface_queue_family = None;
    let mut transfer_queue_family = None;
    for (index, queue_family_properties) in queue_families.into_iter().enumerate() {
        let queue_family = QueueFamily {
            index: index as u32,
            max_count: queue_family_properties.queue_count as usize,
        };
        let surface_support = unsafe {
            profiling::scope!("vk::get_physical_device_surface_support");
            surface_ext.get_physical_device_surface_support(physical_device, index as u32, surface)
        };
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

    let has_feature = |format: vk::Format, feature: vk::FormatFeatureFlags| -> bool {
        let format_properties = unsafe {
            profiling::scope!("vk::get_physical_device_format_properties");
            instance.get_physical_device_format_properties(physical_device, format)
        };
        format_properties.optimal_tiling_features.contains(feature)
    };

    if !has_feature(vk::Format::R8G8B8A8_SRGB, vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR) {
        return None;
    }

    // From the spec:
    // VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT feature...
    // ...must be supported for at least one of VK_FORMAT_D24_UNORM_S8_UINT and VK_FORMAT_D32_SFLOAT_S8_UINT.
    let depth_format = if has_feature(vk::Format::D24_UNORM_S8_UINT, vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT) {
        vk::Format::D24_UNORM_S8_UINT
    } else if has_feature(vk::Format::D32_SFLOAT_S8_UINT, vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT) {
        vk::Format::D32_SFLOAT_S8_UINT
    } else {
        return None;
    };

    let (swapchain_format, swapchain_color_space) = {
        let surface_formats = match unsafe {
            profiling::scope!("vk::get_physical_device_surface_formats");
            surface_ext
                .get_physical_device_surface_formats(physical_device, surface)
                .map_err(Error::VulkanPhysicalDeviceSurfaceQuery)
        } {
            Ok(formats) => formats,
            Err(err) => return Some(Err(err)),
        };
        if let Some(format) = surface_formats.iter().find(|format| is_uncompressed_srgb(format.format)) {
            (format.format, format.color_space)
        } else {
            log::warn!("No SRGB format found for swapchain. The image may look too dark.");
            (surface_formats[0].format, surface_formats[0].color_space)
        }
    };

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

        Some(Ok(PhysicalDevice {
            name,
            uuid,
            inner: physical_device,
            properties,
            graphics_queue_family,
            surface_queue_family,
            transfer_queue_family,
            swapchain_format,
            swapchain_color_space,
            depth_format,
            extensions,
        }))
    } else {
        None
    }
}

#[profiling::function]
fn get_extensions(instance: &Instance, physical_device: vk::PhysicalDevice) -> Vec<String> {
    match unsafe {
        profiling::scope!("vk::enumerate_device_extension_properties");
        instance.enumerate_device_extension_properties(physical_device)
    } {
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

fn is_uncompressed_srgb(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::R8_SRGB
            | vk::Format::R8G8_SRGB
            | vk::Format::R8G8B8_SRGB
            | vk::Format::R8G8B8A8_SRGB
            | vk::Format::B8G8R8_SRGB
            | vk::Format::B8G8R8A8_SRGB
            | vk::Format::A8B8G8R8_SRGB_PACK32
    )
}
