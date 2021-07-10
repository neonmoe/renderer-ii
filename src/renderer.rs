use crate::{Error, Foundation};
use ash::extensions::khr;
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::{vk, Device, Instance};
use std::ffi::CStr;
use std::os::raw::c_char;

#[allow(dead_code)]
pub struct Renderer<'a> {
    /// Held by Renderer to ensure that the Devices are dropped before
    /// the Instance.
    pub foundation: &'a Foundation,
    /// A list of suitable physical devices, for picking between
    /// e.g. a laptop's integrated and discrete GPUs.
    ///
    /// The tuple consists of the display name, and the id passed to a
    /// new Renderer when recreating it with a new physical device.
    pub physical_devices: Vec<(String, [u8; 16])>,

    device: Device,
    swapchain: vk::SwapchainKHR,
    graphics_queue: vk::Queue,
    graphics_family_index: u32,
    surface_queue: vk::Queue,
    surface_family_index: u32,
}

impl Drop for Renderer<'_> {
    fn drop(&mut self) {
        let swapchain_ext = khr::Swapchain::new(&self.foundation.instance, &self.device);
        // TODO: Can use allocator
        unsafe { swapchain_ext.destroy_swapchain(self.swapchain, None) };
    }
}

impl Renderer<'_> {
    pub fn new(
        foundation: &Foundation,
        initial_width: u32,
        initial_height: u32,
        preferred_physical_device: Option<[u8; 16]>,
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> Result<Renderer<'_>, Error> {
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
        let mut physical_devices = all_physical_devices
            .into_iter()
            .filter_map(|physical_device| {
                if !is_extension_supported(
                    &foundation.instance,
                    physical_device,
                    "VK_KHR_swapchain",
                ) {
                    return None;
                }
                let queue_families = unsafe {
                    foundation
                        .instance
                        .get_physical_device_queue_family_properties(physical_device)
                };
                let mut graphics_family_index = None;
                let mut surface_family_index = None;
                for (index, family_index) in queue_families.into_iter().enumerate() {
                    if family_index.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        graphics_family_index = Some(index as u32);
                    }
                    if queue_family_supports_surface(physical_device, index as u32) {
                        surface_family_index = Some(index as u32);
                    }
                    if graphics_family_index == surface_family_index {
                        // If there's a queue which supports both, prefer that one.
                        break;
                    }
                }
                if let (Some(graphics_family_index), Some(surface_family_index)) =
                    (graphics_family_index, surface_family_index)
                {
                    Some((physical_device, graphics_family_index, surface_family_index))
                } else {
                    None
                }
            })
            .collect::<Vec<(vk::PhysicalDevice, u32, u32)>>();

        let (physical_device, graphics_family_index, surface_family_index) =
            if let Some(uuid) = preferred_physical_device {
                physical_devices
                    .iter()
                    .find_map(|tuple| {
                        let properties =
                            unsafe { foundation.instance.get_physical_device_properties(tuple.0) };
                        if properties.pipeline_cache_uuid == uuid {
                            Some(*tuple)
                        } else {
                            None
                        }
                    })
                    .ok_or(Error::VulkanPhysicalDeviceMissing)?
            } else {
                physical_devices.sort_by(|(a, a_gfx, a_surf), (b, b_gfx, b_surf)| {
                    let a_properties =
                        unsafe { foundation.instance.get_physical_device_properties(*a) };
                    let b_properties =
                        unsafe { foundation.instance.get_physical_device_properties(*b) };
                    let type_score =
                        |properties: vk::PhysicalDeviceProperties| match properties.device_type {
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
                    let a_score = type_score(a_properties) + queue_score(a_gfx, a_surf);
                    let b_score = type_score(b_properties) + queue_score(b_gfx, b_surf);
                    // Highest score first.
                    b_score.cmp(&a_score)
                });
                physical_devices
                    .iter()
                    .next()
                    .map(|t| *t)
                    .ok_or(Error::VulkanPhysicalDeviceMissing)?
            };

        let physical_devices = physical_devices
            .into_iter()
            .map(|(pd, _, _)| {
                let properties = unsafe { foundation.instance.get_physical_device_properties(pd) };
                let name = unsafe { CStr::from_ptr((&properties.device_name[..]).as_ptr()) }
                    .to_string_lossy();
                let pd_type = match properties.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => " (Discrete GPU)",
                    vk::PhysicalDeviceType::INTEGRATED_GPU => " (Integrated GPU)",
                    vk::PhysicalDeviceType::VIRTUAL_GPU => " (vCPU)",
                    vk::PhysicalDeviceType::CPU => " (CPU)",
                    _ => "",
                };
                let name = format!("{}{}", name, pd_type);
                log::debug!("Physical device found: {}", name);
                let uuid = properties.pipeline_cache_uuid;
                (name, uuid)
            })
            .collect::<Vec<(String, [u8; 16])>>();

        let queue_priorities = &[1.0, 1.0];
        let make_queue_create_info = |index: u32, count: u32| vk::DeviceQueueCreateInfo {
            queue_family_index: index,
            queue_count: count,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        };
        let queue_create_infos = if graphics_family_index == surface_family_index {
            vec![make_queue_create_info(graphics_family_index, 2)]
        } else {
            vec![
                make_queue_create_info(graphics_family_index, 1),
                make_queue_create_info(surface_family_index, 1),
            ]
        };
        let physical_device_features = &[vk::PhysicalDeviceFeatures {
            ..Default::default()
        }];
        let extensions = &[cstr!("VK_KHR_swapchain")];
        if log::log_enabled!(log::Level::Debug) {
            let cstr_to_str =
                |str_ptr: &*const c_char| unsafe { CStr::from_ptr(*str_ptr) }.to_string_lossy();
            log::debug!(
                "Requested device extensions: {:#?}",
                extensions.iter().map(cstr_to_str).collect::<Vec<_>>()
            );
        }

        let device_create_info = vk::DeviceCreateInfo {
            p_queue_create_infos: queue_create_infos.as_ptr(),
            queue_create_info_count: queue_create_infos.len() as u32,
            p_enabled_features: physical_device_features.as_ptr(),
            pp_enabled_extension_names: extensions.as_ptr(),
            enabled_extension_count: extensions.len() as u32,
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

        let graphics_queue;
        let surface_queue;
        if graphics_family_index == surface_family_index {
            graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };
            surface_queue = unsafe { device.get_device_queue(graphics_family_index, 1) };
        } else {
            graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };
            surface_queue = unsafe { device.get_device_queue(surface_family_index, 0) };
        }

        let swapchain = create_swapchain(
            foundation,
            initial_width,
            initial_height,
            old_swapchain,
            physical_device,
            &device,
            graphics_family_index,
            surface_family_index,
        )?;

        Ok(Renderer {
            foundation,
            physical_devices,
            device,
            swapchain,
            graphics_queue,
            graphics_family_index,
            surface_queue,
            surface_family_index,
        })
    }
}

fn is_extension_supported(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    target_extension_name: &str,
) -> bool {
    match unsafe { instance.enumerate_device_extension_properties(physical_device) } {
        Err(_) => false,
        Ok(extensions) => extensions.iter().any(|extension_properties| {
            let extension_name_slice = &extension_properties.extension_name[..];
            let extension_name =
                unsafe { CStr::from_ptr(extension_name_slice.as_ptr()) }.to_string_lossy();
            extension_name == target_extension_name
        }),
    }
}

fn create_swapchain(
    foundation: &Foundation,
    width: u32,
    height: u32,
    old_swapchain: Option<vk::SwapchainKHR>,
    physical_device: vk::PhysicalDevice,
    device: &Device,
    graphics_family_index: u32,
    surface_family_index: u32,
) -> Result<vk::SwapchainKHR, Error> {
    let surface_ext = khr::Surface::new(&foundation.entry, &foundation.instance);
    let present_mode = {
        let surface_present_modes = unsafe {
            surface_ext
                .get_physical_device_surface_present_modes(physical_device, foundation.surface)
                .map_err(|err| Error::VulkanPhysicalDeviceSurfaceQuery(err))
        }?;
        if surface_present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        }
    };

    let (min_image_count, image_extent) = {
        let surface_capabilities = unsafe {
            surface_ext
                .get_physical_device_surface_capabilities(physical_device, foundation.surface)
                .map_err(|err| Error::VulkanPhysicalDeviceSurfaceQuery(err))
        }?;
        let mut min_image_count = if present_mode == vk::PresentModeKHR::MAILBOX {
            3.max(surface_capabilities.min_image_count)
        } else {
            2.max(surface_capabilities.min_image_count)
        };
        if surface_capabilities.max_image_count != 0
            && surface_capabilities.max_image_count < min_image_count
        {
            min_image_count = surface_capabilities.max_image_count;
        }
        let unset_extent = vk::Extent2D {
            width: u32::MAX,
            height: u32::MAX,
        };
        let image_extent = if surface_capabilities.current_extent != unset_extent {
            surface_capabilities.current_extent
        } else {
            vk::Extent2D { width, height }
        };
        (min_image_count, image_extent)
    };

    let (image_format, image_color_space) = {
        let surface_formats = unsafe {
            surface_ext
                .get_physical_device_surface_formats(physical_device, foundation.surface)
                .map_err(|err| Error::VulkanPhysicalDeviceSurfaceQuery(err))
        }?;
        let format = if let Some(format) = surface_formats.iter().find(|format| {
            format.format == vk::Format::B8G8R8A8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        }) {
            format
        } else {
            &surface_formats[0]
        };
        (format.format, format.color_space)
    };

    let swapchain_ext = khr::Swapchain::new(&foundation.instance, device);
    let mut swapchain_create_info = vk::SwapchainCreateInfoKHR {
        surface: foundation.surface,
        min_image_count,
        image_format,
        image_color_space,
        image_extent,
        image_array_layers: 1,
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
        composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        present_mode,
        clipped: vk::TRUE,
        ..Default::default()
    };
    let queue_family_indices = &[graphics_family_index, surface_family_index];
    if graphics_family_index != surface_family_index {
        swapchain_create_info.image_sharing_mode = vk::SharingMode::CONCURRENT;
        swapchain_create_info.queue_family_index_count = queue_family_indices.len() as u32;
        swapchain_create_info.p_queue_family_indices = queue_family_indices.as_ptr();
    } else {
        swapchain_create_info.image_sharing_mode = vk::SharingMode::EXCLUSIVE;
    }
    if let Some(old_swapchain) = old_swapchain {
        swapchain_create_info.old_swapchain = old_swapchain;
    }
    // TODO: Can use allocator
    unsafe { swapchain_ext.create_swapchain(&swapchain_create_info, None) }
        .map_err(|err| Error::VulkanSwapchainCreation(err))
}
