use arrayvec::ArrayVec;
use ash::extensions::khr;
use ash::vk;
use ash::{Entry, Instance};
use core::ffi::CStr;
use core::fmt::{self, Display, Formatter};
use std::error::Error;

mod device_creation;
pub mod limits;
mod physical_device_features;

use limits::PhysicalDeviceLimitBreak;
use physical_device_features::SupportedFeatures;

pub const HDR_COLOR_ATTACHMENT_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

#[derive(thiserror::Error, Debug)]
pub enum PhysicalDeviceRejectionReason {
    #[error("failed to enumerate vulkan physical devices")]
    Enumeration(#[source] vk::Result),
    #[error("failed to query surface support")]
    SurfaceQuery(#[source] vk::Result),
    #[error("required vulkan version {0}.{1} is not supported")]
    VulkanVersion(u32, u32),
    #[error("graphics driver does not support all of the required features: {0:#?}")]
    DeviceRequirements(SupportedFeatures),
    #[error("graphics driver does not support the device extension: {0}")]
    Extension(&'static str),
    #[error("gpu does not support the amount of resources needed")]
    DeviceLimits(#[from] PhysicalDeviceLimitBreak),
    #[error("the texture format {0:?} is not supported with flags {1:?}")]
    TextureFormatSupport(vk::Format, vk::FormatFeatureFlags),
    #[error("graphics, surface, or transfer queue family not found")]
    QueueFamilyMissing,
}

#[derive(Debug)]
pub struct RejectionReasonList(ArrayVec<PhysicalDeviceRejectionReason, 128>);
impl From<PhysicalDeviceRejectionReason> for RejectionReasonList {
    fn from(reason: PhysicalDeviceRejectionReason) -> Self {
        let mut reasons = ArrayVec::new();
        reasons.push(reason);
        RejectionReasonList(reasons)
    }
}
impl Error for RejectionReasonList {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}
impl Display for RejectionReasonList {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        writeln!(f, "graphics driver or hardware does not support the required features:")?;
        for reason in &self.0 {
            write!(f, "- ")?;
            let mut next_reason: Option<&(dyn Error + 'static)> = Some(reason);
            let mut indents = 0;
            while let Some(reason) = next_reason {
                writeln!(f, "{:indents$}{reason}", "")?;
                next_reason = reason.source();
                indents += 4;
            }
        }
        Ok(())
    }
}

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
            let mut memory_properties = vk::PhysicalDeviceMemoryProperties2::default().push_next(&mut memory_budget_properties);
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
pub fn get_physical_devices(
    entry: &Entry,
    instance: &Instance,
    surface: vk::SurfaceKHR,
) -> Vec<Result<PhysicalDevice, RejectionReasonList>> {
    profiling::scope!("physical device enumeration");
    let surface_ext = khr::Surface::new(entry, instance);
    let physical_devices = {
        profiling::scope!("vk::enumerate_physical_devices");
        match unsafe { instance.enumerate_physical_devices() } {
            Ok(pds) => pds,
            Err(err) => return vec![Err(PhysicalDeviceRejectionReason::Enumeration(err).into())],
        }
    };
    let mut enumerated_physical_devices = physical_devices
        .into_iter()
        .map(|physical_device| filter_capable_device(instance, &surface_ext, surface, physical_device))
        .collect::<Vec<Result<_, RejectionReasonList>>>();
    enumerated_physical_devices.sort_by(|a, b| {
        let a_score;
        let b_score;
        if let (Ok(a), Ok(b)) = (a, b) {
            let type_score = |properties: &vk::PhysicalDeviceProperties| match properties.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 30,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 20,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 10,
                vk::PhysicalDeviceType::CPU => 0,
                _ => 0,
            };
            let queue_score = |gfx, surf| if gfx == surf { 2 } else { 0 };
            a_score = type_score(&a.properties) + queue_score(a.graphics_queue_family.index, a.surface_queue_family.index);
            b_score = type_score(&b.properties) + queue_score(b.graphics_queue_family.index, b.surface_queue_family.index);
        } else {
            a_score = if a.is_ok() { 100 } else { 0 };
            b_score = if b.is_ok() { 100 } else { 0 };
        }
        b_score.cmp(&a_score)
    });
    enumerated_physical_devices
}

#[allow(clippy::result_large_err)]
fn filter_capable_device(
    instance: &Instance,
    surface_ext: &khr::Surface,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> Result<PhysicalDevice, RejectionReasonList> {
    profiling::scope!("physical device capability checks");

    let mut rejection_reasons = ArrayVec::new();
    let mut reject = |reason: PhysicalDeviceRejectionReason| {
        let _ = rejection_reasons.try_push(reason);
    };

    let props = unsafe {
        profiling::scope!("vk::get_physical_device_properties");
        instance.get_physical_device_properties(physical_device)
    };
    if props.api_version < crate::instance::REQUIRED_VULKAN_VERSION {
        reject(PhysicalDeviceRejectionReason::VulkanVersion(1, 3));
    };

    let extensions = get_extensions(instance, physical_device);
    let mut assert_ext_supported = |ext_name: &'static str| {
        if extensions.iter().all(|s| s != ext_name) {
            reject(PhysicalDeviceRejectionReason::Extension(ext_name));
        }
    };
    assert_ext_supported(khr::Swapchain::NAME.to_str().unwrap());

    if let Err(reqs) = physical_device_features::has_required_features(instance, physical_device) {
        reject(PhysicalDeviceRejectionReason::DeviceRequirements(reqs));
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

    // TODO: Add vk-profile checks for formats
    let format_supported = |format: vk::Format, flags: vk::FormatFeatureFlags| -> bool {
        let format_properties = unsafe {
            profiling::scope!("vk::get_physical_device_format_properties");
            instance.get_physical_device_format_properties(physical_device, format)
        };
        format_properties.optimal_tiling_features.contains(flags)
    };
    let mut require_format = |format: vk::Format, flags: vk::FormatFeatureFlags| -> bool {
        if !format_supported(format, flags) {
            reject(PhysicalDeviceRejectionReason::TextureFormatSupport(format, flags));
            false
        } else {
            true
        }
    };

    let texture_usage_features =
        vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR | vk::FormatFeatureFlags::TRANSFER_DST;
    require_format(vk::Format::R8G8B8A8_SRGB, texture_usage_features); // Uncompressed textures
    require_format(vk::Format::R8G8B8A8_UNORM, texture_usage_features); // Uncompressed textures
    require_format(vk::Format::BC7_SRGB_BLOCK, texture_usage_features); // Compressed textures
    require_format(vk::Format::BC7_UNORM_BLOCK, texture_usage_features); // Compressed textures
    require_format(HDR_COLOR_ATTACHMENT_FORMAT, vk::FormatFeatureFlags::COLOR_ATTACHMENT); // HDR color attachments

    // From the spec:
    // VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT feature...
    // ...must be supported for at least one of VK_FORMAT_D24_UNORM_S8_UINT and VK_FORMAT_D32_SFLOAT_S8_UINT.
    let depth_formats = [vk::Format::D24_UNORM_S8_UINT, vk::Format::D32_SFLOAT_S8_UINT];
    let depth_format = if format_supported(depth_formats[0], vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT) {
        depth_formats[0]
    } else {
        depth_formats[1]
    };

    let (swapchain_format, swapchain_color_space) = {
        let surface_formats = unsafe {
            profiling::scope!("vk::get_physical_device_surface_formats");
            match surface_ext.get_physical_device_surface_formats(physical_device, surface) {
                Ok(fmts) => fmts,
                Err(err) => {
                    reject(PhysicalDeviceRejectionReason::SurfaceQuery(err));
                    Vec::with_capacity(0)
                }
            }
        };
        if let Some(format) = surface_formats.iter().find(|format| is_uncompressed_srgb(format.format)) {
            (format.format, format.color_space)
        } else {
            log::warn!("No SRGB format found for swapchain. The image may look too dark.");
            (surface_formats[0].format, surface_formats[0].color_space)
        }
    };

    {
        use limits::*;
        use vk::DescriptorType as D;

        // The limits are set up to be the minimum of the system limit and the
        // targeted profile's limit. This will provide useful errors in two cases:
        // - During development, having accidentally made something require too much resources (over profile)
        // - Debugging user issues, with their GPU being below the minimum spec and hitting limits (over system)
        use vk_profile::pd_limit;
        let mut limits = vk::PhysicalDeviceLimits {
            max_uniform_buffer_range: pd_limit!["max_uniform_buffer_range"].min(props.limits.max_uniform_buffer_range),
            max_storage_buffer_range: pd_limit!["max_storage_buffer_range"].min(props.limits.max_storage_buffer_range),
            max_push_constants_size: pd_limit!["max_push_constants_size"].min(props.limits.max_push_constants_size),
            max_bound_descriptor_sets: pd_limit!["max_bound_descriptor_sets"].min(props.limits.max_bound_descriptor_sets),
            max_per_stage_resources: pd_limit!["max_per_stage_resources"].min(props.limits.max_per_stage_resources),
            max_vertex_input_attributes: pd_limit!["max_vertex_input_attributes"].min(props.limits.max_vertex_input_attributes),
            max_vertex_input_bindings: pd_limit!["max_vertex_input_bindings"].min(props.limits.max_vertex_input_bindings),
            max_vertex_input_attribute_offset: pd_limit!["max_vertex_input_attribute_offset"]
                .min(props.limits.max_vertex_input_attribute_offset),
            max_vertex_input_binding_stride: pd_limit!["max_vertex_input_binding_stride"].min(props.limits.max_vertex_input_binding_stride),
            max_per_stage_descriptor_samplers: pd_limit!["max_per_stage_descriptor_samplers"]
                .min(props.limits.max_per_stage_descriptor_samplers),
            max_per_stage_descriptor_uniform_buffers: pd_limit!["max_per_stage_descriptor_uniform_buffers"]
                .min(props.limits.max_per_stage_descriptor_uniform_buffers),
            max_per_stage_descriptor_storage_buffers: pd_limit!["max_per_stage_descriptor_storage_buffers"]
                .min(props.limits.max_per_stage_descriptor_storage_buffers),
            max_per_stage_descriptor_sampled_images: pd_limit!["max_per_stage_descriptor_sampled_images"]
                .min(props.limits.max_per_stage_descriptor_sampled_images),
            max_per_stage_descriptor_storage_images: pd_limit!["max_per_stage_descriptor_storage_images"]
                .min(props.limits.max_per_stage_descriptor_storage_images),
            max_per_stage_descriptor_input_attachments: pd_limit!["max_per_stage_descriptor_input_attachments"]
                .min(props.limits.max_per_stage_descriptor_input_attachments),
            max_descriptor_set_samplers: pd_limit!["max_descriptor_set_samplers"].min(props.limits.max_descriptor_set_samplers),
            max_descriptor_set_uniform_buffers: pd_limit!["max_descriptor_set_uniform_buffers"]
                .min(props.limits.max_descriptor_set_uniform_buffers),
            max_descriptor_set_uniform_buffers_dynamic: pd_limit!["max_descriptor_set_uniform_buffers_dynamic"]
                .min(props.limits.max_descriptor_set_uniform_buffers_dynamic),
            max_descriptor_set_storage_buffers: pd_limit!["max_descriptor_set_storage_buffers"]
                .min(props.limits.max_descriptor_set_storage_buffers),
            max_descriptor_set_storage_buffers_dynamic: pd_limit!["max_descriptor_set_storage_buffers_dynamic"]
                .min(props.limits.max_descriptor_set_storage_buffers_dynamic),
            max_descriptor_set_sampled_images: pd_limit!["max_descriptor_set_sampled_images"]
                .min(props.limits.max_descriptor_set_sampled_images),
            max_descriptor_set_storage_images: pd_limit!["max_descriptor_set_storage_images"]
                .min(props.limits.max_descriptor_set_storage_images),
            max_descriptor_set_input_attachments: pd_limit!["max_descriptor_set_input_attachments"]
                .min(props.limits.max_descriptor_set_input_attachments),
            ..Default::default()
        };
        limits.max_per_stage_descriptor_uniform_buffers = 26; // TODO: Require less uniform buffers

        let mut check_limit_break = |r: Result<(), PhysicalDeviceLimitBreak>| {
            if let Err(reason) = r {
                reject(reason.into());
            }
        };
        check_limit_break(uniform_buffer_range(limits.max_uniform_buffer_range));
        check_limit_break(storage_buffer_range(limits.max_storage_buffer_range));
        check_limit_break(push_constants_size(limits.max_push_constants_size));
        check_limit_break(bound_descriptor_sets(limits.max_bound_descriptor_sets));
        check_limit_break(per_stage_resources(limits.max_per_stage_resources));
        check_limit_break(vertex_input_attributes(limits.max_vertex_input_attributes));
        check_limit_break(vertex_input_bindings(limits.max_vertex_input_bindings));
        check_limit_break(vertex_input_attribute_offset(limits.max_vertex_input_attribute_offset));
        check_limit_break(vertex_input_binding_stride(limits.max_vertex_input_binding_stride));

        let mut check_per_stage_descs = |dt: vk::DescriptorType, limit: u32| {
            if let Err(reason) = per_stage_descriptors(dt, limit) {
                reject(reason.into());
            }
        };
        check_per_stage_descs(D::SAMPLER, limits.max_per_stage_descriptor_samplers);
        check_per_stage_descs(D::UNIFORM_BUFFER, limits.max_per_stage_descriptor_uniform_buffers);
        check_per_stage_descs(D::STORAGE_BUFFER, limits.max_per_stage_descriptor_storage_buffers);
        check_per_stage_descs(D::SAMPLED_IMAGE, limits.max_per_stage_descriptor_sampled_images);
        check_per_stage_descs(D::STORAGE_IMAGE, limits.max_per_stage_descriptor_storage_images);
        check_per_stage_descs(D::INPUT_ATTACHMENT, limits.max_per_stage_descriptor_input_attachments);

        let mut check_per_set_descs = |dt: vk::DescriptorType, limit: u32| {
            if let Err(reason) = per_set_descriptors(dt, limit) {
                reject(reason.into());
            }
        };
        check_per_set_descs(D::SAMPLER, limits.max_descriptor_set_samplers);
        check_per_set_descs(D::UNIFORM_BUFFER, limits.max_descriptor_set_uniform_buffers);
        check_per_set_descs(D::UNIFORM_BUFFER_DYNAMIC, limits.max_descriptor_set_uniform_buffers_dynamic);
        check_per_set_descs(D::STORAGE_BUFFER, limits.max_descriptor_set_storage_buffers);
        check_per_set_descs(D::STORAGE_BUFFER_DYNAMIC, limits.max_descriptor_set_storage_buffers_dynamic);
        check_per_set_descs(D::SAMPLED_IMAGE, limits.max_descriptor_set_sampled_images);
        check_per_set_descs(D::STORAGE_IMAGE, limits.max_descriptor_set_storage_images);
        check_per_set_descs(D::INPUT_ATTACHMENT, limits.max_descriptor_set_input_attachments);
    }

    if let (Some(graphics_queue_family), Some(surface_queue_family), Some(transfer_queue_family)) =
        (graphics_queue_family, surface_queue_family, transfer_queue_family)
    {
        let name = get_device_name(&props);
        let pd_type = match props.device_type {
            vk::PhysicalDeviceType::DISCRETE_GPU => " (Discrete GPU)",
            vk::PhysicalDeviceType::INTEGRATED_GPU => " (Integrated GPU)",
            vk::PhysicalDeviceType::VIRTUAL_GPU => " (vGPU)",
            vk::PhysicalDeviceType::CPU => " (CPU)",
            _ => "",
        };
        let name = format!("{}{}", name, pd_type);
        let uuid = GpuId(props.pipeline_cache_uuid);

        if rejection_reasons.is_empty() {
            return Ok(PhysicalDevice {
                name,
                uuid,
                inner: physical_device,
                properties: props,
                graphics_queue_family,
                surface_queue_family,
                transfer_queue_family,
                swapchain_format,
                swapchain_color_space,
                depth_format,
                extensions,
            });
        }
    } else {
        reject(PhysicalDeviceRejectionReason::QueueFamilyMissing);
    }

    Err(RejectionReasonList(rejection_reasons))
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

fn get_device_name(properties: &vk::PhysicalDeviceProperties) -> alloc::borrow::Cow<'_, str> {
    unsafe { CStr::from_ptr(properties.device_name[..].as_ptr()) }.to_string_lossy()
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
