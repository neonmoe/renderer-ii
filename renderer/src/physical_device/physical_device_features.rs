use core::ffi::CStr;

use ash::{vk, Instance};

pub const REQUIRED_DEVICE_EXTENSIONS: &[&CStr] = &[
    cstr!("VK_KHR_swapchain"),
    cstr!("VK_KHR_synchronization2"),
    cstr!("VK_KHR_dynamic_rendering"),
];
pub const OPTIONAL_DEVICE_EXTENSIONS: &[&CStr] = &[cstr!("VK_KHR_portability_subset"), cstr!("VK_EXT_memory_budget")];
pub const TOTAL_DEVICE_EXTENSIONS: usize = REQUIRED_DEVICE_EXTENSIONS.len() + OPTIONAL_DEVICE_EXTENSIONS.len();

#[derive(Debug)]
pub struct SupportedFeatures {
    sampler_anisotropy: bool,
    sample_rate_shading: bool,
    texture_array_dynamic_indexing: bool,
    partially_bound_descriptors: bool,
    extended_dynamic_state: bool,
    pipeline_creation_cache_control: bool,
    synchronization2: bool,
    uniform_buffer_standard_layout: bool,
    dynamic_rendering: bool,
    shader_draw_parameters: bool,
}

pub fn create_with_features(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device_create_info: vk::DeviceCreateInfo,
) -> Result<ash::Device, vk::Result> {
    // Note: requested features should match what is checked in has_required_features
    let features = vk::PhysicalDeviceFeatures::default()
        .sampler_anisotropy(true)
        .sample_rate_shading(true)
        .shader_sampled_image_array_dynamic_indexing(true);
    let mut descriptor_indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default().descriptor_binding_partially_bound(true);
    let mut extended_dynamic_state_features = vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default().extended_dynamic_state(true);
    let mut pipeline_creation_cache_control_features =
        vk::PhysicalDevicePipelineCreationCacheControlFeatures::default().pipeline_creation_cache_control(true);
    let mut synchronization2_features = vk::PhysicalDeviceSynchronization2FeaturesKHR::default().synchronization2(true);
    let mut uniform_buffer_standard_layout_features =
        vk::PhysicalDeviceUniformBufferStandardLayoutFeatures::default().uniform_buffer_standard_layout(true);
    let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeaturesKHR::default().dynamic_rendering(true);
    let mut vk11_features = vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
    let device_create_info = device_create_info
        .push_next(&mut descriptor_indexing_features)
        .push_next(&mut extended_dynamic_state_features)
        .push_next(&mut pipeline_creation_cache_control_features)
        .push_next(&mut synchronization2_features)
        .push_next(&mut uniform_buffer_standard_layout_features)
        .push_next(&mut dynamic_rendering_features)
        .push_next(&mut vk11_features)
        .enabled_features(&features);
    {
        profiling::scope!("vk::create_device");
        unsafe { instance.create_device(physical_device, &device_create_info, None) }
    }
}

/// Returns Ok(()) if the physical device does have the required features.
pub fn has_required_features(instance: &Instance, physical_device: vk::PhysicalDevice) -> Result<(), SupportedFeatures> {
    let mut descriptor_indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
    let mut extended_dynamic_state_features = vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default();
    let mut pipeline_creation_cache_control_features = vk::PhysicalDevicePipelineCreationCacheControlFeatures::default();
    let mut synchronization2_features = vk::PhysicalDeviceSynchronization2FeaturesKHR::default();
    let mut uniform_buffer_standard_layout_features = vk::PhysicalDeviceUniformBufferStandardLayoutFeatures::default();
    let mut dynamic_rendering_features = vk::PhysicalDeviceDynamicRenderingFeaturesKHR::default();
    let mut vk11_features = vk::PhysicalDeviceVulkan11Features::default();
    let mut features = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut descriptor_indexing_features)
        .push_next(&mut extended_dynamic_state_features)
        .push_next(&mut pipeline_creation_cache_control_features)
        .push_next(&mut synchronization2_features)
        .push_next(&mut uniform_buffer_standard_layout_features)
        .push_next(&mut dynamic_rendering_features)
        .push_next(&mut vk11_features);
    unsafe { instance.get_physical_device_features2(physical_device, &mut features) };
    // Note: requirements should match what is requested in create_device_with_feature_requirements
    let features = SupportedFeatures {
        sampler_anisotropy: features.features.sampler_anisotropy == vk::TRUE,
        sample_rate_shading: features.features.sample_rate_shading == vk::TRUE,
        texture_array_dynamic_indexing: features.features.shader_sampled_image_array_dynamic_indexing == vk::TRUE,
        partially_bound_descriptors: descriptor_indexing_features.descriptor_binding_partially_bound == vk::TRUE,
        extended_dynamic_state: extended_dynamic_state_features.extended_dynamic_state == vk::TRUE,
        pipeline_creation_cache_control: pipeline_creation_cache_control_features.pipeline_creation_cache_control == vk::TRUE,
        synchronization2: synchronization2_features.synchronization2 == vk::TRUE,
        uniform_buffer_standard_layout: uniform_buffer_standard_layout_features.uniform_buffer_standard_layout == vk::TRUE,
        dynamic_rendering: dynamic_rendering_features.dynamic_rendering == vk::TRUE,
        shader_draw_parameters: vk11_features.shader_draw_parameters == vk::TRUE,
    };
    if features.sampler_anisotropy
        && features.sample_rate_shading
        && features.texture_array_dynamic_indexing
        && features.partially_bound_descriptors
        && features.extended_dynamic_state
        && features.pipeline_creation_cache_control
        && features.synchronization2
        && features.uniform_buffer_standard_layout
        && features.dynamic_rendering
        && features.shader_draw_parameters
    {
        Ok(())
    } else {
        Err(features)
    }
}
