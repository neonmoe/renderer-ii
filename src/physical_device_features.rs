use crate::Error;
use ash::vk;
use ash::{Device, Instance};

pub fn create_device_with_feature_requirements(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    device_create_info_builder: vk::DeviceCreateInfoBuilder,
) -> Result<Device, Error> {
    // Note: requested features should match what is checked in has_required_features
    let features = vk::PhysicalDeviceFeatures::builder()
        .sampler_anisotropy(true)
        .shader_sampled_image_array_dynamic_indexing(true);
    let mut descriptor_indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::builder().descriptor_binding_partially_bound(true);
    let mut extended_dynamic_state_features = vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::builder().extended_dynamic_state(true);
    let device_create_info = device_create_info_builder
        .push_next(&mut descriptor_indexing_features)
        .push_next(&mut extended_dynamic_state_features)
        .enabled_features(&features);
    {
        profiling::scope!("vk::create_device");
        unsafe { instance.create_device(physical_device, &device_create_info, None) }.map_err(Error::VulkanDeviceCreation)
    }
}

pub fn has_required_features(instance: &Instance, physical_device: vk::PhysicalDevice) -> bool {
    let mut descriptor_indexing_features = vk::PhysicalDeviceDescriptorIndexingFeatures::default();
    let mut extended_dynamic_state_features = vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::default();
    let mut features = vk::PhysicalDeviceFeatures2::builder()
        .push_next(&mut descriptor_indexing_features)
        .push_next(&mut extended_dynamic_state_features)
        .build();
    unsafe { instance.get_physical_device_features2(physical_device, &mut features) };
    // Note: requirements should match what is requested in create_device_with_feature_requirements
    features.features.sampler_anisotropy == vk::TRUE
        && features.features.shader_sampled_image_array_dynamic_indexing == vk::TRUE
        && descriptor_indexing_features.descriptor_binding_partially_bound == vk::TRUE
        && extended_dynamic_state_features.extended_dynamic_state == vk::TRUE
}
