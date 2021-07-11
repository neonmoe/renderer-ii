use ash::vk;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("query for the window's required extensions failed")]
    WindowRequiredExtensions(#[source] vk::Result),
    #[error("vulkan instance creation failed")]
    VulkanInstanceCreation(#[from] ash::InstanceError),
    #[error("vulkan surface creation failed")]
    VulkanSurfaceCreation(#[source] vk::Result),
    #[error("could not list physical devices")]
    VulkanEnumeratePhysicalDevices(#[source] vk::Result),
    #[error("could not find a GPU that can render to the screen")]
    VulkanPhysicalDeviceMissing,
    #[error("vulkan logical device creation failed")]
    VulkanDeviceCreation(#[source] vk::Result),
    #[error("physical device surface query failed")]
    VulkanPhysicalDeviceSurfaceQuery(#[source] vk::Result),
    #[error("vulkan swapchain creation failed")]
    VulkanSwapchainCreation(#[source] vk::Result),
    #[error("could not get swapchain images")]
    VulkanGetSwapchainImages(#[source] vk::Result),
    #[error("swapchain image view creation failed")]
    VulkanSwapchainImageViewCreation(#[source] vk::Result),
}
