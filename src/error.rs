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
    #[error("could not create the shader module")]
    VulkanShaderModuleCreation(#[source] vk::Result),
    #[error("could not create the pipeline layout")]
    VulkanPipelineLayoutCreation(#[source] vk::Result),
    #[error("could not create the render pass")]
    VulkanRenderPassCreation(#[source] vk::Result),
    #[error("could not create the graphics pipeline")]
    VulkanGraphicsPipelineCreation(#[source] vk::Result),
    #[error("could not create the framebuffer")]
    VulkanFramebufferCreation(#[source] vk::Result),
    #[error("could not create the command pool")]
    VulkanCommandPoolCreation(#[source] vk::Result),
    #[error("could not allocate the command buffers")]
    VulkanCommandBuffersAllocation(#[source] vk::Result),
    #[error("failed to begin command buffer recording")]
    VulkanBeginCommandBuffer(#[source] vk::Result),
    #[error("failed to end command buffer recording")]
    VulkanEndCommandBuffer(#[source] vk::Result),
}
