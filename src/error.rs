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
    #[error("could not create the descriptor set layout")]
    VulkanDescriptorSetLayoutCreation(#[source] vk::Result),
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
    #[error("could not reset the command buffer")]
    VulkanResetCommandBuffer(#[source] vk::Result),
    #[error("could not begin command buffer recording")]
    VulkanBeginCommandBuffer(#[source] vk::Result),
    #[error("failed to record command buffer")]
    VulkanEndCommandBuffer(#[source] vk::Result),
    #[error("could not create semaphore")]
    VulkanSemaphoreCreation(#[source] vk::Result),
    #[error("could not acquire next frame's image")]
    VulkanAcquireImage(#[source] vk::Result),
    #[error("could not submit the queue")]
    VulkanQueueSubmit(#[source] vk::Result),
    #[error("could not present the queue")]
    VulkanQueuePresent(#[source] vk::Result),
    #[error("swapchain is out of date, cannot present")]
    VulkanSwapchainOutOfDate(#[source] vk::Result),
    #[error("could not wait until the device is idle")]
    VulkanDeviceWaitIdle(#[source] vk::Result),
    #[error("could not create the fence")]
    VulkanFenceCreation(#[source] vk::Result),
    #[error("could not reset the fence")]
    VulkanFenceReset(#[source] vk::Result),
    #[error("could not wait for the fence")]
    VulkanFenceWait(#[source] vk::Result),
    #[error("could not create descriptor pool")]
    VulkanDescriptorPoolCreation(#[source] vk::Result),
    #[error("could not allocate descriptor sets")]
    VulkanAllocateDescriptorSets(#[source] vk::Result),
    #[error("vma (via vk-mem-rs) allocator creation failed")]
    VmaAllocatorCreation(#[source] vk_mem::error::Error),
    #[error("vma (via vk-mem-rs) allocator pool creation failed")]
    VmaPoolCreation(#[source] vk_mem::error::Error),
    #[error("vma (via vk-mem-rs) buffer allocation failed")]
    VmaBufferAllocation(#[source] vk_mem::error::Error),
    #[error("vma (via vk-mem-rs) could not flush allocation")]
    VmaFlushAllocation(#[source] vk_mem::error::Error),
    #[error("vma (via vk-mem-rs) stats calculation failed")]
    VmaCalculateStats(#[source] vk_mem::error::Error),
    #[error("vma (via vk-mem-rs) could not find a memory type index (gpu doesn't support required memory features)")]
    VmaFindMemoryType(#[source] vk_mem::error::Error),
    #[error("tried to update vertices, buffer is not editable (see Buffer::new)")]
    BufferNotEditable,
}
