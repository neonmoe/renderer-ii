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
    #[error("depth image view creation failed")]
    VulkanDepthImageViewCreation(#[source] vk::Result),
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
    #[error("could not create image view")]
    VulkanImageViewCreation(#[source] vk::Result),
    #[error("could not create sampler")]
    VulkanSamplerCreation(#[source] vk::Result),
    #[error("could not get fence status")]
    VulkanFenceStatus(#[source] vk::Result),
    #[error("tried to update vertices, buffer is not editable (see Buffer::new)")]
    BufferNotEditable,
    #[error("too many textures: failed to reserve a texture index")]
    TextureIndexReserve,
    #[error("not a glb file")]
    InvalidGlbHeader,
    #[error("glb header length mismatch")]
    InvalidGlbLength,
    #[error("glb chunk length mismatch")]
    InvalidGlbChunkLength,
    #[error("invalid glb chunk type")]
    InvalidGlbChunkType,
    #[error("too many glb json chunks")]
    TooManyGlbJsonChunks,
    #[error("too many glb binary chunks")]
    TooManyGlbBinaryChunks,
    #[error("glb json is not valid utf-8")]
    InvalidGlbJson(#[source] std::str::Utf8Error),
    #[error("glb json chunk missing")]
    MissingGlbJson,
    #[error("failed to deserialize gltf json")]
    GltfJsonDeserialization(#[source] miniserde::Error),
    #[error("unsupported gltf minimum version ({0}), 2.0 is supported")]
    UnsupportedGltfVersion(String),
    #[error("gltf has buffer without an uri but no glb BIN buffer")]
    GlbBinMissing,
    #[error("gltf refers to external data ({0}) but no directory was given in from_gltf/from_glb")]
    GltfMissingDirectory(String),
    #[error("could not load gltf buffer from {0}")]
    GltfBufferLoading(String, #[source] std::io::Error),
    #[error("gltf node has multiple parents, which is not allowed by the 2.0 spec")]
    GltfInvalidNodeGraph,
    #[error("gltf has an out-of-bounds index ({0})")]
    GltfOob(&'static str),
    #[error("gltf does not conform to the 2.0 spec: {0}")]
    GltfSpec(&'static str),
    #[error("unimplemented gltf feature: {0}")]
    GltfMisc(&'static str),
    #[error("error during image decoding: {0}")]
    MiscImageDecoding(&'static str),
    #[error("malformed ktx file")]
    BadKtx,
    #[error("could not load ktx: {0}")]
    UnsupportedKtxFeature(&'static str),
}
