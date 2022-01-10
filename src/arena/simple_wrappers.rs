use ash::vk;

/// A Vulkan image allocated from an [Arena].
pub struct ImageAllocation {
    pub image: vk::Image,
}
