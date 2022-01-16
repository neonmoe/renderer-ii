use ash::vk;

/// A Vulkan image allocated from an [Arena].
pub struct ImageAllocation {
    pub image: vk::Image,
}

/// An ImageView of an [Arena]-allocated image.
pub struct ImageView {
    pub image_view: vk::ImageView,
}
