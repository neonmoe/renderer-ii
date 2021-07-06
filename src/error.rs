use ash::vk;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("query for the window's required extensions failed ({0})")]
    WindowRequiredExtensions(vk::Result),
    #[error("vulkan instance creation failed ({0})")]
    VulkanInstanceCreation(ash::InstanceError),
    #[error("vulkan surface creation failed ({0})")]
    VulkanSurfaceCreation(vk::Result),
}
