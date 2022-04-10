use ash::vk;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("vulkan surface creation failed")]
    SurfaceCreation(#[source] vk::Result),
    #[error("vulkan device creation failed")]
    DeviceCreation(#[source] vk::Result),
    #[error("waiting for vulkan device to become idle failed")]
    DeviceWaitIdle(#[source] vk::Result),
}
