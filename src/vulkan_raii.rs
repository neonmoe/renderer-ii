//! Wrappers for Vulkan objects that enforce proper destruction order
//! via refcounting. Non-atomic refcounting ([Rc]s) is used for
//! performance.
//!
//! Take care to get rid of these asap! Leaving Rc's lying around is a
//! recipe for memory leaks.

use ash::extensions::khr;
use ash::vk;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

pub struct Device {
    pub inner: ash::Device,
    pub graphics_queue: vk::Queue,
    pub surface_queue: vk::Queue,
    pub transfer_queue: vk::Queue,
}
impl std::ops::Deref for Device {
    type Target = ash::Device;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl Drop for Device {
    fn drop(&mut self) {
        unsafe { self.inner.destroy_device(None) };
    }
}
impl Device {
    /// Wait until the device is idle. Should be called before
    /// swapchain recreation and after the game loop is over.
    #[profiling::function]
    pub fn wait_idle(&self) -> Result<(), crate::Error> {
        unsafe { self.inner.device_wait_idle() }.map_err(crate::Error::VulkanDeviceWaitIdle)
    }
}

macro_rules! trivial_drop_impl {
    ($struct_name:ident, $destroy_func_name:ident) => {
        impl Drop for $struct_name {
            fn drop(&mut self) {
                profiling::scope!(concat!("vk::", stringify!($destroy_func_name)));
                unsafe { self.device.$destroy_func_name(self.inner, None) }
            }
        }
    };
}

macro_rules! inner_and_device_based_eq_impl {
    ($struct_name:ident) => {
        impl PartialEq for $struct_name {
            fn eq(&self, other: &Self) -> bool {
                // I think just comparing the inner objects should be
                // enough, but at the very least, it should be enough
                // if the devices match as well.
                self.inner == other.inner && self.device.handle() == other.device.handle()
            }
        }

        impl Eq for $struct_name {}
    };
}

macro_rules! inner_and_device_based_hash_impl {
    ($struct_name:ident) => {
        impl Hash for $struct_name {
            fn hash<H: Hasher>(&self, state: &mut H) {
                // Same logic as with eq().
                self.inner.hash(state);
                self.device.handle().hash(state);
            }
        }
    };
}

pub enum AnyImage {
    Regular(Image),
    Swapchain(vk::Image, Rc<Swapchain>),
}

impl AnyImage {
    pub fn inner(&self) -> vk::Image {
        match self {
            AnyImage::Regular(image) => image.inner,
            AnyImage::Swapchain(image, _) => *image,
        }
    }
}

impl From<Image> for AnyImage {
    fn from(image: Image) -> AnyImage {
        AnyImage::Regular(image)
    }
}

pub struct Surface {
    pub inner: vk::SurfaceKHR,
    pub device: khr::Surface,
}
trivial_drop_impl!(Surface, destroy_surface);

pub struct Swapchain {
    pub inner: vk::SwapchainKHR,
    pub device: khr::Swapchain,
    pub surface: Rc<Surface>,
}
trivial_drop_impl!(Swapchain, destroy_swapchain);

pub struct DeviceMemory {
    pub inner: vk::DeviceMemory,
    pub device: Rc<Device>,
}
trivial_drop_impl!(DeviceMemory, free_memory);

pub struct Buffer {
    pub inner: vk::Buffer,
    pub device: Rc<Device>,
    pub memory: Rc<DeviceMemory>,
}
trivial_drop_impl!(Buffer, destroy_buffer);
inner_and_device_based_eq_impl!(Buffer);
inner_and_device_based_hash_impl!(Buffer);

pub struct Image {
    pub inner: vk::Image,
    pub device: Rc<Device>,
    pub memory: Rc<DeviceMemory>,
}
trivial_drop_impl!(Image, destroy_image);
inner_and_device_based_eq_impl!(Image);
inner_and_device_based_hash_impl!(Image);

pub struct ImageView {
    pub inner: vk::ImageView,
    pub device: Rc<Device>,
    pub image: Rc<AnyImage>,
}
trivial_drop_impl!(ImageView, destroy_image_view);
inner_and_device_based_eq_impl!(ImageView);
inner_and_device_based_hash_impl!(ImageView);

pub struct RenderPass {
    pub inner: vk::RenderPass,
    pub device: Rc<Device>,
}
trivial_drop_impl!(RenderPass, destroy_render_pass);

pub struct Framebuffer {
    pub inner: vk::Framebuffer,
    pub device: Rc<Device>,
    pub render_pass: Rc<RenderPass>,
    // NOTE: Not an Rc because every attachment image view probably
    // maps to just one framebuffer.
    pub attachments: Vec<ImageView>,
}
trivial_drop_impl!(Framebuffer, destroy_framebuffer);

pub struct Pipeline {
    pub inner: vk::Pipeline,
    pub device: Rc<Device>,
    pub render_pass: Rc<RenderPass>,
}
trivial_drop_impl!(Pipeline, destroy_pipeline);

pub struct Sampler {
    pub inner: vk::Sampler,
    pub device: Rc<Device>,
}
trivial_drop_impl!(Sampler, destroy_sampler);

pub struct DescriptorSetLayouts {
    pub inner: Vec<vk::DescriptorSetLayout>,
    pub device: Rc<Device>,
    pub immutable_samplers: Vec<Rc<Sampler>>,
}
impl Drop for DescriptorSetLayouts {
    fn drop(&mut self) {
        for descriptor_set_layout in &self.inner {
            unsafe { self.device.destroy_descriptor_set_layout(*descriptor_set_layout, None) };
        }
    }
}

pub struct PipelineLayout {
    pub inner: vk::PipelineLayout,
    pub device: Rc<Device>,
    pub descriptor_set_layouts: Rc<DescriptorSetLayouts>,
}
trivial_drop_impl!(PipelineLayout, destroy_pipeline_layout);

pub struct DescriptorPool {
    pub inner: vk::DescriptorPool,
    pub device: Rc<Device>,
}
trivial_drop_impl!(DescriptorPool, destroy_descriptor_pool);

/// Wrapper for an array of descriptor sets.
///
/// Does not implement drop, as the resources are freed when the pool
/// is destroyed.
pub struct DescriptorSets {
    pub inner: Vec<vk::DescriptorSet>,
    pub device: Rc<Device>,
    pub descriptor_pool: Rc<DescriptorPool>,
}

pub struct CommandPool {
    pub inner: vk::CommandPool,
    pub device: Rc<Device>,
}
trivial_drop_impl!(CommandPool, destroy_command_pool);

pub struct Semaphore {
    pub inner: vk::Semaphore,
    pub device: Rc<Device>,
}
trivial_drop_impl!(Semaphore, destroy_semaphore);

pub struct Fence {
    pub inner: vk::Fence,
    pub device: Rc<Device>,
}
trivial_drop_impl!(Fence, destroy_fence);
