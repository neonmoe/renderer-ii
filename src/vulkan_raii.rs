//! Wrappers for Vulkan objects that enforce proper destruction order
//! via refcounting. Non-atomic refcounting ([Rc]s) is used for
//! performance.
//!
//! Take care to get rid of these asap! Leaving Rc's lying around is a
//! recipe for memory leaks.

use ash::extensions::khr;
use ash::version::DeviceV1_0;
use ash::vk;
use ash::Device;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

macro_rules! trivial_drop_impl {
    ($struct_name:ident, $destroy_func_name:ident) => {
        impl Drop for $struct_name {
            fn drop(&mut self) {
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

pub struct Swapchain {
    pub inner: vk::SwapchainKHR,
    pub device: Rc<khr::Swapchain>,
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
