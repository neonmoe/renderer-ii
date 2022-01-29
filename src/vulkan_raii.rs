//! Wrappers for Vulkan objects that enforce proper destruction order
//! via refcounting. Non-atomic refcounting ([Rc]s) is used for
//! performance.
//!
//! Take care to get rid of these asap! Leaving Rc's lying around is a
//! recipe for memory leaks.

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

macro_rules! trivial_eq_impl {
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

macro_rules! trivial_hash_impl {
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

pub struct DeviceMemory {
    pub inner: vk::DeviceMemory,
    pub device: Rc<Device>,
}
trivial_drop_impl!(DeviceMemory, free_memory);
trivial_eq_impl!(DeviceMemory);
trivial_hash_impl!(DeviceMemory);

pub struct Buffer {
    pub inner: vk::Buffer,
    pub device: Rc<Device>,
    pub memory: Rc<DeviceMemory>,
}
trivial_drop_impl!(Buffer, destroy_buffer);
trivial_eq_impl!(Buffer);
trivial_hash_impl!(Buffer);

pub struct Image {
    pub inner: vk::Image,
    pub device: Rc<Device>,
    pub memory: Option<Rc<DeviceMemory>>,
}
trivial_drop_impl!(Image, destroy_image);
trivial_eq_impl!(Image);
trivial_hash_impl!(Image);

pub struct ImageView {
    pub inner: vk::ImageView,
    pub device: Rc<Device>,
    pub parent_image: Rc<Image>,
}
trivial_drop_impl!(ImageView, destroy_image_view);
trivial_eq_impl!(ImageView);
trivial_hash_impl!(ImageView);
