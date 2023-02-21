//! Wrappers for Vulkan objects that enforce proper destruction order
//! via refcounting. Non-atomic refcounting ([Rc]s) is used for
//! performance.
//!
//! Take care to get rid of these asap! Leaving Rc's lying around is a
//! recipe for memory leaks.

use alloc::rc::Rc;
use ash::extensions::khr;
use ash::vk;
use core::hash::{Hash, Hasher};
use core::sync::atomic::Ordering;
use smallvec::SmallVec;

/// The Vulkan device, which is used to make pretty much all Vulkan calls after
/// its creation.
///
/// It's safe to clone to share, until destruction time, as the only function
/// where it must be externally synchronized according to the
/// [spec](https://www.khronos.org/registry/vulkan/specs/1.3-extensions/html/chap3.html#fundamentals-threadingbehavior)
/// is vkDestroyDevice.
///
/// The reason Device isn't shared via borrows or Rcs is that it is pretty much
/// as global as a structure as it gets, and the ergonomics of just copying it
/// everywhere outweigh the possibility that destroy does not get called.
#[derive(Clone)]
pub struct Device {
    pub inner: &'static ash::Device,
    pub graphics_queue: vk::Queue,
    pub surface_queue: vk::Queue,
    pub transfer_queue: vk::Queue,
}
impl Device {
    pub fn destroy(&mut self) {
        unsafe { self.inner.destroy_device(None) };
    }
}
impl core::ops::Deref for Device {
    type Target = ash::Device;
    fn deref(&self) -> &Self::Target {
        self.inner
    }
}
impl Device {
    /// Wait until the device is idle. Should be called before swapchain
    /// recreation and after the game loop is over to make sure none of the
    /// resources are in use before destroying anything.
    #[profiling::function]
    pub fn wait_idle(&self) -> Result<(), crate::Error> {
        unsafe { self.inner.device_wait_idle() }.map_err(crate::Error::DeviceWaitIdle)
    }
}

macro_rules! trivial_drop_impl {
    ($struct_name:ident, $destroy_func_name:ident) => {
        impl Drop for $struct_name {
            fn drop(&mut self) {
                profiling::scope!(concat!("vk::", stringify!($destroy_func_name)));
                log::trace!("{}({:?})", concat!("vk::", stringify!($destroy_func_name)), self.inner);
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

// It's actually AnyImage::Swapchain that is unexpectdly small, thanks to the
// Rc. The "large variant" is the expected size of an AnyImage, as it's the size
// of an Image.
#[allow(clippy::large_enum_variant)]
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
    /// This is an Option so that `surface` can be recovered from a Swapchain
    /// before it's dropped, to be reused for another Swapchain.
    pub surface: Option<Surface>,
}
trivial_drop_impl!(Swapchain, destroy_swapchain);

pub struct DeviceMemory {
    pub inner: vk::DeviceMemory,
    pub device: Device,
    size: u64,
}
impl DeviceMemory {
    pub fn new(inner: vk::DeviceMemory, device: Device, size: u64) -> DeviceMemory {
        crate::allocation::ALLOCATED.fetch_add(size, Ordering::Relaxed);
        crate::allocation::IN_USE.fetch_add(size, Ordering::Relaxed);
        DeviceMemory { inner, device, size }
    }
}
impl Drop for DeviceMemory {
    fn drop(&mut self) {
        profiling::scope!("vk::free_memory");
        log::trace!("vk::free_memory({:?}) [{} bytes]", self.inner, self.size);
        crate::allocation::IN_USE.fetch_sub(self.size, Ordering::Relaxed);
        crate::allocation::ALLOCATED.fetch_sub(self.size, Ordering::Relaxed);
        unsafe { self.device.free_memory(self.inner, None) };
    }
}

pub struct Buffer {
    pub inner: vk::Buffer,
    pub device: Device,
    pub memory: Rc<DeviceMemory>,
    pub size: vk::DeviceSize,
}
trivial_drop_impl!(Buffer, destroy_buffer);
inner_and_device_based_eq_impl!(Buffer);
inner_and_device_based_hash_impl!(Buffer);

pub struct Image {
    pub inner: vk::Image,
    pub device: Device,
    pub memory: Rc<DeviceMemory>,
}
trivial_drop_impl!(Image, destroy_image);
inner_and_device_based_eq_impl!(Image);
inner_and_device_based_hash_impl!(Image);

pub struct ImageView {
    pub inner: vk::ImageView,
    pub device: Device,
    pub image: Rc<AnyImage>,
}
trivial_drop_impl!(ImageView, destroy_image_view);
inner_and_device_based_eq_impl!(ImageView);
inner_and_device_based_hash_impl!(ImageView);

pub struct RenderPass {
    pub inner: vk::RenderPass,
    pub device: Device,
}
trivial_drop_impl!(RenderPass, destroy_render_pass);

pub struct Framebuffer {
    pub inner: vk::Framebuffer,
    pub device: Device,
    pub render_pass: Rc<RenderPass>,
    pub attachments: SmallVec<[Rc<ImageView>; 4]>,
}
trivial_drop_impl!(Framebuffer, destroy_framebuffer);

pub struct Pipeline {
    pub inner: vk::Pipeline,
    pub device: Device,
    pub render_pass: Rc<RenderPass>,
}
trivial_drop_impl!(Pipeline, destroy_pipeline);

pub struct Sampler {
    pub inner: vk::Sampler,
    pub device: Device,
}
trivial_drop_impl!(Sampler, destroy_sampler);

pub struct DescriptorSetLayouts {
    pub inner: SmallVec<[vk::DescriptorSetLayout; 8]>,
    pub device: Device,
    pub immutable_samplers: SmallVec<[Rc<Sampler>; 1]>,
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
    pub device: Device,
    pub descriptor_set_layouts: Rc<DescriptorSetLayouts>,
}
trivial_drop_impl!(PipelineLayout, destroy_pipeline_layout);

pub struct DescriptorPool {
    pub inner: vk::DescriptorPool,
    pub device: Device,
}
trivial_drop_impl!(DescriptorPool, destroy_descriptor_pool);

/// Wrapper for an array of descriptor sets.
///
/// Does not implement drop, as the resources are freed when the pool
/// is destroyed.
pub struct DescriptorSets {
    pub inner: SmallVec<[vk::DescriptorSet; 8]>,
    pub device: Device,
    pub descriptor_pool: Rc<DescriptorPool>,
}

pub struct CommandPool {
    pub inner: vk::CommandPool,
    pub device: Device,
}
trivial_drop_impl!(CommandPool, destroy_command_pool);

pub struct CommandBuffer {
    pub inner: vk::CommandBuffer,
    pub device: Device,
    pub command_pool: Rc<CommandPool>,
}
impl Drop for CommandBuffer {
    fn drop(&mut self) {
        unsafe { self.device.free_command_buffers(self.command_pool.inner, &[self.inner]) };
    }
}

pub struct Semaphore {
    pub inner: vk::Semaphore,
    pub device: Device,
}
trivial_drop_impl!(Semaphore, destroy_semaphore);

pub struct Fence {
    pub inner: vk::Fence,
    pub device: Device,
}
trivial_drop_impl!(Fence, destroy_fence);

pub struct PipelineCache {
    pub inner: vk::PipelineCache,
    pub device: Device,
}
trivial_drop_impl!(PipelineCache, destroy_pipeline_cache);
