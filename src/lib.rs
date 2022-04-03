//! A Vulkan renderer for 3D games. The mission statement will
//! probably narrow down over time.

macro_rules! cstr {
    ($string:literal) => {
        unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(concat!($string, "\0").as_bytes()) }
    };
}

// internal modules:

mod debug_utils;
mod mesh;
mod physical_device_features;
mod vulkan_raii;

// public-facing modules:

mod arena;
pub use arena::{ForBuffers, ForImages, VulkanArena};

pub use ash::vk;

mod framebuffers;
pub use framebuffers::Framebuffers;

mod descriptors;
pub use descriptors::{Descriptors, Material, PbrDefaults};

mod device;
pub use device::create_device;

mod instance;
pub use instance::Instance;

mod error;
pub use error::Error;

mod gltf;
pub use gltf::{Gltf, MeshIter};

pub mod image_loading;

mod physical_device;
pub use physical_device::{get_physical_devices, GpuId, PhysicalDevice};

mod pipelines;
pub use pipelines::Pipelines;

mod pipeline_parameters;
pub use pipeline_parameters::PipelineIndex;

mod renderer;
pub use renderer::{FrameIndex, Renderer};

mod scene;
pub use scene::Scene;

mod swapchain;
pub use swapchain::{Swapchain, SwapchainSettings};

mod uploader;
pub use uploader::Uploader;

mod surface {
    use crate::vulkan_raii::Surface;
    use crate::Error;
    use ash::extensions::khr;
    use ash::{Entry, Instance};
    use raw_window_handle::HasRawWindowHandle;

    pub fn create_surface(entry: &Entry, instance: &Instance, window: &dyn HasRawWindowHandle) -> Result<Surface, Error> {
        profiling::scope!("window surface creation");
        let surface = unsafe { ash_window::create_surface(entry, instance, window, None) }.map_err(Error::VulkanSurfaceCreation)?;
        let surface_ext = khr::Surface::new(entry, instance);
        Ok(Surface {
            inner: surface,
            device: surface_ext,
        })
    }
}
pub use surface::create_surface;

mod allocation {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub(crate) static ALLOCATED: AtomicU64 = AtomicU64::new(0);
    pub(crate) static IN_USE: AtomicU64 = AtomicU64::new(0);

    pub fn get_allocated_vram() -> u64 {
        ALLOCATED.load(Ordering::Relaxed)
    }

    pub fn get_allocated_vram_in_use() -> u64 {
        IN_USE.load(Ordering::Relaxed)
    }
}
pub use allocation::{get_allocated_vram, get_allocated_vram_in_use};
