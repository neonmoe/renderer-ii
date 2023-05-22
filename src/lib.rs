//! A Vulkan renderer for 3D games. The mission statement will
//! probably narrow down over time.

#![feature(int_roundings)] // seems like this will get merged soon enough

// TODO: #![warn(clippy::pedantic)]
// TODO: #![no_std]

extern crate alloc;

macro_rules! cstr {
    ($string:expr) => {
        unsafe { core::ffi::CStr::from_bytes_with_nul_unchecked(concat!($string, "\0").as_bytes()) }
    };
}

// internal modules:

mod debug_utils;
mod physical_device_features;

// TODO: Remove memmap2
// TODO: Add a file loading module to replace std::fs in the gltf loading code.

// public-facing modules:

mod vram_usage {
    use core::sync::atomic::{AtomicU64, Ordering};
    pub(crate) static ALLOCATED: AtomicU64 = AtomicU64::new(0);
    pub(crate) static IN_USE: AtomicU64 = AtomicU64::new(0);
    pub(crate) static ALLOCATED_PEAK: AtomicU64 = AtomicU64::new(0);

    pub fn get_allocated_vram() -> u64 {
        ALLOCATED.load(Ordering::Relaxed)
    }
    pub fn get_allocated_vram_in_use() -> u64 {
        IN_USE.load(Ordering::Relaxed)
    }
    pub fn get_allocated_vram_peak() -> u64 {
        ALLOCATED_PEAK.load(Ordering::Relaxed)
    }
}
pub use vram_usage::{get_allocated_vram, get_allocated_vram_in_use, get_allocated_vram_peak};

mod arena;
pub use arena::{ForBuffers, ForImages, MemoryProps, VulkanArena};

pub use ash::vk;

mod framebuffers;
pub use framebuffers::Framebuffers;

mod descriptors;
pub use descriptors::{AlphaMode, DescriptorError, Descriptors, GltfFactors, Material, PipelineSpecificData};

mod device;
pub use device::create_device;

pub mod display_utils {
    use core::fmt::{Display, Formatter, Result};

    #[derive(Debug)]
    pub struct Bytes(pub u64);

    impl Display for Bytes {
        fn fmt(&self, fmt: &mut Formatter) -> Result {
            const KIBI: u64 = 1_024;
            const MEBI: u64 = KIBI * KIBI;
            const GIBI: u64 = MEBI * KIBI;
            const TIBI: u64 = GIBI * KIBI;
            match self.0 {
                bytes if bytes < KIBI => write!(fmt, "{:.0} bytes", bytes as f32),
                bytes if bytes < MEBI => write!(fmt, "{:.2} KiB", bytes as f32 / KIBI as f32),
                bytes if bytes < GIBI => write!(fmt, "{:.2} MiB", bytes as f32 / MEBI as f32),
                bytes if bytes < TIBI => write!(fmt, "{:.2} GiB", bytes as f32 / GIBI as f32),
                bytes => write!(fmt, "{:.3} TiB", bytes as f32 / TIBI as f32),
            }
        }
    }
}

pub mod image_loading;

pub mod include_words;

mod instance;
pub use instance::Instance;

mod error;
pub use error::Error;

mod gltf;
pub use gltf::{Animation, AnimationError, Gltf, Keyframes};

mod memory_measurement {
    mod arena;
    pub use arena::{VulkanArenaMeasurementError, VulkanArenaMeasurer};

    mod gltf;
    pub use gltf::{measure_glb_memory_usage, measure_gltf_memory_usage};
}
pub use memory_measurement::*;

mod mesh;
pub use mesh::Mesh;

mod physical_device;
pub use physical_device::{get_physical_devices, GpuId, PhysicalDevice};

mod pipelines;
pub use pipelines::Pipelines;

mod pipeline_parameters;
pub use pipeline_parameters::PipelineIndex;

mod renderer;
pub use renderer::{FrameIndex, Renderer, RendererError};

mod scene;
pub use scene::Scene;

mod surface {
    use crate::Error;
    use ash::extensions::khr;
    use ash::{Entry, Instance};
    use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

    pub use crate::vulkan_raii::Surface;

    pub fn create_surface(
        entry: &Entry,
        instance: &Instance,
        display: &dyn HasRawDisplayHandle,
        window: &dyn HasRawWindowHandle,
    ) -> Result<Surface, Error> {
        profiling::scope!("window surface creation");
        let surface =
            unsafe { ash_window::create_surface(entry, instance, display.raw_display_handle(), window.raw_window_handle(), None) }
                .map_err(Error::SurfaceCreation)?;
        let surface_ext = khr::Surface::new(entry, instance);
        Ok(Surface {
            inner: surface,
            device: surface_ext,
        })
    }
}
pub use surface::{create_surface, Surface};

mod swapchain;
pub use swapchain::{Swapchain, SwapchainBase, SwapchainSettings};

mod uploader;
pub use uploader::Uploader;

mod vulkan_raii;
pub use vulkan_raii::{Buffer, Device, ImageView};
