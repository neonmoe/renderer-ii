//! A Vulkan renderer for 3D games. The mission statement will probably narrow
//! down over time.
//!
//! Currently targeting the
//! [VP_LUNARG_desktop_baseline_2022](https://vulkan.lunarg.com/doc/sdk/1.3.246.1/windows/profiles_definitions.html)
//! Vulkan Profile.

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

use debug_utils::*;

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
pub use arena::{ForBuffers, ForImages, MemoryProps, VulkanArena, VulkanArenaError};

pub use ash::vk;

mod display_utils {
    use core::fmt::{Display, Formatter, Result};

    /// Wrapper around u64 for pretty-printing byte amount with the appropriate
    /// size prefix (KiB, MiB, etc.).
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
pub use display_utils::*;

pub mod image_loading;

pub mod include_words;

mod instance;
pub use instance::Instance;

mod memory_measurement {
    mod arena;
    pub use arena::{VulkanArenaMeasurementError, VulkanArenaMeasurer};
}
pub use memory_measurement::*;

mod physical_device;
pub use physical_device::{get_physical_devices, GpuId, PhysicalDevice};

mod renderer;
pub use renderer::coordinate_system::CoordinateSystem;
pub use renderer::descriptors::material::{AlphaMode, Material, PbrFactors, PipelineSpecificData};
pub use renderer::descriptors::{DescriptorError, Descriptors, PbrDefaults};
pub use renderer::framebuffers::{FramebufferCreationError, Framebuffers};
pub use renderer::mesh::Mesh;
pub use renderer::pipelines::{PipelineCreationError, Pipelines};
pub use renderer::scene::{JointOffset, Scene, SkinnedModel, StaticMeshMap};
pub use renderer::swapchain::{Swapchain, SwapchainBase, SwapchainError, SwapchainSettings};
pub use renderer::{Renderer, RendererError};

mod surface {
    use ash::extensions::khr;
    use ash::{Entry, Instance};
    use raw_window_handle::{HasRawDisplayHandle, HasRawWindowHandle};

    pub use crate::vulkan_raii::Surface;

    pub fn create_surface(
        entry: &Entry,
        instance: &Instance,
        display: &dyn HasRawDisplayHandle,
        window: &dyn HasRawWindowHandle,
    ) -> Result<Surface, ash::vk::Result> {
        profiling::scope!("window surface creation");
        let surface =
            unsafe { ash_window::create_surface(entry, instance, display.raw_display_handle(), window.raw_window_handle(), None) }?;
        let surface_ext = khr::Surface::new(entry, instance);
        Ok(Surface {
            inner: surface,
            device: surface_ext,
        })
    }
}
pub use surface::{create_surface, Surface};

mod uploader;
pub use uploader::Uploader;

mod vulkan_raii;
pub use vulkan_raii::{Buffer, Device, ImageView};
