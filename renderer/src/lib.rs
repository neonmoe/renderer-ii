//! A Vulkan renderer for 3D games. The mission statement will probably narrow
//! down over time.
//!
//! Currently targeting the
//! [VP_LUNARG_desktop_baseline_2022](https://vulkan.lunarg.com/doc/sdk/latest/windows/profiles_definitions.html)
//! Vulkan Profile, with the following changes:
//!
//! - VkPhysicalDeviceVulkan11Features.shaderDrawParameters is required (for
//!   indirect rendering, `gl_BaseInstanceARB` in shaders)
//!
//! (See
//! [`VP_CUSTOM_profile_based_on_2022_baseline.json`](../../../renderer/src/vk-profiles/VP_CUSTOM_profile_based_on_2022_baseline.json)
//! for the actual profile JSON.)
//!
//! Vulkan error handling policy: if the returned error is "out of host/device
//! memory", "surface lost", or "device lost", it is handled by panicking. If
//! the function is specifically for allocating a `VkDeviceMemory`, "out of
//! device memory" is handled by returning a Result with an appropriate error.

#![warn(clippy::pedantic)]
#![warn(clippy::std_instead_of_alloc)]
#![warn(clippy::std_instead_of_core)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::module_name_repetitions,
    clippy::struct_excessive_bools,
    clippy::from_iter_instead_of_collect,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::cast_lossless,
    clippy::too_many_lines,
    clippy::similar_names,
    clippy::wildcard_imports
)]
// TODO: #![no_std]

extern crate alloc;
extern crate core;

macro_rules! cstr {
    ($string:expr) => {
        unsafe { core::ffi::CStr::from_bytes_with_nul_unchecked(concat!($string, "\0").as_bytes()) }
    };
}

// internal modules:

#[cfg(feature = "vulkan-debug-utils")]
mod debug_utils;

#[cfg(not(feature = "vulkan-debug-utils"))]
mod debug_utils {
    use core::fmt::Arguments;

    use ash::{vk, Device};

    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn name_vulkan_object<H: vk::Handle>(_device: &Device, _object: H, _name: Arguments) {}
}

use debug_utils::name_vulkan_object;
#[cfg(feature = "vulkan-debug-utils")]
use debug_utils::{create_debug_utils_messenger_info, init_debug_utils};

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

pub use arena::buffers::ForBuffers;
pub use arena::images::ForImages;
pub use arena::{MemoryProps, VulkanArena, VulkanArenaError};
pub use ash::vk;

mod display_utils {
    use core::fmt::{Display, Formatter, Result};

    /// Wrapper around u64 for pretty-printing byte amount with the appropriate
    /// size prefix (KiB, MiB, etc.).
    #[derive(Debug)]
    pub struct Bytes(pub u64);

    impl Display for Bytes {
        fn fmt(&self, fmt: &mut Formatter) -> Result {
            const KILO: u64 = 1000;
            const MEGA: u64 = KILO * KILO;
            const GIGA: u64 = MEGA * KILO;
            const TERA: u64 = GIGA * KILO;
            match self.0 {
                bytes if bytes < KILO => write!(fmt, "{:.0} bytes", bytes as f32),
                bytes if bytes < MEGA => write!(fmt, "{:.2} KB", bytes as f32 / KILO as f32),
                bytes if bytes < GIGA => write!(fmt, "{:.2} MB", bytes as f32 / MEGA as f32),
                bytes if bytes < TERA => write!(fmt, "{:.2} GB", bytes as f32 / GIGA as f32),
                bytes => write!(fmt, "{:.3} TiB", bytes as f32 / TERA as f32),
            }
        }
    }
}

pub use display_utils::*;

pub mod image_loading;

pub mod include_words;

mod instance;

pub use instance::Instance;

mod memory_measurement;

pub use memory_measurement::VulkanArenaMeasurer;

mod physical_device;

pub use physical_device::{get_physical_devices, GpuId, PhysicalDevice};

mod renderer;

pub use renderer::descriptors::material::{AlphaMode, Material, PbrFactors, PipelineSpecificData};
pub use renderer::descriptors::{Descriptors, PbrDefaults};
pub use renderer::framebuffers::Framebuffers;
pub use renderer::pipeline_parameters::constants::{MAX_DRAW_CALLS, MAX_JOINT_COUNT};
pub use renderer::pipeline_parameters::PipelineIndex;
pub use renderer::pipelines::Pipelines;
pub use renderer::scene::camera::Camera;
pub use renderer::scene::coordinate_system::CoordinateSystem;
pub use renderer::scene::mesh::{IndexType, Mesh};
pub use renderer::scene::{JointsOffset, Scene};
pub use renderer::swapchain::{Swapchain, SwapchainBase, SwapchainError, SwapchainSettings};
pub use renderer::Renderer;

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
        Ok(Surface { inner: surface, device: surface_ext })
    }
}

pub use surface::{create_surface, Surface};

mod uploader;

pub use uploader::Uploader;

mod vertex_library;

pub use vertex_library::{VertexLibraryBuilder, VertexLibraryMeasurer};

mod vulkan_raii;

pub use vulkan_raii::{Buffer, Device, ImageView};
