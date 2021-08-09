//! A Vulkan renderer for 3D games. The mission statement will
//! probably narrow down over time.

macro_rules! cstr {
    ($string:literal) => {
        unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(concat!($string, "\0").as_bytes()) }
    };
}

// internal modules:

mod buffer;
mod debug_utils;
mod descriptors;

// public-facing modules:

mod camera;
pub use camera::Camera;

mod canvas;
pub use canvas::Canvas;

mod driver;
pub use driver::Driver;

mod error;
pub use error::Error;

mod gpu;
pub use gpu::{FrameIndex, Gpu, GpuId, GpuInfo};

mod mesh;
pub use mesh::Mesh;

mod pipeline;
pub use pipeline::Pipeline;
