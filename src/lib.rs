//! A Vulkan renderer for 3D games. The mission statement will
//! probably narrow down over time.

macro_rules! cstr {
    ($string:literal) => {
        unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(concat!($string, "\0").as_bytes()) }
    };
}

// internal modules:

mod debug_utils;

// public-facing modules:

mod error;
pub use error::Error;

mod driver;
pub use driver::Driver;

mod gpu;
pub use gpu::{Gpu, GpuId, GpuInfo};

mod canvas;
pub use canvas::Canvas;

mod pipeline;
pub use pipeline::Pipeline;

mod mesh;
pub use mesh::Mesh;
