//! A Vulkan renderer for 3D games. The mission statement will
//! probably narrow down over time.

macro_rules! cstr {
    ($string:literal) => {
        unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(concat!($string, "\0").as_bytes()) }
            .as_ptr()
    };
}

// internal modules:

mod debug_utils;

// public-facing modules:

mod error;
pub use error::Error;

mod foundation;
pub use foundation::Foundation;

mod renderer;
pub use renderer::Renderer;
