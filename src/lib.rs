//! A Vulkan renderer for 3D games. The mission statement will
//! probably narrow down over time.

macro_rules! cstr {
    ($string:literal) => {
        unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(concat!($string, "\0").as_bytes()) }
    };
}

// internal modules:

mod debug_utils;
mod descriptors;

// public-facing modules:

mod arena;
pub use arena::Arena;

pub use ash::vk;

mod camera;
pub use camera::Camera;

mod canvas;
pub use canvas::Canvas;

mod driver;
pub use driver::Driver;

mod error;
pub use error::Error;

mod gltf;
pub use gltf::{Gltf, GltfResources, MeshIter};

mod gpu;
pub use gpu::{FrameIndex, Gpu, GpuId, GpuInfo};

pub mod image_loading;

mod material;
pub use material::Material;

mod mesh;
pub use mesh::{IndexType, Mesh};

mod pipeline;
pub use pipeline::Pipeline;

mod scene;
pub use scene::Scene;
