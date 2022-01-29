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
mod mesh;
mod vulkan_raii;

// public-facing modules:

mod arena;
pub use arena::VulkanArena;

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

mod pipeline;
pub use pipeline::Pipeline;

mod scene;
pub use scene::Scene;

// docs modules:

pub mod plan_for_refcountification {
    //! The plan is to refactor this crate's way of modeling Vulkan
    //! object relations, somehow.
    //!
    //! Those relations refer to the order of creation and
    //! destruction, where e.g. a VkBuffer must be created by a
    //! VkDevice, and destroyed (using a VkDevice) before VkDevice is
    //! destroyed.
    //!
    //! I'm writing this doc to clarify my current idea of the
    //! situation, as I've been mulling over it for a few months now
    //! while being busy with other projects.
    //!
    //! # The issue
    //!
    //! Currently this is loosely managed with two techniques: borrows
    //! (Foo::new, Drop impl pairs), and manual creation-destruction
    //! code (Foo::new, Foo::clean_up). The Driver/Gpu/Canvas divide
    //! is defined by the fact that each of them borrows the "parent",
    //! and must be held in some external scope. This means that you
    //! couldn't make a struct that owns all three (or any two even),
    //! because you cannot make self-referential structs in
    //! Rust. Other resources (images, buffers) do not directly own
    //! their parent VkDevice or such, they are managed manually by
    //! the [Gltf] object, which borrows [Arena], which borrows an
    //! [ash::Device]. This is all a great pain to manage, and is
    //! half-safe, half-manual-memory-management.
    //!
    //! # What to do?
    //!
    //! Fundamentally, to make a safe Buffer object, it needs to be
    //! able to access a VkDevice in its Drop implementation. For
    //! this, it needs a reference, either a borrow or an Rc. The main
    //! issue with borrows is that you need to separate the owner and
    //! the borrower of a thing. The main issue with Rcs is that bugs
    //! relating to lifetimes get shoved to runtime instead of being
    //! detected by the borrow checker, which I would really like to
    //! avoid.
    //!
    //! The current architecture really makes it feel like Rcs are
    //! required, but now that I've written the issue down like this,
    //! it might just be that the crate is too centralized re: structs
    //! having to own so much stuff. Maybe the borrows aren't actually
    //! a problem, maybe the humongous [Gpu] struct is? It should be
    //! noted that my ideas for the architecture have also changed in
    //! the meantime: I used to place a lot of value on the idea that
    //! the crate would have a very small API, and very few structs
    //! for the dependent to manage, but now I think that might be
    //! misguided.
    #[allow(unused_imports)] // actually used by the doc above
    use super::*;
}
