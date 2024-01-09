use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec4};

use crate::renderer::pipeline_parameters::constants::{MAX_DRAW_CALLS, MAX_TEXTURE_COUNT};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct ProjViewTransforms {
    pub projection: Mat4,
    pub view: Mat4,
}

/// The per-frame uniform buffer.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct RenderSettings {
    pub debug_value: u32,
}

/// The per-draw-call uniform buffer, accessible in the fragment shader.
///
/// This is the most "dynamic" data accessible in the shader which is still
/// dynamically uniform. Stored in a structure-of-arrays layout, indexed via the
/// `gl_BaseInstanceARB` variable which can be included in indirect draws.
///
/// When rendering many instances, all instances of a specific draw will refer to the same base
/// instance, so the rest is left zeroed.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct DrawCallFragParams {
    pub material_index: [u32; MAX_DRAW_CALLS as usize],
}

/// The per-draw-call uniform buffer, accessible in the vertex shader.
///
/// This is the most "dynamic" data accessible in the shader which is still
/// dynamically uniform. Stored in a structure-of-arrays layout, indexed via the
/// `gl_BaseInstanceARB` variable which can be included in indirect draws.
///
/// When rendering many instances, all instances of a specific draw will refer to the same base
/// instance, so the rest is left zeroed.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct DrawCallVertParams {
    pub joints_offset: [u32; MAX_DRAW_CALLS as usize],
}

/// Rust-side representation of the std430-layout `PbrFactorsSoa` struct in
/// main.frag.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PbrFactors {
    /// (r, g, b, a).
    pub base_color: [Vec4; MAX_TEXTURE_COUNT as usize],
    /// (emissive r, .. g, .. b, occlusion strength)
    pub emissive_and_occlusion: [Vec4; MAX_TEXTURE_COUNT as usize],
    /// (alpha cutoff, roughness, metallic, normal scale)
    pub alpha_rgh_mtl_normal: [Vec4; MAX_TEXTURE_COUNT as usize],
}
