use alloc::rc::Rc;
use core::cmp::Ordering;

use arrayvec::{ArrayString, ArrayVec};
use glam::Vec4;

use crate::renderer::descriptors::MAX_PIPELINES_PER_MATERIAL;
use crate::renderer::pipeline_parameters::vertex_buffers::VertexLayout;
use crate::renderer::pipeline_parameters::PipelineIndex;
use crate::vulkan_raii::ImageView;

#[derive(Clone, Copy, Default)]
pub struct PbrFactors {
    /// (r, g, b, a).
    pub base_color: Vec4,
    /// (emissive r, .. g, .. b, occlusion strength)
    pub emissive_and_occlusion: Vec4,
    /// (alpha cutoff, roughness, metallic, normal scale)
    pub alpha_rgh_mtl_normal: Vec4,
}

#[derive(Clone, Copy)]
pub enum AlphaMode {
    Opaque,
    AlphaToCoverage,
    Blend,
}

#[derive(Clone)]
pub enum PipelineSpecificData {
    Pbr {
        base_color: Option<Rc<ImageView>>,
        metallic_roughness: Option<Rc<ImageView>>,
        normal: Option<Rc<ImageView>>,
        occlusion: Option<Rc<ImageView>>,
        emissive: Option<Rc<ImageView>>,
        factors: PbrFactors,
        alpha_mode: AlphaMode,
    },
}

/// A unique index into one pipeline's textures and other material data.
pub struct Material {
    pub name: ArrayString<64>,
    pub(crate) array_indices: ArrayVec<(PipelineIndex, u32), { MAX_PIPELINES_PER_MATERIAL }>,
    pub(crate) data: PipelineSpecificData,
}

impl Material {
    pub(crate) fn array_index(&self, pipeline: PipelineIndex) -> Option<u32> {
        for &(pipeline_, index) in &self.array_indices {
            if pipeline == pipeline_ {
                return Some(index);
            }
        }
        None
    }

    pub fn pipeline(&self, vertex_layout: VertexLayout) -> PipelineIndex {
        let skinned = vertex_layout == VertexLayout::SkinnedMesh;
        for &(pipeline, _) in &self.array_indices {
            if pipeline.skinned() == skinned {
                return pipeline;
            }
        }
        // The array_indices vec is filled out with get_pipelines(), which
        // always returns both a skinned and a non-skinned pipeline.
        unreachable!()
    }
}

impl PartialEq for Material {
    fn eq(&self, other: &Self) -> bool {
        self.array_indices == other.array_indices
    }
}

impl Eq for Material {}

impl Ord for Material {
    fn cmp(&self, other: &Self) -> Ordering {
        self.array_indices.cmp(&other.array_indices)
    }
}

impl PartialOrd for Material {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
