use crate::renderer::descriptors::MAX_PIPELINES_PER_MATERIAL;
use crate::renderer::pipelines::pipeline_parameters::PipelineIndex;
use crate::vulkan_raii::{Buffer, ImageView};
use alloc::rc::Rc;
use arrayvec::{ArrayString, ArrayVec};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use core::hash::{Hash, Hasher};
use glam::Vec4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GltfFactors {
    /// (r, g, b, a).
    pub base_color: Vec4,
    /// (r, g, b, _). Vec4 to make sure there's no padding/alignment issues.
    pub emissive: Vec4,
    /// (metallic, roughness, alpha_cutoff, _). Vec4 to make sure there's no padding.
    pub metallic_roughness_alpha_cutoff: Vec4,
}

#[derive(Clone, Copy)]
pub enum AlphaMode {
    Opaque,
    AlphaToCoverage,
    Blend,
}

#[derive(Clone)]
pub enum PipelineSpecificData {
    Gltf {
        base_color: Option<Rc<ImageView>>,
        metallic_roughness: Option<Rc<ImageView>>,
        normal: Option<Rc<ImageView>>,
        occlusion: Option<Rc<ImageView>>,
        emissive: Option<Rc<ImageView>>,
        /// (Buffer, offset, size) that contains a [GltfFactors].
        factors: (Rc<Buffer>, vk::DeviceSize, vk::DeviceSize),
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

    pub fn pipeline(&self, skinned: bool) -> PipelineIndex {
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
impl Hash for Material {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.array_indices.hash(state);
    }
}
