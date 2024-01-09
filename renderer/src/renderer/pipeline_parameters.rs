use core::mem;

use ash::vk;
use enum_map::EnumMap;
use glam::Mat4;

/// Constants from renderer/shaders/glsl/constants.glsl
pub(crate) mod constants {
    #![allow(unused_parens)]
    include!(concat!(env!("OUT_DIR"), "/shader_constants.rs"));
}
pub(crate) mod render_passes;
pub(crate) mod uniforms;
pub(crate) mod vertex_buffers;

use constants::*;
use render_passes::RenderPass;
use vertex_buffers::*;

pub const SKINNED_PIPELINES: [PipelineIndex; 3] =
    [PipelineIndex::PbrSkinnedOpaque, PipelineIndex::PbrSkinnedAlphaToCoverage, PipelineIndex::PbrSkinnedBlended];

pub const PBR_PIPELINES: [PipelineIndex; 6] = [
    PipelineIndex::PbrOpaque,
    PipelineIndex::PbrAlphaToCoverage,
    PipelineIndex::PbrBlended,
    PipelineIndex::PbrSkinnedOpaque,
    PipelineIndex::PbrSkinnedAlphaToCoverage,
    PipelineIndex::PbrSkinnedBlended,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, enum_map::Enum)]
pub enum PipelineIndex {
    /// Opaque geometry pass.
    PbrOpaque,
    /// Skinned opaque geometry pass.
    PbrSkinnedOpaque,
    /// Alpha-to-coverage "fake transparent" geometry pass.
    PbrAlphaToCoverage,
    /// Skinned alpha-to-coverage "fake transparent" geometry pass.
    PbrSkinnedAlphaToCoverage,
    /// Transparent geomtry pass.
    PbrBlended,
    /// Skinned transparent geomtry pass.
    PbrSkinnedBlended,
    /// Post-processing pass before MSAA resolve and up/downsampling.
    RenderResolutionPostProcess,
}

impl PipelineIndex {
    /// A pipeline whose first descriptor set is written to and read
    /// from, where the shared descriptor set is concerned.
    pub const SHARED_DESCRIPTOR_PIPELINE: PipelineIndex = PipelineIndex::PbrOpaque;

    pub(crate) fn skinned(self) -> bool {
        use PipelineIndex::{PbrSkinnedAlphaToCoverage, PbrSkinnedBlended, PbrSkinnedOpaque};
        [PbrSkinnedOpaque, PbrSkinnedAlphaToCoverage, PbrSkinnedBlended].contains(&self)
    }

    pub fn vertex_layout(self) -> VertexLayout {
        match self {
            PipelineIndex::PbrOpaque | PipelineIndex::PbrAlphaToCoverage | PipelineIndex::PbrBlended => VertexLayout::StaticMesh,
            PipelineIndex::PbrSkinnedOpaque | PipelineIndex::PbrSkinnedAlphaToCoverage | PipelineIndex::PbrSkinnedBlended => {
                VertexLayout::SkinnedMesh
            }
            PipelineIndex::RenderResolutionPostProcess => VertexLayout::FullscreenQuad,
        }
    }
}

/// Maps every `PipelineIndex` to a T.
pub type PipelineMap<T> = EnumMap<PipelineIndex, T>;

pub(crate) struct DescriptorSetLayoutParams {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub descriptor_count: u32,
    pub stage_flags: vk::ShaderStageFlags,
    pub binding_flags: vk::DescriptorBindingFlags,
    /// Uniform and storage buffer size limit in bytes. This is used to check
    /// that the hardware can support a descriptor containing a buffer this big
    /// (maxUniformBufferRange and maxStorageBufferRange), and a warning is
    /// emitted if the renderer uploads a buffer larger than this.
    pub descriptor_size: Option<vk::DeviceSize>,
}

/// Shader file names paired with the SPIR-V binaries. There are variants for
/// cases where a different shader is used based on other pipeline parameters,
/// e.g. single-sample and multi-sample.
#[derive(Clone, Copy)]
pub(crate) enum Shader {
    SingleVariant((&'static str, &'static [u32])),
    MsaaVariants { single_sample: (&'static str, &'static [u32]), multi_sample: (&'static str, &'static [u32]) },
}
macro_rules! shader {
    ($shader_name:literal) => {{
        use crate::include_words;
        static SPIRV: &[u32] = include_words!(concat!("../../shaders/spirv/", $shader_name, ".spv"));
        ($shader_name, SPIRV)
    }};
}

#[derive(Clone, Copy)]
pub(crate) struct PipelineParameters {
    pub alpha_to_coverage: bool,
    pub blended: bool,
    pub depth_test: bool,
    pub depth_write: bool,
    pub sample_shading: bool,
    pub min_sample_shading_factor: f32,
    pub render_pass: RenderPass,
    pub vertex_shader: Shader,
    pub fragment_shader: Shader,
    pub bindings: &'static [vk::VertexInputBindingDescription],
    pub attributes: &'static [vk::VertexInputAttributeDescription],
    pub descriptor_sets: &'static [&'static [DescriptorSetLayoutParams]],
}

/// A descriptor set that should be used as the first set for every
/// pipeline, so that global state (projection, view transforms) can
/// be bound once and never touched again during a frame.
///
/// In concrete terms, this maps to uniforms in shaders with the
/// layout `set = 0`, and the bindings are in order.
static SHARED_DESCRIPTOR_SET_0: &[DescriptorSetLayoutParams] = &[
    DescriptorSetLayoutParams {
        binding: UF_TRANSFORMS_BINDING,
        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::VERTEX,
        binding_flags: vk::DescriptorBindingFlags::empty(),
        descriptor_size: Some(mem::size_of::<uniforms::ProjViewTransforms>() as vk::DeviceSize),
    },
    DescriptorSetLayoutParams {
        binding: UF_RENDER_SETTINGS_BINDING,
        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        binding_flags: vk::DescriptorBindingFlags::empty(),
        descriptor_size: Some(mem::size_of::<uniforms::RenderSettings>() as vk::DeviceSize),
    },
    DescriptorSetLayoutParams {
        binding: UF_DRAW_CALL_VERT_PARAMS_BINDING,
        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::VERTEX,
        binding_flags: vk::DescriptorBindingFlags::empty(),
        descriptor_size: Some(mem::size_of::<uniforms::DrawCallVertParams>() as vk::DeviceSize),
    },
    DescriptorSetLayoutParams {
        binding: UF_DRAW_CALL_FRAG_PARAMS_BINDING,
        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        binding_flags: vk::DescriptorBindingFlags::empty(),
        descriptor_size: Some(mem::size_of::<uniforms::DrawCallFragParams>() as vk::DeviceSize),
    },
];

static PBR_DESCRIPTOR_SET_1: &[DescriptorSetLayoutParams] = &[
    DescriptorSetLayoutParams {
        binding: UF_SAMPLER_BINDING,
        descriptor_type: vk::DescriptorType::SAMPLER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        binding_flags: vk::DescriptorBindingFlags::empty(),
        descriptor_size: None,
    },
    DescriptorSetLayoutParams {
        binding: UF_TEX_BASE_COLOR_BINDING,
        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
        descriptor_count: MAX_TEXTURE_COUNT,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        binding_flags: vk::DescriptorBindingFlags::PARTIALLY_BOUND,
        descriptor_size: None,
    },
    DescriptorSetLayoutParams {
        binding: UF_TEX_METALLIC_ROUGHNESS_BINDING,
        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
        descriptor_count: MAX_TEXTURE_COUNT,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        binding_flags: vk::DescriptorBindingFlags::PARTIALLY_BOUND,
        descriptor_size: None,
    },
    DescriptorSetLayoutParams {
        binding: UF_TEX_NORMAL_BINDING,
        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
        descriptor_count: MAX_TEXTURE_COUNT,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        binding_flags: vk::DescriptorBindingFlags::PARTIALLY_BOUND,
        descriptor_size: None,
    },
    DescriptorSetLayoutParams {
        binding: UF_TEX_OCCLUSION_BINDING,
        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
        descriptor_count: MAX_TEXTURE_COUNT,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        binding_flags: vk::DescriptorBindingFlags::PARTIALLY_BOUND,
        descriptor_size: None,
    },
    DescriptorSetLayoutParams {
        binding: UF_TEX_EMISSIVE_BINDING,
        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
        descriptor_count: MAX_TEXTURE_COUNT,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        binding_flags: vk::DescriptorBindingFlags::PARTIALLY_BOUND,
        descriptor_size: None,
    },
    DescriptorSetLayoutParams {
        binding: UF_PBR_FACTORS_BINDING,
        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        binding_flags: vk::DescriptorBindingFlags::empty(),
        descriptor_size: Some(mem::size_of::<uniforms::PbrFactors>() as vk::DeviceSize),
    },
];

static OPAQUE_PARAMETERS: PipelineParameters = PipelineParameters {
    alpha_to_coverage: false,
    blended: false,
    depth_test: true,
    depth_write: true,
    sample_shading: false,
    min_sample_shading_factor: 0.0,
    render_pass: RenderPass::Geometry,
    vertex_shader: Shader::SingleVariant(shader!("variants/main-static.vert")),
    fragment_shader: Shader::SingleVariant(shader!("main.frag")),
    bindings: VERTEX_BINDING_DESCRIPTIONS.as_array()[VertexLayout::StaticMesh as usize],
    attributes: VERTEX_ATTRIBUTE_DESCRIPTIONS.as_array()[VertexLayout::StaticMesh as usize],
    descriptor_sets: &[SHARED_DESCRIPTOR_SET_0, PBR_DESCRIPTOR_SET_1],
};

static SKINNED_OPAQUE_PARAMETERS: PipelineParameters = PipelineParameters {
    alpha_to_coverage: false,
    blended: false,
    depth_test: true,
    depth_write: true,
    sample_shading: false,
    min_sample_shading_factor: 0.0,
    render_pass: RenderPass::Geometry,
    vertex_shader: Shader::SingleVariant(shader!("variants/main-skinned.vert")),
    fragment_shader: Shader::SingleVariant(shader!("main.frag")),
    bindings: VERTEX_BINDING_DESCRIPTIONS.as_array()[VertexLayout::SkinnedMesh as usize],
    attributes: VERTEX_ATTRIBUTE_DESCRIPTIONS.as_array()[VertexLayout::SkinnedMesh as usize],
    descriptor_sets: &[
        SHARED_DESCRIPTOR_SET_0,
        PBR_DESCRIPTOR_SET_1,
        &[DescriptorSetLayoutParams {
            binding: UF_SKELETON_BINDING,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            binding_flags: vk::DescriptorBindingFlags::empty(),
            descriptor_size: Some(mem::size_of::<Mat4>() as vk::DeviceSize * MAX_JOINT_COUNT as vk::DeviceSize),
        }],
    ],
};

static CLIPPED_PARAMETERS: PipelineParameters = PipelineParameters { alpha_to_coverage: true, ..OPAQUE_PARAMETERS };

static SKINNED_CLIPPED_PARAMETERS: PipelineParameters = PipelineParameters { alpha_to_coverage: true, ..SKINNED_OPAQUE_PARAMETERS };

static BLENDED_PARAMETERS: PipelineParameters = PipelineParameters { blended: true, ..OPAQUE_PARAMETERS };

static SKINNED_BLENDED_PARAMETERS: PipelineParameters = PipelineParameters { blended: true, ..SKINNED_OPAQUE_PARAMETERS };

static RENDER_RESOLUTION_POST_PROCESS: PipelineParameters = PipelineParameters {
    alpha_to_coverage: false,
    blended: false,
    depth_test: false,
    depth_write: false,
    sample_shading: true,
    min_sample_shading_factor: 1.0,
    render_pass: RenderPass::PostProcess,
    vertex_shader: Shader::SingleVariant(shader!("fullscreen.vert")),
    fragment_shader: Shader::MsaaVariants {
        single_sample: shader!("variants/render_res_pp-singlesample.frag"),
        multi_sample: shader!("variants/render_res_pp-multisample.frag"),
    },
    bindings: &[],
    attributes: &[],
    descriptor_sets: &[
        SHARED_DESCRIPTOR_SET_0,
        &[DescriptorSetLayoutParams {
            binding: UF_HDR_FRAMEBUFFER_BINDING,
            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
            binding_flags: vk::DescriptorBindingFlags::empty(),
            descriptor_size: None,
        }],
    ],
};

pub(crate) static PIPELINE_PARAMETERS: PipelineMap<PipelineParameters> = PipelineMap::from_array([
    OPAQUE_PARAMETERS,
    SKINNED_OPAQUE_PARAMETERS,
    CLIPPED_PARAMETERS,
    SKINNED_CLIPPED_PARAMETERS,
    BLENDED_PARAMETERS,
    SKINNED_BLENDED_PARAMETERS,
    RENDER_RESOLUTION_POST_PROCESS,
]);
