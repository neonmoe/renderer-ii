use core::mem::{self, MaybeUninit};

use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2, Vec3, Vec4};

pub(crate) mod constants;
pub(crate) mod render_passes;

use constants::*;
use render_passes::RenderPass;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    #[doc(hidden)]
    Count,
}

pub const PIPELINE_COUNT: usize = PipelineIndex::Count as usize;

pub enum VertexBinding {
    Transform,
    Position,
    Texcoord0,
    Normal,
    Tangent,
    Joints0,
    Weights0,
    #[doc(hidden)]
    Count,
}

pub const VERTEX_BINDING_COUNT: usize = VertexBinding::Count as usize;

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

/// The per-draw-call uniform buffer.
///
/// This is the most "dynamic" data accessible in the shader which is still
/// dynamically uniform. Stored in a structure-of-arrays layout, indexed via the
/// `gl_BaseInstanceARB` variable which can be included in indirect draws.
///
/// When rendering many instances, all instances of a specific draw will refer to the same base
/// instance, so the rest is left zeroed.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct DrawCallParametersSoa {
    pub material_index: [u32; MAX_DRAWS as usize],
}

impl DrawCallParametersSoa {
    pub const MATERIAL_INDEX_OFFSET: vk::DeviceSize = 0;
    pub const MATERIAL_INDEX_ELEMENT_SIZE: vk::DeviceSize = mem::size_of::<u32>() as vk::DeviceSize;
}

/// Rust-side representation of the std430-layout `PbrFactorsSoa` struct in
/// main.frag.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PbrFactorsSoa {
    /// (r, g, b, a).
    pub base_color: [Vec4; MAX_TEXTURE_COUNT as usize],
    /// (emissive r, .. g, .. b, occlusion strength)
    pub emissive_and_occlusion: [Vec4; MAX_TEXTURE_COUNT as usize],
    /// (alpha cutoff, roughness, metallic, normal scale)
    pub alpha_rgh_mtl_normal: [Vec4; MAX_TEXTURE_COUNT as usize],
}

pub const ALL_PIPELINES: [PipelineIndex; PIPELINE_COUNT] = [
    PipelineIndex::PbrOpaque,
    PipelineIndex::PbrSkinnedOpaque,
    PipelineIndex::PbrAlphaToCoverage,
    PipelineIndex::PbrSkinnedAlphaToCoverage,
    PipelineIndex::PbrBlended,
    PipelineIndex::PbrSkinnedBlended,
    PipelineIndex::RenderResolutionPostProcess,
];

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

impl PipelineIndex {
    /// A pipeline whose first descriptor set is written to and read
    /// from, where the shared descriptor set is concerned.
    pub const SHARED_DESCRIPTOR_PIPELINE: PipelineIndex = PipelineIndex::PbrOpaque;

    pub(crate) fn skinned(self) -> bool {
        use PipelineIndex::{PbrSkinnedAlphaToCoverage, PbrSkinnedBlended, PbrSkinnedOpaque};
        [PbrSkinnedOpaque, PbrSkinnedAlphaToCoverage, PbrSkinnedBlended].contains(&self)
    }
}

/// Maps every `PipelineIndex` to a T.
pub struct PipelineMap<T> {
    buffer: [MaybeUninit<T>; PIPELINE_COUNT],
}

impl<T> Drop for PipelineMap<T> {
    fn drop(&mut self) {
        for buffer in &mut self.buffer {
            unsafe { buffer.as_mut_ptr().drop_in_place() };
        }
    }
}

impl<T> PipelineMap<T> {
    pub fn from_infallible<F: FnMut(PipelineIndex) -> T>(mut f: F) -> PipelineMap<T> {
        PipelineMap::new::<(), _>(|pipeline| Ok(f(pipeline))).unwrap()
    }
    pub fn new<E, F: FnMut(PipelineIndex) -> Result<T, E>>(mut f: F) -> Result<PipelineMap<T>, E> {
        let mut buffer = [
            MaybeUninit::uninit(),
            MaybeUninit::uninit(),
            MaybeUninit::uninit(),
            MaybeUninit::uninit(),
            MaybeUninit::uninit(),
            MaybeUninit::uninit(),
            MaybeUninit::uninit(),
        ];
        for (value, pipeline) in buffer.iter_mut().zip(ALL_PIPELINES) {
            value.write(f(pipeline)?);
        }
        Ok(PipelineMap { buffer })
    }
    #[allow(clippy::unused_self)]
    pub const fn len(&self) -> usize {
        PIPELINE_COUNT
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        // Safety: initialized in PipelineMap::new
        self.buffer.iter().map(|o| unsafe { o.assume_init_ref() })
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        // Safety: initialized in PipelineMap::new
        self.buffer.iter_mut().map(|o| unsafe { o.assume_init_mut() })
    }
    pub fn iter_with_pipeline(&self) -> impl Iterator<Item = (PipelineIndex, &T)> {
        self.buffer
            .iter()
            // Safety: initialized in PipelineMap::new
            .map(|o| unsafe { o.assume_init_ref() })
            .zip(ALL_PIPELINES)
            .map(|(t, pl)| (pl, t))
    }
}

impl<T> core::ops::Index<PipelineIndex> for PipelineMap<T> {
    type Output = T;
    fn index(&self, index: PipelineIndex) -> &Self::Output {
        // Safety: initialized in PipelineMap::new
        unsafe { self.buffer[index as usize].assume_init_ref() }
    }
}

impl<T> core::ops::IndexMut<PipelineIndex> for PipelineMap<T> {
    fn index_mut(&mut self, index: PipelineIndex) -> &mut Self::Output {
        // Safety: initialized in PipelineMap::new
        unsafe { self.buffer[index as usize].assume_init_mut() }
    }
}

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

static INSTANCED_TRANSFORM_BINDING_0: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::Transform as u32,
    stride: mem::size_of::<Mat4>() as u32,
    input_rate: vk::VertexInputRate::INSTANCE,
};
static POSITION_BINDING_1: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::Position as u32,
    stride: mem::size_of::<Vec3>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};
static TEXCOORD0_BINDING_2: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::Texcoord0 as u32,
    stride: mem::size_of::<Vec2>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};
static NORMAL_BINDING_3: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::Normal as u32,
    stride: mem::size_of::<Vec3>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};
static TANGENT_BINDING_4: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::Tangent as u32,
    stride: mem::size_of::<Vec4>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};
static JOINTS0_BINDING_5: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::Joints0 as u32,
    stride: mem::size_of::<[u8; 4]>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};
static WEIGHTS0_BINDING_6: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::Weights0 as u32,
    stride: mem::size_of::<[f32; 4]>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};

static INSTANCED_TRANSFORM_BINDING_0_ATTRIBUTES: [vk::VertexInputAttributeDescription; 4] = [
    vk::VertexInputAttributeDescription {
        binding: VertexBinding::Transform as u32,
        location: IN_TRANSFORM_LOCATION,
        format: vk::Format::R32G32B32A32_SFLOAT,
        offset: 0,
    },
    vk::VertexInputAttributeDescription {
        binding: VertexBinding::Transform as u32,
        location: IN_TRANSFORM_LOCATION + 1,
        format: vk::Format::R32G32B32A32_SFLOAT,
        offset: mem::size_of::<[Vec4; 1]>() as u32,
    },
    vk::VertexInputAttributeDescription {
        binding: VertexBinding::Transform as u32,
        location: IN_TRANSFORM_LOCATION + 2,
        format: vk::Format::R32G32B32A32_SFLOAT,
        offset: mem::size_of::<[Vec4; 2]>() as u32,
    },
    vk::VertexInputAttributeDescription {
        binding: VertexBinding::Transform as u32,
        location: IN_TRANSFORM_LOCATION + 3,
        format: vk::Format::R32G32B32A32_SFLOAT,
        offset: mem::size_of::<[Vec4; 3]>() as u32,
    },
];
static POSITION_BINDING_1_ATTRIBUTE: vk::VertexInputAttributeDescription = vk::VertexInputAttributeDescription {
    binding: VertexBinding::Position as u32,
    location: IN_POSITION_LOCATION,
    format: vk::Format::R32G32B32_SFLOAT,
    offset: 0,
};
static TEXCOORD0_BINDING_2_ATTRIBUTE: vk::VertexInputAttributeDescription = vk::VertexInputAttributeDescription {
    binding: VertexBinding::Texcoord0 as u32,
    location: IN_TEXCOORD_0_LOCATION,
    format: vk::Format::R32G32_SFLOAT,
    offset: 0,
};
static NORMAL_BINDING_3_ATTRIBUTE: vk::VertexInputAttributeDescription = vk::VertexInputAttributeDescription {
    binding: VertexBinding::Normal as u32,
    location: IN_NORMAL_LOCATION,
    format: vk::Format::R32G32B32_SFLOAT,
    offset: 0,
};
static TANGENT_BINDING_4_ATTRIBUTE: vk::VertexInputAttributeDescription = vk::VertexInputAttributeDescription {
    binding: VertexBinding::Tangent as u32,
    location: IN_TANGENT_LOCATION,
    format: vk::Format::R32G32B32A32_SFLOAT,
    offset: 0,
};
static JOINTS0_BINDING_5_ATTRIBUTE: vk::VertexInputAttributeDescription = vk::VertexInputAttributeDescription {
    binding: VertexBinding::Joints0 as u32,
    location: IN_JOINTS_0_LOCATION,
    format: vk::Format::R8G8B8A8_UINT,
    offset: 0,
};
static WEIGHTS0_BINDING_6_ATTRIBUTE: vk::VertexInputAttributeDescription = vk::VertexInputAttributeDescription {
    binding: VertexBinding::Weights0 as u32,
    location: IN_WEIGHTS_0_LOCATION,
    format: vk::Format::R32G32B32A32_SFLOAT,
    offset: 0,
};

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
        descriptor_size: Some(mem::size_of::<ProjViewTransforms>() as vk::DeviceSize),
    },
    DescriptorSetLayoutParams {
        binding: UF_RENDER_SETTINGS_BINDING,
        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        binding_flags: vk::DescriptorBindingFlags::empty(),
        descriptor_size: Some(mem::size_of::<RenderSettings>() as vk::DeviceSize),
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
        descriptor_size: Some(mem::size_of::<PbrFactorsSoa>() as vk::DeviceSize),
    },
    DescriptorSetLayoutParams {
        binding: UF_DRAW_CALL_PARAMS_BINDING,
        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        binding_flags: vk::DescriptorBindingFlags::empty(),
        descriptor_size: Some(mem::size_of::<DrawCallParametersSoa>() as vk::DeviceSize),
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
    bindings: &[INSTANCED_TRANSFORM_BINDING_0, POSITION_BINDING_1, TEXCOORD0_BINDING_2, NORMAL_BINDING_3, TANGENT_BINDING_4],
    attributes: &[
        INSTANCED_TRANSFORM_BINDING_0_ATTRIBUTES[0],
        INSTANCED_TRANSFORM_BINDING_0_ATTRIBUTES[1],
        INSTANCED_TRANSFORM_BINDING_0_ATTRIBUTES[2],
        INSTANCED_TRANSFORM_BINDING_0_ATTRIBUTES[3],
        POSITION_BINDING_1_ATTRIBUTE,
        TEXCOORD0_BINDING_2_ATTRIBUTE,
        NORMAL_BINDING_3_ATTRIBUTE,
        TANGENT_BINDING_4_ATTRIBUTE,
    ],
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
    bindings: &[
        INSTANCED_TRANSFORM_BINDING_0,
        POSITION_BINDING_1,
        TEXCOORD0_BINDING_2,
        NORMAL_BINDING_3,
        TANGENT_BINDING_4,
        JOINTS0_BINDING_5,
        WEIGHTS0_BINDING_6,
    ],
    attributes: &[
        INSTANCED_TRANSFORM_BINDING_0_ATTRIBUTES[0],
        INSTANCED_TRANSFORM_BINDING_0_ATTRIBUTES[1],
        INSTANCED_TRANSFORM_BINDING_0_ATTRIBUTES[2],
        INSTANCED_TRANSFORM_BINDING_0_ATTRIBUTES[3],
        POSITION_BINDING_1_ATTRIBUTE,
        TEXCOORD0_BINDING_2_ATTRIBUTE,
        NORMAL_BINDING_3_ATTRIBUTE,
        TANGENT_BINDING_4_ATTRIBUTE,
        JOINTS0_BINDING_5_ATTRIBUTE,
        WEIGHTS0_BINDING_6_ATTRIBUTE,
    ],
    descriptor_sets: &[
        SHARED_DESCRIPTOR_SET_0,
        PBR_DESCRIPTOR_SET_1,
        &[DescriptorSetLayoutParams {
            binding: UF_SKELETON_BINDING,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            binding_flags: vk::DescriptorBindingFlags::empty(),
            descriptor_size: Some(mem::size_of::<Mat4>() as vk::DeviceSize * MAX_BONE_COUNT as vk::DeviceSize),
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

pub(crate) static PIPELINE_PARAMETERS: PipelineMap<PipelineParameters> = PipelineMap {
    buffer: [
        MaybeUninit::new(OPAQUE_PARAMETERS),
        MaybeUninit::new(SKINNED_OPAQUE_PARAMETERS),
        MaybeUninit::new(CLIPPED_PARAMETERS),
        MaybeUninit::new(SKINNED_CLIPPED_PARAMETERS),
        MaybeUninit::new(BLENDED_PARAMETERS),
        MaybeUninit::new(SKINNED_BLENDED_PARAMETERS),
        MaybeUninit::new(RENDER_RESOLUTION_POST_PROCESS),
    ],
};
