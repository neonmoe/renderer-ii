use ash::vk;
use glam::{Mat4, Vec2, Vec3, Vec4};
use std::mem;
use std::mem::MaybeUninit;

pub const MAX_TEXTURE_COUNT: u32 = 128; // Keep in sync with shaders/constants.glsl.

/// The push constant pushed to the fragment shader for every draw
/// call.
#[derive(Clone, Copy)]
pub struct PushConstantStruct {
    // NOTE: Careful with changing this struct, the bytemuck impls are very strict!
    pub texture_index: u32,
    pub debug_value: u32,
}

unsafe impl bytemuck::Zeroable for PushConstantStruct {}
unsafe impl bytemuck::Pod for PushConstantStruct {}

const ALL_PIPELINES: [PipelineIndex; PipelineIndex::Count as usize] = [
    PipelineIndex::Opaque,
    PipelineIndex::SkinnedOpaque,
    PipelineIndex::Clipped,
    PipelineIndex::SkinnedClipped,
    PipelineIndex::Blended,
    PipelineIndex::SkinnedBlended,
    PipelineIndex::RenderResolutionPostProcess,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineIndex {
    /// Opaque geometry pass.
    Opaque,
    /// Animated opaque geometry pass.
    SkinnedOpaque,
    /// Alpha-to-coverage "fake transparent" geometry pass.
    Clipped,
    /// Animated alpha-to-coverage "fake transparent" geometry pass.
    SkinnedClipped,
    /// Transparent geomtry pass.
    Blended,
    /// Animated transparent geomtry pass.
    SkinnedBlended,
    /// Post-processing pass before MSAA resolve and up/downsampling.
    RenderResolutionPostProcess,
    #[doc(hidden)]
    Count,
}

impl PipelineIndex {
    /// A pipeline whose first descriptor set is written to and read
    /// from, where the shared descriptor set is concerned.
    pub const SHARED_DESCRIPTOR_PIPELINE: PipelineIndex = PipelineIndex::Opaque;
}

/// Maps every PipelineIndex to a T.
pub struct PipelineMap<T> {
    buffer: [MaybeUninit<T>; PipelineIndex::Count as usize],
}

impl<T> Drop for PipelineMap<T> {
    fn drop(&mut self) {
        for buffer in &mut self.buffer {
            unsafe { buffer.as_mut_ptr().drop_in_place() };
        }
    }
}

impl<T> PipelineMap<T> {
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
    pub const fn len(&self) -> usize {
        PipelineIndex::Count as usize
    }
    pub fn get(&self, pipeline: PipelineIndex) -> &T {
        // Safety: initialized in PipelineMap::new
        unsafe { self.buffer[pipeline as usize].assume_init_ref() }
    }
    pub fn get_mut(&mut self, pipeline: PipelineIndex) -> &mut T {
        // Safety: initialized in PipelineMap::new
        unsafe { self.buffer[pipeline as usize].assume_init_mut() }
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        // Safety: initialized in PipelineMap::new
        self.buffer.iter().map(|o| unsafe { o.assume_init_ref() })
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

pub(crate) struct DescriptorSetLayoutParams {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub descriptor_count: u32,
    pub stage_flags: vk::ShaderStageFlags,
    pub binding_flags: vk::DescriptorBindingFlags,
}

#[derive(Clone, Copy)]
pub(crate) struct PipelineParameters {
    pub alpha_to_coverage: bool,
    pub blended: bool,
    pub depth_test: bool,
    pub depth_write: bool,
    pub sample_shading: bool,
    pub min_sample_shading_factor: f32,
    pub subpass: u32,
    pub vertex_shader: (&'static str, &'static [u32]),
    pub fragment_shader: (&'static str, &'static [u32]),
    pub bindings: &'static [vk::VertexInputBindingDescription],
    pub attributes: &'static [vk::VertexInputAttributeDescription],
    pub descriptor_sets: &'static [&'static [DescriptorSetLayoutParams]],
}

static INSTANCED_TRANSFORM_BINDING_0: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: 0,
    stride: mem::size_of::<Mat4>() as u32,
    input_rate: vk::VertexInputRate::INSTANCE,
};
static POSITION_BINDING_1: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: 1,
    stride: mem::size_of::<Vec3>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};
static TEXCOORD0_BINDING_2: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: 2,
    stride: mem::size_of::<Vec2>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};
static NORMAL_BINDING_3: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: 3,
    stride: mem::size_of::<Vec3>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};
static TANGENT_BINDING_4: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: 4,
    stride: mem::size_of::<Vec4>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};

static INSTANCED_TRANSFORM_BINDING_0_ATTRIBUTES: [vk::VertexInputAttributeDescription; 4] = [
    vk::VertexInputAttributeDescription {
        binding: 0,
        location: 0,
        format: vk::Format::R32G32B32A32_SFLOAT,
        offset: 0,
    },
    vk::VertexInputAttributeDescription {
        binding: 0,
        location: 1,
        format: vk::Format::R32G32B32A32_SFLOAT,
        offset: std::mem::size_of::<[Vec4; 1]>() as u32,
    },
    vk::VertexInputAttributeDescription {
        binding: 0,
        location: 2,
        format: vk::Format::R32G32B32A32_SFLOAT,
        offset: std::mem::size_of::<[Vec4; 2]>() as u32,
    },
    vk::VertexInputAttributeDescription {
        binding: 0,
        location: 3,
        format: vk::Format::R32G32B32A32_SFLOAT,
        offset: std::mem::size_of::<[Vec4; 3]>() as u32,
    },
];
static POSITION_BINDING_1_ATTRIBUTE: vk::VertexInputAttributeDescription = vk::VertexInputAttributeDescription {
    binding: 1,
    location: 4,
    format: vk::Format::R32G32B32_SFLOAT,
    offset: 0,
};
static TEXCOORD0_BINDING_2_ATTRIBUTE: vk::VertexInputAttributeDescription = vk::VertexInputAttributeDescription {
    binding: 2,
    location: 5,
    format: vk::Format::R32G32_SFLOAT,
    offset: 0,
};
static NORMAL_BINDING_3_ATTRIBUTE: vk::VertexInputAttributeDescription = vk::VertexInputAttributeDescription {
    binding: 3,
    location: 6,
    format: vk::Format::R32G32B32_SFLOAT,
    offset: 0,
};
static TANGENT_BINDING_4_ATTRIBUTE: vk::VertexInputAttributeDescription = vk::VertexInputAttributeDescription {
    binding: 4,
    location: 7,
    format: vk::Format::R32G32B32A32_SFLOAT,
    offset: 0,
};

/// A descriptor set that should be used as the first set for every
/// pipeline, so that global state (projection, view transforms) can
/// be bound once and never touched again during a frame.
///
/// In concrete terms, this maps to uniforms in shaders with the
/// layout `set = 0`, and the bindings are in order.
static SHARED_DESCRIPTOR_SET_0: &[DescriptorSetLayoutParams] = &[DescriptorSetLayoutParams {
    binding: 0,
    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
    descriptor_count: 1,
    stage_flags: vk::ShaderStageFlags::VERTEX,
    binding_flags: vk::DescriptorBindingFlags::empty(),
}];

static PBR_DESCRIPTOR_SET_1: &[DescriptorSetLayoutParams] = &[
    DescriptorSetLayoutParams {
        binding: 0,
        descriptor_type: vk::DescriptorType::SAMPLER,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        binding_flags: vk::DescriptorBindingFlags::empty(),
    },
    DescriptorSetLayoutParams {
        binding: 1,
        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
        descriptor_count: MAX_TEXTURE_COUNT,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        binding_flags: vk::DescriptorBindingFlags::PARTIALLY_BOUND,
    },
    DescriptorSetLayoutParams {
        binding: 2,
        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
        descriptor_count: MAX_TEXTURE_COUNT,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        binding_flags: vk::DescriptorBindingFlags::PARTIALLY_BOUND,
    },
    DescriptorSetLayoutParams {
        binding: 3,
        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
        descriptor_count: MAX_TEXTURE_COUNT,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        binding_flags: vk::DescriptorBindingFlags::PARTIALLY_BOUND,
    },
    DescriptorSetLayoutParams {
        binding: 4,
        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
        descriptor_count: MAX_TEXTURE_COUNT,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        binding_flags: vk::DescriptorBindingFlags::PARTIALLY_BOUND,
    },
    DescriptorSetLayoutParams {
        binding: 5,
        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
        descriptor_count: MAX_TEXTURE_COUNT,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        binding_flags: vk::DescriptorBindingFlags::PARTIALLY_BOUND,
    },
    DescriptorSetLayoutParams {
        binding: 6,
        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
        descriptor_count: MAX_TEXTURE_COUNT,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        binding_flags: vk::DescriptorBindingFlags::PARTIALLY_BOUND,
    },
];

static OPAQUE_PARAMETERS: PipelineParameters = PipelineParameters {
    alpha_to_coverage: false,
    blended: false,
    depth_test: true,
    depth_write: true,
    sample_shading: false,
    min_sample_shading_factor: 0.0,
    subpass: 0,
    vertex_shader: shaders::include_spirv!("shaders/main.vert"),
    fragment_shader: shaders::include_spirv!("shaders/main.frag"),
    bindings: &[
        INSTANCED_TRANSFORM_BINDING_0,
        POSITION_BINDING_1,
        TEXCOORD0_BINDING_2,
        NORMAL_BINDING_3,
        TANGENT_BINDING_4,
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
    ],
    descriptor_sets: &[SHARED_DESCRIPTOR_SET_0, PBR_DESCRIPTOR_SET_1],
};

static SKINNED_OPAQUE_PARAMETERS: PipelineParameters = PipelineParameters {
    vertex_shader: shaders::include_spirv!("shaders/main.vert", "ANIMATED"),
    fragment_shader: shaders::include_spirv!("shaders/main.frag", "ANIMATED"),
    descriptor_sets: &[
        SHARED_DESCRIPTOR_SET_0,
        PBR_DESCRIPTOR_SET_1,
        &[DescriptorSetLayoutParams {
            binding: 1,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            binding_flags: vk::DescriptorBindingFlags::empty(),
        }],
    ],
    ..OPAQUE_PARAMETERS
};

static CLIPPED_PARAMETERS: PipelineParameters = PipelineParameters {
    alpha_to_coverage: true,
    ..OPAQUE_PARAMETERS
};

static SKINNED_CLIPPED_PARAMETERS: PipelineParameters = PipelineParameters {
    alpha_to_coverage: true,
    ..SKINNED_OPAQUE_PARAMETERS
};

static BLENDED_PARAMETERS: PipelineParameters = PipelineParameters {
    blended: true,
    ..OPAQUE_PARAMETERS
};

static SKINNED_BLENDED_PARAMETERS: PipelineParameters = PipelineParameters {
    blended: true,
    ..SKINNED_OPAQUE_PARAMETERS
};

static RENDER_RESOLUTION_POST_PROCESS: PipelineParameters = PipelineParameters {
    alpha_to_coverage: false,
    blended: false,
    depth_test: false,
    depth_write: false,
    sample_shading: true,
    min_sample_shading_factor: 1.0,
    subpass: 1,
    vertex_shader: shaders::include_spirv!("shaders/fullscreen.vert"),
    fragment_shader: shaders::include_spirv!("shaders/render_res_pp.frag"),
    bindings: &[],
    attributes: &[],
    descriptor_sets: &[&[DescriptorSetLayoutParams {
        binding: 0,
        descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
        descriptor_count: 1,
        stage_flags: vk::ShaderStageFlags::FRAGMENT,
        binding_flags: vk::DescriptorBindingFlags::empty(),
    }]],
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
