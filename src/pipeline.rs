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

const ALL_PIPELINES: [Pipeline; Pipeline::Count as usize] = [Pipeline::Gltf];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pipeline {
    Gltf,
    #[doc(hidden)]
    Count,
}

impl Pipeline {
    /// A pipeline whose first descriptor set is written to and read
    /// from, where the shared descriptor set is concerned.
    pub const SHARED_DESCRIPTOR_PIPELINE: Pipeline = Pipeline::Gltf;
}

/// Maps every pipeline to a T.
pub struct PipelineMap<T> {
    buffer: [MaybeUninit<T>; Pipeline::Count as usize],
}

impl<T> Drop for PipelineMap<T> {
    fn drop(&mut self) {
        for buffer in &mut self.buffer {
            unsafe { buffer.as_mut_ptr().drop_in_place() };
        }
    }
}

impl<T> PipelineMap<T> {
    pub fn new<E, F: FnMut(Pipeline) -> Result<T, E>>(mut f: F) -> Result<PipelineMap<T>, E> {
        let mut buffer = [MaybeUninit::uninit(); Pipeline::Count as usize];
        for (value, pipeline) in buffer.iter_mut().zip(ALL_PIPELINES) {
            value.write(f(pipeline)?);
        }
        Ok(PipelineMap { buffer })
    }
    pub const fn len(&self) -> usize {
        Pipeline::Count as usize
    }
    pub fn get(&self, pipeline: Pipeline) -> &T {
        // Safety: initialized in PipelineMap::new
        unsafe { self.buffer[pipeline as usize].assume_init_ref() }
    }
    pub fn get_mut(&mut self, pipeline: Pipeline) -> &mut T {
        // Safety: initialized in PipelineMap::new
        unsafe { self.buffer[pipeline as usize].assume_init_mut() }
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        // Safety: initialized in PipelineMap::new
        self.buffer.iter().map(|o| unsafe { o.assume_init_ref() })
    }
    pub fn iter_with_pipeline(&self) -> impl Iterator<Item = (Pipeline, &T)> {
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

pub(crate) struct PipelineParameters {
    pub vertex_shader: &'static [u32],
    pub vertex_shader_name: &'static str,
    pub fragment_shader: &'static [u32],
    pub fragment_shader_name: &'static str,
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
static SHARED_DESCRIPTOR_SET_0: &[DescriptorSetLayoutParams] = &[DescriptorSetLayoutParams {
    binding: 0,
    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
    descriptor_count: 1,
    stage_flags: vk::ShaderStageFlags::VERTEX,
    binding_flags: vk::DescriptorBindingFlags::empty(),
}];

static INSTANCED_TRANSFORM_BINDING_0: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: 0,
    stride: mem::size_of::<Mat4>() as u32,
    input_rate: vk::VertexInputRate::INSTANCE,
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

pub(crate) static PIPELINE_PARAMETERS: PipelineMap<PipelineParameters> = PipelineMap {
    buffer: [MaybeUninit::new(PipelineParameters {
        vertex_shader: shaders::include_spirv!("shaders/textured.vert"),
        vertex_shader_name: "shaders/textured.vert",
        fragment_shader: shaders::include_spirv!("shaders/textured.frag"),
        fragment_shader_name: "shaders/textured.frag",
        bindings: &[
            INSTANCED_TRANSFORM_BINDING_0,
            vk::VertexInputBindingDescription {
                binding: 1,
                stride: mem::size_of::<Vec3>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            },
            vk::VertexInputBindingDescription {
                binding: 2,
                stride: mem::size_of::<Vec2>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            },
            vk::VertexInputBindingDescription {
                binding: 3,
                stride: mem::size_of::<Vec3>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            },
            vk::VertexInputBindingDescription {
                binding: 4,
                stride: mem::size_of::<Vec4>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            },
        ],
        attributes: &[
            INSTANCED_TRANSFORM_BINDING_0_ATTRIBUTES[0],
            INSTANCED_TRANSFORM_BINDING_0_ATTRIBUTES[1],
            INSTANCED_TRANSFORM_BINDING_0_ATTRIBUTES[2],
            INSTANCED_TRANSFORM_BINDING_0_ATTRIBUTES[3],
            vk::VertexInputAttributeDescription {
                binding: 1,
                location: 4,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                binding: 2,
                location: 5,
                format: vk::Format::R32G32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                binding: 3,
                location: 6,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                binding: 4,
                location: 7,
                format: vk::Format::R32G32B32A32_SFLOAT,
                offset: 0,
            },
        ],
        descriptor_sets: &[
            SHARED_DESCRIPTOR_SET_0,
            &[
                DescriptorSetLayoutParams {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::SAMPLER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::FRAGMENT,
                    binding_flags: vk::DescriptorBindingFlags::PARTIALLY_BOUND,
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
            ],
        ],
    })],
};
