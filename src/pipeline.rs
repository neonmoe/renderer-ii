use ash::vk;
use std::mem;
use ultraviolet::{Mat4, Vec2, Vec3, Vec4};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pipeline {
    Default,
    #[doc(hidden)]
    Count,
}

pub(crate) struct DescriptorSetLayoutParams {
    pub descriptor_type: vk::DescriptorType,
    pub descriptor_count: u32,
    pub stage_flags: vk::ShaderStageFlags,
    // This matches DescriptorSetLayoutBinding, except for the immutable samplers.
    // I don't see a practical use case for them, and this is simpler.
}

pub(crate) struct PipelineParameters {
    pub vertex_shader: &'static [u32],
    pub fragment_shader: &'static [u32],
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
    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
    descriptor_count: 1,
    stage_flags: vk::ShaderStageFlags::VERTEX,
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

pub(crate) static PIPELINE_PARAMETERS: [PipelineParameters; Pipeline::Count as usize] = [PipelineParameters {
    vertex_shader: shaders::include_spirv!("shaders/textured.vert"),
    fragment_shader: shaders::include_spirv!("shaders/textured.frag"),
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
    ],
    descriptor_sets: &[
        SHARED_DESCRIPTOR_SET_0,
        &[DescriptorSetLayoutParams {
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::FRAGMENT,
        }],
    ],
}];
