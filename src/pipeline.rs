use ash::vk;
use std::mem;
use ultraviolet::Vec3;

#[derive(Clone, Copy)]
pub enum Pipeline {
    PlainVertexColor,
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

pub(crate) static PIPELINE_PARAMETERS: [PipelineParameters; Pipeline::Count as usize] =
    [PipelineParameters {
        vertex_shader: shaders::include_spirv!("shaders/plain_color.vert"),
        fragment_shader: shaders::include_spirv!("shaders/plain_color.frag"),
        bindings: &[vk::VertexInputBindingDescription {
            binding: 0,
            stride: mem::size_of::<[Vec3; 2]>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }],
        attributes: &[
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: mem::size_of::<[Vec3; 1]>() as u32,
            },
        ],
        descriptor_sets: &[SHARED_DESCRIPTOR_SET_0],
    }];
