use ash::vk;
use std::mem;
use ultraviolet::Vec3;

#[derive(Clone, Copy)]
pub enum Material {
    PlainVertexColor,
    #[doc(hidden)]
    Length,
}

pub(crate) struct PipelineParameters {
    pub vertex_shader: &'static [u32],
    pub fragment_shader: &'static [u32],
    pub bindings: &'static [vk::VertexInputBindingDescription],
    pub attributes: &'static [vk::VertexInputAttributeDescription],
}

pub(crate) static PIPELINE_PARAMETERS: [PipelineParameters; Material::Length as usize] =
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
    }];
