use core::mem;

use arrayvec::ArrayVec;
use ash::vk;
use enum_map::{Enum, EnumMap};
use glam::{Vec2, Vec3, Vec4};

use crate::renderer::pipeline_parameters::constants::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, enum_map::Enum)]
pub enum VertexLayout {
    StaticMesh,
    SkinnedMesh,
    FullscreenQuad,
}

impl VertexLayout {
    /// Returns a list of the [`VertexBinding`]s that are required for a mesh
    /// using this vertex layout.
    pub fn required_inputs(self) -> ArrayVec<VertexBinding, { VertexBinding::LENGTH }> {
        match self {
            VertexLayout::StaticMesh => {
                ArrayVec::from_iter([VertexBinding::Position, VertexBinding::Texcoord0, VertexBinding::Normal, VertexBinding::Tangent])
            }
            VertexLayout::SkinnedMesh => ArrayVec::from_iter([
                VertexBinding::Position,
                VertexBinding::Texcoord0,
                VertexBinding::Normal,
                VertexBinding::Tangent,
                VertexBinding::Joints0,
                VertexBinding::Weights0,
            ]),
            VertexLayout::FullscreenQuad => ArrayVec::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, enum_map::Enum)]
pub enum VertexBinding {
    /// Plural, because this binding has both the regular transform *and* the
    /// inverse transposes of their 3x3 part.
    Transforms,
    Position,
    Texcoord0,
    Normal,
    Tangent,
    Joints0,
    Weights0,
}

impl VertexBinding {
    pub(crate) fn description(self, vertex_layout: VertexLayout) -> Option<vk::VertexInputBindingDescription> {
        VERTEX_BINDING_DESCRIPTIONS[vertex_layout].iter().find(|desc| desc.binding == self as u32).copied()
    }
}

pub type VertexLayoutMap<T> = EnumMap<VertexLayout, T>;
pub type VertexBindingMap<T> = EnumMap<VertexBinding, Option<T>>;

pub static VERTEX_BINDING_DESCRIPTIONS: VertexLayoutMap<&[vk::VertexInputBindingDescription]> =
    EnumMap::from_array([STATIC_MESH_VERTEX_BINDINGS, SKINNED_MESH_VERTEX_BINDINGS, &[]]);

pub static VERTEX_ATTRIBUTE_DESCRIPTIONS: VertexLayoutMap<&[vk::VertexInputAttributeDescription]> =
    EnumMap::from_array([STATIC_MESH_VERTEX_ATTRIBUTES, SKINNED_MESH_VERTEX_ATTRIBUTES, &[]]);

static INSTANCED_TRANSFORM_BINDING: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::Transforms as u32,
    // The regular transform matrix is 4x3 + normal transform is 3x3 = 7x3.
    stride: (mem::size_of::<Vec3>() * 7) as u32,
    input_rate: vk::VertexInputRate::INSTANCE,
};
static POSITION_BINDING: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::Position as u32,
    stride: mem::size_of::<Vec3>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};
static TEXCOORD0_BINDING: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::Texcoord0 as u32,
    stride: mem::size_of::<Vec2>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};
static NORMAL_BINDING: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::Normal as u32,
    stride: mem::size_of::<Vec3>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};
static TANGENT_BINDING: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::Tangent as u32,
    stride: mem::size_of::<Vec4>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};
static JOINTS0_BINDING: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::Joints0 as u32,
    stride: mem::size_of::<[u8; 4]>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};
static WEIGHTS0_BINDING: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::Weights0 as u32,
    stride: mem::size_of::<[f32; 4]>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};

static STATIC_MESH_VERTEX_BINDINGS: &[vk::VertexInputBindingDescription] =
    &[INSTANCED_TRANSFORM_BINDING, POSITION_BINDING, TEXCOORD0_BINDING, NORMAL_BINDING, TANGENT_BINDING];
static SKINNED_MESH_VERTEX_BINDINGS: &[vk::VertexInputBindingDescription] =
    &[INSTANCED_TRANSFORM_BINDING, POSITION_BINDING, TEXCOORD0_BINDING, NORMAL_BINDING, TANGENT_BINDING, JOINTS0_BINDING, WEIGHTS0_BINDING];

static INSTANCED_TRANSFORM_BINDING_ATTRIBUTES: [vk::VertexInputAttributeDescription; 7] = [
    vk::VertexInputAttributeDescription {
        binding: VertexBinding::Transforms as u32,
        location: IN_TRANSFORMS_LOCATION,
        format: vk::Format::R32G32B32_SFLOAT,
        offset: 0,
    },
    vk::VertexInputAttributeDescription {
        binding: VertexBinding::Transforms as u32,
        location: IN_TRANSFORMS_LOCATION + 1,
        format: vk::Format::R32G32B32_SFLOAT,
        offset: mem::size_of::<Vec3>() as u32,
    },
    vk::VertexInputAttributeDescription {
        binding: VertexBinding::Transforms as u32,
        location: IN_TRANSFORMS_LOCATION + 2,
        format: vk::Format::R32G32B32_SFLOAT,
        offset: 2 * mem::size_of::<Vec3>() as u32,
    },
    vk::VertexInputAttributeDescription {
        binding: VertexBinding::Transforms as u32,
        location: IN_TRANSFORMS_LOCATION + 3,
        format: vk::Format::R32G32B32_SFLOAT,
        offset: 3 * mem::size_of::<Vec3>() as u32,
    },
    vk::VertexInputAttributeDescription {
        binding: VertexBinding::Transforms as u32,
        location: IN_TRANSFORMS_LOCATION + 4,
        format: vk::Format::R32G32B32_SFLOAT,
        offset: 4 * mem::size_of::<Vec3>() as u32,
    },
    vk::VertexInputAttributeDescription {
        binding: VertexBinding::Transforms as u32,
        location: IN_TRANSFORMS_LOCATION + 5,
        format: vk::Format::R32G32B32_SFLOAT,
        offset: 5 * mem::size_of::<Vec3>() as u32,
    },
    vk::VertexInputAttributeDescription {
        binding: VertexBinding::Transforms as u32,
        location: IN_TRANSFORMS_LOCATION + 6,
        format: vk::Format::R32G32B32_SFLOAT,
        offset: 6 * mem::size_of::<Vec3>() as u32,
    },
];
static POSITION_BINDING_ATTRIBUTE: vk::VertexInputAttributeDescription = vk::VertexInputAttributeDescription {
    binding: VertexBinding::Position as u32,
    location: IN_POSITION_LOCATION,
    format: vk::Format::R32G32B32_SFLOAT,
    offset: 0,
};
static TEXCOORD0_BINDING_ATTRIBUTE: vk::VertexInputAttributeDescription = vk::VertexInputAttributeDescription {
    binding: VertexBinding::Texcoord0 as u32,
    location: IN_TEXCOORD_0_LOCATION,
    format: vk::Format::R32G32_SFLOAT,
    offset: 0,
};
static NORMAL_BINDING_ATTRIBUTE: vk::VertexInputAttributeDescription = vk::VertexInputAttributeDescription {
    binding: VertexBinding::Normal as u32,
    location: IN_NORMAL_LOCATION,
    format: vk::Format::R32G32B32_SFLOAT,
    offset: 0,
};
static TANGENT_BINDING_ATTRIBUTE: vk::VertexInputAttributeDescription = vk::VertexInputAttributeDescription {
    binding: VertexBinding::Tangent as u32,
    location: IN_TANGENT_LOCATION,
    format: vk::Format::R32G32B32A32_SFLOAT,
    offset: 0,
};
static JOINTS0_BINDING_ATTRIBUTE: vk::VertexInputAttributeDescription = vk::VertexInputAttributeDescription {
    binding: VertexBinding::Joints0 as u32,
    location: IN_JOINTS_0_LOCATION,
    format: vk::Format::R8G8B8A8_UINT,
    offset: 0,
};
static WEIGHTS0_BINDING_ATTRIBUTE: vk::VertexInputAttributeDescription = vk::VertexInputAttributeDescription {
    binding: VertexBinding::Weights0 as u32,
    location: IN_WEIGHTS_0_LOCATION,
    format: vk::Format::R32G32B32A32_SFLOAT,
    offset: 0,
};

static STATIC_MESH_VERTEX_ATTRIBUTES: &[vk::VertexInputAttributeDescription] = &[
    INSTANCED_TRANSFORM_BINDING_ATTRIBUTES[0],
    INSTANCED_TRANSFORM_BINDING_ATTRIBUTES[1],
    INSTANCED_TRANSFORM_BINDING_ATTRIBUTES[2],
    INSTANCED_TRANSFORM_BINDING_ATTRIBUTES[3],
    INSTANCED_TRANSFORM_BINDING_ATTRIBUTES[4],
    INSTANCED_TRANSFORM_BINDING_ATTRIBUTES[5],
    INSTANCED_TRANSFORM_BINDING_ATTRIBUTES[6],
    POSITION_BINDING_ATTRIBUTE,
    TEXCOORD0_BINDING_ATTRIBUTE,
    NORMAL_BINDING_ATTRIBUTE,
    TANGENT_BINDING_ATTRIBUTE,
];
static SKINNED_MESH_VERTEX_ATTRIBUTES: &[vk::VertexInputAttributeDescription] = &[
    INSTANCED_TRANSFORM_BINDING_ATTRIBUTES[0],
    INSTANCED_TRANSFORM_BINDING_ATTRIBUTES[1],
    INSTANCED_TRANSFORM_BINDING_ATTRIBUTES[2],
    INSTANCED_TRANSFORM_BINDING_ATTRIBUTES[3],
    INSTANCED_TRANSFORM_BINDING_ATTRIBUTES[4],
    INSTANCED_TRANSFORM_BINDING_ATTRIBUTES[5],
    INSTANCED_TRANSFORM_BINDING_ATTRIBUTES[6],
    POSITION_BINDING_ATTRIBUTE,
    TEXCOORD0_BINDING_ATTRIBUTE,
    NORMAL_BINDING_ATTRIBUTE,
    TANGENT_BINDING_ATTRIBUTE,
    JOINTS0_BINDING_ATTRIBUTE,
    WEIGHTS0_BINDING_ATTRIBUTE,
];
