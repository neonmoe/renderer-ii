use core::mem;

use arrayvec::ArrayVec;
use ash::vk;
use enum_map::EnumMap;
use glam::{Mat4, Vec2, Vec3, Vec4};

use crate::renderer::pipeline_parameters::constants::*;

#[derive(Clone, Copy, Debug, enum_map::Enum)]
pub enum VertexLayout {
    StaticMesh,
    SkinnedMesh,
    FullscreenQuad,
}

pub type VertexLayoutMap<T> = EnumMap<VertexLayout, T>;
pub type VertexBindingVec<T> = ArrayVec<T, { VertexBinding::Count as usize }>;

pub static VERTEX_BINDING_DESCRIPTIONS: VertexLayoutMap<&[vk::VertexInputBindingDescription]> =
    EnumMap::from_array([STATIC_MESH_VERTEX_BINDINGS, SKINNED_MESH_VERTEX_BINDINGS, &[]]);

pub static VERTEX_ATTRIBUTE_DESCRIPTIONS: VertexLayoutMap<&[vk::VertexInputAttributeDescription]> =
    EnumMap::from_array([STATIC_MESH_VERTEX_ATTRIBUTES, SKINNED_MESH_VERTEX_ATTRIBUTES, &[]]);

pub const VERTEX_BINDING_COUNT: usize = VertexBinding::Count as usize;

enum VertexBinding {
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

static STATIC_MESH_VERTEX_BINDINGS: &[vk::VertexInputBindingDescription] =
    &[INSTANCED_TRANSFORM_BINDING_0, POSITION_BINDING_1, TEXCOORD0_BINDING_2, NORMAL_BINDING_3, TANGENT_BINDING_4];
static SKINNED_MESH_VERTEX_BINDINGS: &[vk::VertexInputBindingDescription] = &[
    INSTANCED_TRANSFORM_BINDING_0,
    POSITION_BINDING_1,
    TEXCOORD0_BINDING_2,
    NORMAL_BINDING_3,
    TANGENT_BINDING_4,
    JOINTS0_BINDING_5,
    WEIGHTS0_BINDING_6,
];

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

static STATIC_MESH_VERTEX_ATTRIBUTES: &[vk::VertexInputAttributeDescription] = &[
    INSTANCED_TRANSFORM_BINDING_0_ATTRIBUTES[0],
    INSTANCED_TRANSFORM_BINDING_0_ATTRIBUTES[1],
    INSTANCED_TRANSFORM_BINDING_0_ATTRIBUTES[2],
    INSTANCED_TRANSFORM_BINDING_0_ATTRIBUTES[3],
    POSITION_BINDING_1_ATTRIBUTE,
    TEXCOORD0_BINDING_2_ATTRIBUTE,
    NORMAL_BINDING_3_ATTRIBUTE,
    TANGENT_BINDING_4_ATTRIBUTE,
];
static SKINNED_MESH_VERTEX_ATTRIBUTES: &[vk::VertexInputAttributeDescription] = &[
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
];
