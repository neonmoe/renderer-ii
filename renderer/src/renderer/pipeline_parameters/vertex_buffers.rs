use core::mem;

use arrayvec::ArrayVec;
use ash::vk;
use enum_map::{Enum, EnumMap};
use glam::Vec3;
use half::f16;

use crate::renderer::pipeline_parameters::constants::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, enum_map::Enum)]
pub enum VertexLayout {
    StaticMesh,
    SkinnedMesh,
    FullscreenQuad,
    ImGui,
}

impl VertexLayout {
    /// Returns a list of the [`VertexBinding`]s that are required for a mesh
    /// using this vertex layout.
    pub fn required_inputs(self) -> ArrayVec<VertexBinding, { VertexBinding::LENGTH }> {
        match self {
            VertexLayout::StaticMesh => ArrayVec::from_iter([
                VertexBinding::Position,
                VertexBinding::Texcoord0,
                VertexBinding::NormalOrColor,
                VertexBinding::Tangent,
            ]),
            VertexLayout::SkinnedMesh => ArrayVec::from_iter([
                VertexBinding::Position,
                VertexBinding::Texcoord0,
                VertexBinding::NormalOrColor,
                VertexBinding::Tangent,
                VertexBinding::Joints0,
                VertexBinding::Weights0,
            ]),
            VertexLayout::FullscreenQuad => ArrayVec::new(),
            VertexLayout::ImGui => ArrayVec::from_iter([VertexBinding::Position, VertexBinding::Texcoord0, VertexBinding::NormalOrColor]),
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
    NormalOrColor,
    Tangent,
    Joints0,
    Weights0,
}

pub(crate) struct VertexSizes {
    pub in_vertex_size: usize,
    pub out_vertex_size: usize,
    pub out_vertex_alignment: usize,
}

impl VertexSizes {
    pub fn from_types<In, Out>() -> VertexSizes {
        VertexSizes {
            in_vertex_size: mem::size_of::<In>(),
            out_vertex_size: mem::size_of::<Out>(),
            out_vertex_alignment: mem::align_of::<Out>(),
        }
    }
}

/// Returns the size of a single vertex expected by [`write_vertex_buffer`], and
/// the size of a single vertex written out by it, in that order.
#[allow(clippy::match_same_arms)]
pub fn get_vertex_sizes(vertex_layout: VertexLayout, vertex_binding: VertexBinding) -> VertexSizes {
    use VertexLayout::{ImGui, SkinnedMesh, StaticMesh};
    match (vertex_layout, vertex_binding) {
        (StaticMesh | SkinnedMesh, VertexBinding::Position) => VertexSizes::from_types::<[f32; 3], [f16; 3]>(),
        (StaticMesh | SkinnedMesh, VertexBinding::Texcoord0) => VertexSizes::from_types::<[f32; 2], [f16; 2]>(),
        (StaticMesh | SkinnedMesh, VertexBinding::NormalOrColor) => VertexSizes::from_types::<[f32; 3], u32>(),
        (StaticMesh | SkinnedMesh, VertexBinding::Tangent) => VertexSizes::from_types::<[f32; 4], u32>(),
        (SkinnedMesh, VertexBinding::Joints0) => VertexSizes::from_types::<[u8; 4], [u8; 4]>(),
        (SkinnedMesh, VertexBinding::Weights0) => VertexSizes::from_types::<[f32; 4], [u8; 4]>(),
        (ImGui, VertexBinding::Position) => VertexSizes::from_types::<[u8; 20], [f32; 2]>(),
        (ImGui, VertexBinding::Texcoord0) => VertexSizes::from_types::<[u8; 20], [f32; 2]>(),
        (ImGui, VertexBinding::NormalOrColor) => VertexSizes::from_types::<[u8; 20], [u8; 4]>(),
        _ => unimplemented!("binding {vertex_binding:?} is not used in {vertex_layout:?}"),
    }
}

/// Writes the `src`, assumed to be arrays of `f32`s, with implied strides e.g.
/// positions being tightly packed `[f32; 3]`, to `dst` in the format that the
/// renderer expects.
#[profiling::function]
pub fn write_vertices(vertex_layout: VertexLayout, binding: VertexBinding, src: &[u8], dst: &mut [u8]) {
    // Relevant bits from the vulkan spec:

    // VK_FORMAT_A2B10G10R10_SNORM_PACK32 specifies a four-component, 32-bit
    // packed signed normalized format that has a 2-bit A component in bits
    // 30..31, a 10-bit B component in bits 20..29, a 10-bit G component in bits
    // 10..19, and a 10-bit R component in bits 0..9.

    #[inline]
    fn pack<const N: u32>(f: f32) -> u32 {
        let max = 2u32.pow(N - 1) - 1;
        let mask = 2u32.pow(N) - 1;
        (max as f32 * f.clamp(-1.0, 1.0)) as i32 as u32 & mask
    }

    use VertexLayout::{ImGui, SkinnedMesh, StaticMesh};
    match (vertex_layout, binding) {
        (StaticMesh | SkinnedMesh, VertexBinding::Position) => {
            let src = bytemuck::cast_slice::<u8, [f32; 3]>(src);
            let dst = bytemuck::cast_slice_mut::<u8, [f16; 3]>(dst);
            for (&src, dst) in src.iter().zip(dst) {
                dst[0] = f16::from_f32(src[0]);
                dst[1] = f16::from_f32(src[1]);
                dst[2] = f16::from_f32(src[2]);
            }
        }
        (StaticMesh | SkinnedMesh, VertexBinding::Texcoord0) => {
            let src = bytemuck::cast_slice::<u8, [f32; 2]>(src);
            let dst = bytemuck::cast_slice_mut::<u8, [f16; 2]>(dst);
            for (&src, dst) in src.iter().zip(dst) {
                dst[0] = f16::from_f32(src[0]);
                dst[1] = f16::from_f32(src[1]);
            }
        }
        (StaticMesh | SkinnedMesh, VertexBinding::NormalOrColor) => {
            let src = bytemuck::cast_slice::<u8, [f32; 3]>(src);
            let dst = bytemuck::cast_slice_mut::<u8, u32>(dst);
            for (&src, dst) in src.iter().zip(dst) {
                *dst = (pack::<10>(src[2]) << 20) | (pack::<10>(src[1]) << 10) | pack::<10>(src[0]);
            }
        }
        (StaticMesh | SkinnedMesh, VertexBinding::Tangent) => {
            let src = bytemuck::cast_slice::<u8, [f32; 4]>(src);
            let dst = bytemuck::cast_slice_mut::<u8, u32>(dst);
            for (&src, dst) in src.iter().zip(dst) {
                *dst = (pack::<2>(src[3]) << 30) | (pack::<10>(src[2]) << 20) | (pack::<10>(src[1]) << 10) | pack::<10>(src[0]);
            }
        }
        (SkinnedMesh, VertexBinding::Weights0) => {
            let src = bytemuck::cast_slice::<u8, f32>(src);
            for (&src, dst) in src.iter().zip(dst) {
                *dst = (src * 0xFF as f32) as u8;
            }
        }
        (SkinnedMesh, VertexBinding::Joints0) => dst.copy_from_slice(src),
        (ImGui, VertexBinding::Position) => {
            let src = bytemuck::cast_slice::<u8, [f32; 5]>(src);
            let dst = bytemuck::cast_slice_mut::<u8, [f32; 2]>(dst);
            for (&src, dst) in src.iter().zip(dst) {
                dst[0] = src[0];
                dst[1] = src[1];
            }
        }
        (ImGui, VertexBinding::Texcoord0) => {
            let src = bytemuck::cast_slice::<u8, [f32; 5]>(src);
            let dst = bytemuck::cast_slice_mut::<u8, [f32; 2]>(dst);
            for (&src, dst) in src.iter().zip(dst) {
                dst[0] = src[2];
                dst[1] = src[3];
            }
        }
        (ImGui, VertexBinding::NormalOrColor) => {
            let src = bytemuck::cast_slice::<u8, [u8; 20]>(src);
            let dst = bytemuck::cast_slice_mut::<u8, [u8; 4]>(dst);
            for (&src, dst) in src.iter().zip(dst) {
                dst[0] = src[16];
                dst[1] = src[17];
                dst[2] = src[18];
                dst[3] = src[19];
            }
        }
        _ => unimplemented!("binding {binding:?} is not used in {vertex_layout:?}"),
    }
}

pub type VertexLayoutMap<T> = EnumMap<VertexLayout, T>;
pub type VertexBindingMap<T> = EnumMap<VertexBinding, Option<T>>;

pub static VERTEX_BINDING_DESCRIPTIONS: VertexLayoutMap<&[vk::VertexInputBindingDescription]> =
    EnumMap::from_array([STATIC_MESH_VERTEX_BINDINGS, SKINNED_MESH_VERTEX_BINDINGS, &[], IMGUI_VERTEX_BINDINGS]);

pub static VERTEX_ATTRIBUTE_DESCRIPTIONS: VertexLayoutMap<&[vk::VertexInputAttributeDescription]> =
    EnumMap::from_array([STATIC_MESH_VERTEX_ATTRIBUTES, SKINNED_MESH_VERTEX_ATTRIBUTES, &[], IMGUI_VERTEX_ATTRIBUTES]);

static INSTANCED_TRANSFORM_BINDING: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::Transforms as u32,
    // The regular transform matrix is 4x3 + normal transform is 3x3 = 7x3.
    stride: (mem::size_of::<Vec3>() * 7) as u32,
    input_rate: vk::VertexInputRate::INSTANCE,
};
static POSITION_BINDING: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::Position as u32,
    stride: mem::size_of::<[f16; 3]>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};
static TEXCOORD0_BINDING: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::Texcoord0 as u32,
    stride: mem::size_of::<[f16; 2]>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};
static NORMAL_BINDING: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::NormalOrColor as u32,
    stride: mem::size_of::<u32>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};
static TANGENT_BINDING: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::Tangent as u32,
    stride: mem::size_of::<u32>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};
static JOINTS0_BINDING: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::Joints0 as u32,
    stride: mem::size_of::<[u8; 4]>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};
static WEIGHTS0_BINDING: vk::VertexInputBindingDescription = vk::VertexInputBindingDescription {
    binding: VertexBinding::Weights0 as u32,
    stride: mem::size_of::<[u8; 4]>() as u32,
    input_rate: vk::VertexInputRate::VERTEX,
};

static STATIC_MESH_VERTEX_BINDINGS: &[vk::VertexInputBindingDescription] =
    &[INSTANCED_TRANSFORM_BINDING, POSITION_BINDING, TEXCOORD0_BINDING, NORMAL_BINDING, TANGENT_BINDING];
static SKINNED_MESH_VERTEX_BINDINGS: &[vk::VertexInputBindingDescription] =
    &[INSTANCED_TRANSFORM_BINDING, POSITION_BINDING, TEXCOORD0_BINDING, NORMAL_BINDING, TANGENT_BINDING, JOINTS0_BINDING, WEIGHTS0_BINDING];
static IMGUI_VERTEX_BINDINGS: &[vk::VertexInputBindingDescription] = &[
    INSTANCED_TRANSFORM_BINDING,
    vk::VertexInputBindingDescription {
        binding: VertexBinding::Position as u32,
        stride: mem::size_of::<[f32; 2]>() as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    },
    vk::VertexInputBindingDescription {
        binding: VertexBinding::Texcoord0 as u32,
        stride: mem::size_of::<[f32; 2]>() as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    },
    vk::VertexInputBindingDescription {
        binding: VertexBinding::NormalOrColor as u32,
        stride: mem::size_of::<[u8; 4]>() as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    },
];

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
    format: vk::Format::R16G16B16_SFLOAT,
    offset: 0,
};
static TEXCOORD0_BINDING_ATTRIBUTE: vk::VertexInputAttributeDescription = vk::VertexInputAttributeDescription {
    binding: VertexBinding::Texcoord0 as u32,
    location: IN_TEXCOORD_0_LOCATION,
    format: vk::Format::R16G16_SFLOAT,
    offset: 0,
};
static NORMAL_BINDING_ATTRIBUTE: vk::VertexInputAttributeDescription = vk::VertexInputAttributeDescription {
    binding: VertexBinding::NormalOrColor as u32,
    location: IN_NORMAL_LOCATION,
    format: vk::Format::A2B10G10R10_SNORM_PACK32,
    offset: 0,
};
static TANGENT_BINDING_ATTRIBUTE: vk::VertexInputAttributeDescription = vk::VertexInputAttributeDescription {
    binding: VertexBinding::Tangent as u32,
    location: IN_TANGENT_LOCATION,
    format: vk::Format::A2B10G10R10_SNORM_PACK32,
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
    format: vk::Format::R8G8B8A8_UNORM,
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
static IMGUI_VERTEX_ATTRIBUTES: &[vk::VertexInputAttributeDescription] = &[
    INSTANCED_TRANSFORM_BINDING_ATTRIBUTES[0],
    INSTANCED_TRANSFORM_BINDING_ATTRIBUTES[1],
    INSTANCED_TRANSFORM_BINDING_ATTRIBUTES[2],
    INSTANCED_TRANSFORM_BINDING_ATTRIBUTES[3],
    vk::VertexInputAttributeDescription {
        binding: VertexBinding::Position as u32,
        location: IN_POSITION_LOCATION,
        format: vk::Format::R32G32_SFLOAT,
        offset: 0,
    },
    vk::VertexInputAttributeDescription {
        binding: VertexBinding::Texcoord0 as u32,
        location: IN_TEXCOORD_0_LOCATION,
        format: vk::Format::R32G32_SFLOAT,
        offset: 0,
    },
    vk::VertexInputAttributeDescription {
        binding: VertexBinding::NormalOrColor as u32,
        location: IN_COLOR_LOCATION,
        format: vk::Format::R8G8B8A8_UNORM,
        offset: 0,
    },
];
