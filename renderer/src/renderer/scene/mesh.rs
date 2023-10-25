use alloc::rc::Rc;

use arrayvec::ArrayVec;
use ash::vk;

use crate::vulkan_raii::Buffer;

const VERTEX_BUFFERS: usize = 6;

// TODO: Differentiate meshes by vertex attribute binding layout?

// Previously they were differentiated by pipelines, but then they were
// generalized and pipelines were moved to the material. But really, the
// pipeline choice is a combination of both: vertex bindings change for meshes,
// uniforms change for both. So maybe meshes should have some identifier that
// can be compared against a PipelineIndex to see if it's compatible, when
// trying to render with a specific material?

#[derive(PartialEq, Eq, Hash)]
pub struct Mesh {
    pub(crate) vertex_buffer: Rc<Buffer>,
    pub(crate) vertex_buffer_offsets: ArrayVec<vk::DeviceSize, VERTEX_BUFFERS>,
    pub(crate) vertex_offset: i32,
    pub(crate) index_buffer: Rc<Buffer>,
    pub(crate) index_buffer_offset: vk::DeviceSize,
    pub(crate) first_index: u32,
    pub(crate) index_count: u32,
    pub(crate) index_type: vk::IndexType,
}

pub trait IndexType {
    const SIZE: usize;
    /// Returns the appropriate [`vk::IndexType`] for the implementing type.
    fn vk_index_type() -> vk::IndexType;
    /// Converts the implementing type to a u32.
    fn to_u32(self) -> u32;
}

impl IndexType for u16 {
    const SIZE: usize = core::mem::size_of::<Self>();
    fn vk_index_type() -> vk::IndexType {
        vk::IndexType::UINT16
    }
    fn to_u32(self) -> u32 {
        self as u32
    }
}

impl IndexType for u32 {
    const SIZE: usize = core::mem::size_of::<Self>();
    fn vk_index_type() -> vk::IndexType {
        vk::IndexType::UINT32
    }
    fn to_u32(self) -> u32 {
        self
    }
}
