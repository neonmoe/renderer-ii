use alloc::rc::Rc;

use ash::vk;

use crate::renderer::pipeline_parameters::vertex_buffers::VertexLayout;
use crate::vertex_library::VertexLibrary;

#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct Mesh {
    pub(crate) library: Rc<VertexLibrary>,
    pub(crate) vertex_layout: VertexLayout,
    pub(crate) vertex_offset: i32,
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
