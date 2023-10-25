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
    pub(crate) vertex_buffers: ArrayVec<Rc<Buffer>, VERTEX_BUFFERS>,
    pub(crate) vertices_offsets: ArrayVec<vk::DeviceSize, VERTEX_BUFFERS>,
    pub(crate) index_buffer: Rc<Buffer>,
    pub(crate) index_buffer_offset: vk::DeviceSize,
    pub(crate) index_count: u32,
    pub(crate) index_type: vk::IndexType,
}

impl Mesh {
    /// Creates a new mesh. Ensure that the vertices match the
    /// pipeline.
    pub fn new<I: IndexType>(
        vertex_buffers: ArrayVec<Rc<Buffer>, VERTEX_BUFFERS>,
        vertices_offsets: ArrayVec<vk::DeviceSize, VERTEX_BUFFERS>,
        index_buffer: Rc<Buffer>,
        index_buffer_offset: vk::DeviceSize,
        index_buffer_size: vk::DeviceSize,
    ) -> Mesh {
        profiling::scope!("new_mesh");
        Mesh {
            vertex_buffers,
            vertices_offsets,
            index_buffer,
            index_buffer_offset,
            index_count: (index_buffer_size / core::mem::size_of::<I>() as u64) as u32,
            index_type: I::vk_index_type(),
        }
    }
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
