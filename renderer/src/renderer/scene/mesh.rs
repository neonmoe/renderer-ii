use crate::vulkan_raii::Buffer;
use alloc::rc::Rc;
use arrayvec::ArrayVec;
use ash::vk;

const VERTEX_BUFFERS: usize = 6;

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
    fn vk_index_type() -> vk::IndexType;
}

impl IndexType for u16 {
    fn vk_index_type() -> vk::IndexType {
        vk::IndexType::UINT16
    }
}

impl IndexType for u32 {
    fn vk_index_type() -> vk::IndexType {
        vk::IndexType::UINT32
    }
}
