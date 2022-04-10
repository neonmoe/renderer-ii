use crate::vulkan_raii::Buffer;
use crate::PipelineIndex;
use ash::vk;
use std::rc::Rc;

#[derive(PartialEq, Eq, Hash)]
pub struct Mesh {
    pub pipeline: PipelineIndex,
    pub(crate) vertex_buffers: Vec<Rc<Buffer>>,
    pub(crate) vertices_offsets: Vec<vk::DeviceSize>,
    pub(crate) index_buffer: Rc<Buffer>,
    pub(crate) index_buffer_offset: vk::DeviceSize,
    pub(crate) index_count: u32,
    pub(crate) index_type: vk::IndexType,
}

impl Mesh {
    /// Creates a new mesh. Ensure that the vertices match the
    /// pipeline.
    pub fn new<I: IndexType>(
        pipeline: PipelineIndex,
        vertex_buffers: Vec<Rc<Buffer>>,
        vertices_offsets: Vec<vk::DeviceSize>,
        index_buffer: Rc<Buffer>,
        index_buffer_offset: vk::DeviceSize,
        index_buffer_size: vk::DeviceSize,
    ) -> Mesh {
        profiling::scope!("new_mesh");
        Mesh {
            pipeline,
            vertex_buffers,
            vertices_offsets,
            index_buffer,
            index_buffer_offset,
            index_count: (index_buffer_size / std::mem::size_of::<I>() as u64) as u32,
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
