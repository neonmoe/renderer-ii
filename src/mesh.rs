use crate::buffer::Buffer;
use crate::{Error, FrameIndex, Gpu, Pipeline};
use ash::vk;
use std::ptr;

#[derive(PartialEq, Eq, Hash)]
pub struct Mesh<'a> {
    /// Contains the vertices and indices.
    pub(crate) mesh_buffer: Buffer<'a>,

    pub pipeline: Pipeline,
    pub(crate) index_count: u32,
    pub(crate) indices_offset: vk::DeviceSize,
    pub(crate) index_type: vk::IndexType,
    pub(crate) vertices_offsets: Vec<vk::DeviceSize>,
}

impl Mesh<'_> {
    /// Creates a new mesh. Ensure that the vertices match the
    /// pipeline. If not `editable`, call [Gpu::wait_mesh_uploads]
    /// after your mesh creation code, before they're rendered.
    ///
    /// The `vertices` slice should contain a slice for each
    /// attribute, containing the values for that attribute tightly
    /// packed.
    #[profiling::function]
    pub fn new<'a, I: IndexType>(
        gpu: &'a Gpu<'_>,
        frame_index: FrameIndex,
        vertices: &[&[u8]],
        indices: &[u8],
        pipeline: Pipeline,
    ) -> Result<Mesh<'a>, Error> {
        let indices_size = indices.len();
        let total_vertices_size = vertices.iter().map(|vertices| vertices.len()).sum::<usize>();
        let mut data: Vec<u8> = vec![0; total_vertices_size + indices_size];
        let mut buffer_dst_ptr = data.as_mut_ptr();
        let indices_src_ptr = indices.as_ptr();
        unsafe {
            ptr::copy_nonoverlapping(indices_src_ptr, buffer_dst_ptr, indices_size);
        }
        buffer_dst_ptr = unsafe { buffer_dst_ptr.add(indices_size) };

        let mut vertices_offsets = Vec::with_capacity(vertices.len());
        let mut vertices_offset = indices_size as vk::DeviceSize;
        for vertices in vertices {
            let vertices_size = vertices.len();
            let vertices_src_ptr = vertices.as_ptr();
            unsafe {
                ptr::copy_nonoverlapping(vertices_src_ptr, buffer_dst_ptr, vertices_size);
            }
            vertices_offsets.push(vertices_offset);
            vertices_offset += vertices_size as vk::DeviceSize;
            buffer_dst_ptr = unsafe { buffer_dst_ptr.add(vertices_size) };
        }

        Ok(Mesh {
            mesh_buffer: Buffer::new(gpu, frame_index, &data)?,
            pipeline,
            index_count: indices.len() as u32,
            indices_offset: 0,
            index_type: I::vk_index_type(),
            vertices_offsets,
        })
    }

    pub(crate) fn buffer(&self) -> vk::Buffer {
        self.mesh_buffer.buffer
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
