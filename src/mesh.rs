use crate::buffer::Buffer;
use crate::{Error, FrameIndex, Gpu, Pipeline};
use ash::vk;
use std::{mem, ptr};

#[derive(PartialEq, Eq, Hash)]
pub struct Mesh<'a> {
    /// Contains the vertices and indices.
    pub(crate) mesh_buffer: Buffer<'a>,

    pub pipeline: Pipeline,
    pub(crate) index_count: u32,
    pub(crate) indices_offset: vk::DeviceSize,
    pub(crate) index_type: vk::IndexType,
}

impl Mesh<'_> {
    /// Creates a new mesh. Ensure that the vertices match the
    /// pipeline. If not `editable`, call [Gpu::wait_mesh_uploads]
    /// after your mesh creation code, before they're rendered.
    #[profiling::function]
    pub fn new<'a, V, I: IndexType>(
        gpu: &'a Gpu<'_>,
        frame_index: FrameIndex,
        vertices: &[V],
        indices: &[I],
        pipeline: Pipeline,
        editable: bool,
    ) -> Result<Mesh<'a>, Error> {
        let vertices_size = vertices.len() * mem::size_of::<V>();
        let indices_size = indices.len() * mem::size_of::<I>();
        let mut data: Vec<u8> = vec![0; vertices_size + indices_size];
        let vertices_dst_ptr = data.as_mut_ptr();
        let vertices_src_ptr = vertices.as_ptr() as *const u8;
        let indices_dst_ptr = unsafe { vertices_dst_ptr.add(vertices_size) };
        let indices_src_ptr = indices.as_ptr() as *const u8;
        unsafe {
            // Fill out the buffer with the vertices and
            // indices. Feels unnecessarily unsafe, but apparently
            // it's better to use a single buffer, and the vertices
            // and indices aren't of the same type, so here we are.
            ptr::copy_nonoverlapping(vertices_src_ptr, vertices_dst_ptr, vertices_size);
            ptr::copy_nonoverlapping(indices_src_ptr, indices_dst_ptr, indices_size);
        }

        Ok(Mesh {
            mesh_buffer: Buffer::new(gpu, frame_index, &data, editable)?,
            pipeline,
            index_count: indices.len() as u32,
            indices_offset: vertices_size as vk::DeviceSize,
            index_type: I::vk_index_type(),
        })
    }

    /// Updates the vertices of the mesh. The amount of vertices must
    /// be the same as in [Mesh::new].
    #[profiling::function]
    pub fn update_vertices<V>(&mut self, gpu: &Gpu, new_vertices: &[V]) -> Result<(), Error> {
        self.mesh_buffer.update_data(gpu, new_vertices)
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
