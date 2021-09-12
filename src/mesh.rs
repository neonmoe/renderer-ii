use crate::buffer::Buffer;
use crate::{Error, FrameIndex, Gpu, Pipeline};
use ash::vk;
use std::{mem, ptr};
use ultraviolet::Vec4;

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
    /// pipeline.
    ///
    /// The `vertices` slice should contain a slice for each
    /// attribute, containing the values for that attribute tightly
    /// packed.
    // TODO: Meshes that refer to existing buffers, instead of owning them themselves
    #[profiling::function]
    pub fn new<'a, I: IndexType>(
        gpu: &'a Gpu<'_>,
        frame_index: FrameIndex,
        vertices: &[&[u8]],
        indices: &[u8],
        pipeline: Pipeline,
    ) -> Result<Mesh<'a>, Error> {
        // The destination memory is aligned to 16 bytes (size of
        // Vec4), as well as the individual slices in it. Just to make
        // sure that alignment isn't causing problems.

        let round_size = |len: usize| if len % 16 == 0 { len } else { len + (16 - (len % 16)) };
        let indices_size = indices.len();
        let mut total_size = round_size(indices_size);
        for vertices in vertices {
            total_size += round_size(vertices.len());
        }
        let mut data: Vec<Vec4> = vec![Vec4::zero(); total_size];
        let mut buffer_dst_ptr = data.as_mut_ptr() as *mut u8;
        let indices_src_ptr = indices.as_ptr();
        unsafe {
            ptr::copy_nonoverlapping(indices_src_ptr, buffer_dst_ptr, indices_size);
        }
        buffer_dst_ptr = unsafe { buffer_dst_ptr.add(round_size(indices_size)) };

        let mut vertices_offsets = Vec::with_capacity(vertices.len());
        let mut vertices_offset = round_size(indices_size) as vk::DeviceSize;
        for vertices in vertices {
            let vertices_size = vertices.len();
            let vertices_src_ptr = vertices.as_ptr();
            unsafe {
                ptr::copy_nonoverlapping(vertices_src_ptr, buffer_dst_ptr, vertices_size);
            }
            vertices_offsets.push(vertices_offset);
            let offset = round_size(vertices_size);
            vertices_offset += offset as vk::DeviceSize;
            buffer_dst_ptr = unsafe { buffer_dst_ptr.add(offset) };
        }

        Ok(Mesh {
            mesh_buffer: Buffer::new(gpu, frame_index, &data)?,
            pipeline,
            index_count: (indices.len() / mem::size_of::<I>()) as u32,
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
