use crate::arena::VulkanArena;
use crate::vulkan_raii::Buffer;
use crate::{Error, PipelineIndex, Uploader};
use ash::vk;
use glam::Vec4;
use std::fmt::Arguments;
use std::{mem, ptr};

#[derive(PartialEq, Eq, Hash)]
pub struct Mesh {
    /// Contains the vertices and indices.
    pub(crate) mesh_buffer: Buffer,

    pub pipeline: PipelineIndex,
    pub(crate) index_count: u32,
    pub(crate) indices_offset: vk::DeviceSize,
    pub(crate) index_type: vk::IndexType,
    pub(crate) vertices_offsets: Vec<vk::DeviceSize>,
}

impl Mesh {
    /// Creates a new mesh. Ensure that the vertices match the
    /// pipeline.
    ///
    /// The `vertices` slice should contain a slice for each
    /// attribute, containing the values for that attribute tightly
    /// packed.
    // TODO(high): Meshes that refer to existing buffers, instead of owning them themselves
    pub fn new<I: IndexType>(
        uploader: &mut Uploader,
        arena: &mut VulkanArena,
        vertices: &[&[u8]],
        indices: &[u8],
        pipeline: PipelineIndex,
        name: Arguments,
    ) -> Result<Mesh, Error> {
        profiling::scope!("new_mesh");

        // The destination memory is aligned to 16 bytes (size of
        // Vec4), as well as the individual slices in it. Just to make
        // sure that alignment isn't causing problems.
        let round_size = |len: usize| if len % 16 == 0 { len } else { len + (16 - (len % 16)) };
        let indices_size = indices.len();
        let mut total_size = round_size(indices_size);
        for vertices in vertices {
            total_size += round_size(vertices.len());
        }
        let mut data: Vec<Vec4> = vec![Vec4::ZERO; total_size / mem::size_of::<Vec4>()];
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

        let mesh_buffer = {
            profiling::scope!("allocate gpu buffer");
            let buffer_create_info = vk::BufferCreateInfo::builder()
                .size(total_size as vk::DeviceSize)
                .usage(
                    // Just prepare for everything for simplicity.
                    vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::INDEX_BUFFER
                        | vk::BufferUsageFlags::UNIFORM_BUFFER,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            arena.create_buffer(*buffer_create_info, bytemuck::cast_slice(&data), Some(uploader), name)?
        };

        Ok(Mesh {
            mesh_buffer,
            pipeline,
            index_count: (indices.len() / mem::size_of::<I>()) as u32,
            indices_offset: 0,
            index_type: I::vk_index_type(),
            vertices_offsets,
        })
    }

    pub(crate) fn buffer(&self) -> vk::Buffer {
        self.mesh_buffer.inner
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
