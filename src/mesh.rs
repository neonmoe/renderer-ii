use crate::arena::{Arena, BufferAllocation};
use crate::{Error, FrameIndex, Gpu, Pipeline};
use ash::version::DeviceV1_0;
use ash::vk;
use glam::Vec4;
use std::{mem, ptr};

#[derive(PartialEq, Eq, Hash)]
pub struct Mesh {
    /// Contains the vertices and indices.
    pub(crate) mesh_buffer: BufferAllocation,

    pub pipeline: Pipeline,
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
    // TODO: Meshes that refer to existing buffers, instead of owning them themselves
    pub fn new<I: IndexType>(
        gpu: &Gpu,
        arena: &Arena,
        temp_arenas: &[Arena],
        frame_index: FrameIndex,
        vertices: &[&[u8]],
        indices: &[u8],
        pipeline: Pipeline,
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
        let mut data: Vec<Vec4> = vec![Vec4::ZERO; total_size];
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

        let temp_arena = frame_index.get_arena(temp_arenas);
        let staging_buffer = {
            profiling::scope!("allocate staging buffer");
            let buffer_create_info = vk::BufferCreateInfo::builder()
                .size(total_size as vk::DeviceSize)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            temp_arena.create_buffer(*buffer_create_info)?
        };

        {
            profiling::scope!("write staging buffer");
            if let Err(err) = unsafe { staging_buffer.write(temp_arena, data.as_ptr() as *const u8, 0, vk::WHOLE_SIZE) } {
                staging_buffer.clean_up(temp_arena);
                return Err(err);
            }
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
            match arena.create_buffer(*buffer_create_info) {
                Ok(buffer_allocation) => buffer_allocation,
                Err(err) => {
                    staging_buffer.clean_up(temp_arena);
                    return Err(err);
                }
            }
        };

        let upload_result = gpu.run_command_buffer(frame_index, vk::PipelineStageFlags::VERTEX_INPUT, |command_buffer| {
            profiling::scope!("vkCmdCopyBuffer");
            let src = staging_buffer.buffer;
            let dst = mesh_buffer.buffer;
            let copy_regions = [vk::BufferCopy::builder().size(total_size as vk::DeviceSize).build()];
            unsafe { gpu.device.cmd_copy_buffer(command_buffer, src, dst, &copy_regions) };
        });

        if let Err(err) = upload_result {
            staging_buffer.clean_up(temp_arena);
            mesh_buffer.clean_up(arena);
            return Err(err);
        }

        gpu.add_temp_buffer(frame_index, staging_buffer.buffer);

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
