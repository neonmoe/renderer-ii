use crate::arena::{BufferAllocation, VulkanArena};
use crate::{Error, Pipeline, Uploader};
use ash::vk;
use glam::Vec4;
use std::fmt::Arguments;
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
    // TODO(high): Meshes that refer to existing buffers, instead of owning them themselves
    pub fn new<I: IndexType>(
        uploader: &mut Uploader,
        arena: &mut VulkanArena,
        vertices: &[&[u8]],
        indices: &[u8],
        pipeline: Pipeline,
        name: Arguments,
    ) -> Result<Mesh, Error> {
        profiling::scope!("new_mesh");

        let &mut Uploader {
            graphics_queue_family,
            transfer_queue_family,
            ..
        } = uploader;

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

        let staging_buffer = {
            profiling::scope!("allocate staging buffer");
            let buffer_create_info = vk::BufferCreateInfo::builder()
                .size(total_size as vk::DeviceSize)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            uploader
                .staging_arena
                .create_buffer(*buffer_create_info, format_args!("staging buffer for {}", name))?
        };

        {
            profiling::scope!("write staging buffer");
            unsafe { staging_buffer.write(data.as_ptr() as *const u8, 0, vk::WHOLE_SIZE) }?;
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
            arena.create_buffer(*buffer_create_info, name)?
        };

        uploader.start_upload(
            vk::PipelineStageFlags::VERTEX_INPUT,
            staging_buffer.buffer,
            |device, staging_buffer, command_buffer| {
                profiling::scope!("vkCmdCopyBuffer");
                let src = staging_buffer.inner;
                let dst = mesh_buffer.buffer.inner;
                let copy_regions = [vk::BufferCopy::builder().size(total_size as vk::DeviceSize).build()];
                unsafe { device.cmd_copy_buffer(command_buffer, src, dst, &copy_regions) };
            },
            |device, command_buffer| {
                profiling::scope!("vkCmdPipelineBarrier");
                let barrier_from_transfer_dst_to_draw = vk::BufferMemoryBarrier::builder()
                    .buffer(mesh_buffer.buffer.inner)
                    .offset(0)
                    .size(vk::WHOLE_SIZE)
                    .src_queue_family_index(transfer_queue_family)
                    .dst_queue_family_index(graphics_queue_family)
                    .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                    .dst_access_mask(vk::AccessFlags::VERTEX_ATTRIBUTE_READ)
                    .build();
                unsafe {
                    device.cmd_pipeline_barrier(
                        command_buffer,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::PipelineStageFlags::VERTEX_INPUT,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[barrier_from_transfer_dst_to_draw],
                        &[],
                    );
                }
            },
        )?;

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
        self.mesh_buffer.buffer.inner
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
