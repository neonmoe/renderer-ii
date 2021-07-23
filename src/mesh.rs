use crate::{Error, Gpu, Pipeline};
use ash::vk;
use std::mem;

fn copy_raw<T>(data: &[T], dst_ptr: *mut u8) {
    let length = data.len() * mem::size_of::<T>();
    let src_ptr = data.as_ptr() as *const u8;
    unsafe { std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, length) };
}

pub struct Mesh<'a> {
    /// Held by [Mesh] to be able to destroy resources on drop.
    pub gpu: &'a Gpu<'a>,

    pub pipeline: Pipeline,
    pub(crate) buffer: vk::Buffer,
    pub(crate) vertices: u32,
    allocation: vk_mem::Allocation,
    alloc_info: vk_mem::AllocationInfo,
}

impl Drop for Mesh<'_> {
    fn drop(&mut self) {
        let _ = self
            .gpu
            .allocator
            .destroy_buffer(self.buffer, &self.allocation);
    }
}

impl Mesh<'_> {
    /// Creates a new mesh. Ensure that the vertices match the
    /// pipeline.
    pub fn new<'a, V>(
        gpu: &'a Gpu<'_>,
        vertices: &[V],
        pipeline: Pipeline,
    ) -> Result<Mesh<'a>, Error> {
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size((vertices.len() * mem::size_of::<V>()) as u64)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::CpuToGpu,
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            required_flags: vk::MemoryPropertyFlags::HOST_COHERENT,
            ..Default::default()
        };
        let (buffer, allocation, alloc_info) = gpu
            .allocator
            .create_buffer(&buffer_create_info, &allocation_create_info)
            .map_err(Error::VmaBufferAllocation)?;
        let buffer_ptr = alloc_info.get_mapped_data();
        copy_raw(&vertices, buffer_ptr);
        Ok(Mesh {
            gpu,
            pipeline,
            buffer,
            vertices: vertices.len() as u32,
            allocation,
            alloc_info,
        })
    }

    /// Updates the vertices of the mesh. The amount of vertices must
    /// be the same as in [Mesh::new].
    pub fn update_vertices<V>(&mut self, new_vertices: &[V]) -> Result<(), Error> {
        if self.vertices != new_vertices.len() as u32 {
            Err(Error::MeshVertexCountMismatch)
        } else {
            copy_raw(&new_vertices, self.alloc_info.get_mapped_data());
            Ok(())
        }
    }
}
