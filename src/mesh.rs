use crate::{Error, Gpu, Material};
use ash::vk;
use std::mem;
use ultraviolet::Vec3;

fn copy_raw<T>(data: &[T], pointer: *mut u8) {
    let length = data.len() * mem::size_of::<T>();
    let data_ptr = unsafe { mem::transmute::<*const T, *const u8>(data.as_ptr()) };
    unsafe { std::ptr::copy_nonoverlapping(data_ptr, pointer, length) };
}

pub struct Mesh<'a> {
    /// Held by [Mesh] to be able to destroy resources on drop.
    pub gpu: &'a Gpu<'a>,

    pub material: Material,
    pub(crate) buffer: vk::Buffer,
    pub(crate) allocation: vk_mem::Allocation,
    pub(crate) vertices: u32,
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
    pub fn new<'a, V>(gpu: &'a Gpu<'_>, vertices: &[V]) -> Result<Mesh<'a>, Error> {
        let buffer_using_families = [gpu.graphics_family_index];
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size((vertices.len() * mem::size_of::<[Vec3; 2]>()) as u64)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&buffer_using_families);
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
            material: Material::PlainVertexColor,
            buffer,
            allocation,
            vertices: vertices.len() as u32,
        })
    }
}
