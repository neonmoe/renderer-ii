use crate::{Error, Gpu, Pipeline};
use ash::version::DeviceV1_0;
use ash::vk;
use std::mem;

/// Contains the fence for the upload queue submit, and everything
/// that needs to be cleaned up after it's done.
pub(crate) struct MeshUpload {
    pub finished_upload: vk::Fence,
    pub staging_buffer: vk::Buffer,
    pub staging_allocation: vk_mem::Allocation,
    pub upload_cmdbuf: vk::CommandBuffer,
}

pub struct Mesh<'a> {
    /// Held by [Mesh] to be able to destroy resources on drop.
    pub gpu: &'a Gpu<'a>,

    pub pipeline: Pipeline,
    pub(crate) buffer: vk::Buffer,
    pub(crate) index_count: u32,
    pub(crate) indices_offset: vk::DeviceSize,
    pub(crate) index_type: vk::IndexType,
    allocation: vk_mem::Allocation,
    alloc_info: vk_mem::AllocationInfo,
    editable: bool,
    vertex_count: usize,
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
    /// pipeline. If not `editable`, call [Gpu::wait_mesh_uploads]
    /// after your mesh creation code, before they're rendered.
    pub fn new<'a, V, I: IndexType>(
        gpu: &'a Gpu<'_>,
        vertices: &[V],
        indices: &[I],
        pipeline: Pipeline,
        editable: bool,
    ) -> Result<Mesh<'a>, Error> {
        // TODO: Create separate staging/gpu-only/shared allocation pools

        let buffer_size = mesh_data_size(vertices, indices);
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(buffer_size)
            .usage(if editable {
                vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER
            } else {
                vk::BufferUsageFlags::TRANSFER_SRC
            })
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
        copy_mesh_data(&vertices, Some(&indices), buffer_ptr);
        let mut mesh = Mesh {
            gpu,
            pipeline,
            buffer,
            index_count: indices.len() as u32,
            indices_offset: mesh_indices_offset(&vertices),
            index_type: I::vk_index_type(),
            allocation,
            alloc_info,
            editable,
            vertex_count: vertices.len(),
        };

        if !editable {
            let buffer_create_info = vk::BufferCreateInfo::builder()
                .size(buffer_size)
                .usage(
                    vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::INDEX_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST,
                )
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let allocation_create_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::GpuOnly,
                ..Default::default()
            };
            let (gpu_buffer, gpu_allocation, gpu_alloc_info) = gpu
                .allocator
                .create_buffer(&buffer_create_info, &allocation_create_info)
                .map_err(Error::VmaBufferAllocation)?;

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(gpu.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffers = unsafe {
                gpu.device
                    .allocate_command_buffers(&command_buffer_allocate_info)
                    .map_err(Error::VulkanCommandBuffersAllocation)
            }?;
            let upload_cmdbuf = command_buffers[0];

            let finished_upload = unsafe {
                gpu.device
                    .create_fence(&vk::FenceCreateInfo::default(), None)
                    .map_err(Error::VulkanFenceCreation)
            }?;

            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            let copy_regions = [vk::BufferCopy::builder().size(buffer_size).build()];
            unsafe {
                gpu.device
                    .begin_command_buffer(upload_cmdbuf, &command_buffer_begin_info)
                    .map_err(Error::VulkanBeginCommandBuffer)?;
                gpu.device
                    .cmd_copy_buffer(upload_cmdbuf, buffer, gpu_buffer, &copy_regions);
                gpu.device
                    .end_command_buffer(upload_cmdbuf)
                    .map_err(Error::VulkanEndCommandBuffer)?;
            }

            let command_buffers = [upload_cmdbuf];
            let submit_infos = [vk::SubmitInfo::builder()
                .command_buffers(&command_buffers)
                .build()];
            unsafe {
                gpu.device
                    .queue_submit(gpu.graphics_queue, &submit_infos, finished_upload)
                    .map_err(Error::VulkanQueueSubmit)
            }?;

            let mesh_upload = MeshUpload {
                finished_upload,
                staging_buffer: mesh.buffer,
                staging_allocation: mesh.allocation,
                upload_cmdbuf,
            };
            gpu.add_mesh_upload(mesh_upload);

            mesh.buffer = gpu_buffer;
            mesh.allocation = gpu_allocation;
            mesh.alloc_info = gpu_alloc_info;
        }

        Ok(mesh)
    }

    /// Updates the vertices of the mesh. The amount of vertices must
    /// be the same as in [Mesh::new].
    pub fn update_vertices<V>(&mut self, new_vertices: &[V]) -> Result<(), Error> {
        if !self.editable {
            Err(Error::MeshNotEditable)
        } else if self.vertex_count != new_vertices.len() {
            Err(Error::MeshVertexCountMismatch)
        } else {
            // The index type doesn't matter, but needs to be specified.
            copy_mesh_data::<V, u16>(&new_vertices, None, self.alloc_info.get_mapped_data());
            Ok(())
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

fn mesh_indices_offset<V>(vertices: &[V]) -> vk::DeviceSize {
    (vertices.len() * mem::size_of::<V>()) as vk::DeviceSize
}

fn mesh_data_size<V, I: IndexType>(vertices: &[V], indices: &[I]) -> vk::DeviceSize {
    let vertices_length = vertices.len() * mem::size_of::<V>();
    let indices_length = indices.len() * mem::size_of::<I>();
    (vertices_length + indices_length) as vk::DeviceSize
}

fn copy_mesh_data<V, I: IndexType>(vertices: &[V], indices: Option<&[I]>, dst_ptr: *mut u8) {
    let vertices_length = vertices.len() * mem::size_of::<V>();
    let vertices_src_ptr = vertices.as_ptr() as *const u8;
    unsafe { std::ptr::copy_nonoverlapping(vertices_src_ptr, dst_ptr, vertices_length) };
    if let Some(indices) = indices {
        // TODO: Fix possible alignment issues?
        let indices_length = indices.len() * mem::size_of::<I>();
        let indices_src_ptr = indices.as_ptr() as *const u8;
        let indices_dst_ptr = unsafe { dst_ptr.offset(vertices_length as isize) };
        unsafe { std::ptr::copy_nonoverlapping(indices_src_ptr, indices_dst_ptr, indices_length) };
    }
}
