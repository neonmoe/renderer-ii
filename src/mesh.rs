use crate::{Error, Gpu, Pipeline};
use ash::version::DeviceV1_0;
use ash::vk;
use std::mem;

fn copy_raw<T>(data: &[T], dst_ptr: *mut u8) {
    let length = data.len() * mem::size_of::<T>();
    let src_ptr = data.as_ptr() as *const u8;
    unsafe { std::ptr::copy_nonoverlapping(src_ptr, dst_ptr, length) };
}

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
    pub(crate) vertices: u32,
    allocation: vk_mem::Allocation,
    alloc_info: vk_mem::AllocationInfo,
    editable: bool,
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
    pub fn new<'a, V>(
        gpu: &'a Gpu<'_>,
        vertices: &[V],
        pipeline: Pipeline,
        editable: bool,
    ) -> Result<Mesh<'a>, Error> {
        // TODO: Create separate staging/gpu-only/shared allocation pools

        let buffer_size = (vertices.len() * mem::size_of::<V>()) as u64;
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(buffer_size)
            .usage(if editable {
                vk::BufferUsageFlags::VERTEX_BUFFER
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
        copy_raw(&vertices, buffer_ptr);
        let mut mesh = Mesh {
            gpu,
            pipeline,
            buffer,
            vertices: vertices.len() as u32,
            allocation,
            alloc_info,
            editable,
        };

        if !editable {
            let buffer_create_info = vk::BufferCreateInfo::builder()
                .size((vertices.len() * mem::size_of::<V>()) as u64)
                .usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
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
        } else if self.vertices != new_vertices.len() as u32 {
            Err(Error::MeshVertexCountMismatch)
        } else {
            copy_raw(&new_vertices, self.alloc_info.get_mapped_data());
            Ok(())
        }
    }
}
