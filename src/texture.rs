use crate::buffer_ops::{self, BufferUpload};
use crate::{Error, FrameIndex, Gpu};
use ash::vk;

pub struct Texture<'a> {
    gpu: &'a Gpu<'a>,
    image: vk::Image,
    allocation: vk_mem::Allocation,
}

impl Drop for Texture<'_> {
    #[profiling::function]
    fn drop(&mut self) {
        let _ = self
            .gpu
            .allocator
            .destroy_image(self.image, &self.allocation);
    }
}

impl Texture<'_> {
    #[profiling::function]
    pub fn new<'a>(
        gpu: &'a Gpu<'_>,
        frame_index: FrameIndex,
        pixels: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Texture<'a>, Error> {
        let buffer_size = pixels.len() as vk::DeviceSize;
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            pool: Some(gpu.staging_cpu_buffer_pool.clone()),
            ..Default::default()
        };
        let (staging_buffer, staging_allocation, staging_alloc_info) = gpu
            .allocator
            .create_buffer(&buffer_create_info, &allocation_create_info)
            .map_err(Error::VmaBufferAllocation)?;
        buffer_ops::copy_to_allocation(pixels, gpu, &staging_allocation, &staging_alloc_info)?;
        let (image, allocation, upload_cmdbuf, finished_upload, wait_stage) =
            buffer_ops::start_image_upload(
                gpu,
                gpu.main_gpu_texture_pool.clone(),
                staging_buffer,
                vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
            )?;

        gpu.add_buffer_upload(
            frame_index,
            BufferUpload {
                finished_upload,
                wait_stage,
                upload_cmdbuf,
                staging_buffer: Some(staging_buffer),
                staging_allocation: Some(staging_allocation),
            },
        );

        Ok(Texture {
            gpu,
            image,
            allocation,
        })
    }
}
