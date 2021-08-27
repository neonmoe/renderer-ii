use crate::buffer_ops;
use crate::{Error, FrameIndex, Gpu};
use ash::version::DeviceV1_0;
use ash::vk;

pub struct Texture<'a> {
    gpu: &'a Gpu<'a>,
    image: vk::Image,
    pub(crate) image_view: vk::ImageView,
    allocation: vk_mem::Allocation,
}

impl Drop for Texture<'_> {
    #[profiling::function]
    fn drop(&mut self) {
        unsafe { self.gpu.device.destroy_image_view(self.image_view, None) };
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
        format: vk::Format,
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

        let image_pool = gpu.main_gpu_texture_pool.clone();
        let image_extent = vk::Extent3D {
            width,
            height,
            depth: 1,
        };
        let (image, allocation, mip_levels) = match buffer_ops::start_image_upload(
            gpu,
            frame_index,
            image_pool,
            staging_buffer,
            image_extent,
            format,
        ) {
            Ok(result) => result,
            Err(err) => {
                let _ = gpu
                    .allocator
                    .destroy_buffer(staging_buffer, &staging_allocation);
                return Err(err);
            }
        };
        gpu.add_temporary_buffer(frame_index, staging_buffer, staging_allocation);

        let subresource_range = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(mip_levels)
            .base_array_layer(0)
            .layer_count(1)
            .build();
        let image_view_create_info = vk::ImageViewCreateInfo::builder()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(subresource_range);
        let image_view = unsafe { gpu.device.create_image_view(&image_view_create_info, None) }
            .map_err(Error::VulkanImageViewCreation)?;

        Ok(Texture {
            gpu,
            image,
            image_view,
            allocation,
        })
    }
}
