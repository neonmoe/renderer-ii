use crate::buffer_ops;
use crate::resources::{AllocatedBuffer, AllocatedImage, RefCountedStatus, WeakRefCountedStatus};
use crate::{Error, FrameIndex, Gpu};
use ash::version::DeviceV1_0;
use ash::vk;
use std::ops::Range;
use std::rc::Rc;

#[derive(Clone)]
pub struct Texture {
    pub(crate) image_view: vk::ImageView,
    texture_ref: RefCountedStatus,
}

impl Texture {
    #[profiling::function]
    pub fn new(
        gpu: &Gpu,
        frame_index: FrameIndex,
        pixels: &[u8],
        mipmap_ranges: Option<Vec<Range<usize>>>,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<Texture, Error> {
        let buffer_size = pixels.len() as vk::DeviceSize;
        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(buffer_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE,
            ..Default::default()
        };
        let (staging_buffer, staging_allocation, staging_alloc_info) = gpu
            .allocator
            .create_buffer(&buffer_create_info, &allocation_create_info)
            .map_err(Error::VmaBufferAllocation)?;
        buffer_ops::copy_to_allocation(pixels, &gpu.allocator, &staging_allocation, &staging_alloc_info)?;

        let image_extent = vk::Extent3D { width, height, depth: 1 };
        let (upload_fence, image, allocation, mip_levels) =
            match buffer_ops::start_image_upload(gpu, frame_index, staging_buffer, mipmap_ranges.as_deref(), image_extent, format) {
                Ok(result) => result,
                Err(err) => {
                    let _ = gpu.allocator.destroy_buffer(staging_buffer, &staging_allocation);
                    return Err(err);
                }
            };

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
        let image_view = unsafe { gpu.device.create_image_view(&image_view_create_info, None) }.map_err(Error::VulkanImageViewCreation)?;

        let staging_buffer = AllocatedBuffer(staging_buffer, staging_allocation);
        let image = AllocatedImage(image, image_view, allocation);
        let image_view = image.1;
        let texture_ref = gpu.resources.add_image(upload_fence, Some(staging_buffer), image);
        Ok(Texture { image_view, texture_ref })
    }

    pub(crate) fn create_weak_ref(&self) -> WeakRefCountedStatus {
        Rc::downgrade(&self.texture_ref)
    }
}
