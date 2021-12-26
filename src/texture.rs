use crate::{Error, FrameIndex, Gpu};
use ash::version::DeviceV1_0;
use ash::vk;
use std::ops::Range;

#[derive(Clone)]
pub struct Texture {
    pub(crate) image_view: vk::ImageView,
}

impl Texture {
    pub fn new(
        gpu: &Gpu,
        frame_index: FrameIndex,
        pixels: &[u8],
        mipmap_ranges: Option<Vec<Range<usize>>>,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<Texture, Error> {
        profiling::scope!("new_texture");

        let (mip_levels, image) = {
            profiling::scope!("allocate and create staging buffer");
            let buffer_size = pixels.len() as vk::DeviceSize;
            let buffer_create_info = vk::BufferCreateInfo::builder()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::TRANSFER_SRC)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            todo!()
        };

        let image_view = {
            profiling::scope!("create image view");
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
            unsafe { gpu.device.create_image_view(&image_view_create_info, None) }.map_err(Error::VulkanImageViewCreation)?
        };

        todo!()
    }
}
