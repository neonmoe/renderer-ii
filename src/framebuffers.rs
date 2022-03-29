use crate::arena::VulkanArena;
use crate::debug_utils;
use crate::vulkan_raii::{AnyImage, Device, Framebuffer, ImageView};
use crate::{Error, PhysicalDevice, Pipelines, Swapchain};
use ash::{vk, Instance};
use std::rc::Rc;

pub struct Framebuffers {
    pub extent: vk::Extent2D,
    pub(crate) inner: Vec<Framebuffer>,
}

impl Framebuffers {
    pub fn new(
        instance: &Instance,
        device: &Rc<Device>,
        physical_device: &PhysicalDevice,
        pipelines: &Pipelines,
        swapchain: &Swapchain,
    ) -> Result<Framebuffers, Error> {
        profiling::scope!("new_canvas");

        let swapchain_format = physical_device.swapchain_format;
        let frame_count = swapchain.frame_count();
        let vk::Extent2D { width, height } = swapchain.extent;

        let mut framebuffer_arena = VulkanArena::new(
            instance,
            device,
            physical_device.inner,
            1_000_000_000, // FIXME: query framebuffer memory requirements for arena
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::LAZILY_ALLOCATED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            format_args!("framebuffer arena ({width}x{height}, {swapchain_format:?}, {frame_count} frames)"),
        )?;

        let create_image_view = |aspect_mask: vk::ImageAspectFlags, format: vk::Format| {
            move |image: Rc<AnyImage>| -> Result<ImageView, Error> {
                let subresource_range = vk::ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let image_view_create_info = vk::ImageViewCreateInfo {
                    image: image.inner(),
                    view_type: vk::ImageViewType::TYPE_2D,
                    format,
                    subresource_range,
                    ..Default::default()
                };
                let image_view =
                    unsafe { device.create_image_view(&image_view_create_info, None) }.map_err(Error::VulkanSwapchainImageViewCreation)?;
                Ok(ImageView {
                    inner: image_view,
                    device: device.clone(),
                    image,
                })
            }
        };

        let mut create_image = |format: vk::Format, usage: vk::ImageUsageFlags| {
            let image_create_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width, height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(pipelines.attachment_sample_count)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage);
            // "tbd" is changed to the real name right before framebuffer creation.
            framebuffer_arena.create_image(*image_create_info, format_args!("tbd"))
        };

        // TODO(med): Add another set of images to render to, to allow for post processing
        // Also, consider: render to a linear/higher depth image, then map to SRGB for the swapchain?
        let swapchain_image_views = swapchain
            .images
            .iter()
            .cloned()
            .map(create_image_view(vk::ImageAspectFlags::COLOR, swapchain_format))
            .collect::<Result<Vec<_>, _>>()?;

        let color_images = (0..swapchain_image_views.len())
            .map(|_| create_image(swapchain_format, vk::ImageUsageFlags::COLOR_ATTACHMENT))
            .collect::<Result<Vec<_>, Error>>()?;
        let color_image_views = color_images
            .into_iter()
            .map(|image| Rc::new(AnyImage::from(image)))
            .map(create_image_view(vk::ImageAspectFlags::COLOR, swapchain_format))
            .collect::<Result<Vec<_>, _>>()?;

        let depth_images = (0..swapchain_image_views.len())
            .map(|_| create_image(physical_device.depth_format, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT))
            .collect::<Result<Vec<_>, Error>>()?;
        let depth_image_views = depth_images
            .into_iter()
            .map(|image| Rc::new(AnyImage::from(image)))
            .map(create_image_view(vk::ImageAspectFlags::DEPTH, physical_device.depth_format))
            .collect::<Result<Vec<_>, _>>()?;

        for (((i, sc), color), depth) in swapchain_image_views
            .iter()
            .enumerate()
            .zip(color_image_views.iter())
            .zip(depth_image_views.iter())
        {
            let nth = i + 1;
            debug_utils::name_vulkan_object(device, sc.inner, format_args!("swapchain (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, sc.image.inner(), format_args!("swapchain (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, color.inner, format_args!("color fb (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, color.image.inner(), format_args!("color fb (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, depth.inner, format_args!("depth fb (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, depth.image.inner(), format_args!("depth fb (frame {nth}/{frame_count})"));
        }

        let framebuffers = color_image_views
            .into_iter()
            .enumerate()
            .zip(depth_image_views.into_iter())
            .zip(swapchain_image_views.into_iter())
            .map(|(((i, color_image_view), depth_image_view), swapchain_image_view)| {
                let attachments = [color_image_view.inner, depth_image_view.inner, swapchain_image_view.inner];
                let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(pipelines.render_pass.inner)
                    .attachments(&attachments)
                    .width(width)
                    .height(height)
                    .layers(1);
                let framebuffer =
                    unsafe { device.create_framebuffer(&framebuffer_create_info, None) }.map_err(Error::VulkanFramebufferCreation)?;
                debug_utils::name_vulkan_object(device, framebuffer, format_args!("main fb {}/{frame_count}", i + 1));
                Ok(Framebuffer {
                    inner: framebuffer,
                    device: device.clone(),
                    render_pass: pipelines.render_pass.clone(),
                    attachments: vec![color_image_view, depth_image_view, swapchain_image_view],
                })
            })
            .collect::<Result<Vec<Framebuffer>, Error>>()?;

        Ok(Framebuffers {
            extent: vk::Extent2D { width, height },
            inner: framebuffers,
        })
    }
}
