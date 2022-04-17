use crate::arena::{VulkanArena, VulkanArenaError};
use crate::debug_utils;
use crate::physical_device::HDR_COLOR_ATTACHMENT_FORMAT;
use crate::pipelines::AttachmentLayout;
use crate::vulkan_raii::{AnyImage, Device, Framebuffer, ImageView};
use crate::{PhysicalDevice, Pipelines, Swapchain};
use ash::{vk, Instance};
use std::rc::Rc;

#[derive(thiserror::Error, Debug)]
pub enum FramebufferCreationError {
    #[error("failed to create vulkan arena for storing framebuffer images")]
    Arena(#[source] VulkanArenaError),
    #[error("failed to create temp image to query framebuffer memory requirements")]
    QueryImage(#[source] vk::Result),
    #[error("failed to create image view from framebuffer image")]
    ImageView(#[source] vk::Result),
    #[error("failed to create and allocate framebuffer image")]
    Image(#[source] VulkanArenaError),
    #[error("failed to create framebuffer object")]
    ObjectCreation(#[source] vk::Result),
}

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
    ) -> Result<Framebuffers, FramebufferCreationError> {
        profiling::scope!("framebuffers creation");

        let swapchain_format = physical_device.swapchain_format;
        let frame_count = swapchain.frame_count();
        let vk::Extent2D { width, height } = swapchain.extent;

        let create_image_info = |format: vk::Format, usage: vk::ImageUsageFlags| {
            vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width, height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(pipelines.attachment_sample_count)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage)
                .build()
        };

        let hdr_image_infos = (0..frame_count)
            .map(|_| {
                create_image_info(
                    HDR_COLOR_ATTACHMENT_FORMAT,
                    vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT,
                )
            })
            .collect::<Vec<_>>();
        let depth_image_infos = (0..frame_count)
            .map(|_| create_image_info(physical_device.depth_format, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT))
            .collect::<Vec<_>>();
        let resolve_src_image_infos = match pipelines.render_pass_layout {
            AttachmentLayout::SingleSampled => vec![],
            AttachmentLayout::MultiSampled => (0..frame_count)
                .map(|_| create_image_info(swapchain_format, vk::ImageUsageFlags::COLOR_ATTACHMENT))
                .collect::<Vec<_>>(),
        };

        let mut framebuffer_size = 0;
        {
            profiling::scope!("framebuffer memory requirements querying");
            for framebuffer_image_info in hdr_image_infos
                .iter()
                .chain(depth_image_infos.iter())
                .chain(resolve_src_image_infos.iter())
            {
                let image = unsafe { device.create_image(framebuffer_image_info, None) }.map_err(FramebufferCreationError::QueryImage)?;
                debug_utils::name_vulkan_object(device, image, format_args!("memory requirement querying temp img"));
                let reqs = unsafe { device.get_image_memory_requirements(image) };
                let size_mod = framebuffer_size % reqs.alignment;
                if size_mod != 0 {
                    framebuffer_size += reqs.alignment - size_mod;
                }
                framebuffer_size += reqs.size;
                unsafe { device.destroy_image(image, None) };
            }
        }

        let mut framebuffer_arena = VulkanArena::new(
            instance,
            device,
            physical_device,
            framebuffer_size,
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::LAZILY_ALLOCATED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            format_args!("framebuffer arena ({width}x{height}, {swapchain_format:?}, {frame_count} frames)"),
        )
        .map_err(FramebufferCreationError::Arena)?;

        let create_image_view = |aspect_mask: vk::ImageAspectFlags, format: vk::Format| {
            move |image: Rc<AnyImage>| -> Result<ImageView, FramebufferCreationError> {
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
                    unsafe { device.create_image_view(&image_view_create_info, None) }.map_err(FramebufferCreationError::ImageView)?;
                Ok(ImageView {
                    inner: image_view,
                    device: device.clone(),
                    image,
                })
            }
        };

        let swapchain_image_views = swapchain
            .images
            .iter()
            .cloned()
            .map(create_image_view(vk::ImageAspectFlags::COLOR, swapchain_format))
            .collect::<Result<Vec<_>, _>>()?;

        let hdr_images = hdr_image_infos
            .into_iter()
            .map(|create_info| {
                framebuffer_arena
                    .create_image(create_info, format_args!(""))
                    .map_err(FramebufferCreationError::Image)
            })
            .collect::<Result<Vec<_>, FramebufferCreationError>>()?;
        let hdr_image_views = hdr_images
            .into_iter()
            .map(|image| Rc::new(AnyImage::from(image)))
            .map(create_image_view(vk::ImageAspectFlags::COLOR, HDR_COLOR_ATTACHMENT_FORMAT))
            .collect::<Result<Vec<_>, _>>()?;

        let depth_images = depth_image_infos
            .into_iter()
            .map(|create_info| {
                framebuffer_arena
                    .create_image(create_info, format_args!(""))
                    .map_err(FramebufferCreationError::Image)
            })
            .collect::<Result<Vec<_>, FramebufferCreationError>>()?;
        let depth_image_views = depth_images
            .into_iter()
            .map(|image| Rc::new(AnyImage::from(image)))
            .map(create_image_view(vk::ImageAspectFlags::DEPTH, physical_device.depth_format))
            .collect::<Result<Vec<_>, _>>()?;

        let resolve_src_images = resolve_src_image_infos
            .into_iter()
            .map(|create_info| {
                framebuffer_arena
                    .create_image(create_info, format_args!(""))
                    .map_err(FramebufferCreationError::Image)
            })
            .collect::<Result<Vec<_>, FramebufferCreationError>>()?;
        let resolve_src_image_views = resolve_src_images
            .into_iter()
            .map(|image| Rc::new(AnyImage::from(image)))
            .map(create_image_view(vk::ImageAspectFlags::COLOR, swapchain_format))
            .collect::<Result<Vec<_>, _>>()?;

        for i in 0..frame_count as usize {
            let sc = &swapchain_image_views[i];
            let hdr = &hdr_image_views[i];
            let depth = &depth_image_views[i];
            let nth = i + 1;
            debug_utils::name_vulkan_object(device, sc.inner, format_args!("swapchain (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, sc.image.inner(), format_args!("swapchain (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, hdr.inner, format_args!("hdr fb (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, hdr.image.inner(), format_args!("hdr fb (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, depth.inner, format_args!("depth fb (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, depth.image.inner(), format_args!("depth fb (frame {nth}/{frame_count})"));
            if let AttachmentLayout::MultiSampled = pipelines.render_pass_layout {
                let tm = &resolve_src_image_views[i];
                debug_utils::name_vulkan_object(device, tm.inner, format_args!("tonemapped fb (frame {nth}/{frame_count})"));
                debug_utils::name_vulkan_object(device, tm.image.inner(), format_args!("tonemapped fb (frame {nth}/{frame_count})"));
            }
        }

        let mut hdr_image_views = hdr_image_views.into_iter();
        let mut depth_image_views = depth_image_views.into_iter();
        let mut resolve_src_image_views = resolve_src_image_views.into_iter();
        let mut swapchain_image_views = swapchain_image_views.into_iter();
        let mut framebuffers = Vec::with_capacity(frame_count as usize);
        for i in 0..frame_count as usize {
            profiling::scope!("one frame's framebuffer creation");
            let attachments = match pipelines.render_pass_layout {
                AttachmentLayout::SingleSampled => vec![
                    hdr_image_views.next().unwrap(),
                    depth_image_views.next().unwrap(),
                    swapchain_image_views.next().unwrap(),
                ],
                AttachmentLayout::MultiSampled => vec![
                    hdr_image_views.next().unwrap(),
                    depth_image_views.next().unwrap(),
                    resolve_src_image_views.next().unwrap(),
                    swapchain_image_views.next().unwrap(),
                ],
            };
            let raw_attachments = attachments.iter().map(|image_view| image_view.inner).collect::<Vec<_>>();
            let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(pipelines.render_pass.inner)
                .attachments(&raw_attachments)
                .width(width)
                .height(height)
                .layers(1);
            let framebuffer =
                unsafe { device.create_framebuffer(&framebuffer_create_info, None) }.map_err(FramebufferCreationError::ObjectCreation)?;
            debug_utils::name_vulkan_object(device, framebuffer, format_args!("main fb {}/{frame_count}", i + 1));
            framebuffers.push(Framebuffer {
                inner: framebuffer,
                device: device.clone(),
                render_pass: pipelines.render_pass.clone(),
                attachments,
            });
        }

        Ok(Framebuffers {
            extent: vk::Extent2D { width, height },
            inner: framebuffers,
        })
    }
}
