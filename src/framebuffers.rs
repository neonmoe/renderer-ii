use crate::arena::{VulkanArena, VulkanArenaError};
use crate::debug_utils;
use crate::pipelines::AttachmentLayout;
use crate::vulkan_raii::{AnyImage, Device, Framebuffer, ImageView};
use crate::{PhysicalDevice, Pipelines, Swapchain};
use alloc::rc::Rc;
use arrayvec::ArrayVec;
use ash::{vk, Instance};

pub const HDR_COLOR_ATTACHMENT_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

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
    pub(crate) inner: ArrayVec<Framebuffer, 8>,
}

impl Framebuffers {
    pub fn new(
        instance: &Instance,
        device: &Device,
        physical_device: &PhysicalDevice,
        pipelines: &Pipelines,
        swapchain: &Swapchain,
    ) -> Result<Framebuffers, FramebufferCreationError> {
        profiling::scope!("framebuffers creation");

        let swapchain_format = physical_device.swapchain_format;
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

        let hdr_image_info = create_image_info(
            HDR_COLOR_ATTACHMENT_FORMAT,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::INPUT_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
        );
        let depth_image_info = create_image_info(
            physical_device.depth_format,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
        );
        let resolve_src_image_info = match pipelines.render_pass_layout {
            AttachmentLayout::SingleSampled => None,
            AttachmentLayout::MultiSampled => Some(create_image_info(
                swapchain_format,
                vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            )),
        };

        let mut framebuffer_size = 0;
        {
            profiling::scope!("framebuffer memory requirements querying");
            let mut image_infos: ArrayVec<vk::ImageCreateInfo, 3> = ArrayVec::new();
            image_infos.push(hdr_image_info);
            image_infos.push(depth_image_info);
            if let Some(resolve_src_image_info) = resolve_src_image_info {
                image_infos.push(resolve_src_image_info);
            }
            for framebuffer_image_info in image_infos {
                let image = unsafe { device.create_image(&framebuffer_image_info, None) }.map_err(FramebufferCreationError::QueryImage)?;
                debug_utils::name_vulkan_object(device, image, format_args!("memory requirement querying temp image"));
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
            format_args!("framebuffer arena ({width}x{height})"),
        )
        .map_err(FramebufferCreationError::Arena)?;

        let create_image_view = |image: Rc<AnyImage>,
                                 aspect_mask: vk::ImageAspectFlags,
                                 format: vk::Format|
         -> Result<Rc<ImageView>, FramebufferCreationError> {
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
            Ok(Rc::new(ImageView {
                inner: image_view,
                device: device.clone(),
                image,
            }))
        };

        let swapchain_image_views = swapchain
            .images
            .iter()
            .map(|image| create_image_view(image.clone(), vk::ImageAspectFlags::COLOR, swapchain_format))
            .collect::<Result<ArrayVec<Rc<ImageView>, 8>, _>>()?;
        for (i, sc) in swapchain_image_views.iter().enumerate() {
            let frame_count = swapchain_image_views.len();
            let nth = i + 1;
            debug_utils::name_vulkan_object(device, sc.inner, format_args!("swapchain (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, sc.image.inner(), format_args!("swapchain (frame {nth}/{frame_count})"));
        }

        let hdr_image = framebuffer_arena
            .create_image(hdr_image_info, format_args!(""))
            .map_err(FramebufferCreationError::Image)?;
        let hdr_image_view = create_image_view(
            Rc::new(AnyImage::from(hdr_image)),
            vk::ImageAspectFlags::COLOR,
            HDR_COLOR_ATTACHMENT_FORMAT,
        )?;
        debug_utils::name_vulkan_object(device, hdr_image_view.inner, format_args!("hdr fb"));
        debug_utils::name_vulkan_object(device, hdr_image_view.image.inner(), format_args!("hdr fb"));

        let depth_image = framebuffer_arena
            .create_image(depth_image_info, format_args!(""))
            .map_err(FramebufferCreationError::Image)?;
        let depth_image_view = create_image_view(
            Rc::new(AnyImage::from(depth_image)),
            vk::ImageAspectFlags::DEPTH,
            physical_device.depth_format,
        )?;
        debug_utils::name_vulkan_object(device, depth_image_view.inner, format_args!("depth fb"));
        debug_utils::name_vulkan_object(device, depth_image_view.image.inner(), format_args!("depth fb"));

        let resolve_src_image_view = if let Some(create_info) = resolve_src_image_info {
            let image = framebuffer_arena
                .create_image(create_info, format_args!(""))
                .map_err(FramebufferCreationError::Image)?;
            let image_view = create_image_view(Rc::new(AnyImage::from(image)), vk::ImageAspectFlags::COLOR, swapchain_format)?;
            Some(image_view)
        } else {
            None
        };
        if let Some(image_view) = &resolve_src_image_view {
            debug_utils::name_vulkan_object(device, image_view.inner, format_args!("tonemapped fb"));
            debug_utils::name_vulkan_object(device, image_view.image.inner(), format_args!("tonemapped fb"));
        }

        let framebuffers = swapchain_image_views
            .into_iter()
            .map(|swapchain_image_view| {
                profiling::scope!("one frame's framebuffer creation");
                let mut attachments = ArrayVec::new();
                attachments.push(hdr_image_view.clone());
                attachments.push(depth_image_view.clone());
                if let Some(resolve_src_image_view) = &resolve_src_image_view {
                    attachments.push(resolve_src_image_view.clone());
                }
                attachments.push(swapchain_image_view);
                let raw_attachments: ArrayVec<vk::ImageView, 4> =
                    attachments.iter().map(|image_view: &Rc<ImageView>| image_view.inner).collect();
                let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(pipelines.render_pass.inner)
                    .attachments(&raw_attachments)
                    .width(width)
                    .height(height)
                    .layers(1);
                let framebuffer = unsafe { device.create_framebuffer(&framebuffer_create_info, None) }
                    .map_err(FramebufferCreationError::ObjectCreation)?;
                Ok(Framebuffer {
                    inner: framebuffer,
                    device: device.clone(),
                    render_pass: pipelines.render_pass.clone(),
                    attachments,
                })
            })
            .collect::<Result<ArrayVec<Framebuffer, 8>, FramebufferCreationError>>()?;

        for (i, framebuffer) in framebuffers.iter().enumerate() {
            let frame_count = framebuffers.len();
            let nth = i + 1;
            debug_utils::name_vulkan_object(device, framebuffer.inner, format_args!("main fb {nth}/{frame_count}"));
        }

        Ok(Framebuffers {
            extent: vk::Extent2D { width, height },
            inner: framebuffers,
        })
    }

    // TODO: Add in-place destroy() and reset() to Framebuffers
    // The current drop-then-new is less ergonomic because it requires a move.
    // Maybe make destroy unsafe to ensure it's used properly (i.e. resetted after)?
}
