use std::fmt::Arguments;

use crate::arena::{MemoryProps, VulkanArena, VulkanArenaError};
use crate::physical_device::PhysicalDevice;
use crate::renderer::pipelines::render_passes::{Attachment, AttachmentVec};
use crate::renderer::{Pipelines, Swapchain};
use crate::vulkan_raii::{AnyImage, Device, ImageView};
use alloc::rc::Rc;
use arrayvec::ArrayVec;
use ash::{vk, Instance};

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
    pub hdr_image: ImageView,
    pub depth_image: ImageView,
    pub multisampled_final_image: Option<ImageView>,
    pub swapchain_images: ArrayVec<ImageView, 8>,
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

        let vk::Extent2D { width, height } = swapchain.extent;
        let multisampled = pipelines.attachment_sample_count != vk::SampleCountFlags::TYPE_1;

        let image_info = |attachment: Attachment| {
            vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(attachment.format(physical_device))
                .extent(vk::Extent3D { width, height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(pipelines.attachment_sample_count)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(attachment.usage())
        };

        let mut allocated_image_infos = AttachmentVec::<Attachment>::new();
        allocated_image_infos.push(Attachment::Hdr);
        allocated_image_infos.push(Attachment::Depth);
        if multisampled {
            allocated_image_infos.push(Attachment::PostProcess);
        }

        let mut framebuffer_size = 0;
        {
            profiling::scope!("framebuffer memory requirements querying");
            for attachment in &allocated_image_infos {
                let framebuffer_image_info = image_info(*attachment);
                let image = unsafe { device.create_image(&framebuffer_image_info, None) }.map_err(FramebufferCreationError::QueryImage)?;
                crate::name_vulkan_object(device, image, format_args!("memory requirement querying temp image"));
                let reqs = unsafe { device.get_image_memory_requirements(image) };
                framebuffer_size += reqs.size.next_multiple_of(reqs.alignment);
                unsafe { device.destroy_image(image, None) };
            }
        }

        let mut framebuffer_arena = VulkanArena::new(
            instance,
            device,
            physical_device,
            framebuffer_size,
            MemoryProps::for_framebuffers(),
            format_args!("framebuffer arena ({width}x{height})"),
        )
        .map_err(FramebufferCreationError::Arena)?;

        let create_image_view = |image: Rc<AnyImage>,
                                 aspect_mask: vk::ImageAspectFlags,
                                 format: vk::Format,
                                 debug_identifier: Arguments|
         -> Result<ImageView, FramebufferCreationError> {
            let subresource_range = vk::ImageSubresourceRange::default()
                .aspect_mask(aspect_mask)
                .level_count(1)
                .layer_count(1);
            let image_view_create_info = vk::ImageViewCreateInfo::default()
                .image(image.inner())
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(format)
                .subresource_range(subresource_range);
            let image_view =
                unsafe { device.create_image_view(&image_view_create_info, None) }.map_err(FramebufferCreationError::ImageView)?;
            crate::name_vulkan_object(device, image_view, debug_identifier);
            crate::name_vulkan_object(device, image.inner(), debug_identifier);
            Ok(ImageView {
                inner: image_view,
                device: device.clone(),
                image,
            })
        };

        let mut create_attachment_image = |attachment: Attachment| {
            let image_info = image_info(attachment);
            let image = framebuffer_arena
                .create_image(image_info, format_args!("{attachment:?} attachment"))
                .map_err(FramebufferCreationError::Image)?;
            let image_view = create_image_view(
                Rc::new(AnyImage::Regular(image)),
                attachment.aspect(),
                attachment.format(physical_device),
                format_args!("{attachment:?} render target"),
            )?;
            Ok(image_view)
        };

        let hdr_image = create_attachment_image(Attachment::Hdr)?;
        let depth_image = create_attachment_image(Attachment::Depth)?;
        let multisampled_final_image = if multisampled {
            Some(create_attachment_image(Attachment::PostProcess)?)
        } else {
            None
        };

        let mut swapchain_images = ArrayVec::new();
        for (i, swapchain_image) in swapchain.images.iter().enumerate() {
            let n = i + 1;
            let m = swapchain.images.len();
            let image_view = create_image_view(
                swapchain_image.clone(),
                vk::ImageAspectFlags::COLOR,
                physical_device.swapchain_format,
                format_args!("Swapchain render target {n}/{m}"),
            )?;
            swapchain_images.push(image_view);
        }

        Ok(Framebuffers {
            extent: vk::Extent2D { width, height },
            hdr_image,
            depth_image,
            multisampled_final_image,
            swapchain_images,
        })
    }

    pub(crate) fn attachment_image_views(&self, frame_index: usize) -> [vk::ImageView; Attachment::COUNT] {
        let post_process_image = if let Some(msaa_pp) = &self.multisampled_final_image {
            msaa_pp.inner
        } else {
            self.swapchain_images[frame_index].inner
        };
        [
            self.hdr_image.inner,   // Attachment::Hdr
            self.depth_image.inner, // Attachment::Depth
            post_process_image,     // Attachment::PostProcess
        ]
    }

    pub(crate) fn attachment_images(&self, frame_index: usize) -> [vk::Image; Attachment::COUNT] {
        let post_process_image = if let Some(msaa_pp) = &self.multisampled_final_image {
            msaa_pp.image.inner()
        } else {
            self.swapchain_images[frame_index].image.inner()
        };
        [
            self.hdr_image.image.inner(),   // Attachment::Hdr
            self.depth_image.image.inner(), // Attachment::Depth
            post_process_image,             // Attachment::PostProcess
        ]
    }

    pub(crate) fn swapchain_write_barrier(&self, i: usize) -> [vk::ImageMemoryBarrier2; 1] {
        let color_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        [vk::ImageMemoryBarrier2::default()
            .image(self.swapchain_images[i].image.inner())
            .subresource_range(color_subresource_range)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)]
    }

    pub(crate) fn swapchain_present_barrier(&self, i: usize) -> [vk::ImageMemoryBarrier2; 1] {
        let color_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        [vk::ImageMemoryBarrier2::default()
            .image(self.swapchain_images[i].image.inner())
            .subresource_range(color_subresource_range)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)]
    }

    // TODO: Add in-place destroy() and reset() to Framebuffers
    // The current drop-then-new is less ergonomic because it requires a move.
    // Maybe make destroy unsafe to ensure it's used properly (i.e. resetted after)?
}
