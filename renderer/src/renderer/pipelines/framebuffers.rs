use crate::arena::{MemoryProps, VulkanArena, VulkanArenaError};
use crate::physical_device::PhysicalDevice;
use crate::renderer::pipelines::attachments::{self, AttachmentName, Attachments};
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
    pub attachments: ArrayVec<Attachments<ImageView>, 8>,
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

        let formats = pipelines.attachment_formats.all_attachments();
        let vk::Extent2D { width, height } = swapchain.extent;

        let mut image_infos = ArrayVec::<(AttachmentName, vk::ImageCreateInfo), { attachments::MAX_ATTACHMENTS }>::new();
        for (name, format) in &formats {
            let usage_flags = match name {
                AttachmentName::Hdr => vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
                AttachmentName::TonemappedMultisampled => vk::ImageUsageFlags::COLOR_ATTACHMENT,
                AttachmentName::Depth => vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                AttachmentName::TonemappedSwapchain => continue,
            };
            image_infos.push((
                *name,
                vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(**format)
                    .extent(vk::Extent3D { width, height, depth: 1 })
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(pipelines.attachment_sample_count)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(usage_flags),
            ));
        }

        let mut framebuffer_size = 0;
        {
            profiling::scope!("framebuffer memory requirements querying");
            for (_, framebuffer_image_info) in &image_infos {
                let image = unsafe { device.create_image(framebuffer_image_info, None) }.map_err(FramebufferCreationError::QueryImage)?;
                crate::name_vulkan_object(device, image, format_args!("memory requirement querying temp image"));
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
            MemoryProps::for_framebuffers(),
            format_args!("framebuffer arena ({width}x{height})"),
        )
        .map_err(FramebufferCreationError::Arena)?;

        let create_image_view =
            |image: Rc<AnyImage>, aspect_mask: vk::ImageAspectFlags, format: vk::Format| -> Result<ImageView, FramebufferCreationError> {
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
            };

        let mut non_swapchain_images = ArrayVec::<(AttachmentName, Rc<AnyImage>), { attachments::MAX_ATTACHMENTS }>::new();
        for (name, image_info) in image_infos {
            let image = framebuffer_arena
                .create_image(image_info, format_args!("{name:?} attachment"))
                .map_err(FramebufferCreationError::Image)?;
            non_swapchain_images.push((name, Rc::new(AnyImage::Regular(image))));
        }

        let mut attachments = ArrayVec::new();
        let mut swapchain_images = ArrayVec::new();
        for (i, swapchain_image) in swapchain.images.iter().enumerate() {
            let mut image_views = ArrayVec::<ImageView, { attachments::MAX_ATTACHMENTS }>::new();
            for (name, format) in &formats {
                let image = if name == &AttachmentName::TonemappedSwapchain {
                    swapchain_image
                } else {
                    non_swapchain_images
                        .iter()
                        .find(|(name_, _)| name == name_)
                        .map(|(_, image)| image)
                        .unwrap()
                };
                let image_aspect_flags = if name == &AttachmentName::Depth {
                    vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
                } else {
                    vk::ImageAspectFlags::COLOR
                };
                let image_view = create_image_view(image.clone(), image_aspect_flags, **format)?;
                crate::name_vulkan_object(device, image_view.inner, format_args!("{name:?} attachment"));
                image_views.push(image_view);
            }
            attachments.push(Attachments::create_similar(&pipelines.attachment_formats, image_views));

            let image_view = create_image_view(
                swapchain_image.clone(),
                vk::ImageAspectFlags::COLOR,
                physical_device.swapchain_format,
            )?;
            let n = i + 1;
            let m = swapchain.images.len();
            crate::name_vulkan_object(device, image_view.inner, format_args!("Swapchain {n}/{m}"));
            crate::name_vulkan_object(device, image_view.image.inner(), format_args!("Swapchain {n}/{m}"));
            swapchain_images.push(image_view);
        }

        Ok(Framebuffers {
            extent: vk::Extent2D { width, height },
            attachments,
            swapchain_images,
        })
    }

    pub fn hdr_attachment(&self, i: usize) -> &ImageView {
        match &self.attachments[i] {
            Attachments::SingleSampled { hdr, .. } | Attachments::MultiSampled { hdr, .. } => hdr,
        }
    }

    pub fn insert_initial_barriers(&self, device: &Device, command_buffer: vk::CommandBuffer, i: usize) {
        match &self.attachments[i] {
            Attachments::SingleSampled { hdr, depth, .. } | Attachments::MultiSampled { hdr, depth, .. } => {
                let color_subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1);
                let depth_subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
                    .level_count(1)
                    .layer_count(1);
                let layouts_to_initials = [
                    vk::ImageMemoryBarrier2::default()
                        .image(hdr.image.inner())
                        .subresource_range(color_subresource_range)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
                    vk::ImageMemoryBarrier2::default()
                        .image(depth.image.inner())
                        .subresource_range(depth_subresource_range)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS),
                ];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&layouts_to_initials);
                unsafe { device.sync2.cmd_pipeline_barrier2(command_buffer, &dep_info) };
            }
        }
    }

    pub fn first_pass_color_attachments(
        &self,
        i: usize,
    ) -> ArrayVec<vk::RenderingAttachmentInfoKHR, { attachments::MAX_COLOR_ATTACHMENTS }> {
        let mut attachment_infos = ArrayVec::new();
        match &self.attachments[i] {
            Attachments::SingleSampled { hdr, .. } | Attachments::MultiSampled { hdr, .. } => {
                attachment_infos.push(
                    vk::RenderingAttachmentInfoKHR::default()
                        .image_view(hdr.inner)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue::default()),
                );
            }
        }
        attachment_infos
    }

    pub fn first_pass_depth_attachment(&self, i: usize) -> vk::RenderingAttachmentInfoKHR {
        let mut clear_value = vk::ClearValue::default();
        clear_value.depth_stencil = vk::ClearDepthStencilValue::default().depth(0.0);
        match &self.attachments[i] {
            Attachments::SingleSampled { depth, .. } | Attachments::MultiSampled { depth, .. } => vk::RenderingAttachmentInfoKHR::default()
                .image_view(depth.inner)
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(clear_value),
        }
    }

    pub fn insert_post_processing_barriers(&self, device: &Device, command_buffer: vk::CommandBuffer, i: usize) {
        match &self.attachments[i] {
            Attachments::SingleSampled {
                hdr,
                present_tonemapped: tonemapped,
                ..
            }
            | Attachments::MultiSampled {
                hdr,
                resolve_src_tonemapped: tonemapped,
                ..
            } => {
                let color_subresource_range = vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1);
                let hdr_to_read_layout = [
                    vk::ImageMemoryBarrier2::default()
                        .image(hdr.image.inner())
                        .subresource_range(color_subresource_range)
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::INPUT_ATTACHMENT_READ)
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER),
                    vk::ImageMemoryBarrier2::default()
                        .image(tonemapped.image.inner())
                        .subresource_range(color_subresource_range)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
                    vk::ImageMemoryBarrier2::default()
                        .image(self.swapchain_images[i].image.inner())
                        .subresource_range(color_subresource_range)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
                ];
                let dep_info = vk::DependencyInfo::default().image_memory_barriers(&hdr_to_read_layout);
                unsafe { device.sync2.cmd_pipeline_barrier2(command_buffer, &dep_info) };
            }
        }
    }

    pub fn second_pass_color_attachments(
        &self,
        i: usize,
    ) -> ArrayVec<vk::RenderingAttachmentInfoKHR, { attachments::MAX_COLOR_ATTACHMENTS }> {
        let mut attachment_infos = ArrayVec::new();
        match &self.attachments[i] {
            Attachments::SingleSampled {
                present_tonemapped: post_process_output,
                ..
            } => {
                attachment_infos.push(
                    vk::RenderingAttachmentInfoKHR::default()
                        .image_view(post_process_output.inner)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue::default()),
                );
            }
            Attachments::MultiSampled {
                resolve_src_tonemapped: post_process_output,
                ..
            } => {
                attachment_infos.push(
                    vk::RenderingAttachmentInfoKHR::default()
                        .image_view(post_process_output.inner)
                        .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .resolve_mode(vk::ResolveModeFlagsKHR::AVERAGE)
                        .resolve_image_view(self.swapchain_images[i].inner)
                        .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .clear_value(vk::ClearValue::default()),
                );
            }
        }
        attachment_infos
    }

    pub fn insert_end_of_frame_barriers(&self, device: &Device, command_buffer: vk::CommandBuffer, i: usize) {
        let color_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let swapchain_to_present_layout = [vk::ImageMemoryBarrier2::default()
            .image(self.swapchain_images[i].image.inner())
            .subresource_range(color_subresource_range)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags2::NONE)];
        let dep_info = vk::DependencyInfo::default().image_memory_barriers(&swapchain_to_present_layout);
        unsafe { device.sync2.cmd_pipeline_barrier2(command_buffer, &dep_info) };
    }

    // TODO: Add in-place destroy() and reset() to Framebuffers
    // The current drop-then-new is less ergonomic because it requires a move.
    // Maybe make destroy unsafe to ensure it's used properly (i.e. resetted after)?
}
