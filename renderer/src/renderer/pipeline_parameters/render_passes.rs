use arrayvec::ArrayVec;
use ash::vk;

#[derive(Debug, Clone, Copy)]
#[repr(usize)]
pub enum RenderPass {
    Geometry,
    PostProcess,
}

impl RenderPass {
    /// Returns the rendering information for the color attachments of this
    /// render pass (used at render time). The resolve targets are inserted into
    /// the associated attachments' rendering infos with sane defaults, and the
    /// store op of the affected attachments is set to `DONT_CARE`.
    pub fn color_attachment_infos(
        self,
        attachment_images: &[vk::ImageView; Attachment::COUNT],
        resolve_targets: &[(Attachment, vk::ImageView)],
    ) -> AttachmentVec<vk::RenderingAttachmentInfoKHR<'static>> {
        let mut attachment_names = AttachmentVec::new();
        match self {
            RenderPass::Geometry => attachment_names.push(Attachment::Hdr),
            RenderPass::PostProcess => attachment_names.push(Attachment::PostProcess),
        }
        let mut attachments: AttachmentVec<vk::RenderingAttachmentInfoKHR> =
            attachment_names.iter().map(|a| a.attachment_info(attachment_images)).collect();
        for (attachment_to_resolve, resolve_target_image) in resolve_targets {
            if let Some(i) = attachment_names.iter().position(|a| a == attachment_to_resolve) {
                let attachment_info = &mut attachments[i];
                attachment_info.store_op = vk::AttachmentStoreOp::DONT_CARE;
                attachment_info.resolve_mode = vk::ResolveModeFlags::AVERAGE;
                attachment_info.resolve_image_view = *resolve_target_image;
                attachment_info.resolve_image_layout = vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL;
            } else {
                debug_assert!(
                    false,
                    "resolve target provided for attachment {attachment_to_resolve:?}, \
                    which is not used in this pass ({self:?})"
                );
            }
        }
        attachments
    }

    /// Returns the formats of the clor attachments of this render pass (used at
    /// pipeline creation time).
    pub fn color_attachment_formats(self, attachment_formats: &AttachmentFormats) -> AttachmentVec<vk::Format> {
        let mut formats = ArrayVec::new();
        match self {
            RenderPass::Geometry => formats.push(Attachment::Hdr.format(attachment_formats)),
            RenderPass::PostProcess => formats.push(Attachment::PostProcess.format(attachment_formats)),
        }
        formats
    }

    /// Returns the rendering information for the depth attachment of this
    /// render pass (used at render time).
    pub fn depth_attachment_info(self, attachment_images: &[vk::ImageView; Attachment::COUNT]) -> vk::RenderingAttachmentInfoKHR<'static> {
        self.depth_attachment()
            .map_or_else(vk::RenderingAttachmentInfoKHR::default, |attachment| attachment.attachment_info(attachment_images))
    }

    /// Returns the format of the depth attachment of this render pass (used at
    /// pipeline creation time). This is [`vk::Format::UNDEFINED`] if the render
    /// pass does not use depth.
    pub fn depth_attachment_format(self, formats: &AttachmentFormats) -> vk::Format {
        self.depth_attachment().map_or(vk::Format::UNDEFINED, |attachment| attachment.format(formats))
    }

    fn depth_attachment(self) -> Option<Attachment> {
        match self {
            RenderPass::Geometry => Some(Attachment::Depth),
            RenderPass::PostProcess => None,
        }
    }

    /// Returns the [`ImageMemoryBarriers`] that should be placed before the
    /// draw commands of this render pass.
    pub fn barriers(self, attachment_images: &[vk::Image; Attachment::COUNT]) -> AttachmentVec<vk::ImageMemoryBarrier2<'static>> {
        let color_subresource_range =
            vk::ImageSubresourceRange::default().aspect_mask(vk::ImageAspectFlags::COLOR).level_count(1).layer_count(1);
        let depth_subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
            .level_count(1)
            .layer_count(1);
        let mut barriers = ArrayVec::new();
        match self {
            RenderPass::Geometry => {
                barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .image(attachment_images[Attachment::Hdr as usize])
                        .subresource_range(color_subresource_range)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
                );
                barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .image(attachment_images[Attachment::Depth as usize])
                        .subresource_range(depth_subresource_range)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS),
                );
            }
            RenderPass::PostProcess => {
                barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .image(attachment_images[Attachment::Hdr as usize])
                        .subresource_range(color_subresource_range)
                        .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .dst_access_mask(vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::INPUT_ATTACHMENT_READ)
                        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER),
                );
                barriers.push(
                    vk::ImageMemoryBarrier2::default()
                        .image(attachment_images[Attachment::PostProcess as usize])
                        .subresource_range(color_subresource_range)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                        .src_access_mask(vk::AccessFlags2::NONE)
                        .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                        .src_stage_mask(vk::PipelineStageFlags2::NONE)
                        .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT),
                );
            }
        }
        barriers
    }
}

pub type AttachmentVec<T> = ArrayVec<T, { Attachment::COUNT }>;

pub struct AttachmentFormats {
    pub hdr: vk::Format,
    pub swapchain: vk::Format,
    pub depth: vk::Format,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Attachment {
    Hdr,
    Depth,
    PostProcess,
}

impl Attachment {
    pub const COUNT: usize = 3;

    pub fn usage(self) -> vk::ImageUsageFlags {
        match self {
            Attachment::Hdr => vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
            Attachment::Depth => vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            Attachment::PostProcess => vk::ImageUsageFlags::COLOR_ATTACHMENT,
        }
    }

    pub fn aspect(self) -> vk::ImageAspectFlags {
        match self {
            Attachment::Hdr | Attachment::PostProcess => vk::ImageAspectFlags::COLOR,
            Attachment::Depth => vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL,
        }
    }

    pub fn format(self, formats: &AttachmentFormats) -> vk::Format {
        match self {
            Attachment::Hdr => formats.hdr,
            Attachment::Depth => formats.depth,
            Attachment::PostProcess => formats.swapchain,
        }
    }

    pub fn attachment_info(self, attachment_images: &[vk::ImageView; Attachment::COUNT]) -> vk::RenderingAttachmentInfoKHR<'static> {
        match self {
            Attachment::Hdr => vk::RenderingAttachmentInfoKHR::default()
                .image_view(attachment_images[self as usize])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue::default()),
            Attachment::Depth => vk::RenderingAttachmentInfoKHR::default()
                .image_view(attachment_images[self as usize])
                .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(vk::ClearValue { depth_stencil: vk::ClearDepthStencilValue::default().depth(0.0) }),
            Attachment::PostProcess => vk::RenderingAttachmentInfoKHR::default()
                .image_view(attachment_images[self as usize])
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(vk::ClearValue::default()),
        }
    }
}
