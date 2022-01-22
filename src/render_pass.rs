use crate::arena::ImageAllocation;
use crate::pipeline::{PipelineParameters, PIPELINE_PARAMETERS};
use crate::vulkan_raii::{VkFramebuffer, VkImageView, VkPipelines, VkRenderPass};
use crate::{Error, Swapchain, VulkanArena};
use ash::version::DeviceV1_0;
use ash::vk;
use ash::Device;

const DEPTH_FORMAT: vk::Format = vk::Format::D16_UNORM;
const SAMPLE_COUNT: vk::SampleCountFlags = vk::SampleCountFlags::TYPE_8;

/// Render pass, pipelines, and framebuffers.
///
/// Contains the vkPipelines, vkFramebuffers and vkImages related to a
/// specific render pass. Needs to be changed after refreshing the
/// swapchain and/or device. Changing shaders would require just
/// recreating the RenderPass, the swapchain can be retained.
pub struct RenderPass<'a> {
    pub extent: vk::Extent2D,

    // NOTE: Load-bearing field order ahead. Pipeline relies on
    // RenderPass which relies on the Framebuffers which rely on the
    // ImageViews.
    pub pipelines: VkPipelines<'a>,
    pub final_render_pass: VkRenderPass<'a>,
    pub framebuffers: Vec<VkFramebuffer<'a>>,
    #[allow(dead_code)]
    color_image_views: Vec<VkImageView<'a>>,
    #[allow(dead_code)]
    depth_image_views: Vec<VkImageView<'a>>,

    #[allow(dead_code)]
    /// Owns the swapchain image views used in the render pass.
    swapchain: &'a Swapchain<'a>,
    #[allow(dead_code)]
    /// Owns the backing memory for the images used in the
    /// framebuffers and image views.
    framebuffer_arena: &'a VulkanArena<'a>,
}

impl RenderPass<'_> {
    pub fn new<'a>(
        device: &'a Device,
        framebuffer_arena: &'a VulkanArena,
        swapchain: &'a Swapchain,
        pipeline_layouts: &[vk::PipelineLayout],
        extent: vk::Extent2D,
    ) -> Result<RenderPass<'a>, Error> {
        profiling::scope!("new_renderpass");

        let create_image_view = |aspect_mask: vk::ImageAspectFlags, format: vk::Format| {
            move |image: &&'a ImageAllocation| -> Result<VkImageView, Error> {
                let subresource_range = vk::ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let image_view_create_info = vk::ImageViewCreateInfo {
                    image: image.image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format,
                    subresource_range,
                    ..Default::default()
                };
                unsafe { VkImageView::new(device, Some(image), &image_view_create_info) }
            }
        };

        let create_image = |format: vk::Format, usage: vk::ImageUsageFlags| {
            let image_create_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(SAMPLE_COUNT)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage);
            framebuffer_arena.create_image(*image_create_info)
        };

        let color_images = (0..swapchain.swapchain_image_views.len())
            .map(|_| create_image(swapchain.format, vk::ImageUsageFlags::COLOR_ATTACHMENT))
            .collect::<Result<Vec<_>, Error>>()?;
        let color_image_views = color_images
            .iter()
            .map(create_image_view(vk::ImageAspectFlags::COLOR, swapchain.format))
            .collect::<Result<Vec<_>, _>>()?;

        // TODO: Select depth format based on capabilities
        let depth_images = (0..swapchain.swapchain_image_views.len())
            .map(|_| create_image(DEPTH_FORMAT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT))
            .collect::<Result<Vec<_>, Error>>()?;
        let depth_image_views = depth_images
            .iter()
            .map(create_image_view(vk::ImageAspectFlags::DEPTH, DEPTH_FORMAT))
            .collect::<Result<Vec<_>, _>>()?;

        let final_render_pass = create_render_pass(device, swapchain.format)?;
        let pipelines = create_pipelines(device, *final_render_pass, extent, pipeline_layouts, &PIPELINE_PARAMETERS)?;

        let framebuffers = color_image_views
            .iter()
            .zip(depth_image_views.iter())
            .zip(swapchain.swapchain_image_views.iter())
            .map(|((color_image_view, depth_image_view), swapchain_image_view)| {
                let attachments = [**color_image_view, **depth_image_view, **swapchain_image_view];
                let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(*final_render_pass)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);
                unsafe { VkFramebuffer::new(device, &framebuffer_create_info) }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(RenderPass {
            extent,
            framebuffers,
            final_render_pass,
            pipelines,
            swapchain,
            framebuffer_arena,
            color_image_views,
            depth_image_views,
        })
    }
}

#[profiling::function]
fn create_render_pass<'a>(device: &'a Device, format: vk::Format) -> Result<VkRenderPass<'a>, Error> {
    let color_attachment = vk::AttachmentDescription::builder()
        .format(format)
        .samples(SAMPLE_COUNT)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let depth_attachment = vk::AttachmentDescription::builder()
        .format(DEPTH_FORMAT)
        .samples(SAMPLE_COUNT)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let resolve_attachment = vk::AttachmentDescription::builder()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
    let attachments = [color_attachment.build(), depth_attachment.build(), resolve_attachment.build()];

    let color_attachment_reference = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let color_attachment_references = [color_attachment_reference.build()];
    let resolve_attachment_reference = vk::AttachmentReference::builder()
        .attachment(2)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let resolve_attachment_references = [resolve_attachment_reference.build()];
    let depth_attachment_reference = vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&color_attachment_references)
        .resolve_attachments(&resolve_attachment_references)
        .depth_stencil_attachment(&depth_attachment_reference);
    let subpasses = [subpass.build()];

    let render_pass_create_info = vk::RenderPassCreateInfo::builder().attachments(&attachments).subpasses(&subpasses);
    unsafe { VkRenderPass::new(device, &render_pass_create_info) }
}

#[profiling::function]
fn create_pipelines<'a>(
    device: &'a Device,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
    pipeline_layouts: &[vk::PipelineLayout],
    pipelines_params: &[PipelineParameters],
) -> Result<VkPipelines<'a>, Error> {
    let mut all_shader_modules = Vec::with_capacity(pipelines_params.len() * 2);
    let mut create_shader_module = |spirv: &'static [u32]| -> Result<vk::ShaderModule, Error> {
        let create_info = vk::ShaderModuleCreateInfo::builder().code(spirv);
        let shader_module = unsafe { device.create_shader_module(&create_info, None) }.map_err(Error::VulkanShaderModuleCreation)?;
        all_shader_modules.push(shader_module);
        Ok(shader_module)
    };

    let shader_stages_per_pipeline = pipelines_params
        .iter()
        .map(|pipeline| {
            let vert_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(create_shader_module(pipeline.vertex_shader)?)
                .name(cstr!("main"));
            let frag_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(create_shader_module(pipeline.fragment_shader)?)
                .name(cstr!("main"));
            Ok([vert_shader_stage_create_info.build(), frag_shader_stage_create_info.build()])
        })
        .collect::<Result<Vec<[vk::PipelineShaderStageCreateInfo; 2]>, Error>>()?;

    let vertex_input_per_pipeline = pipelines_params
        .iter()
        .map(|pipeline| {
            vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_binding_descriptions(pipeline.bindings)
                .vertex_attribute_descriptions(pipeline.attributes)
                .build()
        })
        .collect::<Vec<vk::PipelineVertexInputStateCreateInfo>>();

    let pipelines = {
        let input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewports = [vk::Viewport::builder()
            .width(extent.width as f32)
            .height(extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)
            .build()];
        let scissors = [vk::Rect2D::builder().extent(extent).build()];
        let viewport_create_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        let rasterization_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        let multisample_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(SAMPLE_COUNT);

        let pipeline_depth_stencil_create_info = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL);

        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(
                vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B | vk::ColorComponentFlags::A,
            )
            .blend_enable(false)
            .build()];
        let color_blend_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(&color_blend_attachment_states);

        let pipeline_create_infos = shader_stages_per_pipeline
            .iter()
            .zip(vertex_input_per_pipeline.iter())
            .zip(pipeline_layouts.iter())
            .map(|((shader_stages, vertex_input), &pipeline_layout)| {
                vk::GraphicsPipelineCreateInfo::builder()
                    .stages(&shader_stages[..])
                    .vertex_input_state(vertex_input)
                    .input_assembly_state(&input_assembly_create_info)
                    .viewport_state(&viewport_create_info)
                    .rasterization_state(&rasterization_create_info)
                    .multisample_state(&multisample_create_info)
                    .depth_stencil_state(&pipeline_depth_stencil_create_info)
                    .color_blend_state(&color_blend_create_info)
                    .layout(pipeline_layout)
                    .render_pass(render_pass)
                    .subpass(0)
                    .build()
            })
            .collect::<Vec<vk::GraphicsPipelineCreateInfo>>();
        unsafe { VkPipelines::new(device, &pipeline_create_infos) }?
    };

    for shader_module in all_shader_modules {
        unsafe { device.destroy_shader_module(shader_module, None) };
    }

    Ok(pipelines)
}
