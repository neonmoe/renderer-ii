use crate::debug_utils;
use crate::pipeline_parameters::{PipelineMap, PIPELINE_PARAMETERS};
use crate::vulkan_raii::{self, Device, PipelineCache, PipelineLayout, RenderPass};
use crate::{Descriptors, PhysicalDevice};
use ash::vk;
use std::rc::Rc;

pub(crate) enum AttachmentLayout {
    /// Attachments: present color, depth
    SingleSampled,
    /// Attachments: present color (resolved), depth (multisampled), color (multisampled)
    MultiSampled,
}

#[derive(thiserror::Error, Debug)]
pub enum PipelineCreationError {
    #[error("failed to create vulkan render pass object")]
    RenderPass(#[source] vk::Result),
    #[error("failed to create shader module (compilation issue?)")]
    ShaderModule(#[source] vk::Result),
    #[error("failed to create vulkan pipeline object")]
    Object(#[source] vk::Result),
}

pub struct Pipelines {
    pub(crate) render_pass_layout: AttachmentLayout,
    pub(crate) attachment_sample_count: vk::SampleCountFlags,
    pub(crate) render_pass: Rc<RenderPass>,
    pub(crate) pipelines: PipelineMap<vulkan_raii::Pipeline>,
    pipeline_cache: Option<PipelineCache>,
}

impl Pipelines {
    pub fn new(
        device: &Rc<Device>,
        physical_device: &PhysicalDevice,
        descriptors: &Descriptors,
        extent: vk::Extent2D,
        attachment_sample_count: vk::SampleCountFlags,
        old_pipelines: Option<Pipelines>,
    ) -> Result<Pipelines, PipelineCreationError> {
        let (render_pass, render_pass_layout) = create_render_pass(
            device,
            physical_device.swapchain_format,
            physical_device.depth_format,
            attachment_sample_count,
        )?;
        debug_utils::name_vulkan_object(device, render_pass, format_args!("main render pass"));
        let render_pass = Rc::new(RenderPass {
            inner: render_pass,
            device: device.clone(),
        });

        let (vk_pipelines, pipeline_cache) = create_pipelines(
            device,
            render_pass.inner,
            &descriptors.pipeline_layouts,
            attachment_sample_count,
            extent,
            old_pipelines.and_then(|pipelines| pipelines.pipeline_cache),
        )?;
        let mut vk_pipelines_iter = vk_pipelines.into_iter();
        let pipelines = PipelineMap::new::<PipelineCreationError, _>(|name| {
            let pipeline = vk_pipelines_iter.next().unwrap();
            debug_utils::name_vulkan_object(device, pipeline, format_args!("{name:?}"));
            Ok(vulkan_raii::Pipeline {
                inner: pipeline,
                device: device.clone(),
                render_pass: render_pass.clone(),
            })
        })?;
        Ok(Pipelines {
            render_pass_layout,
            attachment_sample_count,
            render_pass,
            pipelines,
            pipeline_cache,
        })
    }
}

#[profiling::function]
fn create_render_pass(
    device: &Device,
    swapchain_format: vk::Format,
    depth_format: vk::Format,
    attachment_sample_count: vk::SampleCountFlags,
) -> Result<(vk::RenderPass, AttachmentLayout), PipelineCreationError> {
    let depth_attachment = vk::AttachmentDescription::builder()
        .format(depth_format)
        .samples(attachment_sample_count)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let depth_attachment_reference = vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .build();

    if attachment_sample_count == vk::SampleCountFlags::TYPE_1 {
        let color_attachment = vk::AttachmentDescription::builder()
            .format(swapchain_format)
            .samples(attachment_sample_count)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let color_attachment_references = [vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build()];

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_references)
            .depth_stencil_attachment(&depth_attachment_reference);

        let attachments = [color_attachment.build(), depth_attachment.build()];
        let subpasses = [subpass.build()];
        let render_pass_create_info = vk::RenderPassCreateInfo::builder().attachments(&attachments).subpasses(&subpasses);
        let render_pass =
            unsafe { device.create_render_pass(&render_pass_create_info, None) }.map_err(PipelineCreationError::RenderPass)?;
        Ok((render_pass, AttachmentLayout::SingleSampled))
    } else {
        let resolve_attachment = vk::AttachmentDescription::builder()
            .format(swapchain_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let resolve_attachment_references = [vk::AttachmentReference::builder()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build()];

        let color_attachment = vk::AttachmentDescription::builder()
            .format(swapchain_format)
            .samples(attachment_sample_count)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let color_attachment_references = [vk::AttachmentReference::builder()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .build()];

        let subpass = vk::SubpassDescription::builder()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_attachment_references)
            .depth_stencil_attachment(&depth_attachment_reference)
            .resolve_attachments(&resolve_attachment_references);

        let attachments = [resolve_attachment.build(), depth_attachment.build(), color_attachment.build()];
        let subpasses = [subpass.build()];
        let render_pass_create_info = vk::RenderPassCreateInfo::builder().attachments(&attachments).subpasses(&subpasses);
        let render_pass =
            unsafe { device.create_render_pass(&render_pass_create_info, None) }.map_err(PipelineCreationError::RenderPass)?;
        Ok((render_pass, AttachmentLayout::MultiSampled))
    }
}

#[profiling::function]
fn create_pipelines(
    device: &Rc<Device>,
    render_pass: vk::RenderPass,
    pipeline_layouts: &PipelineMap<PipelineLayout>,
    attachment_sample_count: vk::SampleCountFlags,
    extent: vk::Extent2D,
    mut pipeline_cache: Option<PipelineCache>,
) -> Result<(Vec<vk::Pipeline>, Option<PipelineCache>), PipelineCreationError> {
    let mut all_shader_modules = Vec::with_capacity(PIPELINE_PARAMETERS.len() * 2);
    let mut create_shader_module = |filename: &'static str, spirv: &'static [u32]| -> Result<vk::ShaderModule, PipelineCreationError> {
        let create_info = vk::ShaderModuleCreateInfo::builder().code(spirv);
        let shader_module = unsafe { device.create_shader_module(&create_info, None) }.map_err(PipelineCreationError::ShaderModule)?;
        debug_utils::name_vulkan_object(device, shader_module, format_args!("{}", filename));
        all_shader_modules.push(shader_module);
        Ok(shader_module)
    };

    let shader_stages_per_pipeline = PIPELINE_PARAMETERS
        .iter()
        .map(|params| {
            let vert_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(create_shader_module(params.vertex_shader_name, params.vertex_shader)?)
                .name(cstr!("main"));
            let frag_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(create_shader_module(params.fragment_shader_name, params.fragment_shader)?)
                .name(cstr!("main"));
            Ok([vert_shader_stage_create_info.build(), frag_shader_stage_create_info.build()])
        })
        .collect::<Result<Vec<[vk::PipelineShaderStageCreateInfo; 2]>, PipelineCreationError>>()?;

    let vertex_input_per_pipeline = PIPELINE_PARAMETERS
        .iter()
        .map(|params| {
            vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_binding_descriptions(params.bindings)
                .vertex_attribute_descriptions(params.attributes)
                .build()
        })
        .collect::<Vec<vk::PipelineVertexInputStateCreateInfo>>();

    let multisample_create_infos = PIPELINE_PARAMETERS
        .iter()
        .map(|params| {
            vk::PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(attachment_sample_count)
                .alpha_to_coverage_enable(params.alpha_to_coverage)
                .build()
        })
        .collect::<Vec<vk::PipelineMultisampleStateCreateInfo>>();

    let color_blend_attachment_states_per_pipeline = PIPELINE_PARAMETERS
        .iter()
        .map(|params| {
            let rgba_mask =
                vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B | vk::ColorComponentFlags::A;
            let mut blend_attachment_state_builder = vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(rgba_mask)
                .blend_enable(params.blended);
            if params.blended {
                blend_attachment_state_builder = blend_attachment_state_builder
                    .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                    .color_blend_op(vk::BlendOp::ADD)
                    .src_alpha_blend_factor(vk::BlendFactor::ONE)
                    .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                    .alpha_blend_op(vk::BlendOp::ADD);
            }
            [blend_attachment_state_builder.build()]
        })
        .collect::<Vec<[vk::PipelineColorBlendAttachmentState; 1]>>();

    let color_blend_create_infos = color_blend_attachment_states_per_pipeline
        .iter()
        .map(|color_blend_attachment_states| {
            vk::PipelineColorBlendStateCreateInfo::builder()
                .logic_op_enable(false)
                .attachments(&*color_blend_attachment_states)
                .build()
        })
        .collect::<Vec<vk::PipelineColorBlendStateCreateInfo>>();

    let input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let rasterization_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let pipeline_depth_stencil_create_info = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL);

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

    pipeline_cache = pipeline_cache.or_else(|| {
        // NOTE: Access to the PipelineCache is synchronized because
        // Pipelines-structs are created on one thread, and the cache is
        // per-Pipelines. Old Pipelines can't be inherited by multiple
        // Pipelines because they get moved, and Rust's ownership semantics
        // take care of the rest.
        let create_info = vk::PipelineCacheCreateInfo::builder().flags(vk::PipelineCacheCreateFlags::EXTERNALLY_SYNCHRONIZED);
        let pipeline_cache = unsafe { device.create_pipeline_cache(&create_info, None) }.ok()?;
        Some(PipelineCache {
            inner: pipeline_cache,
            device: device.clone(),
        })
    });
    let mut pipeline_create_infos = Vec::with_capacity(pipeline_layouts.len());
    for (i, pipeline_layout) in pipeline_layouts.iter().enumerate() {
        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages_per_pipeline[i][..])
            .vertex_input_state(&vertex_input_per_pipeline[i])
            .input_assembly_state(&input_assembly_create_info)
            .viewport_state(&viewport_create_info)
            .rasterization_state(&rasterization_create_info)
            .multisample_state(&multisample_create_infos[i])
            .depth_stencil_state(&pipeline_depth_stencil_create_info)
            .color_blend_state(&color_blend_create_infos[i])
            .layout(pipeline_layout.inner)
            .render_pass(render_pass)
            .subpass(0)
            .build();
        pipeline_create_infos.push(pipeline_create_info);
    }

    let pipelines = unsafe {
        device
            .create_graphics_pipelines(
                pipeline_cache.as_ref().map(|pc| pc.inner).unwrap_or_else(vk::PipelineCache::null),
                &pipeline_create_infos,
                None,
            )
            .map_err(|(_, err)| PipelineCreationError::Object(err))
    }?;

    for shader_module in all_shader_modules {
        unsafe { device.destroy_shader_module(shader_module, None) };
    }

    Ok((pipelines, pipeline_cache))
}
