use crate::debug_utils;
use crate::pipeline_parameters::{PipelineMap, PIPELINE_PARAMETERS};
use crate::vulkan_raii::{self, Device, PipelineLayout, RenderPass};
use crate::{Descriptors, PhysicalDevice};
use ash::vk;
use std::rc::Rc;

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
    pub(crate) attachment_sample_count: vk::SampleCountFlags,
    pub(crate) render_pass: Rc<RenderPass>,
    pub(crate) pipelines: PipelineMap<vulkan_raii::Pipeline>,
}

impl Pipelines {
    pub fn new(
        device: &Rc<Device>,
        physical_device: &PhysicalDevice,
        descriptors: &Descriptors,
    ) -> Result<Pipelines, PipelineCreationError> {
        let attachment_sample_count = vk::SampleCountFlags::TYPE_8;
        let final_render_pass = create_render_pass(
            device,
            physical_device.swapchain_format,
            physical_device.depth_format,
            attachment_sample_count,
        )?;
        debug_utils::name_vulkan_object(device, final_render_pass, format_args!("main render pass"));
        let final_render_pass = Rc::new(RenderPass {
            inner: final_render_pass,
            device: device.clone(),
        });

        let vk_pipelines = create_pipelines(
            device,
            final_render_pass.inner,
            &descriptors.pipeline_layouts,
            attachment_sample_count,
        )?;
        let mut vk_pipelines_iter = vk_pipelines.into_iter();
        let pipelines = PipelineMap::new::<PipelineCreationError, _>(|name| {
            let pipeline = vk_pipelines_iter.next().unwrap();
            debug_utils::name_vulkan_object(device, pipeline, format_args!("{name:?}"));
            Ok(vulkan_raii::Pipeline {
                inner: pipeline,
                device: device.clone(),
                render_pass: final_render_pass.clone(),
            })
        })?;
        Ok(Pipelines {
            attachment_sample_count,
            render_pass: final_render_pass,
            pipelines,
        })
    }
}

#[profiling::function]
fn create_render_pass(
    device: &Device,
    swapchain_format: vk::Format,
    depth_format: vk::Format,
    attachment_sample_count: vk::SampleCountFlags,
) -> Result<vk::RenderPass, PipelineCreationError> {
    let color_attachment = vk::AttachmentDescription::builder()
        .format(swapchain_format)
        .samples(attachment_sample_count)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let depth_attachment = vk::AttachmentDescription::builder()
        .format(depth_format)
        .samples(attachment_sample_count)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
    let resolve_attachment = vk::AttachmentDescription::builder()
        .format(swapchain_format)
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
    unsafe { device.create_render_pass(&render_pass_create_info, None) }.map_err(PipelineCreationError::RenderPass)
}

#[profiling::function]
fn create_pipelines(
    device: &Device,
    render_pass: vk::RenderPass,
    pipeline_layouts: &PipelineMap<PipelineLayout>,
    attachment_sample_count: vk::SampleCountFlags,
) -> Result<Vec<vk::Pipeline>, PipelineCreationError> {
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

    let pipelines = {
        let input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let rasterization_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        let multisample_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(attachment_sample_count);

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

        let viewport_create_info = vk::PipelineViewportStateCreateInfo::default();
        let dynamic_states = [vk::DynamicState::VIEWPORT_WITH_COUNT, vk::DynamicState::SCISSOR_WITH_COUNT];
        let dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&dynamic_states);

        let pipeline_create_infos = shader_stages_per_pipeline
            .iter()
            .zip(vertex_input_per_pipeline.iter())
            .zip(pipeline_layouts.iter())
            .map(|((shader_stages, vertex_input), pipeline_layout)| {
                vk::GraphicsPipelineCreateInfo::builder()
                    .stages(&shader_stages[..])
                    .vertex_input_state(vertex_input)
                    .input_assembly_state(&input_assembly_create_info)
                    .viewport_state(&viewport_create_info)
                    .rasterization_state(&rasterization_create_info)
                    .multisample_state(&multisample_create_info)
                    .depth_stencil_state(&pipeline_depth_stencil_create_info)
                    .color_blend_state(&color_blend_create_info)
                    .dynamic_state(&dynamic_state_create_info)
                    .layout(pipeline_layout.inner)
                    .render_pass(render_pass)
                    .subpass(0)
                    .build()
            })
            .collect::<Vec<vk::GraphicsPipelineCreateInfo>>();
        unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_create_infos, None)
                .map_err(|(_, err)| PipelineCreationError::Object(err))
        }?
    };

    for shader_module in all_shader_modules {
        unsafe { device.destroy_shader_module(shader_module, None) };
    }

    Ok(pipelines)
}
