use crate::physical_device::{PhysicalDevice, HDR_COLOR_ATTACHMENT_FORMAT};
use crate::renderer::descriptors::Descriptors;
use crate::vulkan_raii::{self, Device, PipelineCache, PipelineLayout, RenderPass};
use alloc::rc::Rc;
use arrayvec::ArrayVec;
use ash::vk;
use hashbrown::HashMap;

pub mod pipeline_parameters;

use pipeline_parameters::{PipelineIndex, PipelineMap, Shader, ALL_PIPELINES, PIPELINE_PARAMETERS};

pub enum AttachmentLayout {
    /// Attachments:
    /// - hdr color,
    /// - depth,
    /// - tonemapped color (presented),
    SingleSampled,
    /// Attachments:
    /// - hdr color (multisampled),
    /// - depth (multisampled),
    /// - tonemapped color (multisampled, resolve source),
    /// - present color (resolved, presented),
    MultiSampled,
}

#[derive(thiserror::Error, Debug, Clone, Copy)]
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
        device: &Device,
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
        crate::name_vulkan_object(device, render_pass, format_args!("main render pass"));
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
            crate::name_vulkan_object(device, pipeline, format_args!("{name:?}"));
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
    let hdr_index = 0;
    let depth_index = 1;
    let tonemapped_index = 2;
    let resolve_index = 3;

    let hdr_attachment = vk::AttachmentDescription2::default()
        .format(HDR_COLOR_ATTACHMENT_FORMAT)
        .samples(attachment_sample_count)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let hdr_pass_color_attachment_references = [vk::AttachmentReference2::default()
        .attachment(hdr_index)
        .layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
        .aspect_mask(vk::ImageAspectFlags::COLOR)];
    let tonemapping_pass_input_attachment_references = [vk::AttachmentReference2::default()
        .attachment(hdr_index)
        .layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .aspect_mask(vk::ImageAspectFlags::COLOR)];

    let depth_attachment = vk::AttachmentDescription2::default()
        .format(depth_format)
        .samples(attachment_sample_count)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL);
    let depth_attachment_reference = vk::AttachmentReference2::default()
        .attachment(depth_index)
        .layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
        .aspect_mask(vk::ImageAspectFlags::DEPTH);

    let hdr_subpass = vk::SubpassDescription2::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&hdr_pass_color_attachment_references)
        .depth_stencil_attachment(&depth_attachment_reference);

    let mut hdr_to_tonemapped_attachment_barrier = vk::MemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
        .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
        .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
        .dst_access_mask(vk::AccessFlags2::INPUT_ATTACHMENT_READ);
    let hdr_to_tonemapped_attachment_dependency = vk::SubpassDependency2::default()
        .push_next(&mut hdr_to_tonemapped_attachment_barrier)
        .src_subpass(0)
        .dst_subpass(1)
        .dependency_flags(vk::DependencyFlags::BY_REGION);
    let dependencies = [hdr_to_tonemapped_attachment_dependency];

    if attachment_sample_count == vk::SampleCountFlags::TYPE_1 {
        let tonemapped_color_attachment = vk::AttachmentDescription2::default()
            .format(swapchain_format)
            .samples(attachment_sample_count)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let tonemapping_pass_color_attachment_references = [vk::AttachmentReference2::default()
            .attachment(tonemapped_index)
            .layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .aspect_mask(vk::ImageAspectFlags::COLOR)];

        let tonemapping_subpass = vk::SubpassDescription2::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&tonemapping_pass_color_attachment_references)
            .input_attachments(&tonemapping_pass_input_attachment_references);

        let attachments = [hdr_attachment, depth_attachment, tonemapped_color_attachment];
        let subpasses = [hdr_subpass, tonemapping_subpass];
        let render_pass_create_info = vk::RenderPassCreateInfo2::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);
        let render_pass =
            unsafe { device.create_render_pass2(&render_pass_create_info, None) }.map_err(PipelineCreationError::RenderPass)?;
        Ok((render_pass, AttachmentLayout::SingleSampled))
    } else {
        let tonemapped_color_attachment = vk::AttachmentDescription2::default()
            .format(swapchain_format)
            .samples(attachment_sample_count)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL);
        let tonemapping_pass_color_attachment_references = [vk::AttachmentReference2::default()
            .attachment(tonemapped_index)
            .layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .aspect_mask(vk::ImageAspectFlags::COLOR)];

        let resolve_attachment = vk::AttachmentDescription2::default()
            .format(swapchain_format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let resolve_attachment_references = [vk::AttachmentReference2::default()
            .attachment(resolve_index)
            .layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .aspect_mask(vk::ImageAspectFlags::COLOR)];

        let tonemapping_subpass = vk::SubpassDescription2::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&tonemapping_pass_color_attachment_references)
            .resolve_attachments(&resolve_attachment_references)
            .input_attachments(&tonemapping_pass_input_attachment_references);

        let attachments = [hdr_attachment, depth_attachment, tonemapped_color_attachment, resolve_attachment];
        let subpasses = [hdr_subpass, tonemapping_subpass];
        let render_pass_create_info = vk::RenderPassCreateInfo2::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);
        let render_pass =
            unsafe { device.create_render_pass2(&render_pass_create_info, None) }.map_err(PipelineCreationError::RenderPass)?;
        Ok((render_pass, AttachmentLayout::MultiSampled))
    }
}

#[profiling::function]
fn create_pipelines(
    device: &Device,
    render_pass: vk::RenderPass,
    pipeline_layouts: &PipelineMap<PipelineLayout>,
    attachment_sample_count: vk::SampleCountFlags,
    extent: vk::Extent2D,
    mut pipeline_cache: Option<PipelineCache>,
) -> Result<(Vec<vk::Pipeline>, Option<PipelineCache>), PipelineCreationError> {
    let mut all_shader_modules = HashMap::with_capacity(PIPELINE_PARAMETERS.len() * 2);
    let mut create_shader_module = |(filename, spirv): (&'static str, &'static [u32])| -> Result<vk::ShaderModule, PipelineCreationError> {
        *all_shader_modules.entry((filename, spirv)).or_insert_with(|| {
            #[cfg(target_endian = "big")]
            let spirv = &ash::util::read_spv(&mut std::io::Cursor::new(bytemuck::cast_slice(spirv))).unwrap();
            let create_info = vk::ShaderModuleCreateInfo::default().code(spirv);
            let shader_module = unsafe { device.create_shader_module(&create_info, None) }.map_err(PipelineCreationError::ShaderModule)?;
            crate::name_vulkan_object(device, shader_module, format_args!("{}", filename));
            Ok(shader_module)
        })
    };

    let shader_stages_per_pipeline = PipelineMap::new(|pipeline| {
        let params = &PIPELINE_PARAMETERS[pipeline];
        let multisampled = attachment_sample_count != vk::SampleCountFlags::TYPE_1;
        let mut create_from_shader_variant = |shader| match shader {
            Shader::MsaaVariants { multi_sample: shader, .. } if multisampled => create_shader_module(shader),
            Shader::SingleVariant(shader) | Shader::MsaaVariants { single_sample: shader, .. } => create_shader_module(shader),
        };
        let vertex_module = create_from_shader_variant(params.vertex_shader)?;
        let fragment_module = create_from_shader_variant(params.fragment_shader)?;
        let vert_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vertex_module)
            .name(cstr!("main"));
        let frag_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(fragment_module)
            .name(cstr!("main"));
        Ok([vert_shader_stage_create_info, frag_shader_stage_create_info])
    })?;

    let vertex_input_per_pipeline = PipelineMap::new(|pipeline| {
        let params = &PIPELINE_PARAMETERS[pipeline];
        Ok(vk::PipelineVertexInputStateCreateInfo::default()
            .vertex_binding_descriptions(params.bindings)
            .vertex_attribute_descriptions(params.attributes))
    })?;

    let multisample_create_infos = PipelineMap::new(|pipeline| {
        let params = &PIPELINE_PARAMETERS[pipeline];
        Ok(vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(attachment_sample_count)
            .alpha_to_coverage_enable(params.alpha_to_coverage)
            .sample_shading_enable(params.sample_shading)
            .min_sample_shading(params.min_sample_shading_factor))
    })?;

    let color_blend_attachment_states_per_pipeline = PipelineMap::new(|pipeline| {
        let params = &PIPELINE_PARAMETERS[pipeline];
        let rgba_mask = vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B | vk::ColorComponentFlags::A;
        let mut blend_attachment_state_builder = vk::PipelineColorBlendAttachmentState::default()
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
        Ok([blend_attachment_state_builder])
    })?;

    let color_blend_create_infos = PipelineMap::new(|pipeline| {
        let color_blend_attachment_states = &color_blend_attachment_states_per_pipeline[pipeline];
        Ok(vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(color_blend_attachment_states))
    })?;

    let input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let rasterization_create_info = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);

    let pipeline_depth_stencil_create_infos = PipelineMap::new(|pipeline| {
        let params = &PIPELINE_PARAMETERS[pipeline];
        Ok(vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(params.depth_test)
            .depth_write_enable(params.depth_write)
            .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL))
    })?;

    let viewports = [vk::Viewport::default()
        .width(extent.width as f32)
        .height(extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0)];
    let scissors = [vk::Rect2D::default().extent(extent)];
    let viewport_create_info = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);

    let subpasses = PipelineMap::new(|pipeline| {
        let params = PIPELINE_PARAMETERS[pipeline];
        Ok(params.subpass)
    })?;

    pipeline_cache = pipeline_cache.or_else(|| {
        // NOTE: Access to the PipelineCache is synchronized because
        // Pipelines-structs are created on one thread, and the cache is
        // per-Pipelines. Old Pipelines can't be inherited by multiple
        // Pipelines because they get moved, and Rust's ownership semantics
        // take care of the rest.
        let create_info = vk::PipelineCacheCreateInfo::default().flags(vk::PipelineCacheCreateFlags::EXTERNALLY_SYNCHRONIZED);
        let pipeline_cache = unsafe { device.create_pipeline_cache(&create_info, None) }.ok()?;
        crate::name_vulkan_object(device, pipeline_cache, format_args!("all pipelines"));
        Some(PipelineCache {
            inner: pipeline_cache,
            device: device.clone(),
        })
    });
    let mut pipeline_create_infos = ArrayVec::<_, { PipelineIndex::Count as usize }>::new();
    for i in ALL_PIPELINES {
        let pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&shader_stages_per_pipeline[i][..])
            .vertex_input_state(&vertex_input_per_pipeline[i])
            .input_assembly_state(&input_assembly_create_info)
            .viewport_state(&viewport_create_info)
            .rasterization_state(&rasterization_create_info)
            .multisample_state(&multisample_create_infos[i])
            .depth_stencil_state(&pipeline_depth_stencil_create_infos[i])
            .color_blend_state(&color_blend_create_infos[i])
            .layout(pipeline_layouts[i].inner)
            .render_pass(render_pass)
            .subpass(subpasses[i]);
        pipeline_create_infos.push(pipeline_create_info);
    }

    let pipelines = unsafe {
        device
            .create_graphics_pipelines(
                pipeline_cache.as_ref().map_or_else(vk::PipelineCache::null, |pc| pc.inner),
                &pipeline_create_infos,
                None,
            )
            .map_err(|(_, err)| PipelineCreationError::Object(err))
    }?;

    for shader_module in all_shader_modules.values().flatten() {
        unsafe { device.destroy_shader_module(*shader_module, None) };
    }

    Ok((pipelines, pipeline_cache))
}
