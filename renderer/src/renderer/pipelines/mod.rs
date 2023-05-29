use crate::physical_device::{PhysicalDevice, HDR_COLOR_ATTACHMENT_FORMAT};
use crate::renderer::descriptors::Descriptors;
use crate::vulkan_raii::{self, Device, PipelineCache, PipelineLayout};
use arrayvec::ArrayVec;
use ash::vk;
use hashbrown::HashMap;

pub(crate) mod attachments;
pub(crate) mod framebuffers;
pub mod pipeline_parameters;

use attachments::Attachments;
use pipeline_parameters::{PipelineIndex, PipelineMap, Shader, ALL_PIPELINES, PIPELINE_PARAMETERS};

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
    pub(crate) pipelines: PipelineMap<vulkan_raii::Pipeline>,
    pub(crate) attachment_formats: Attachments<vk::Format>,
    pub(crate) attachment_sample_count: vk::SampleCountFlags,
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
        let attachment_formats = if attachment_sample_count == vk::SampleCountFlags::TYPE_1 {
            Attachments::SingleSampled {
                hdr: HDR_COLOR_ATTACHMENT_FORMAT,
                depth: physical_device.depth_format,
                present_tonemapped: physical_device.swapchain_format,
            }
        } else {
            Attachments::MultiSampled {
                hdr: HDR_COLOR_ATTACHMENT_FORMAT,
                depth: physical_device.depth_format,
                resolve_src_tonemapped: physical_device.swapchain_format,
            }
        };

        let (vk_pipelines, pipeline_cache) = create_pipelines(
            device,
            &descriptors.pipeline_layouts,
            attachment_sample_count,
            extent,
            &attachment_formats,
            old_pipelines.and_then(|pipelines| pipelines.pipeline_cache),
        )?;
        let mut vk_pipelines_iter = vk_pipelines.into_iter();
        let pipelines = PipelineMap::new::<PipelineCreationError, _>(|name| {
            let pipeline = vk_pipelines_iter.next().unwrap();
            crate::name_vulkan_object(device, pipeline, format_args!("{name:?}"));
            Ok(vulkan_raii::Pipeline {
                inner: pipeline,
                device: device.clone(),
            })
        })?;

        Ok(Pipelines {
            pipelines,
            attachment_formats,
            attachment_sample_count,
            pipeline_cache,
        })
    }
}

#[profiling::function]
fn create_pipelines(
    device: &Device,
    pipeline_layouts: &PipelineMap<PipelineLayout>,
    attachment_sample_count: vk::SampleCountFlags,
    extent: vk::Extent2D,
    attachment_layout: &Attachments<vk::Format>,
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

    let color_attachment_formats = attachment_layout
        .color_attachments()
        .into_iter()
        .map(|(_, format)| *format)
        .collect::<ArrayVec<vk::Format, { attachments::MAX_COLOR_ATTACHMENTS }>>();
    let depth_attachment_format = *attachment_layout.depth_attachment();
    let color_attachment_formats = PipelineMap::new(|pipeline| {
        let params = &PIPELINE_PARAMETERS[pipeline];
        let mut base = color_attachment_formats.clone();
        let mut ldr_index = 1;
        if !params.writes_hdr {
            base.remove(0);
            ldr_index -= 1;
        }
        if !params.writes_ldr {
            base.remove(ldr_index);
        }
        Ok(base)
    })?;
    let mut pipeline_rendering_create_infos = PipelineMap::new(|pipeline| {
        let params = &PIPELINE_PARAMETERS[pipeline];
        Ok(vk::PipelineRenderingCreateInfoKHR::default()
            .color_attachment_formats(&color_attachment_formats[pipeline])
            .depth_attachment_format(if params.writes_depth {
                depth_attachment_format
            } else {
                vk::Format::UNDEFINED
            }))
    })?;

    let color_blend_attachment_states_per_pipeline = PipelineMap::new(|pipeline| {
        let params = &PIPELINE_PARAMETERS[pipeline];
        let rgba_mask = vk::ColorComponentFlags::R | vk::ColorComponentFlags::G | vk::ColorComponentFlags::B | vk::ColorComponentFlags::A;
        let mut blend_attachment_state = vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(rgba_mask)
            .blend_enable(params.blended);
        if params.blended {
            blend_attachment_state = blend_attachment_state
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
                .alpha_blend_op(vk::BlendOp::ADD);
        }
        let mut blend_attachment_states = ArrayVec::<vk::PipelineColorBlendAttachmentState, { attachments::MAX_COLOR_ATTACHMENTS }>::new();
        blend_attachment_states.push(blend_attachment_state);
        while blend_attachment_states.len() < color_attachment_formats[pipeline].len() {
            blend_attachment_states.push(blend_attachment_state);
        }
        Ok(blend_attachment_states)
    })?;

    let color_blend_create_infos = PipelineMap::new(|pipeline| {
        Ok(vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(&color_blend_attachment_states_per_pipeline[pipeline]))
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
    for (&i, pipeline_rendering_create_info) in ALL_PIPELINES.iter().zip(pipeline_rendering_create_infos.iter_mut()) {
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
            .push_next(pipeline_rendering_create_info);
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
