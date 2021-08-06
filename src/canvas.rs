use crate::pipeline::{PipelineParameters, PIPELINE_PARAMETERS};
use crate::{Error, Gpu};
use ash::extensions::khr;
use ash::version::DeviceV1_0;
use ash::{vk, Device};

pub const SWAPCHAIN_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;

/// The shorter-lived half of the rendering pair, along with [Gpu].
///
/// This struct has the concrete rendering objects, like the render
/// passes, framebuffers, command buffers and so on.
pub struct Canvas<'a> {
    /// Held by [Canvas] to ensure that the swapchain and command
    /// buffers are dropped before the device.
    pub gpu: &'a Gpu<'a>,

    pub extent: vk::Extent2D,

    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) swapchain_image_views: Vec<vk::ImageView>,
    pub(crate) swapchain_framebuffers: Vec<vk::Framebuffer>,
    pub(crate) final_render_pass: vk::RenderPass,
    pub(crate) pipelines: Vec<vk::Pipeline>,
    pub(crate) command_buffers: Vec<vk::CommandBuffer>,
}

impl Drop for Canvas<'_> {
    #[profiling::function]
    fn drop(&mut self) {
        let device = &self.gpu.device;

        for &pipeline in &self.pipelines {
            unsafe { device.destroy_pipeline(pipeline, None) };
        }

        unsafe {
            device.destroy_render_pass(self.final_render_pass, None);
            device.free_command_buffers(self.gpu.command_pool, &self.command_buffers);
        }

        for &framebuffer in &self.swapchain_framebuffers {
            unsafe { device.destroy_framebuffer(framebuffer, None) };
        }

        for &image_view in &self.swapchain_image_views {
            unsafe { device.destroy_image_view(image_view, None) };
        }

        unsafe {
            self.gpu
                .swapchain_ext
                .destroy_swapchain(self.swapchain, None)
        };
    }
}

impl Canvas<'_> {
    /// Creates a new Canvas. Should be recreated when the window size
    /// changes.
    ///
    /// The fallback width and height parameters are used when Vulkan
    /// can't get the window size, e.g. when creating a new window in
    /// Wayland (when the window size is specified by the size of the
    /// initial framebuffer).
    #[profiling::function]
    pub fn new<'a>(
        gpu: &'a Gpu,
        old_canvas: Option<&Canvas>,
        fallback_width: u32,
        fallback_height: u32,
    ) -> Result<Canvas<'a>, Error> {
        let device = &gpu.device;
        let swapchain_ext = &gpu.swapchain_ext;
        let queue_family_indices = [gpu.graphics_family_index, gpu.surface_family_index];
        let (swapchain, swapchain_format, extent, frame_count) = create_swapchain(
            &gpu.surface_ext,
            &swapchain_ext,
            gpu.driver.surface,
            vk::Extent2D {
                width: fallback_width,
                height: fallback_height,
            },
            old_canvas.map(|r| r.swapchain),
            gpu.physical_device,
            &queue_family_indices,
        )?;

        gpu.set_frame_count(frame_count);

        // TODO: Add another set of images to render to, to allow for post processing
        // Also, consider: render to a linear/higher depth image, then map to SRGB for the swapchain?
        let swapchain_images = unsafe { swapchain_ext.get_swapchain_images(swapchain) }
            .map_err(Error::VulkanGetSwapchainImages)?;
        let swapchain_image_views = swapchain_images
            .into_iter()
            .map(|image| {
                let subresource_range = vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let image_view_create_info = vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: swapchain_format,
                    subresource_range,
                    ..Default::default()
                };
                unsafe { device.create_image_view(&image_view_create_info, None) }
                    .map_err(Error::VulkanSwapchainImageViewCreation)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let final_render_pass = create_render_pass(&device, crate::canvas::SWAPCHAIN_FORMAT)?;
        let pipelines = create_pipelines(
            &device,
            final_render_pass,
            extent,
            &gpu.descriptors.pipeline_layouts,
            &PIPELINE_PARAMETERS,
        )?;

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(gpu.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(gpu.frame_sync_objects_vec.len() as u32);
        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .map_err(Error::VulkanCommandBuffersAllocation)
        }?;

        let swapchain_framebuffers = swapchain_image_views
            .iter()
            .map(|image_view| {
                let attachments = [*image_view];
                let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(final_render_pass)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);
                unsafe { device.create_framebuffer(&framebuffer_create_info, None) }
                    .map_err(Error::VulkanFramebufferCreation)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Canvas {
            gpu,
            extent,
            swapchain,
            swapchain_image_views,
            swapchain_framebuffers,
            final_render_pass,
            pipelines,
            command_buffers,
        })
    }
}

#[profiling::function]
fn create_swapchain(
    surface_ext: &khr::Surface,
    swapchain_ext: &khr::Swapchain,
    surface: vk::SurfaceKHR,
    extent: vk::Extent2D,
    old_swapchain: Option<vk::SwapchainKHR>,
    physical_device: vk::PhysicalDevice,
    queue_family_indices: &[u32],
) -> Result<(vk::SwapchainKHR, vk::Format, vk::Extent2D, u32), Error> {
    // NOTE: The following combinations should be presented as a config option:
    // - FIFO + 2 (traditional double-buffered vsync)
    //   - no tearing, good latency, bad for perf when running under refresh rate
    // - FIFO + 3 (like double-buffering, but longer queue)
    //   - no tearing, bad latency, no perf issues when running under refresh rate
    // - MAILBOX + 3 (render-constantly, discard frames when waiting for vsync)
    //   - no tearing, great latency, optimal choice when available
    // - IMMEDIATE + 2 (render-constantly, ignore vsync (probably causes tearing))
    //   - possible tearing, best latency
    // With the non-available ones grayed out, of course.
    let present_mode = vk::PresentModeKHR::FIFO;
    let min_image_count = 3;

    let (image_format, image_color_space) = {
        let surface_formats = unsafe {
            surface_ext
                .get_physical_device_surface_formats(physical_device, surface)
                .map_err(Error::VulkanPhysicalDeviceSurfaceQuery)
        }?;
        let color_space = if let Some(format) = surface_formats
            .iter()
            .find(|format| format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
        {
            format.color_space
        } else {
            surface_formats[0].color_space
        };
        (SWAPCHAIN_FORMAT, color_space)
    };

    let get_image_extent = || -> Result<vk::Extent2D, Error> {
        let surface_capabilities = unsafe {
            surface_ext
                .get_physical_device_surface_capabilities(physical_device, surface)
                .map_err(Error::VulkanPhysicalDeviceSurfaceQuery)
        }?;
        let unset_extent = vk::Extent2D {
            width: u32::MAX,
            height: u32::MAX,
        };
        let image_extent = if surface_capabilities.current_extent != unset_extent {
            surface_capabilities.current_extent
        } else {
            extent
        };
        Ok(image_extent)
    };

    let mut swapchain_create_info = vk::SwapchainCreateInfoKHR::builder()
        .surface(surface)
        .min_image_count(min_image_count)
        .image_format(image_format)
        .image_color_space(image_color_space)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        .clipped(true)
        .image_extent(get_image_extent()?);
    if queue_family_indices.windows(2).any(|indices| {
        if let [a, b] = *indices {
            a != b
        } else {
            unreachable!()
        }
    }) {
        swapchain_create_info = swapchain_create_info
            .image_sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(&queue_family_indices);
    } else {
        swapchain_create_info =
            swapchain_create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE);
    }
    if let Some(old_swapchain) = old_swapchain {
        swapchain_create_info = swapchain_create_info.old_swapchain(old_swapchain);
    }
    let swapchain = unsafe { swapchain_ext.create_swapchain(&swapchain_create_info, None) }
        .map_err(Error::VulkanSwapchainCreation)?;

    Ok((
        swapchain,
        image_format,
        swapchain_create_info.image_extent,
        min_image_count,
    ))
}

#[profiling::function]
fn create_render_pass(device: &Device, format: vk::Format) -> Result<vk::RenderPass, Error> {
    let surface_color_attachment = vk::AttachmentDescription::builder()
        .format(format)
        .samples(vk::SampleCountFlags::TYPE_1) // NOTE: Multisampling
        .load_op(vk::AttachmentLoadOp::CLEAR) // NOTE: Shadow maps probably don't care
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);
    let attachments = [surface_color_attachment.build()];

    let surface_color_attachment_reference = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
    let attachment_references = [surface_color_attachment_reference.build()];
    let surface_subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(&attachment_references); // NOTE: resolve_attachments for multisampling?
    let subpasses = [surface_subpass.build()];

    // NOTE: This subpass dependency ensures that the layout of
    // the swapchain image is set up properly for rendering to
    // it. The spec says it should be inserted by the
    // implementation if not provided by the application, but
    // Android seems to be buggy in this regard. Source:
    //
    // https://www.reddit.com/r/vulkan/comments/701qqz/vk_subpass_external_presentation_question/dmzovoh/
    let color_attachment_write_dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .dst_subpass(0)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
        .build();
    let dependencies = [color_attachment_write_dependency];

    let render_pass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses)
        .dependencies(&dependencies);
    unsafe { device.create_render_pass(&render_pass_create_info, None) }
        .map_err(Error::VulkanRenderPassCreation)
}

#[profiling::function]
fn create_pipelines(
    device: &Device,
    render_pass: vk::RenderPass,
    extent: vk::Extent2D,
    pipeline_layouts: &[vk::PipelineLayout],
    pipelines_params: &[PipelineParameters],
) -> Result<Vec<vk::Pipeline>, Error> {
    let mut all_shader_modules = Vec::with_capacity(pipelines_params.len() * 2);
    let mut create_shader_module = |spirv: &'static [u32]| -> Result<vk::ShaderModule, Error> {
        let create_info = vk::ShaderModuleCreateInfo::builder().code(spirv);
        let shader_module = unsafe { device.create_shader_module(&create_info, None) }
            .map_err(Error::VulkanShaderModuleCreation)?;
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
            Ok([
                vert_shader_stage_create_info.build(),
                frag_shader_stage_create_info.build(),
            ])
        })
        .collect::<Result<Vec<[vk::PipelineShaderStageCreateInfo; 2]>, Error>>()?;

    let vertex_input_per_pipeline = pipelines_params
        .iter()
        .map(|pipeline| {
            vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_binding_descriptions(&pipeline.bindings)
                .vertex_attribute_descriptions(&pipeline.attributes)
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

        // NOTE: Shadow maps would want to configure this for clamping and biasing depth values
        let rasterization_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);

        // TODO: Add multisampling
        let multisample_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
            .sample_shading_enable(false)
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);

        // NOTE: Shadow maps may need a vk::PipelineDepthStencilStateCreateInfo

        let color_blend_attachment_states = [vk::PipelineColorBlendAttachmentState::builder()
            .color_write_mask(
                vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
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
                    .vertex_input_state(&vertex_input)
                    .input_assembly_state(&input_assembly_create_info)
                    .viewport_state(&viewport_create_info)
                    .rasterization_state(&rasterization_create_info)
                    .multisample_state(&multisample_create_info)
                    .color_blend_state(&color_blend_create_info)
                    .layout(pipeline_layout)
                    .render_pass(render_pass)
                    .subpass(0)
                    .build()
            })
            .collect::<Vec<vk::GraphicsPipelineCreateInfo>>();
        unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_create_infos, None)
                .map_err(|(_, err)| Error::VulkanGraphicsPipelineCreation(err))
        }?
    };

    for shader_module in all_shader_modules {
        unsafe { device.destroy_shader_module(shader_module, None) };
    }

    Ok(pipelines)
}
