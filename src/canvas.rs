use crate::{Error, Gpu};
use ash::extensions::khr;
use ash::version::DeviceV1_0;
use ash::{vk, Device};

/// The shorter-lived half of the rendering pair, along with [Gpu].
///
/// This struct has the concrete rendering objects, like the render
/// passes, framebuffers, command buffers and so on.
pub struct Canvas<'a> {
    /// Held by [Canvas] to ensure that the swapchain and command
    /// buffers are dropped before the device.
    pub gpu: &'a Gpu<'a>,

    pub(crate) swapchain: vk::SwapchainKHR,
    pub(crate) swapchain_image_views: Vec<vk::ImageView>,
    pub(crate) swapchain_framebuffers: Vec<vk::Framebuffer>,
    pub(crate) surface_pipeline_render_pass: vk::RenderPass,
    pub(crate) surface_pipeline: vk::Pipeline,
    pub(crate) command_buffers: Vec<vk::CommandBuffer>,

    surface_pipeline_layout: vk::PipelineLayout,
}

impl Drop for Canvas<'_> {
    fn drop(&mut self) {
        let device = &self.gpu.device;

        for framebuffer in &self.swapchain_framebuffers {
            unsafe { device.destroy_framebuffer(*framebuffer, None) };
        }

        unsafe {
            device.destroy_pipeline(self.surface_pipeline, None);
            device.destroy_pipeline_layout(self.surface_pipeline_layout, None);
            device.destroy_render_pass(self.surface_pipeline_render_pass, None);
            device.free_command_buffers(self.gpu.command_pool, &self.command_buffers);
        };

        for image_view in &self.swapchain_image_views {
            unsafe { device.destroy_image_view(*image_view, None) };
        }

        unsafe {
            self.gpu
                .swapchain_ext
                .destroy_swapchain(self.swapchain, None)
        };
    }
}

impl Canvas<'_> {
    pub fn new<'a>(
        gpu: &'a Gpu,
        old_canvas: Option<Canvas>,
        width: u32,
        height: u32,
    ) -> Result<Canvas<'a>, Error> {
        let device = &gpu.device;
        let swapchain_ext = &gpu.swapchain_ext;
        let queue_family_indices = [gpu.graphics_family_index, gpu.surface_family_index];
        let (swapchain, swapchain_format, final_extent) = create_swapchain(
            &gpu.surface_ext,
            &swapchain_ext,
            gpu.driver.surface,
            vk::Extent2D { width, height },
            old_canvas.map(|r| r.swapchain),
            gpu.physical_device,
            &queue_family_indices,
        )?;
        // The width and height may change from the ones passed in,
        // because they're queried during swapchain creation.
        let vk::Extent2D { width, height } = final_extent;

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

        let (surface_pipeline_layout, surface_pipeline_render_pass, surface_pipeline) =
            create_pipelines(device, width, height, swapchain_format)?;

        let swapchain_framebuffers = swapchain_image_views
            .iter()
            .map(|image_view| {
                let attachments = [*image_view];
                let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(surface_pipeline_render_pass)
                    .attachments(&attachments)
                    .width(width)
                    .height(height)
                    .layers(1);
                unsafe { device.create_framebuffer(&framebuffer_create_info, None) }
                    .map_err(Error::VulkanFramebufferCreation)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(gpu.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(swapchain_framebuffers.len() as u32);
        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .map_err(Error::VulkanCommandBuffersAllocation)
        }?;

        for (command_buffer, framebuffer) in command_buffers.iter().zip(&swapchain_framebuffers) {
            let command_buffer = *command_buffer;
            let framebuffer = *framebuffer;

            let begin_info = vk::CommandBufferBeginInfo::default();
            unsafe { device.begin_command_buffer(command_buffer, &begin_info) }
                .map_err(Error::VulkanBeginCommandBuffer)?;

            let render_area = vk::Rect2D::builder().extent(vk::Extent2D { width, height });
            let clear_colors = [vk::ClearValue::default()];
            let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
                .render_pass(surface_pipeline_render_pass)
                .framebuffer(framebuffer)
                .render_area(*render_area)
                .clear_values(&clear_colors);
            unsafe {
                device.cmd_begin_render_pass(
                    command_buffer,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );
                device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    surface_pipeline,
                );
                device.cmd_draw(command_buffer, 3, 1, 0, 0);
                device.cmd_end_render_pass(command_buffer);
            };

            unsafe { device.end_command_buffer(command_buffer) }
                .map_err(Error::VulkanEndCommandBuffer)?;
        }

        Ok(Canvas {
            gpu,
            swapchain,
            swapchain_image_views,
            swapchain_framebuffers,
            surface_pipeline_render_pass,
            surface_pipeline,
            command_buffers,
            surface_pipeline_layout,
        })
    }
}

fn create_swapchain(
    surface_ext: &khr::Surface,
    swapchain_ext: &khr::Swapchain,
    surface: vk::SurfaceKHR,
    extent: vk::Extent2D,
    old_swapchain: Option<vk::SwapchainKHR>,
    physical_device: vk::PhysicalDevice,
    queue_family_indices: &[u32],
) -> Result<(vk::SwapchainKHR, vk::Format, vk::Extent2D), Error> {
    let present_mode = vk::PresentModeKHR::FIFO;
    let min_image_count = 2;

    let (image_format, image_color_space) = {
        let surface_formats = unsafe {
            surface_ext
                .get_physical_device_surface_formats(physical_device, surface)
                .map_err(Error::VulkanPhysicalDeviceSurfaceQuery)
        }?;
        let format = if let Some(format) = surface_formats.iter().find(|format| {
            format.format == vk::Format::B8G8R8A8_SRGB
                && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        }) {
            format
        } else {
            &surface_formats[0]
        };
        (format.format, format.color_space)
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

    let mut swapchain_create_info = vk::SwapchainCreateInfoKHR {
        surface,
        min_image_count,
        image_format,
        image_color_space,
        image_array_layers: 1,
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
        composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        present_mode,
        clipped: vk::TRUE,
        ..Default::default()
    };
    if queue_family_indices.windows(2).any(|indices| {
        if let [a, b] = *indices {
            a != b
        } else {
            unreachable!()
        }
    }) {
        swapchain_create_info.image_sharing_mode = vk::SharingMode::CONCURRENT;
        swapchain_create_info.queue_family_index_count = queue_family_indices.len() as u32;
        swapchain_create_info.p_queue_family_indices = queue_family_indices.as_ptr();
    } else {
        swapchain_create_info.image_sharing_mode = vk::SharingMode::EXCLUSIVE;
    }
    if let Some(old_swapchain) = old_swapchain {
        swapchain_create_info.old_swapchain = old_swapchain;
    }
    // Get image extent at the latest possible time to avoid getting
    // an outdated extent. Bummer that this is an issue, but I haven't
    // found a good way to avoid it.
    swapchain_create_info.image_extent = get_image_extent()?;
    let swapchain = unsafe { swapchain_ext.create_swapchain(&swapchain_create_info, None) }
        .map_err(Error::VulkanSwapchainCreation)?;

    Ok((swapchain, image_format, swapchain_create_info.image_extent))
}

fn create_pipelines(
    device: &Device,
    width: u32,
    height: u32,
    format: vk::Format,
) -> Result<(vk::PipelineLayout, vk::RenderPass, vk::Pipeline), Error> {
    let vert_spirv = shaders::include_spirv!("shaders/triangle.vert");
    let frag_spirv = shaders::include_spirv!("shaders/triangle.frag");
    let vert_shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(vert_spirv);
    let frag_shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(frag_spirv);
    let vert_shader_module = unsafe {
        device
            .create_shader_module(&vert_shader_module_create_info, None)
            .map_err(Error::VulkanShaderModuleCreation)
    }?;
    let frag_shader_module = unsafe {
        device
            .create_shader_module(&frag_shader_module_create_info, None)
            .map_err(Error::VulkanShaderModuleCreation)
    }?;
    let vert_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_shader_module)
        .name(cstr!("main"));
    let frag_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_shader_module)
        .name(cstr!("main"));
    let shader_stages = [
        vert_shader_stage_create_info.build(),
        frag_shader_stage_create_info.build(),
    ];

    let surface_pipeline_layout = {
        // TODO: Insert/describe uniforms here?
        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default();

        unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }
            .map_err(Error::VulkanPipelineLayoutCreation)?
    };

    let surface_pipeline_render_pass = {
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

        let subpass_dependency = vk::SubpassDependency::builder()
            .dst_subpass(0) // The subpass at index 0 (surface_subpass) should wait before
            .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE) // writing to the color attachment
            .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT) // during the color output stage.
            .src_subpass(vk::SUBPASS_EXTERNAL) // Because whatever came before
            .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT) // might still be in the color output stage.
            .build();
        let dependencies = [subpass_dependency];

        let render_pass_create_info = vk::RenderPassCreateInfo::builder()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);
        unsafe { device.create_render_pass(&render_pass_create_info, None) }
            .map_err(Error::VulkanRenderPassCreation)?
    };

    let surface_pipeline = {
        let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
            .vertex_binding_descriptions(&[])
            .vertex_attribute_descriptions(&[]);

        let input_assembly_state_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewports = [vk::Viewport::builder()
            .width(width as f32)
            .height(height as f32)
            .min_depth(0.0)
            .max_depth(1.0)
            .build()];
        let scissors = [vk::Rect2D::builder()
            .extent(vk::Extent2D { width, height })
            .build()];
        let viewport_state_create_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        // NOTE: Shadow maps would want to configure this for clamping and biasing depth values
        let rasterization_state_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
            .line_width(1.0);

        // TODO: Add multisampling
        let multisample_state_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
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
        let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
            .logic_op_enable(false)
            .attachments(&color_blend_attachment_states);

        let surface_pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
            .stages(&shader_stages)
            .vertex_input_state(&vertex_input_state_create_info)
            .input_assembly_state(&input_assembly_state_create_info)
            .viewport_state(&viewport_state_create_info)
            .rasterization_state(&rasterization_state_create_info)
            .multisample_state(&multisample_state_create_info)
            .color_blend_state(&color_blend_state_create_info)
            .layout(surface_pipeline_layout)
            .render_pass(surface_pipeline_render_pass)
            .subpass(0);
        let pipeline_create_infos = [surface_pipeline_create_info.build()];
        let pipelines = unsafe {
            device
                .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_create_infos, None)
                .map_err(|(_, err)| Error::VulkanGraphicsPipelineCreation(err))
        }?;
        pipelines[0]
    };

    unsafe { device.destroy_shader_module(vert_shader_module, None) };
    unsafe { device.destroy_shader_module(frag_shader_module, None) };

    Ok((
        surface_pipeline_layout,
        surface_pipeline_render_pass,
        surface_pipeline,
    ))
}
