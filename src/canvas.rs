use crate::pipeline::{PipelineParameters, PIPELINE_PARAMETERS};
use crate::{Error, Gpu};
use ash::extensions::khr;
use ash::version::DeviceV1_0;
use ash::{vk, Device};

pub const SWAPCHAIN_FORMAT: vk::Format = vk::Format::B8G8R8A8_SRGB;

// According to vulkan.gpuinfo.org, this is the most supported depth
// attachment format by a large margin at 99.94%. If the accuracy is
// not enough, D32_SFLOAT comes in second at 87.78%, and if stencil is
// needed, D24_UNORM_S8_UINT is supported by 71.16%.
const DEPTH_FORMAT: vk::Format = vk::Format::D16_UNORM;

const SAMPLE_COUNT: vk::SampleCountFlags = vk::SampleCountFlags::TYPE_8;

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
    swapchain_image_views: Vec<vk::ImageView>,
    color_images: Vec<(vk::Image, vk_mem::Allocation)>,
    color_image_views: Vec<vk::ImageView>,
    depth_images: Vec<(vk::Image, vk_mem::Allocation)>,
    depth_image_views: Vec<vk::ImageView>,
    pub(crate) framebuffers: Vec<vk::Framebuffer>,
    pub(crate) final_render_pass: vk::RenderPass,
    pub(crate) pipelines: Vec<vk::Pipeline>,
    pub(crate) command_buffers: Vec<vk::CommandBuffer>,
}

impl Drop for Canvas<'_> {
    #[profiling::function]
    fn drop(&mut self) {
        let device = &self.gpu.device;

        for &pipeline in &self.pipelines {
            profiling::scope!("destroy pipeline");
            unsafe { device.destroy_pipeline(pipeline, None) };
        }

        {
            profiling::scope!("destroy render pass");
            unsafe { device.destroy_render_pass(self.final_render_pass, None) };
        }

        {
            profiling::scope!("free command buffers");
            unsafe { device.free_command_buffers(self.gpu.command_pool, &self.command_buffers) };
        }

        for &framebuffer in &self.framebuffers {
            profiling::scope!("destroy framebuffers");
            unsafe { device.destroy_framebuffer(framebuffer, None) };
        }

        for &image_view in &self.depth_image_views {
            profiling::scope!("destroy depth image view");
            unsafe { device.destroy_image_view(image_view, None) };
        }

        for (image, allocation) in &self.depth_images {
            profiling::scope!("destroy depth image");
            let _ = self.gpu.allocator.destroy_image(*image, allocation);
        }

        for &image_view in &self.color_image_views {
            profiling::scope!("destroy main render target image view");
            unsafe { device.destroy_image_view(image_view, None) };
        }

        for (image, allocation) in &self.color_images {
            profiling::scope!("destroy main render target image");
            let _ = self.gpu.allocator.destroy_image(*image, allocation);
        }

        for &image_view in &self.swapchain_image_views {
            profiling::scope!("destroy swapchain image view");
            unsafe { device.destroy_image_view(image_view, None) };
        }

        unsafe {
            profiling::scope!("destroy swapchain");
            self.gpu.swapchain_ext.destroy_swapchain(self.swapchain, None)
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
    ///
    /// If `immediate_present` is true, the immediate present mode is
    /// used. Otherwise, FIFO. FIFO only releases frames after they've
    /// been displayed on screen, so it caps the fps to the screen's
    /// refresh rate.
    #[profiling::function]
    pub fn new<'a>(
        gpu: &'a Gpu,
        old_canvas: Option<&Canvas>,
        fallback_width: u32,
        fallback_height: u32,
        immediate_present: bool,
    ) -> Result<Canvas<'a>, Error> {
        let device = &gpu.device;
        let swapchain_ext = &gpu.swapchain_ext;
        let queue_family_indices = [gpu.graphics_family_index, gpu.surface_family_index];
        let (swapchain, swapchain_format, extent, frame_count) = create_swapchain(
            &gpu.surface_ext,
            swapchain_ext,
            gpu.driver.surface,
            vk::Extent2D {
                width: fallback_width,
                height: fallback_height,
            },
            old_canvas.map(|r| r.swapchain),
            gpu.physical_device,
            &queue_family_indices,
            immediate_present,
        )?;

        let create_image_view = |aspect_mask: vk::ImageAspectFlags, format: vk::Format| {
            move |image: vk::Image| -> Result<vk::ImageView, Error> {
                let subresource_range = vk::ImageSubresourceRange {
                    aspect_mask,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                };
                let image_view_create_info = vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format,
                    subresource_range,
                    ..Default::default()
                };
                unsafe { device.create_image_view(&image_view_create_info, None) }.map_err(Error::VulkanSwapchainImageViewCreation)
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
            let allocation_create_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::GpuOnly,
                flags: vk_mem::AllocationCreateFlags::STRATEGY_MIN_FRAGMENTATION,
                ..Default::default()
            };
            gpu.allocator.create_image(&image_create_info, &allocation_create_info)
        };

        // TODO: Add another set of images to render to, to allow for post processing
        // Also, consider: render to a linear/higher depth image, then map to SRGB for the swapchain?
        let swapchain_images = unsafe { swapchain_ext.get_swapchain_images(swapchain) }.map_err(Error::VulkanGetSwapchainImages)?;
        let swapchain_image_views = swapchain_images
            .into_iter()
            .map(create_image_view(vk::ImageAspectFlags::COLOR, swapchain_format))
            .collect::<Result<Vec<_>, _>>()?;

        let color_images = (0..swapchain_image_views.len())
            .map(|_| {
                let (image, allocation, _) =
                    create_image(SWAPCHAIN_FORMAT, vk::ImageUsageFlags::COLOR_ATTACHMENT).map_err(Error::VmaColorImageCreation)?;
                Ok((image, allocation))
            })
            .collect::<Result<Vec<(vk::Image, vk_mem::Allocation)>, Error>>()?;
        let color_image_views = color_images
            .iter()
            .map(|&(image, _)| image)
            .map(create_image_view(vk::ImageAspectFlags::COLOR, SWAPCHAIN_FORMAT))
            .collect::<Result<Vec<_>, _>>()?;

        let depth_images = (0..swapchain_image_views.len())
            .map(|_| {
                let (image, allocation, _) =
                    create_image(DEPTH_FORMAT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT).map_err(Error::VmaDepthImageCreation)?;
                Ok((image, allocation))
            })
            .collect::<Result<Vec<(vk::Image, vk_mem::Allocation)>, Error>>()?;
        let depth_image_views = depth_images
            .iter()
            .map(|&(image, _)| image)
            .map(create_image_view(vk::ImageAspectFlags::DEPTH, DEPTH_FORMAT))
            .collect::<Result<Vec<_>, _>>()?;

        let final_render_pass = create_render_pass(device)?;
        let pipelines = create_pipelines(
            device,
            final_render_pass,
            extent,
            &gpu.descriptors.pipeline_layouts,
            &PIPELINE_PARAMETERS,
        )?;

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(gpu.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(frame_count);
        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .map_err(Error::VulkanCommandBuffersAllocation)
        }?;

        let framebuffers = color_image_views
            .iter()
            .zip(depth_image_views.iter())
            .zip(swapchain_image_views.iter())
            .map(|((&color_image_view, &depth_image_view), &swapchain_image_view)| {
                let attachments = [color_image_view, depth_image_view, swapchain_image_view];
                let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(final_render_pass)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);
                unsafe { device.create_framebuffer(&framebuffer_create_info, None) }.map_err(Error::VulkanFramebufferCreation)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Canvas {
            gpu,
            extent,
            swapchain,
            swapchain_image_views,
            color_images,
            color_image_views,
            depth_images,
            depth_image_views,
            framebuffers,
            final_render_pass,
            pipelines,
            command_buffers,
        })
    }
}

#[allow(clippy::too_many_arguments)]
#[profiling::function]
fn create_swapchain(
    surface_ext: &khr::Surface,
    swapchain_ext: &khr::Swapchain,
    surface: vk::SurfaceKHR,
    extent: vk::Extent2D,
    old_swapchain: Option<vk::SwapchainKHR>,
    physical_device: vk::PhysicalDevice,
    queue_family_indices: &[u32],
    immediate_present: bool,
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
    let present_mode = if immediate_present {
        vk::PresentModeKHR::IMMEDIATE
    } else {
        vk::PresentModeKHR::FIFO
    };
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
    if queue_family_indices
        .windows(2)
        .any(|indices| if let [a, b] = *indices { a != b } else { unreachable!() })
    {
        swapchain_create_info = swapchain_create_info
            .image_sharing_mode(vk::SharingMode::CONCURRENT)
            .queue_family_indices(queue_family_indices);
    } else {
        swapchain_create_info = swapchain_create_info.image_sharing_mode(vk::SharingMode::EXCLUSIVE);
    }
    if let Some(old_swapchain) = old_swapchain {
        swapchain_create_info = swapchain_create_info.old_swapchain(old_swapchain);
    }
    let swapchain = unsafe { swapchain_ext.create_swapchain(&swapchain_create_info, None) }.map_err(Error::VulkanSwapchainCreation)?;

    Ok((swapchain, image_format, swapchain_create_info.image_extent, min_image_count))
}

#[profiling::function]
fn create_render_pass(device: &Device) -> Result<vk::RenderPass, Error> {
    let color_attachment = vk::AttachmentDescription::builder()
        .format(SWAPCHAIN_FORMAT)
        .samples(SAMPLE_COUNT)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::STORE)
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
        .format(SWAPCHAIN_FORMAT)
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
    unsafe { device.create_render_pass(&render_pass_create_info, None) }.map_err(Error::VulkanRenderPassCreation)
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
            .depth_compare_op(vk::CompareOp::LESS);

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
