use crate::arena::VulkanArena;
use crate::debug_utils;
use crate::pipeline::{PipelineMap, PIPELINE_PARAMETERS};
use crate::vulkan_raii::{self, AnyImage, Device, Framebuffer, ImageView, PipelineLayout, RenderPass, Surface, Swapchain};
use crate::{Descriptors, Error, PhysicalDevice};
use ash::extensions::khr;
use ash::{vk, Entry, Instance};
use std::rc::Rc;

// According to vulkan.gpuinfo.org, this is the most supported depth
// attachment format by a large margin at 99.94%. If the accuracy is
// not enough, D32_SFLOAT comes in second at 87.78%, and if stencil is
// needed, D24_UNORM_S8_UINT is supported by 71.16%.
const DEPTH_FORMAT: vk::Format = vk::Format::D16_UNORM;

const SAMPLE_COUNT: vk::SampleCountFlags = vk::SampleCountFlags::TYPE_8;

fn is_uncompressed_srgb(format: vk::Format) -> bool {
    matches!(
        format,
        vk::Format::R8_SRGB
            | vk::Format::R8G8_SRGB
            | vk::Format::R8G8B8_SRGB
            | vk::Format::R8G8B8A8_SRGB
            | vk::Format::B8G8R8_SRGB
            | vk::Format::B8G8R8A8_SRGB
            | vk::Format::A8B8G8R8_SRGB_PACK32
    )
}

struct SwapchainSettings {
    extent: vk::Extent2D,
    immediate_present: bool,
}

/// The shorter-lived half of the rendering pair, along with [Gpu].
///
/// This struct has the concrete rendering objects, like the render
/// passes, framebuffers, command buffers and so on.
pub struct Canvas {
    pub extent: vk::Extent2D,
    pub frame_count: u32,

    pub(crate) swapchain: Rc<Swapchain>,
    pub(crate) framebuffers: Vec<Framebuffer>,
    pub(crate) final_render_pass: Rc<RenderPass>,
    pub(crate) pipelines: PipelineMap<vulkan_raii::Pipeline>,
}

impl Canvas {
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
    pub fn new(
        entry: &Entry,
        instance: &Instance,
        surface: &Rc<Surface>,
        device: &Rc<Device>,
        physical_device: &PhysicalDevice,
        descriptors: &Descriptors,
        old_canvas: Option<&Canvas>,
        fallback_width: u32,
        fallback_height: u32,
        immediate_present: bool,
    ) -> Result<Canvas, Error> {
        profiling::scope!("new_canvas");
        let surface_ext = khr::Surface::new(entry, instance);
        let swapchain_ext = khr::Swapchain::new(instance, &device.inner);
        let queue_family_indices = [
            physical_device.graphics_queue_family.index,
            physical_device.surface_queue_family.index,
        ];
        let (swapchain, swapchain_format, extent) = create_swapchain(
            &surface_ext,
            &swapchain_ext,
            surface.inner,
            old_canvas.map(|r| r.swapchain.inner),
            physical_device.inner,
            &queue_family_indices,
            &SwapchainSettings {
                extent: vk::Extent2D {
                    width: fallback_width,
                    height: fallback_height,
                },
                immediate_present,
            },
        )?;
        let vk::Extent2D { width, height } = extent;
        let swapchain_images = unsafe { swapchain_ext.get_swapchain_images(swapchain) }.map_err(Error::VulkanGetSwapchainImages)?;
        let frame_count = swapchain_images.len() as u32;
        debug_utils::name_vulkan_object(
            device,
            swapchain,
            format_args!("{width}x{height}, {swapchain_format:?}, {frame_count} frames"),
        );
        let swapchain = Rc::new(Swapchain {
            inner: swapchain,
            device: swapchain_ext,
            surface: surface.clone(),
        });

        // TODO(high): Split Canvas:
        // - Into Pipelines+RenderPass. Pipelines' viewport needs to be made dynamic.
        //   See: https://www.khronos.org/registry/vulkan/specs/1.3-extensions/man/html/vkCmdSetViewport.html
        // - Into Swapchain. (create_swapchain is probably enough.)
        // - Into Framebuffers.
        // Only the last two need to be recreated per resize, and framebuffers
        // should be destroyed before swapchains, and created after.
        let mut framebuffer_arena = VulkanArena::new(
            instance,
            device,
            physical_device.inner,
            1_000_000_000, // FIXME: query framebuffer memory requirements for arena
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::LAZILY_ALLOCATED,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            format_args!("framebuffer arena ({width}x{height}, {swapchain_format:?}, {frame_count} frames)"),
        )?;

        let create_image_view = |aspect_mask: vk::ImageAspectFlags, format: vk::Format| {
            move |image: AnyImage| -> Result<ImageView, Error> {
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
                    unsafe { device.create_image_view(&image_view_create_info, None) }.map_err(Error::VulkanSwapchainImageViewCreation)?;
                Ok(ImageView {
                    inner: image_view,
                    device: device.clone(),
                    image: Rc::new(image),
                })
            }
        };

        let mut create_image = |format: vk::Format, usage: vk::ImageUsageFlags| {
            let image_create_info = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(format)
                .extent(vk::Extent3D { width, height, depth: 1 })
                .mip_levels(1)
                .array_layers(1)
                .samples(SAMPLE_COUNT)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(usage);
            framebuffer_arena.create_image(*image_create_info, format_args!("tbd"))
        };

        // TODO(med): Add another set of images to render to, to allow for post processing
        // Also, consider: render to a linear/higher depth image, then map to SRGB for the swapchain?
        let swapchain_image_views = swapchain_images
            .into_iter()
            .map(|image| AnyImage::Swapchain(image, swapchain.clone()))
            .map(create_image_view(vk::ImageAspectFlags::COLOR, swapchain_format))
            .collect::<Result<Vec<_>, _>>()?;

        let color_images = (0..swapchain_image_views.len())
            .map(|_| create_image(swapchain_format, vk::ImageUsageFlags::COLOR_ATTACHMENT))
            .collect::<Result<Vec<_>, Error>>()?;
        let color_image_views = color_images
            .into_iter()
            .map(AnyImage::from)
            .map(create_image_view(vk::ImageAspectFlags::COLOR, swapchain_format))
            .collect::<Result<Vec<_>, _>>()?;

        let depth_images = (0..swapchain_image_views.len())
            .map(|_| create_image(DEPTH_FORMAT, vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT))
            .collect::<Result<Vec<_>, Error>>()?;
        let depth_image_views = depth_images
            .into_iter()
            .map(AnyImage::from)
            .map(create_image_view(vk::ImageAspectFlags::DEPTH, DEPTH_FORMAT))
            .collect::<Result<Vec<_>, _>>()?;

        for (((i, sc), color), depth) in swapchain_image_views
            .iter()
            .enumerate()
            .zip(color_image_views.iter())
            .zip(depth_image_views.iter())
        {
            let nth = i + 1;
            debug_utils::name_vulkan_object(device, sc.inner, format_args!("swapchain (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, sc.image.inner(), format_args!("swapchain (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, color.inner, format_args!("color fb (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, color.image.inner(), format_args!("color fb (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, depth.inner, format_args!("depth fb (frame {nth}/{frame_count})"));
            debug_utils::name_vulkan_object(device, depth.image.inner(), format_args!("depth fb (frame {nth}/{frame_count})"));
        }

        let final_render_pass = create_render_pass(device, swapchain_format)?;
        debug_utils::name_vulkan_object(device, final_render_pass, format_args!("main render pass"));
        let final_render_pass = Rc::new(RenderPass {
            inner: final_render_pass,
            device: device.clone(),
        });

        let mut vk_pipelines = create_pipelines(device, final_render_pass.inner, &descriptors.pipeline_layouts)?.into_iter();
        let pipelines = PipelineMap::new::<Error, _>(|name| {
            let pipeline = vk_pipelines.next().unwrap();
            debug_utils::name_vulkan_object(device, pipeline, format_args!("{name:?}"));
            Ok(vulkan_raii::Pipeline {
                inner: pipeline,
                device: device.clone(),
                render_pass: final_render_pass.clone(),
            })
        })?;

        let framebuffers = color_image_views
            .into_iter()
            .enumerate()
            .zip(depth_image_views.into_iter())
            .zip(swapchain_image_views.into_iter())
            .map(|(((i, color_image_view), depth_image_view), swapchain_image_view)| {
                let attachments = [color_image_view.inner, depth_image_view.inner, swapchain_image_view.inner];
                let framebuffer_create_info = vk::FramebufferCreateInfo::builder()
                    .render_pass(final_render_pass.inner)
                    .attachments(&attachments)
                    .width(extent.width)
                    .height(extent.height)
                    .layers(1);
                let framebuffer =
                    unsafe { device.create_framebuffer(&framebuffer_create_info, None) }.map_err(Error::VulkanFramebufferCreation)?;
                debug_utils::name_vulkan_object(device, framebuffer, format_args!("main fb {}/{frame_count}", i + 1));
                Ok(Framebuffer {
                    inner: framebuffer,
                    device: device.clone(),
                    render_pass: final_render_pass.clone(),
                    attachments: vec![color_image_view, depth_image_view, swapchain_image_view],
                })
            })
            .collect::<Result<Vec<Framebuffer>, Error>>()?;

        Ok(Canvas {
            extent,
            frame_count,
            swapchain,
            framebuffers,
            final_render_pass,
            pipelines,
        })
    }
}

#[profiling::function]
fn create_swapchain(
    surface_ext: &khr::Surface,
    swapchain_ext: &khr::Swapchain,
    surface: vk::SurfaceKHR,
    old_swapchain: Option<vk::SwapchainKHR>,
    physical_device: vk::PhysicalDevice,
    queue_family_indices: &[u32],
    settings: &SwapchainSettings,
) -> Result<(vk::SwapchainKHR, vk::Format, vk::Extent2D), Error> {
    let present_modes = unsafe { surface_ext.get_physical_device_surface_present_modes(physical_device, surface) }
        .map_err(Error::VulkanPhysicalDeviceSurfaceQuery)?;
    let mut present_mode = vk::PresentModeKHR::FIFO;
    if settings.immediate_present {
        // TODO(med): Remove immediate present, use proper gpu profiling instead.
        if present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            present_mode = vk::PresentModeKHR::MAILBOX;
        } else if present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
            present_mode = vk::PresentModeKHR::IMMEDIATE;
        }
    }

    let (image_format, image_color_space) = {
        let surface_formats = unsafe {
            surface_ext
                .get_physical_device_surface_formats(physical_device, surface)
                .map_err(Error::VulkanPhysicalDeviceSurfaceQuery)
        }?;
        if let Some(format) = surface_formats.iter().find(|format| is_uncompressed_srgb(format.format)) {
            (format.format, format.color_space)
        } else {
            log::warn!("No SRGB format found for swapchain. The image may look too dark.");
            (surface_formats[0].format, surface_formats[0].color_space)
        }
    };

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
        settings.extent
    };
    let mut min_image_count = 2.max(surface_capabilities.min_image_count);
    if surface_capabilities.max_image_count > 0 {
        min_image_count = min_image_count.min(surface_capabilities.max_image_count)
    }

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
        .image_extent(image_extent);
    if queue_family_indices[0] != queue_family_indices[1] {
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

    Ok((swapchain, image_format, swapchain_create_info.image_extent))
}

#[profiling::function]
fn create_render_pass(device: &Device, swapchain_format: vk::Format) -> Result<vk::RenderPass, Error> {
    let color_attachment = vk::AttachmentDescription::builder()
        .format(swapchain_format)
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
    unsafe { device.create_render_pass(&render_pass_create_info, None) }.map_err(Error::VulkanRenderPassCreation)
}

#[profiling::function]
fn create_pipelines(
    device: &Device,
    render_pass: vk::RenderPass,
    pipeline_layouts: &PipelineMap<PipelineLayout>,
) -> Result<Vec<vk::Pipeline>, Error> {
    let mut all_shader_modules = Vec::with_capacity(PIPELINE_PARAMETERS.len() * 2);
    let mut create_shader_module = |filename: &'static str, spirv: &'static [u32]| -> Result<vk::ShaderModule, Error> {
        let create_info = vk::ShaderModuleCreateInfo::builder().code(spirv);
        let shader_module = unsafe { device.create_shader_module(&create_info, None) }.map_err(Error::VulkanShaderModuleCreation)?;
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
        .collect::<Result<Vec<[vk::PipelineShaderStageCreateInfo; 2]>, Error>>()?;

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
                .map_err(|(_, err)| Error::VulkanGraphicsPipelineCreation(err))
        }?
    };

    for shader_module in all_shader_modules {
        unsafe { device.destroy_shader_module(shader_module, None) };
    }

    Ok(pipelines)
}
