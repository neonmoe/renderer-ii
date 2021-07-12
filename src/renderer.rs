use crate::{Error, Foundation};
use ash::extensions::{ext, khr};
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk::Handle;
use ash::{vk, Device, Instance};
use std::ffi::CStr;
use std::os::raw::c_char;

struct Pipeline {
    layouts: Vec<vk::PipelineLayout>,
    render_passes: Vec<vk::RenderPass>,
    pipelines: Vec<vk::Pipeline>,
}

#[allow(dead_code)]
pub struct Renderer<'a> {
    /// Held by Renderer to ensure that the Devices are dropped before
    /// the Instance.
    pub foundation: &'a Foundation,
    /// A list of suitable physical devices, for picking between
    /// e.g. a laptop's integrated and discrete GPUs.
    ///
    /// The tuple consists of the display name, and the id passed to a
    /// new Renderer when recreating it with a new physical device.
    pub physical_devices: Vec<(String, [u8; 16])>,

    device: Device,
    swapchain: vk::SwapchainKHR,
    swapchain_image_views: Vec<vk::ImageView>,
    graphics_queue: vk::Queue,
    graphics_family_index: u32,
    surface_queue: vk::Queue,
    surface_family_index: u32,
    pipelines: Vec<Pipeline>,
}

impl Drop for Renderer<'_> {
    fn drop(&mut self) {
        // TODO: Can use allocator
        for pipeline in &self.pipelines {
            for pipeline in &pipeline.pipelines {
                unsafe { self.device.destroy_pipeline(*pipeline, None) };
            }
            for layout in &pipeline.layouts {
                unsafe { self.device.destroy_pipeline_layout(*layout, None) };
            }
            for render_pass in &pipeline.render_passes {
                unsafe { self.device.destroy_render_pass(*render_pass, None) };
            }
        }

        for image_view in &self.swapchain_image_views {
            // TODO: Can use allocator
            unsafe { self.device.destroy_image_view(*image_view, None) };
        }

        let swapchain_ext = khr::Swapchain::new(&self.foundation.instance, &self.device);
        // TODO: Can use allocator
        unsafe { swapchain_ext.destroy_swapchain(self.swapchain, None) };

        // TODO: Can use allocator
        unsafe { self.device.destroy_device(None) };
    }
}

impl Renderer<'_> {
    pub fn new(
        foundation: &Foundation,
        initial_width: u32,
        initial_height: u32,
        preferred_physical_device: Option<[u8; 16]>,
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> Result<Renderer<'_>, Error> {
        let surface_ext = khr::Surface::new(&foundation.entry, &foundation.instance);
        let queue_family_supports_surface = |pd: vk::PhysicalDevice, index: u32| {
            let support = unsafe {
                surface_ext.get_physical_device_surface_support(pd, index, foundation.surface)
            };
            if let Ok(true) = support {
                true
            } else {
                false
            }
        };

        let all_physical_devices = unsafe { foundation.instance.enumerate_physical_devices() }
            .map_err(|err| Error::VulkanEnumeratePhysicalDevices(err))?;
        let mut physical_devices = all_physical_devices
            .into_iter()
            .filter_map(|physical_device| {
                if !is_extension_supported(
                    &foundation.instance,
                    physical_device,
                    "VK_KHR_swapchain",
                ) {
                    return None;
                }
                let queue_families = unsafe {
                    foundation
                        .instance
                        .get_physical_device_queue_family_properties(physical_device)
                };
                let mut graphics_family_index = None;
                let mut surface_family_index = None;
                for (index, family_index) in queue_families.into_iter().enumerate() {
                    if family_index.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                        graphics_family_index = Some(index as u32);
                    }
                    if queue_family_supports_surface(physical_device, index as u32) {
                        surface_family_index = Some(index as u32);
                    }
                    if graphics_family_index == surface_family_index {
                        // If there's a queue which supports both, prefer that one.
                        break;
                    }
                }
                if let (Some(graphics_family_index), Some(surface_family_index)) =
                    (graphics_family_index, surface_family_index)
                {
                    Some((physical_device, graphics_family_index, surface_family_index))
                } else {
                    None
                }
            })
            .collect::<Vec<(vk::PhysicalDevice, u32, u32)>>();

        let (physical_device, graphics_family_index, surface_family_index) =
            if let Some(uuid) = preferred_physical_device {
                physical_devices
                    .iter()
                    .find_map(|tuple| {
                        let properties =
                            unsafe { foundation.instance.get_physical_device_properties(tuple.0) };
                        if properties.pipeline_cache_uuid == uuid {
                            Some(*tuple)
                        } else {
                            None
                        }
                    })
                    .ok_or(Error::VulkanPhysicalDeviceMissing)?
            } else {
                physical_devices.sort_by(|(a, a_gfx, a_surf), (b, b_gfx, b_surf)| {
                    let a_properties =
                        unsafe { foundation.instance.get_physical_device_properties(*a) };
                    let b_properties =
                        unsafe { foundation.instance.get_physical_device_properties(*b) };
                    let type_score =
                        |properties: vk::PhysicalDeviceProperties| match properties.device_type {
                            vk::PhysicalDeviceType::DISCRETE_GPU => 30,
                            vk::PhysicalDeviceType::INTEGRATED_GPU => 20,
                            vk::PhysicalDeviceType::VIRTUAL_GPU => 10,
                            vk::PhysicalDeviceType::CPU => 0,
                            _ => 0,
                        };
                    let queue_score = |graphics_queue, surface_queue| {
                        if graphics_queue == surface_queue {
                            1
                        } else {
                            0
                        }
                    };
                    let a_score = type_score(a_properties) + queue_score(a_gfx, a_surf);
                    let b_score = type_score(b_properties) + queue_score(b_gfx, b_surf);
                    // Highest score first.
                    b_score.cmp(&a_score)
                });
                physical_devices
                    .iter()
                    .next()
                    .map(|t| *t)
                    .ok_or(Error::VulkanPhysicalDeviceMissing)?
            };

        let physical_devices = physical_devices
            .into_iter()
            .map(|(pd, _, _)| {
                let properties = unsafe { foundation.instance.get_physical_device_properties(pd) };
                let name = unsafe { CStr::from_ptr((&properties.device_name[..]).as_ptr()) }
                    .to_string_lossy();
                let pd_type = match properties.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => " (Discrete GPU)",
                    vk::PhysicalDeviceType::INTEGRATED_GPU => " (Integrated GPU)",
                    vk::PhysicalDeviceType::VIRTUAL_GPU => " (vCPU)",
                    vk::PhysicalDeviceType::CPU => " (CPU)",
                    _ => "",
                };
                let name = format!("{}{}", name, pd_type);
                log::debug!("Physical device found: {}", name);
                let uuid = properties.pipeline_cache_uuid;
                (name, uuid)
            })
            .collect::<Vec<(String, [u8; 16])>>();

        let queue_priorities = [1.0, 1.0];
        let queue_create_infos = if graphics_family_index == surface_family_index {
            vec![vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(graphics_family_index)
                .queue_priorities(&queue_priorities)
                .build()]
        } else {
            vec![
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(graphics_family_index)
                    .queue_priorities(&queue_priorities[0..1])
                    .build(),
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(surface_family_index)
                    .queue_priorities(&queue_priorities[1..2])
                    .build(),
            ]
        };
        let physical_device_features = vk::PhysicalDeviceFeatures::default();
        let extensions = &[cstr!("VK_KHR_swapchain").as_ptr()];
        if log::log_enabled!(log::Level::Debug) {
            let cstr_to_str =
                |str_ptr: &*const c_char| unsafe { CStr::from_ptr(*str_ptr) }.to_string_lossy();
            log::debug!(
                "Requested device extensions: {:?}",
                extensions.iter().map(cstr_to_str).collect::<Vec<_>>()
            );
        }

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&physical_device_features)
            .enabled_extension_names(extensions);
        // TODO: Can use allocator
        let allocation_callbacks = None;
        let device = unsafe {
            foundation
                .instance
                .create_device(physical_device, &device_create_info, allocation_callbacks)
                .map_err(|err| Error::VulkanDeviceCreation(err))
        }?;

        let graphics_queue;
        let surface_queue;
        if graphics_family_index == surface_family_index {
            graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };
            surface_queue = unsafe { device.get_device_queue(graphics_family_index, 1) };
        } else {
            graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };
            surface_queue = unsafe { device.get_device_queue(surface_family_index, 0) };
        }
        if foundation.debug_utils_available {
            let debug_utils_ext = ext::DebugUtils::new(&foundation.entry, &foundation.instance);
            let graphics_name_info = vk::DebugUtilsObjectNameInfoEXT {
                object_type: vk::ObjectType::QUEUE,
                object_handle: graphics_queue.as_raw(),
                p_object_name: cstr!("Graphics Queue").as_ptr(),
                ..Default::default()
            };
            let surface_name_info = vk::DebugUtilsObjectNameInfoEXT {
                object_type: vk::ObjectType::QUEUE,
                object_handle: surface_queue.as_raw(),
                p_object_name: cstr!("Surface Presentation Queue").as_ptr(),
                ..Default::default()
            };
            unsafe {
                let _ = debug_utils_ext
                    .debug_utils_set_object_name(device.handle(), &graphics_name_info);
                let _ = debug_utils_ext
                    .debug_utils_set_object_name(device.handle(), &surface_name_info);
            }
        }

        let (swapchain, swapchain_image_views, swapchain_format) =
            create_swapchain_and_image_views(
                foundation,
                initial_width,
                initial_height,
                old_swapchain,
                physical_device,
                &device,
                graphics_family_index,
                surface_family_index,
            )?;

        let pipelines = vec![create_pipeline(
            &device,
            initial_width,
            initial_height,
            swapchain_format,
        )?];

        Ok(Renderer {
            foundation,
            physical_devices,
            device,
            swapchain,
            swapchain_image_views,
            graphics_queue,
            graphics_family_index,
            surface_queue,
            surface_family_index,
            pipelines,
        })
    }
}

fn is_extension_supported(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    target_extension_name: &str,
) -> bool {
    match unsafe { instance.enumerate_device_extension_properties(physical_device) } {
        Err(_) => false,
        Ok(extensions) => extensions.iter().any(|extension_properties| {
            let extension_name_slice = &extension_properties.extension_name[..];
            let extension_name =
                unsafe { CStr::from_ptr(extension_name_slice.as_ptr()) }.to_string_lossy();
            extension_name == target_extension_name
        }),
    }
}

fn create_swapchain_and_image_views(
    foundation: &Foundation,
    width: u32,
    height: u32,
    old_swapchain: Option<vk::SwapchainKHR>,
    physical_device: vk::PhysicalDevice,
    device: &Device,
    graphics_family_index: u32,
    surface_family_index: u32,
) -> Result<(vk::SwapchainKHR, Vec<vk::ImageView>, vk::Format), Error> {
    let surface_ext = khr::Surface::new(&foundation.entry, &foundation.instance);
    let present_mode = {
        let surface_present_modes = unsafe {
            surface_ext
                .get_physical_device_surface_present_modes(physical_device, foundation.surface)
                .map_err(|err| Error::VulkanPhysicalDeviceSurfaceQuery(err))
        }?;
        if surface_present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
            vk::PresentModeKHR::MAILBOX
        } else {
            vk::PresentModeKHR::FIFO
        }
    };

    let (min_image_count, image_extent) = {
        let surface_capabilities = unsafe {
            surface_ext
                .get_physical_device_surface_capabilities(physical_device, foundation.surface)
                .map_err(|err| Error::VulkanPhysicalDeviceSurfaceQuery(err))
        }?;
        let mut min_image_count = if present_mode == vk::PresentModeKHR::MAILBOX {
            3.max(surface_capabilities.min_image_count)
        } else {
            2.max(surface_capabilities.min_image_count)
        };
        if surface_capabilities.max_image_count != 0
            && surface_capabilities.max_image_count < min_image_count
        {
            min_image_count = surface_capabilities.max_image_count;
        }
        let unset_extent = vk::Extent2D {
            width: u32::MAX,
            height: u32::MAX,
        };
        let image_extent = if surface_capabilities.current_extent != unset_extent {
            surface_capabilities.current_extent
        } else {
            vk::Extent2D { width, height }
        };
        (min_image_count, image_extent)
    };

    let (image_format, image_color_space) = {
        let surface_formats = unsafe {
            surface_ext
                .get_physical_device_surface_formats(physical_device, foundation.surface)
                .map_err(|err| Error::VulkanPhysicalDeviceSurfaceQuery(err))
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

    let swapchain_ext = khr::Swapchain::new(&foundation.instance, device);
    let mut swapchain_create_info = vk::SwapchainCreateInfoKHR {
        surface: foundation.surface,
        min_image_count,
        image_format,
        image_color_space,
        image_extent,
        image_array_layers: 1,
        image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT,
        pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
        composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
        present_mode,
        clipped: vk::TRUE,
        ..Default::default()
    };
    let queue_family_indices = &[graphics_family_index, surface_family_index];
    if graphics_family_index != surface_family_index {
        swapchain_create_info.image_sharing_mode = vk::SharingMode::CONCURRENT;
        swapchain_create_info.queue_family_index_count = queue_family_indices.len() as u32;
        swapchain_create_info.p_queue_family_indices = queue_family_indices.as_ptr();
    } else {
        swapchain_create_info.image_sharing_mode = vk::SharingMode::EXCLUSIVE;
    }
    if let Some(old_swapchain) = old_swapchain {
        swapchain_create_info.old_swapchain = old_swapchain;
    }
    // TODO: Can use allocator
    let swapchain = unsafe { swapchain_ext.create_swapchain(&swapchain_create_info, None) }
        .map_err(|err| Error::VulkanSwapchainCreation(err))?;

    let swapchain_images = unsafe { swapchain_ext.get_swapchain_images(swapchain) }
        .map_err(|err| Error::VulkanGetSwapchainImages(err))?;
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
                format: image_format,
                subresource_range,
                ..Default::default()
            };
            // TODO: Can use allocator
            unsafe { device.create_image_view(&image_view_create_info, None) }
                .map_err(|err| Error::VulkanSwapchainImageViewCreation(err))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok((swapchain, swapchain_image_views, image_format))
}

fn create_pipeline(
    device: &Device,
    width: u32,
    height: u32,
    format: vk::Format,
) -> Result<Pipeline, Error> {
    let vert_spirv = shaders::include_spirv!("shaders/triangle.vert");
    let frag_spirv = shaders::include_spirv!("shaders/triangle.frag");
    let vert_shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(vert_spirv);
    let frag_shader_module_create_info = vk::ShaderModuleCreateInfo::builder().code(frag_spirv);
    // TODO: Can use allocator
    let vert_shader_module =
        unsafe { device.create_shader_module(&vert_shader_module_create_info, None) }
            .map_err(|err| Error::VulkanShaderModuleCreation(err))?;
    // TODO: Can use allocator
    let frag_shader_module =
        unsafe { device.create_shader_module(&frag_shader_module_create_info, None) }
            .map_err(|err| Error::VulkanShaderModuleCreation(err))?;
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

    // TODO: vk::PipelineDynamicStateCreateInfo for dynamic viewport, for resizing?

    // TODO: Insert/describe uniforms here?
    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default();

    // TODO: Can use allocator
    let layout = unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }
        .map_err(|err| Error::VulkanPipelineLayoutCreation(err))?;
    let layouts = vec![layout];

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

    let render_pass_create_info = vk::RenderPassCreateInfo::builder()
        .attachments(&attachments)
        .subpasses(&subpasses);
    let render_pass = unsafe { device.create_render_pass(&render_pass_create_info, None) }
        .map_err(|err| Error::VulkanRenderPassCreation(err))?;
    let render_passes = vec![render_pass];

    let pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&shader_stages)
        .vertex_input_state(&vertex_input_state_create_info)
        .input_assembly_state(&input_assembly_state_create_info)
        .viewport_state(&viewport_state_create_info)
        .rasterization_state(&rasterization_state_create_info)
        .multisample_state(&multisample_state_create_info)
        .color_blend_state(&color_blend_state_create_info)
        .layout(layout)
        .render_pass(render_pass)
        .subpass(0);
    let pipeline_create_infos = [pipeline_create_info.build()];
    // TODO: Can use allocator
    let pipelines = unsafe {
        device
            .create_graphics_pipelines(vk::PipelineCache::null(), &pipeline_create_infos, None)
            .map_err(|(_, err)| Error::VulkanGraphicsPipelineCreation(err))
    }?;

    // TODO: Can use allocator
    unsafe { device.destroy_shader_module(vert_shader_module, None) };
    // TODO: Can use allocator
    unsafe { device.destroy_shader_module(frag_shader_module, None) };

    Ok(Pipeline {
        layouts,
        render_passes,
        pipelines,
    })
}
