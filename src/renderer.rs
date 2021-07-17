use crate::{Error, Foundation};
use ash::extensions::{ext, khr};
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk::Handle;
use ash::{vk, Device, Instance};
use std::ffi::CStr;

pub struct Swapchain<'a> {
    renderer: &'a Renderer<'a>,
    swapchain: vk::SwapchainKHR,
    swapchain_image_views: Vec<vk::ImageView>,
    swapchain_framebuffers: Vec<vk::Framebuffer>,
    surface_pipeline_layout: vk::PipelineLayout,
    surface_pipeline_render_pass: vk::RenderPass,
    surface_pipeline: vk::Pipeline,
    command_buffers: Vec<vk::CommandBuffer>,
}

impl Drop for Swapchain<'_> {
    fn drop(&mut self) {
        let device = &self.renderer.device;

        for framebuffer in &self.swapchain_framebuffers {
            unsafe { device.destroy_framebuffer(*framebuffer, None) };
        }

        unsafe {
            device.destroy_pipeline(self.surface_pipeline, None);
            device.destroy_pipeline_layout(self.surface_pipeline_layout, None);
            device.destroy_render_pass(self.surface_pipeline_render_pass, None);
            device.free_command_buffers(self.renderer.command_pool, &self.command_buffers);
        };

        for image_view in &self.swapchain_image_views {
            unsafe { device.destroy_image_view(*image_view, None) };
        }

        unsafe {
            self.renderer
                .swapchain_ext
                .destroy_swapchain(self.swapchain, None)
        };
    }
}

impl Swapchain<'_> {
    pub fn new<'a>(
        renderer: &'a Renderer,
        old_swapchain: Option<Swapchain>,
        width: u32,
        height: u32,
    ) -> Result<Swapchain<'a>, Error> {
        let device = &renderer.device;
        let swapchain_ext = &renderer.swapchain_ext;
        let (swapchain, swapchain_image_views, swapchain_format, final_extent) =
            create_swapchain_and_image_views(
                &renderer.surface_ext,
                &swapchain_ext,
                renderer.foundation.surface,
                width,
                height,
                old_swapchain.map(|r| r.swapchain),
                renderer.physical_device,
                device,
                renderer.graphics_family_index,
                renderer.surface_family_index,
            )?;
        // The width and height may change from the ones passed in,
        // because they're queried during swapchain creation.
        let vk::Extent2D { width, height } = final_extent;

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
                    .map_err(|err| Error::VulkanFramebufferCreation(err))
            })
            .collect::<Result<Vec<_>, _>>()?;

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(renderer.command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(swapchain_framebuffers.len() as u32);
        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .map_err(|err| Error::VulkanCommandBuffersAllocation(err))
        }?;

        for (command_buffer, framebuffer) in command_buffers.iter().zip(&swapchain_framebuffers) {
            let command_buffer = *command_buffer;
            let framebuffer = *framebuffer;

            let begin_info = vk::CommandBufferBeginInfo::default();
            unsafe { device.begin_command_buffer(command_buffer, &begin_info) }
                .map_err(|err| Error::VulkanBeginCommandBuffer(err))?;

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
                .map_err(|err| Error::VulkanEndCommandBuffer(err))?;
        }

        Ok(Swapchain {
            renderer,
            swapchain,
            swapchain_image_views,
            swapchain_framebuffers,
            surface_pipeline_layout,
            surface_pipeline_render_pass,
            surface_pipeline,
            command_buffers,
        })
    }
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

    surface_ext: khr::Surface,
    swapchain_ext: khr::Swapchain,
    physical_device: vk::PhysicalDevice,
    device: Device,
    graphics_queue: vk::Queue,
    graphics_family_index: u32,
    surface_queue: vk::Queue,
    surface_family_index: u32,

    command_pool: vk::CommandPool,
    acquired_image_sp: vk::Semaphore,
    finished_command_buffers_sp: vk::Semaphore,
}

impl Drop for Renderer<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_semaphore(self.acquired_image_sp, None);
            self.device
                .destroy_semaphore(self.finished_command_buffers_sp, None);
        }

        unsafe { self.device.destroy_command_pool(self.command_pool, None) };

        let _ = unsafe { self.device.queue_wait_idle(self.graphics_queue) };
        let _ = unsafe { self.device.queue_wait_idle(self.surface_queue) };

        unsafe { self.device.destroy_device(None) };
    }
}

impl Renderer<'_> {
    pub fn new<'a>(
        foundation: &'a Foundation,
        preferred_physical_device: Option<[u8; 16]>,
    ) -> Result<Renderer<'a>, Error> {
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

        let device_create_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&physical_device_features)
            .enabled_extension_names(extensions);
        let device = unsafe {
            foundation
                .instance
                .create_device(physical_device, &device_create_info, None)
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

        let command_pool_create_info =
            vk::CommandPoolCreateInfo::builder().queue_family_index(graphics_family_index);
        let command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None) }
            .map_err(|err| Error::VulkanCommandPoolCreation(err))?;

        let acquired_image_sp = unsafe {
            device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .map_err(|err| Error::VulkanSemaphoreCreation(err))
        }?;
        let finished_command_buffers_sp = unsafe {
            device
                .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                .map_err(|err| Error::VulkanSemaphoreCreation(err))
        }?;

        let swapchain_ext = khr::Swapchain::new(&foundation.instance, &device);

        Ok(Renderer {
            foundation,
            physical_devices,
            surface_ext,
            swapchain_ext,
            physical_device,
            device,
            graphics_queue,
            graphics_family_index,
            surface_queue,
            surface_family_index,
            command_pool,
            acquired_image_sp,
            finished_command_buffers_sp,
        })
    }

    /// Wait until the device is idle. Should be called before
    /// swapchain recreation and after the game loop is over.
    pub fn wait_idle(&self) -> Result<(), Error> {
        unsafe { self.device.device_wait_idle() }.map_err(|err| Error::VulkanDeviceWaitIdle(err))
    }

    /// Wait until the latest rendered frame is on screen.
    ///
    /// This will eventually be changed to just wait until we can
    /// start rendering the next frame, not necessarily until the
    /// previous one has been presented.
    pub fn wait_frame(&self) {
        // NOTE: Not using fences to sync for simplicity and ease of
        // thinking about the game loop: after render_frame is done,
        // the frame is on the screen, full stop. Fences should be
        // easy enough to add in later.
        let device = &self.device;
        let _ = unsafe { device.queue_wait_idle(self.graphics_queue) };
        let _ = unsafe { device.queue_wait_idle(self.surface_queue) };
    }

    /// Queue up all the rendering commands.
    ///
    /// The rendered frame will appear on the screen some time in the
    /// future. Use [Renderer::wait_frame] to block until that
    /// happens.
    pub fn render_frame(&self, swapchain: &Swapchain) -> Result<(), Error> {
        let (image_index, _) = unsafe {
            self.swapchain_ext
                .acquire_next_image(
                    swapchain.swapchain,
                    u64::MAX,
                    self.acquired_image_sp,
                    vk::Fence::null(),
                )
                .map_err(|err| Error::VulkanAcquireImage(err))
        }?;

        // TODO: Re-record command buffer if out-of-date

        let wait_semaphores = [self.acquired_image_sp];
        let signal_semaphores = [self.finished_command_buffers_sp];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [swapchain.command_buffers[image_index as usize]];
        let submit_infos = [vk::SubmitInfo::builder()
            .wait_semaphores(&wait_semaphores)
            .signal_semaphores(&signal_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .build()];
        unsafe {
            self.device
                .queue_submit(self.graphics_queue, &submit_infos, vk::Fence::null())
                .map_err(|err| Error::VulkanSubmitQueue(err))
        }?;

        let swapchains = [swapchain.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        let present_result = unsafe {
            self.swapchain_ext
                .queue_present(self.surface_queue, &present_info)
        };

        match present_result {
            Err(err @ vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                return Err(Error::VulkanSwapchainOutOfDate(err))
            }
            Err(err) => return Err(Error::VulkanQueuePresent(err)),
            _ => {}
        }

        Ok(())
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
    surface_ext: &khr::Surface,
    swapchain_ext: &khr::Swapchain,
    surface: vk::SurfaceKHR,
    width: u32,
    height: u32,
    old_swapchain: Option<vk::SwapchainKHR>,
    physical_device: vk::PhysicalDevice,
    device: &Device,
    graphics_family_index: u32,
    surface_family_index: u32,
) -> Result<
    (
        vk::SwapchainKHR,
        Vec<vk::ImageView>,
        vk::Format,
        vk::Extent2D,
    ),
    Error,
> {
    let present_mode = vk::PresentModeKHR::FIFO;
    let min_image_count = 2;

    let (image_format, image_color_space) = {
        let surface_formats = unsafe {
            surface_ext
                .get_physical_device_surface_formats(physical_device, surface)
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

    let get_image_extent = || -> Result<vk::Extent2D, Error> {
        let surface_capabilities = unsafe {
            surface_ext
                .get_physical_device_surface_capabilities(physical_device, surface)
                .map_err(|err| Error::VulkanPhysicalDeviceSurfaceQuery(err))
        }?;
        let unset_extent = vk::Extent2D {
            width: u32::MAX,
            height: u32::MAX,
        };
        let image_extent = if surface_capabilities.current_extent != unset_extent {
            surface_capabilities.current_extent
        } else {
            vk::Extent2D { width, height }
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
    // Get image extent at the latest possible time to avoid getting
    // an outdated extent. Bummer that this is an issue, but I haven't
    // found a good way to avoid it.
    swapchain_create_info.image_extent = get_image_extent()?;
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
            unsafe { device.create_image_view(&image_view_create_info, None) }
                .map_err(|err| Error::VulkanSwapchainImageViewCreation(err))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok((
        swapchain,
        swapchain_image_views,
        image_format,
        swapchain_create_info.image_extent,
    ))
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
            .map_err(|err| Error::VulkanShaderModuleCreation(err))
    }?;
    let frag_shader_module = unsafe {
        device
            .create_shader_module(&frag_shader_module_create_info, None)
            .map_err(|err| Error::VulkanShaderModuleCreation(err))
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
            .map_err(|err| Error::VulkanPipelineLayoutCreation(err))?
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
            .map_err(|err| Error::VulkanRenderPassCreation(err))?
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
