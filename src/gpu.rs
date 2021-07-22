use crate::{Canvas, Driver, Error};
use ash::extensions::{ext, khr};
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk::Handle;
use ash::{vk, Device, Instance};
use std::cell::Cell;
use std::ffi::CStr;
use std::mem;
use ultraviolet::Vec3;
use vk_mem::{Allocator, AllocatorCreateFlags, AllocatorCreateInfo};

struct Material {
    vertex_shader: &'static [u32],
    fragment_shader: &'static [u32],
    bindings: &'static [vk::VertexInputBindingDescription],
    attributes: &'static [vk::VertexInputAttributeDescription],
}

enum MaterialIndex {
    PlainVertexColor,
    Length,
}

static MATERIALS: [Material; MaterialIndex::Length as usize] = [Material {
    vertex_shader: shaders::include_spirv!("shaders/plain_color.vert"),
    fragment_shader: shaders::include_spirv!("shaders/plain_color.frag"),
    bindings: &[vk::VertexInputBindingDescription {
        binding: 0,
        stride: mem::size_of::<[Vec3; 2]>() as u32,
        input_rate: vk::VertexInputRate::VERTEX,
    }],
    attributes: &[
        vk::VertexInputAttributeDescription {
            binding: 0,
            location: 0,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: 0,
        },
        vk::VertexInputAttributeDescription {
            binding: 0,
            location: 1,
            format: vk::Format::R32G32B32_SFLOAT,
            offset: mem::size_of::<[Vec3; 1]>() as u32,
        },
    ],
}];

/// A unique id for every distinct GPU.
pub struct GpuId([u8; 16]);

/// Describes a GPU that can be used as a [Gpu]. Queried in
/// [Gpu::new].
pub struct GpuInfo {
    pub in_use: bool,
    pub name: String,
    pub id: GpuId,
}

/// The main half of the rendering pair, along with [Canvas].
///
/// Each instance of [Gpu] contains a handle to a single physical
/// device in Vulkan terms, i.e. a GPU, and everything else is built
/// off of that.
pub struct Gpu<'a> {
    /// Held by [Gpu] to ensure that the devices are dropped before
    /// the instance.
    pub driver: &'a Driver,

    pub(crate) surface_ext: khr::Surface,
    pub(crate) swapchain_ext: khr::Swapchain,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) device: Device,
    pub(crate) graphics_family_index: u32,
    pub(crate) surface_family_index: u32,
    pub(crate) command_pool: vk::CommandPool,

    graphics_queue: vk::Queue,
    surface_queue: vk::Queue,
    allocator: Allocator,
    frame_index: Cell<u32>,
    frame_sync_objects_vec: Vec<FrameSyncObjects>,

    pub(crate) final_render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipelines: Vec<vk::Pipeline>,
    command_buffers: Vec<vk::CommandBuffer>,

    buffers: Vec<(vk::Buffer, vk_mem::Allocation)>,
}

#[derive(Clone, Copy)]
struct FrameSyncObjects {
    acquired_image_sp: vk::Semaphore,
    finished_command_buffers_sp: vk::Semaphore,
    finished_queue_fence: vk::Fence,
}

impl Drop for Gpu<'_> {
    fn drop(&mut self) {
        let _ = self.wait_idle();

        for &(buffer, allocation) in &self.buffers {
            let _ = self.allocator.destroy_buffer(buffer, &allocation);
        }

        for &pipeline in &self.pipelines {
            unsafe { self.device.destroy_pipeline(pipeline, None) };
        }

        unsafe {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_render_pass(self.final_render_pass, None);
            self.device
                .free_command_buffers(self.command_pool, &self.command_buffers);
            self.device.destroy_command_pool(self.command_pool, None);
        }

        self.allocator.destroy();

        for frame_sync_objects in &self.frame_sync_objects_vec {
            let FrameSyncObjects {
                acquired_image_sp,
                finished_command_buffers_sp,
                finished_queue_fence,
            } = *frame_sync_objects;

            unsafe {
                self.device.destroy_semaphore(acquired_image_sp, None);
                self.device
                    .destroy_semaphore(finished_command_buffers_sp, None);
                self.device.destroy_fence(finished_queue_fence, None);
            }
        }

        unsafe { self.device.destroy_device(None) };
    }
}

impl Gpu<'_> {
    /// Creates a new instance of [Gpu], optionally the specified
    /// one. Only one should exist at a time.
    ///
    /// The tuple's second part is a list of usable physical
    /// devices, for picking between e.g. a laptop's integrated and
    /// discrete GPUs.
    ///
    /// The inner tuples consist of: whether the gpu is the one picked
    /// in this function call, the display name, and the id passed to
    /// a new [Gpu] when recreating it with a new physical device.
    pub fn new(
        driver: &Driver,
        preferred_physical_device: Option<[u8; 16]>,
    ) -> Result<(Gpu<'_>, Vec<GpuInfo>), Error> {
        let surface_ext = khr::Surface::new(&driver.entry, &driver.instance);
        let queue_family_supports_surface = |pd: vk::PhysicalDevice, index: u32| {
            let support = unsafe {
                surface_ext.get_physical_device_surface_support(pd, index, driver.surface)
            };
            matches!(support, Ok(true))
        };

        let all_physical_devices = unsafe { driver.instance.enumerate_physical_devices() }
            .map_err(Error::VulkanEnumeratePhysicalDevices)?;
        let mut physical_devices = all_physical_devices
            .into_iter()
            .filter_map(|physical_device| {
                if !is_extension_supported(&driver.instance, physical_device, "VK_KHR_swapchain") {
                    return None;
                }
                let queue_families = unsafe {
                    driver
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

        let (physical_device, graphics_family_index, surface_family_index) = if let Some(uuid) =
            preferred_physical_device
        {
            physical_devices
                .iter()
                .find_map(|tuple| {
                    let properties =
                        unsafe { driver.instance.get_physical_device_properties(tuple.0) };
                    if properties.pipeline_cache_uuid == uuid {
                        Some(*tuple)
                    } else {
                        None
                    }
                })
                .ok_or(Error::VulkanPhysicalDeviceMissing)?
        } else {
            physical_devices.sort_by(|(a, a_gfx, a_surf), (b, b_gfx, b_surf)| {
                let a_properties = unsafe { driver.instance.get_physical_device_properties(*a) };
                let b_properties = unsafe { driver.instance.get_physical_device_properties(*b) };
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
                .get(0)
                .copied()
                .ok_or(Error::VulkanPhysicalDeviceMissing)?
        };
        let physical_device_properties = unsafe {
            driver
                .instance
                .get_physical_device_properties(physical_device)
        };

        let physical_devices = physical_devices
            .into_iter()
            .map(|(pd, _, _)| {
                let properties = unsafe { driver.instance.get_physical_device_properties(pd) };
                let name = unsafe { CStr::from_ptr((&properties.device_name[..]).as_ptr()) }
                    .to_string_lossy();
                let pd_type = match properties.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => " (Discrete GPU)",
                    vk::PhysicalDeviceType::INTEGRATED_GPU => " (Integrated GPU)",
                    vk::PhysicalDeviceType::VIRTUAL_GPU => " (vCPU)",
                    vk::PhysicalDeviceType::CPU => " (CPU)",
                    _ => "",
                };
                let in_use = properties.pipeline_cache_uuid
                    == physical_device_properties.pipeline_cache_uuid;
                let name = format!("{}{}", name, pd_type);
                let id = GpuId(properties.pipeline_cache_uuid);
                GpuInfo { in_use, name, id }
            })
            .collect::<Vec<GpuInfo>>();

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
            driver
                .instance
                .create_device(physical_device, &device_create_info, None)
                .map_err(Error::VulkanDeviceCreation)
        }?;
        let swapchain_ext = khr::Swapchain::new(&driver.instance, &device);

        let graphics_queue = unsafe { device.get_device_queue(graphics_family_index, 0) };
        let surface_queue;
        if graphics_family_index == surface_family_index {
            surface_queue = unsafe { device.get_device_queue(graphics_family_index, 1) };
        } else {
            surface_queue = unsafe { device.get_device_queue(surface_family_index, 0) };
        }
        if driver.debug_utils_available {
            let debug_utils_ext = ext::DebugUtils::new(&driver.entry, &driver.instance);
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

        let allocator_create_info = AllocatorCreateInfo {
            physical_device,
            device: device.clone(),
            instance: driver.instance.clone(),
            flags: AllocatorCreateFlags::EXTERNALLY_SYNCHRONIZED,
            preferred_large_heap_block_size: 128 * 1024 * 1024,
            frame_in_use_count: 1,
            heap_size_limits: None,
        };
        let allocator =
            Allocator::new(&allocator_create_info).map_err(Error::VmaAllocatorCreation)?;

        let frame_sync_objects_vec = (0..2)
            .map(|_| {
                let acquired_image_sp = unsafe {
                    device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .map_err(Error::VulkanSemaphoreCreation)
                }?;
                let finished_command_buffers_sp = unsafe {
                    device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .map_err(Error::VulkanSemaphoreCreation)
                }?;
                let finished_queue_fence = unsafe {
                    device
                        .create_fence(
                            &vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED),
                            None,
                        )
                        .map_err(Error::VulkanFenceCreation)
                }?;
                Ok(FrameSyncObjects {
                    acquired_image_sp,
                    finished_command_buffers_sp,
                    finished_queue_fence,
                })
            })
            .collect::<Result<Vec<FrameSyncObjects>, Error>>()?;

        let final_render_pass = create_render_pass(&device, crate::canvas::SWAPCHAIN_FORMAT)?;
        let pipeline_layout = {
            // TODO: Insert/describe uniforms here?
            let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default();
            unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }
                .map_err(Error::VulkanPipelineLayoutCreation)?
        };
        let pipelines = create_pipelines(&device, pipeline_layout, final_render_pass, &MATERIALS)?;

        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(graphics_family_index)
            .flags(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            );
        let command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None) }
            .map_err(Error::VulkanCommandPoolCreation)?;

        let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(frame_sync_objects_vec.len() as u32);
        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .map_err(Error::VulkanCommandBuffersAllocation)
        }?;

        // Setup testing meshes
        let triangle_vertices: [[Vec3; 2]; 3] = [
            [Vec3::new(0.0, -0.5, 0.0), Vec3::new(1.0, 0.0, 0.0)],
            [Vec3::new(0.5, 0.5, 0.0), Vec3::new(0.0, 1.0, 0.0)],
            [Vec3::new(-0.5, 0.5, 0.0), Vec3::new(0.0, 0.0, 1.0)],
        ];
        let buffer_using_families = [graphics_family_index];
        let triangle_create_info = vk::BufferCreateInfo::builder()
            .size((triangle_vertices.len() * mem::size_of::<[Vec3; 2]>()) as u64)
            .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&buffer_using_families);
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            usage: vk_mem::MemoryUsage::CpuToGpu,
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            required_flags: vk::MemoryPropertyFlags::HOST_COHERENT,
            ..Default::default()
        };
        let (triangle_buffer, triangle_allocation, alloc_info) = allocator
            .create_buffer(&triangle_create_info, &allocation_create_info)
            .map_err(Error::VmaBufferAllocation)?;
        let buffer_ptr = alloc_info.get_mapped_data();
        fn copy_raw<T>(data: &[T], pointer: *mut u8) {
            let length = data.len() * mem::size_of::<T>();
            let data_ptr = unsafe { mem::transmute::<*const T, *const u8>(data.as_ptr()) };
            unsafe { std::ptr::copy_nonoverlapping(data_ptr, pointer, length) };
        }
        copy_raw(&triangle_vertices, buffer_ptr);
        let buffers = vec![(triangle_buffer, triangle_allocation)];

        Ok((
            Gpu {
                driver,
                surface_ext,
                swapchain_ext,
                physical_device,
                device,
                graphics_family_index,
                surface_family_index,
                command_pool,
                graphics_queue,
                surface_queue,
                allocator,
                frame_index: Cell::new(0),
                frame_sync_objects_vec,
                final_render_pass,
                pipelines,
                command_buffers,
                pipeline_layout,
                buffers,
            },
            physical_devices,
        ))
    }

    /// Wait until the device is idle. Should be called before
    /// swapchain recreation and after the game loop is over.
    pub fn wait_idle(&self) -> Result<(), Error> {
        unsafe { self.device.device_wait_idle() }.map_err(Error::VulkanDeviceWaitIdle)
    }

    /// Wait until the next frame can start rendering.
    pub fn wait_frame(&self) -> Result<(), Error> {
        let frame_index = self.frame_index.get().wrapping_add(1);
        let next_sync_objects =
            self.frame_sync_objects_vec[frame_index as usize % self.frame_sync_objects_vec.len()];
        unsafe {
            self.device
                .wait_for_fences(&[next_sync_objects.finished_queue_fence], true, u64::MAX)
                .map_err(Error::VulkanFenceWait)
        }?;
        Ok(())
    }

    /// Queue up all the rendering commands.
    ///
    /// The rendered frame will appear on the screen some time in the
    /// future. Use [Gpu::wait_frame] to block until that
    /// happens.
    pub fn render_frame(&self, canvas: &Canvas) -> Result<(), Error> {
        // NOTE: This will cause self.allocator to mis-diagnose lost
        // allocations once per ~828 days (assuming 60 fps) and cause
        // the slowest memory leak ever.
        let frame_index = self.frame_index.get().wrapping_add(1);
        self.frame_index.set(frame_index);
        let _ = self.allocator.set_current_frame_index(frame_index);

        let FrameSyncObjects {
            acquired_image_sp,
            finished_command_buffers_sp,
            finished_queue_fence,
        } = self.frame_sync_objects_vec[frame_index as usize % self.frame_sync_objects_vec.len()];

        let (image_index, _) = unsafe {
            self.swapchain_ext
                .acquire_next_image(
                    canvas.swapchain,
                    u64::MAX,
                    acquired_image_sp,
                    vk::Fence::null(),
                )
                .map_err(Error::VulkanAcquireImage)
        }?;

        unsafe { self.device.reset_fences(&[finished_queue_fence]) }
            .map_err(Error::VulkanFenceReset)?;

        let command_buffer = self.command_buffers[image_index as usize];
        let framebuffer = canvas.swapchain_framebuffers[image_index as usize];
        self.record_commmand_buffer(command_buffer, framebuffer, canvas.extent)?;

        let wait_semaphores = [acquired_image_sp];
        let signal_semaphores = [finished_command_buffers_sp];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = [command_buffer];
        let submit_infos = [vk::SubmitInfo::builder()
            .wait_semaphores(&wait_semaphores)
            .signal_semaphores(&signal_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .build()];
        unsafe {
            self.device
                .queue_submit(self.graphics_queue, &submit_infos, finished_queue_fence)
                .map_err(Error::VulkanSubmitQueue)
        }?;

        let swapchains = [canvas.swapchain];
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

    fn record_commmand_buffer(
        &self,
        command_buffer: vk::CommandBuffer,
        framebuffer: vk::Framebuffer,
        extent: vk::Extent2D,
    ) -> Result<(), Error> {
        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(Error::VulkanResetCommandBuffer)?;
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(Error::VulkanBeginCommandBuffer)?;
        }

        let render_area = vk::Rect2D::builder().extent(extent).build();
        let clear_colors = [vk::ClearValue::default()];
        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.final_render_pass)
            .framebuffer(framebuffer)
            .render_area(render_area)
            .clear_values(&clear_colors);
        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            );
            let index = MaterialIndex::PlainVertexColor as usize;
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipelines[index],
            );
            self.device
                .cmd_bind_vertex_buffers(command_buffer, 0, &[self.buffers[0].0], &[0]);
            self.device.cmd_draw(command_buffer, 3, 1, 0, 0);
            self.device.cmd_end_render_pass(command_buffer);
        }

        unsafe { self.device.end_command_buffer(command_buffer) }
            .map_err(Error::VulkanEndCommandBuffer)?;
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

fn create_pipelines(
    device: &Device,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    materials: &[Material],
) -> Result<Vec<vk::Pipeline>, Error> {
    let mut all_shader_modules = Vec::with_capacity(materials.len() * 2);
    let mut create_shader_module = |spirv: &'static [u32]| -> Result<vk::ShaderModule, Error> {
        let create_info = vk::ShaderModuleCreateInfo::builder().code(spirv);
        let shader_module = unsafe { device.create_shader_module(&create_info, None) }
            .map_err(Error::VulkanShaderModuleCreation)?;
        all_shader_modules.push(shader_module);
        Ok(shader_module)
    };

    let shader_stages_per_material = materials
        .iter()
        .map(|material| {
            let vert_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(create_shader_module(material.vertex_shader)?)
                .name(cstr!("main"));
            let frag_shader_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(create_shader_module(material.fragment_shader)?)
                .name(cstr!("main"));
            Ok([
                vert_shader_stage_create_info.build(),
                frag_shader_stage_create_info.build(),
            ])
        })
        .collect::<Result<Vec<[vk::PipelineShaderStageCreateInfo; 2]>, Error>>()?;

    let vertex_input_per_material = materials
        .iter()
        .map(|material| {
            vk::PipelineVertexInputStateCreateInfo::builder()
                .vertex_binding_descriptions(&material.bindings)
                .vertex_attribute_descriptions(&material.attributes)
                .build()
        })
        .collect::<Vec<vk::PipelineVertexInputStateCreateInfo>>();

    let pipelines = {
        let input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        // FIXME: Viewport state should be dynamic
        let viewports = [vk::Viewport::builder()
            .width(800.0)
            .height(600.0)
            .min_depth(0.0)
            .max_depth(1.0)
            .build()];
        let scissors = [vk::Rect2D::builder()
            .extent(vk::Extent2D {
                width: 800,
                height: 600,
            })
            .build()];
        let viewport_create_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewports(&viewports)
            .scissors(&scissors);

        // NOTE: Shadow maps would want to configure this for clamping and biasing depth values
        let rasterization_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::CLOCKWISE)
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

        let pipeline_create_infos = shader_stages_per_material
            .iter()
            .zip(vertex_input_per_material.iter())
            .map(|(shader_stages, vertex_input)| {
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
