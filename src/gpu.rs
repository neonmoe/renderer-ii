use crate::descriptors::Descriptors;
use crate::pipeline::PushConstantStruct;
use crate::vulkan_raii::{CommandPool, Device};
use crate::{debug_utils, Camera, Canvas, Driver, Error, PhysicalDevice, Pipeline, Scene, VulkanArena};
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk;
use glam::Mat4;
use std::mem;
use std::rc::Rc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, Receiver, Sender};

/// Get from [Gpu::wait_frame].
#[derive(Clone, Copy)]
pub struct FrameIndex {
    /// The index of the swapchain image we're rendering to this
    /// frame. Can be used to index into frame-specific buffers that
    /// need to make sure they don't overwrite stuff still in use by
    /// previous frames.
    pub index: usize,
}

impl FrameIndex {
    fn new(index: usize) -> FrameIndex {
        FrameIndex { index }
    }

    pub fn get_arena(self, arenas: &mut [VulkanArena]) -> &mut VulkanArena {
        &mut arenas[self.index as usize]
    }
}

struct WaitSemaphore(vk::Semaphore, vk::PipelineStageFlags);

/// Synchronization objects and buffers used during a single frame,
/// which is only cleaned up after enough frames have passed that the
/// specific FrameLocal struct is reused. This allows for processing a
/// frame while the previous one is still being rendered.
struct FrameLocal {
    in_use: AtomicBool,
    ready_for_present_sp: vk::Semaphore,
    frame_end_fence: vk::Fence,
    temp_arena: VulkanArena,
    command_pool: CommandPool,
}

#[derive(PartialEq, Eq, Hash)]
pub(crate) struct TextureIndex(u32);

/// The main half of the rendering pair, along with [Canvas].
///
/// Each instance of [Gpu] contains a handle to a single physical
/// device in Vulkan terms, i.e. a GPU, and everything else is built
/// off of that.
pub struct Gpu {
    /// Held by [Gpu] to ensure that the devices are dropped before
    /// the instance.
    pub driver: Rc<Driver>,

    pub device: Rc<Device>,

    pub graphics_queue: vk::Queue,
    surface_queue: vk::Queue,
    pub transfer_queue: vk::Queue,
    frame_start_fence: vk::Fence,

    frame_locals: Vec<FrameLocal>,
    render_wait_semaphores: (Sender<WaitSemaphore>, Receiver<WaitSemaphore>),
}

impl Drop for Gpu {
    #[profiling::function]
    fn drop(&mut self) {
        let _ = self.wait_idle();

        {
            profiling::scope!("destroy frame start fence");
            unsafe { self.device.destroy_fence(self.frame_start_fence, None) };
        }

        for frame_local in &self.frame_locals {
            profiling::scope!("destroy frame locals");

            unsafe {
                self.device.destroy_semaphore(frame_local.ready_for_present_sp, None);
                self.device.destroy_fence(frame_local.frame_end_fence, None);
            }
        }
    }
}

impl Gpu {
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
    pub fn new(driver: &Rc<Driver>, physical_device: &PhysicalDevice) -> Result<Gpu, Error> {
        profiling::scope!("new_gpu");

        fn create_device_queue_create_infos(queue_family_indices: &[u32], ones: &[f32]) -> Vec<vk::DeviceQueueCreateInfo> {
            let mut results: Vec<vk::DeviceQueueCreateInfo> = Vec::with_capacity(queue_family_indices.len());
            'queue_families: for &queue_family_index in queue_family_indices {
                for create_info in &results {
                    if create_info.queue_family_index == queue_family_index {
                        continue 'queue_families;
                    }
                }
                let count = queue_family_indices.iter().filter(|index| **index == queue_family_index).count();
                let create_info = vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(queue_family_index)
                    .queue_priorities(&ones[..count])
                    .build();
                results.push(create_info);
            }
            results
        }

        fn get_device_queues<const N: usize>(device: &Device, family_indices: &[u32; N], queues: &mut [vk::Queue; N]) {
            let mut picks = Vec::with_capacity(N);
            for (&queue_family_index, queue) in family_indices.iter().zip(queues.iter_mut()) {
                let queue_index = picks.iter().filter(|index| **index == queue_family_index).count() as u32;
                *queue = unsafe { device.get_device_queue(queue_family_index, queue_index) };
                picks.push(queue_family_index);
            }
        }

        // Just to have an array to point at for the queue priorities.
        let ones = [1.0, 1.0, 1.0];
        let queue_family_indices = [
            physical_device.graphics_family_index,
            physical_device.surface_family_index,
            physical_device.transfer_family_index,
        ];
        let queue_create_infos = create_device_queue_create_infos(&queue_family_indices, &ones);
        let mut extensions = vec![cstr!("VK_KHR_swapchain").as_ptr()];
        log::debug!("Device extension: VK_KHR_swapchain");
        if physical_device.extension_supported("VK_EXT_memory_budget") {
            extensions.push(cstr!("VK_EXT_memory_budget").as_ptr());
            log::debug!("Device extension (optional): VK_EXT_memory_budget");
        }

        let mut physical_device_descriptor_indexing_features =
            vk::PhysicalDeviceDescriptorIndexingFeatures::builder().descriptor_binding_partially_bound(true);
        let device_create_info = vk::DeviceCreateInfo::builder()
            .push_next(&mut physical_device_descriptor_indexing_features)
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&physical_device.features)
            .enabled_extension_names(&extensions);
        let device = unsafe {
            driver
                .instance
                .create_device(physical_device.inner, &device_create_info, None)
                .map_err(Error::VulkanDeviceCreation)
        }?;
        let device = Rc::new(Device { inner: device });

        let mut queues = [vk::Queue::default(); 3];
        get_device_queues(&device, &queue_family_indices, &mut queues);
        let [graphics_queue, surface_queue, transfer_queue] = queues;
        if driver.debug_utils_available {
            debug_utils::name_vulkan_object(&device, graphics_queue, format_args!("graphics"));
            debug_utils::name_vulkan_object(&device, surface_queue, format_args!("present"));
        }

        // TODO: Move frame local stuff to its own module
        // <frame local stuff>
        let frame_in_use_count = 3;
        let frame_locals = (0..frame_in_use_count)
            .map(|_| {
                let finished_command_buffers_sp = unsafe {
                    device
                        .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                        .map_err(Error::VulkanSemaphoreCreation)
                }?;
                let fence_create_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
                let frame_end_fence = unsafe {
                    device
                        .create_fence(&fence_create_info, None)
                        .map_err(Error::VulkanSemaphoreCreation)
                }?;
                let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
                    .queue_family_index(physical_device.graphics_family_index)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT);
                let command_pool =
                    unsafe { device.create_command_pool(&command_pool_create_info, None) }.map_err(Error::VulkanCommandPoolCreation)?;
                let command_pool = CommandPool {
                    inner: command_pool,
                    device: device.clone(),
                };
                let temp_arena = VulkanArena::new(
                    &driver.instance,
                    &device,
                    physical_device.inner,
                    10_000_000,
                    vk::MemoryPropertyFlags::HOST_VISIBLE,
                    vk::MemoryPropertyFlags::HOST_VISIBLE,
                    "frame local arena",
                )?;
                Ok(FrameLocal {
                    in_use: AtomicBool::new(false),
                    ready_for_present_sp: finished_command_buffers_sp,
                    frame_end_fence,
                    command_pool,
                    temp_arena,
                })
            })
            .collect::<Result<Vec<FrameLocal>, Error>>()?;
        // </frame local stuff>

        let frame_start_fence = unsafe {
            device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(Error::VulkanSemaphoreCreation)
        }?;

        Ok(Gpu {
            driver: driver.clone(),

            device,

            graphics_queue,
            surface_queue,
            transfer_queue,
            frame_start_fence,

            frame_locals,
            render_wait_semaphores: mpsc::channel(),
        })
    }

    // TODO: Refactor Gpu to handle just the pub fns, rename to maybe Renderer?
    // Pub fns meaning the following:
    // - wait until gpu is idle
    // - wait for next frame
    // - render next frame (+ the command buffer recording function, which is kind of a part of this)
    //
    // Would be ideal if a Renderer object wouldn't even be
    // needed. Might be that once everything else is split out, these
    // could be just their owned functions and everything required
    // could be passed in. OTOH, one of Gpu's functions is to hold the
    // vulkan Device. Though maybe it would be more ideal for the user
    // to hold the Device, it could be passed to various subsystems
    // more directly.

    /// Wait until the device is idle. Should be called before
    /// swapchain recreation and after the game loop is over.
    #[profiling::function]
    pub fn wait_idle(&self) -> Result<(), Error> {
        unsafe { self.device.device_wait_idle() }.map_err(Error::VulkanDeviceWaitIdle)
    }

    /// Wait until the next frame can start rendering.
    ///
    /// After ensuring that the next frame can be rendered, this also
    /// frees the resources that can now be freed up.
    #[profiling::function]
    pub fn wait_frame(&mut self, canvas: &Canvas) -> Result<FrameIndex, Error> {
        let (image_index, _) = unsafe {
            profiling::scope!("acquire next image");
            canvas
                .swapchain
                .device
                .acquire_next_image(canvas.swapchain.inner, u64::MAX, vk::Semaphore::null(), self.frame_start_fence)
                .map_err(Error::VulkanAcquireImage)
        }?;
        let frame_index = FrameIndex::new(image_index as usize);
        let frame_local = &mut self.frame_locals[frame_index.index];

        let fences = [self.frame_start_fence, frame_local.frame_end_fence];
        unsafe {
            profiling::scope!("wait for the image");
            self.device
                .wait_for_fences(&fences, true, u64::MAX)
                .map_err(Error::VulkanFenceWait)?;
            self.device.reset_fences(&fences).map_err(Error::VulkanFenceReset)?;
        }
        frame_local.temp_arena.reset()?;

        Ok(frame_index)
    }

    /// Queue up all the rendering commands.
    ///
    /// The rendered frame will appear on the screen some time in the
    /// future. Use [Gpu::wait_frame] to block until that
    /// happens.
    #[profiling::function]
    pub fn render_frame(
        &mut self,
        frame_index: FrameIndex,
        descriptors: &mut Descriptors,
        canvas: &Canvas,
        camera: &Camera,
        scene: &Scene,
        debug_value: u32,
    ) -> Result<(), Error> {
        let frame_local = &mut self.frame_locals[frame_index.index];
        frame_local.in_use.store(true, Ordering::Relaxed);
        camera.update(descriptors, canvas, &mut frame_local.temp_arena, frame_index)?;

        let framebuffer = &canvas.framebuffers[frame_index.index];
        let command_buffer = self.record_command_buffer(frame_index, framebuffer.inner, descriptors, canvas, scene, debug_value)?;

        let frame_local = &self.frame_locals[frame_index.index];

        let render_wait_semaphores = self.render_wait_semaphores.1.try_iter().collect::<Vec<WaitSemaphore>>();
        let mut wait_semaphores = Vec::with_capacity(render_wait_semaphores.len());
        let mut wait_stages = Vec::with_capacity(render_wait_semaphores.len());
        for WaitSemaphore(semaphore, wait_stage) in render_wait_semaphores {
            wait_semaphores.push(semaphore);
            wait_stages.push(wait_stage);
        }

        let signal_semaphores = [frame_local.ready_for_present_sp];
        let command_buffers = [command_buffer];
        let submit_infos = [vk::SubmitInfo::builder()
            .wait_semaphores(&wait_semaphores)
            .signal_semaphores(&signal_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(&command_buffers)
            .build()];
        unsafe {
            profiling::scope!("queue render");
            self.device
                .queue_submit(self.graphics_queue, &submit_infos, frame_local.frame_end_fence)
                .map_err(Error::VulkanQueueSubmit)
        }?;

        let swapchains = [canvas.swapchain.inner];
        let image_indices = [frame_index.index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        let present_result = unsafe {
            profiling::scope!("queue present");
            canvas.swapchain.device.queue_present(self.surface_queue, &present_info)
        };

        match present_result {
            Err(err @ vk::Result::ERROR_OUT_OF_DATE_KHR) => return Err(Error::VulkanSwapchainOutOfDate(err)),
            Err(err) => return Err(Error::VulkanQueuePresent(err)),
            _ => {}
        }

        Ok(())
    }

    #[profiling::function]
    fn record_command_buffer(
        &mut self,
        frame_index: FrameIndex,
        framebuffer: vk::Framebuffer,
        descriptors: &mut Descriptors,
        canvas: &Canvas,
        scene: &Scene,
        debug_value: u32,
    ) -> Result<vk::CommandBuffer, Error> {
        let frame_local = &mut self.frame_locals[frame_index.index];
        let temp_arena = &mut frame_local.temp_arena;

        descriptors.write_descriptors(frame_index);

        unsafe {
            profiling::scope!("allocate command buffer");
            self.device
                .reset_command_pool(frame_local.command_pool.inner, vk::CommandPoolResetFlags::empty())
                .map_err(Error::VulkanResetCommandPool)
        }?;

        let command_buffers = unsafe {
            profiling::scope!("allocate command buffer");
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(frame_local.command_pool.inner)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            self.device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .map_err(Error::VulkanCommandBuffersAllocation)
        }?;
        let command_buffer = command_buffers[0];

        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            profiling::scope!("begin command buffer");
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(Error::VulkanBeginCommandBuffer)?;
        }

        let render_area = vk::Rect2D::builder().extent(canvas.extent).build();
        let mut depth_clear_value = vk::ClearValue::default();
        depth_clear_value.depth_stencil.depth = 0.0;
        let clear_colors = [vk::ClearValue::default(), depth_clear_value, vk::ClearValue::default()];
        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(canvas.final_render_pass.inner)
            .framebuffer(framebuffer)
            .render_area(render_area)
            .clear_values(&clear_colors);
        unsafe {
            profiling::scope!("begin render pass");
            self.device
                .cmd_begin_render_pass(command_buffer, &render_pass_begin_info, vk::SubpassContents::INLINE);
        }

        // Bind the shared descriptor set
        {
            let pipeline = Pipeline::SHARED_DESCRIPTOR_PIPELINE;
            let shared_descriptor_set = descriptors.descriptor_sets(frame_index, pipeline)[0];
            unsafe {
                profiling::scope!("bind shared descriptor set");
                self.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    descriptors.pipeline_layouts.get(pipeline).inner,
                    0,
                    &[shared_descriptor_set],
                    &[],
                );
            }
        }

        for (&pipeline, meshes) in &scene.pipeline_map {
            profiling::scope!("pipeline");
            if meshes.is_empty() {
                continue;
            }

            unsafe {
                self.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    canvas.pipelines.get(pipeline).inner,
                )
            };
            let bind_point = vk::PipelineBindPoint::GRAPHICS;
            let layout = descriptors.pipeline_layouts.get(pipeline).inner;
            let descriptor_sets = descriptors.descriptor_sets(frame_index, pipeline);
            if descriptor_sets.len() > 1 {
                unsafe {
                    self.device
                        .cmd_bind_descriptor_sets(command_buffer, bind_point, layout, 1, &descriptor_sets[1..], &[])
                };
            }

            for (&(mesh, material), transforms) in meshes {
                profiling::scope!("mesh");
                let transform_buffer = {
                    profiling::scope!("create transform buffer");
                    let buffer_size = (transforms.len() * mem::size_of::<Mat4>()) as vk::DeviceSize;
                    let buffer_create_info = vk::BufferCreateInfo::builder()
                        .size(buffer_size)
                        .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE);
                    temp_arena.create_buffer(*buffer_create_info)?
                };

                {
                    profiling::scope!("write transform buffer");
                    unsafe { transform_buffer.write(transforms.as_ptr() as *const u8, 0, vk::WHOLE_SIZE) }?;
                }

                let push_constants = [PushConstantStruct {
                    texture_index: material.array_index,
                    debug_value,
                }];
                let push_constants = bytemuck::bytes_of(&push_constants);
                unsafe {
                    profiling::scope!("push constants");
                    self.device
                        .cmd_push_constants(command_buffer, layout, vk::ShaderStageFlags::FRAGMENT, 0, push_constants);
                }

                let mut vertex_buffers = vec![mesh.buffer(); mesh.vertices_offsets.len() + 1];
                vertex_buffers[0] = transform_buffer.buffer.inner;
                let mut vertex_offsets = Vec::with_capacity(mesh.vertices_offsets.len() + 1);
                vertex_offsets.push(0);
                vertex_offsets.extend_from_slice(&mesh.vertices_offsets);

                unsafe {
                    profiling::scope!("draw");
                    self.device
                        .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &vertex_offsets);
                    self.device
                        .cmd_bind_index_buffer(command_buffer, mesh.buffer(), mesh.indices_offset, mesh.index_type);
                    self.device
                        .cmd_draw_indexed(command_buffer, mesh.index_count, transforms.len() as u32, 0, 0, 0);
                }

                temp_arena.add_buffer(transform_buffer.buffer);
            }
        }

        unsafe {
            profiling::scope!("end render pass");
            self.device.cmd_end_render_pass(command_buffer);
        }

        unsafe {
            profiling::scope!("end command buffer");
            self.device
                .end_command_buffer(command_buffer)
                .map_err(Error::VulkanEndCommandBuffer)?;
        }

        Ok(command_buffer)
    }
}
