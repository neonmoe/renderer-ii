use crate::descriptors::Descriptors;
use crate::pipeline::PushConstantStruct;
use crate::vulkan_raii::{Buffer, Device};
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

    pub fn get_arena(self, arenas: &[VulkanArena]) -> &VulkanArena {
        &arenas[self.index as usize]
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
    // NOTE: These temporary resources should be removable after the async resource loading system is up and running
    temp_command_buffers: (Sender<vk::CommandBuffer>, Receiver<vk::CommandBuffer>),
    temp_semaphores: (Sender<vk::Semaphore>, Receiver<vk::Semaphore>),
    temp_buffers: (Sender<Buffer>, Receiver<Buffer>),
    // TODO: Add per-frame rendering command pools which get reset as pools intead of just resetting the individual command buffers
    // Creating many buffers (which should be cleaned up after) will be needed for multithreaded draw submission too
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
    pub(crate) command_pool: vk::CommandPool,

    pub(crate) graphics_queue: vk::Queue,
    surface_queue: vk::Queue,
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

            self.cleanup_temp_command_buffers(frame_local);
            self.cleanup_temp_semaphores(frame_local);
            self.cleanup_temp_buffers(frame_local);

            unsafe {
                self.device.destroy_semaphore(frame_local.ready_for_present_sp, None);
                self.device.destroy_fence(frame_local.frame_end_fence, None);
            }
        }

        {
            profiling::scope!("destroy command pools");
            unsafe { self.device.destroy_command_pool(self.command_pool, None) };
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

        let queue_priorities = [1.0, 1.0];
        let queue_create_infos = if physical_device.graphics_family_index == physical_device.surface_family_index {
            vec![vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(physical_device.graphics_family_index)
                .queue_priorities(&queue_priorities)
                .build()]
        } else {
            vec![
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(physical_device.graphics_family_index)
                    .queue_priorities(&queue_priorities[0..1])
                    .build(),
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(physical_device.surface_family_index)
                    .queue_priorities(&queue_priorities[1..2])
                    .build(),
            ]
        };
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
        let device = Device { inner: device };

        let graphics_queue = unsafe { device.get_device_queue(physical_device.graphics_family_index, 0) };
        let surface_queue;
        if physical_device.graphics_family_index == physical_device.surface_family_index {
            surface_queue = unsafe { device.get_device_queue(physical_device.graphics_family_index, 1) };
        } else {
            surface_queue = unsafe { device.get_device_queue(physical_device.surface_family_index, 0) };
        }
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
                Ok(FrameLocal {
                    in_use: AtomicBool::new(false),
                    ready_for_present_sp: finished_command_buffers_sp,
                    frame_end_fence,
                    temp_command_buffers: mpsc::channel(),
                    temp_semaphores: mpsc::channel(),
                    temp_buffers: mpsc::channel(),
                })
            })
            .collect::<Result<Vec<FrameLocal>, Error>>()?;
        // </frame local stuff>

        let frame_start_fence = unsafe {
            device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(Error::VulkanSemaphoreCreation)
        }?;

        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(physical_device.graphics_family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool =
            unsafe { device.create_command_pool(&command_pool_create_info, None) }.map_err(Error::VulkanCommandPoolCreation)?;

        Ok(Gpu {
            driver: driver.clone(),

            device: Rc::new(device),
            command_pool,

            graphics_queue,
            surface_queue,
            frame_start_fence,

            frame_locals,
            render_wait_semaphores: mpsc::channel(),
        })
    }

    pub fn temp_arena_count(&self) -> usize {
        self.frame_locals.len()
    }

    fn frame_local(&self, frame_index: FrameIndex) -> &FrameLocal {
        &self.frame_locals[frame_index.index]
    }

    // TODO: Add a proper way to create resources asynchronously
    // For:
    // - disposing of staging buffers after upload
    // - disposing of uploading command buffers after they've run
    // - creating resources in an "incomplete" state, turning them into "complete" after the creation command buffer has run?
    //
    // Thoughts: a Loading<T> where T is an Image or Buffer, would be
    // neat, but hard to generalize further. Would Gltf become a
    // Loading<Gltf>, and how would it communicate with the actual
    // buffers and images? Another idea: simply tracking uploaded-ness
    // via a fence alongisde all resources that need a command buffer
    // for creation (like meshes and images, though meshes need a
    // rewrite anyways). Gltf could then report its status (3/5
    // resources ready), and perhaps its rendering could be
    // conditional on the upload status.
    //
    // Should replace add_temp_buffer and run_command_buffer here.
    //
    // See also: FrameLocal, the temp stuff can be removed after this
    // is implemented. Really temp stuff, like transform buffers, may
    // require something like them though.

    pub(crate) fn add_temp_buffer(&self, frame_index: FrameIndex, buffer: Buffer) {
        let _ = self.frame_local(frame_index).temp_buffers.0.send(buffer);
    }

    pub(crate) fn run_command_buffer<F: FnOnce(vk::CommandBuffer)>(
        &self,
        frame_index: FrameIndex,
        wait_stage: vk::PipelineStageFlags,
        f: F,
    ) -> Result<(), Error> {
        let command_buffers = {
            profiling::scope!("allocate command buffer");
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            unsafe { self.device.allocate_command_buffers(&command_buffer_allocate_info) }.map_err(Error::VulkanCommandBuffersAllocation)?
        };
        let temp_command_buffer = command_buffers[0];

        let signal_semaphore = {
            profiling::scope!("create semaphore");
            unsafe { self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }.map_err(Error::VulkanSemaphoreCreation)?
        };

        {
            profiling::scope!("begin command buffer recording");
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe { self.device.begin_command_buffer(temp_command_buffer, &command_buffer_begin_info) }
                .map_err(Error::VulkanBeginCommandBuffer)?;
        }

        f(temp_command_buffer);

        {
            profiling::scope!("end command buffer recording");
            unsafe { self.device.end_command_buffer(temp_command_buffer) }.map_err(Error::VulkanEndCommandBuffer)?;
        }

        {
            profiling::scope!("submit command buffer");
            let command_buffers = [temp_command_buffer];
            let signal_semaphores = [signal_semaphore];
            let submit_infos = [vk::SubmitInfo::builder()
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores)
                .build()];
            unsafe {
                self.device
                    .queue_submit(self.graphics_queue, &submit_infos, vk::Fence::null())
                    .map_err(Error::VulkanQueueSubmit)
            }?;
        }

        {
            profiling::scope!("queue for gc");
            let temp_cmdbuf_sender = &self.frame_local(frame_index).temp_command_buffers.0;
            let _ = temp_cmdbuf_sender.send(temp_command_buffer);
            let temp_semaphore_sender = &self.frame_local(frame_index).temp_semaphores.0;
            let _ = temp_semaphore_sender.send(signal_semaphore);
            let render_wait_semaphore_sender = &self.render_wait_semaphores.0;
            let _ = render_wait_semaphore_sender.send(WaitSemaphore(signal_semaphore, wait_stage));
        }
        Ok(())
    }

    #[profiling::function]
    fn cleanup_temp_command_buffers(&self, frame_local: &FrameLocal) {
        for command_buffer in frame_local.temp_command_buffers.1.try_iter() {
            let cmdbufs = [command_buffer];
            unsafe {
                self.device.free_command_buffers(self.command_pool, &cmdbufs);
            }
        }
    }

    #[profiling::function]
    fn cleanup_temp_semaphores(&self, frame_local: &FrameLocal) {
        for semaphore in frame_local.temp_semaphores.1.try_iter() {
            unsafe { self.device.destroy_semaphore(semaphore, None) };
        }
    }

    #[profiling::function]
    fn cleanup_temp_buffers(&self, frame_local: &FrameLocal) {
        for buffer in frame_local.temp_buffers.1.try_iter() {
            drop(buffer);
        }
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
    pub fn wait_frame(&self, canvas: &Canvas) -> Result<FrameIndex, Error> {
        let (image_index, _) = unsafe {
            profiling::scope!("acquire next image");
            canvas
                .swapchain
                .device
                .acquire_next_image(canvas.swapchain.inner, u64::MAX, vk::Semaphore::null(), self.frame_start_fence)
                .map_err(Error::VulkanAcquireImage)
        }?;
        let frame_index = FrameIndex::new(image_index as usize);
        let frame_local = self.frame_local(frame_index);

        let fences = [self.frame_start_fence, frame_local.frame_end_fence];
        unsafe {
            profiling::scope!("wait for the image");
            self.device
                .wait_for_fences(&fences, true, u64::MAX)
                .map_err(Error::VulkanFenceWait)?;
            self.device.reset_fences(&fences).map_err(Error::VulkanFenceReset)?;
        }

        if frame_local.in_use.load(Ordering::Relaxed) {
            self.cleanup_temp_command_buffers(frame_local);
            self.cleanup_temp_semaphores(frame_local);
            self.cleanup_temp_buffers(frame_local);
        }

        Ok(frame_index)
    }

    /// Queue up all the rendering commands.
    ///
    /// The rendered frame will appear on the screen some time in the
    /// future. Use [Gpu::wait_frame] to block until that
    /// happens.
    #[profiling::function]
    pub fn render_frame(
        &self,
        temp_arenas: &[VulkanArena],
        frame_index: FrameIndex,
        descriptors: &mut Descriptors,
        canvas: &Canvas,
        camera: &Camera,
        scene: &Scene,
        debug_value: u32,
    ) -> Result<(), Error> {
        let frame_local = self.frame_local(frame_index);
        frame_local.in_use.store(true, Ordering::Relaxed);
        camera.update(descriptors, canvas, temp_arenas, frame_index)?;

        let command_buffer = canvas.command_buffers[frame_index.index];
        let framebuffer = &canvas.framebuffers[frame_index.index];
        self.record_command_buffer(
            temp_arenas,
            frame_index,
            command_buffer,
            framebuffer.inner,
            descriptors,
            canvas,
            scene,
            debug_value,
        )?;

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
        &self,
        temp_arenas: &[VulkanArena],
        frame_index: FrameIndex,
        command_buffer: vk::CommandBuffer,
        framebuffer: vk::Framebuffer,
        descriptors: &mut Descriptors,
        canvas: &Canvas,
        scene: &Scene,
        debug_value: u32,
    ) -> Result<(), Error> {
        let temp_arena = frame_index.get_arena(temp_arenas);

        unsafe {
            profiling::scope!("reset command buffer");
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .unwrap(); // FIXME: Use per-frame command pools instead of resettable buffers
        }

        descriptors.write_descriptors(frame_index);

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

                self.add_temp_buffer(frame_index, transform_buffer.buffer);

                unsafe {
                    profiling::scope!("draw");
                    self.device
                        .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &vertex_offsets);
                    self.device
                        .cmd_bind_index_buffer(command_buffer, mesh.buffer(), mesh.indices_offset, mesh.index_type);
                    self.device
                        .cmd_draw_indexed(command_buffer, mesh.index_count, transforms.len() as u32, 0, 0, 0);
                }
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

        Ok(())
    }
}
