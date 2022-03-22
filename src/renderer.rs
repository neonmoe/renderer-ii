use crate::debug_utils;
use crate::descriptors::Descriptors;
use crate::pipeline::PushConstantStruct;
use crate::vulkan_raii::{CommandBuffer, CommandPool, Device, Fence, Semaphore};
use crate::{Camera, Canvas, Error, PhysicalDevice, Pipeline, Scene, VulkanArena};
use ash::{vk, Instance};
use glam::Mat4;
use std::mem;
use std::rc::Rc;

/// Get from [Gpu::wait_frame].
#[derive(Clone, Copy)]
pub struct FrameIndex {
    index: usize,
}

impl FrameIndex {
    fn new(index: usize) -> FrameIndex {
        FrameIndex { index }
    }

    /// The index of the swapchain image we're rendering to this
    /// frame. Can be used to index into frame-specific buffers that
    /// need to make sure they don't overwrite stuff still in use by
    /// previous frames.
    pub fn index(&self) -> usize {
        self.index
    }
}

type PerFrame<T> = Vec<T>;

/// The main half of the rendering pair, along with [Canvas].
///
/// Each instance of [Gpu] contains a handle to a single physical
/// device in Vulkan terms, i.e. a GPU, and everything else is built
/// off of that.
pub struct Renderer {
    device: Rc<Device>,
    frame_start_fence: Fence,
    ready_for_present: PerFrame<Semaphore>,
    frame_end_fence: PerFrame<Fence>,
    temp_arena: PerFrame<VulkanArena>,
    command_pool: PerFrame<Rc<CommandPool>>,
    command_buffers_in_use: PerFrame<Vec<CommandBuffer>>,
    command_buffers_unused: PerFrame<Vec<CommandBuffer>>,

    // Renderer relies on the canvas frame count.
    _canvas: Rc<Canvas>,
}

impl Renderer {
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
    pub fn new(instance: &Instance, device: &Rc<Device>, physical_device: &PhysicalDevice, canvas: &Rc<Canvas>) -> Result<Renderer, Error> {
        profiling::scope!("new_gpu");

        let frames = canvas.frame_count;
        let ready_for_present = (1..frames + 1)
            .map(|nth| {
                let semaphore = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                    .map_err(Error::VulkanSemaphoreCreation)?;
                debug_utils::name_vulkan_object(device, semaphore, format_args!("render finish semaphore ({nth}/{frames})"));
                Ok(Semaphore {
                    inner: semaphore,
                    device: device.clone(),
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;

        let frame_end_fence = (1..frames + 1)
            .map(|nth| {
                let fence_create_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
                let fence = unsafe { device.create_fence(&fence_create_info, None) }.map_err(Error::VulkanFenceCreation)?;
                debug_utils::name_vulkan_object(device, fence, format_args!("frame end fence ({nth}/{frames})"));
                Ok(Fence {
                    inner: fence,
                    device: device.clone(),
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;

        let command_pool = (1..frames + 1)
            .map(|nth| {
                let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
                    .queue_family_index(physical_device.graphics_queue_family.index)
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT);
                let command_pool =
                    unsafe { device.create_command_pool(&command_pool_create_info, None) }.map_err(Error::VulkanCommandPoolCreation)?;
                debug_utils::name_vulkan_object(device, command_pool, format_args!("rendering cmds per frame ({nth}/{frames})"));
                Ok(Rc::new(CommandPool {
                    inner: command_pool,
                    device: device.clone(),
                }))
            })
            .collect::<Result<Vec<_>, Error>>()?;

        let temp_arena = (1..frames + 1)
            .map(|nth| {
                VulkanArena::new(
                    instance,
                    device,
                    physical_device.inner,
                    10_000_000,
                    vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    vk::MemoryPropertyFlags::HOST_VISIBLE,
                    format_args!("frame local arena ({nth}/{frames})"),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let frame_start_fence = unsafe {
            device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(Error::VulkanSemaphoreCreation)
        }?;
        debug_utils::name_vulkan_object(device, frame_start_fence, format_args!("wait_frame fence"));
        let frame_start_fence = Fence {
            inner: frame_start_fence,
            device: device.clone(),
        };

        Ok(Renderer {
            device: device.clone(),
            frame_start_fence,
            ready_for_present,
            frame_end_fence,
            temp_arena,
            command_pool,
            command_buffers_in_use: (0..frames).map(|_| Vec::new()).collect::<Vec<Vec<_>>>(),
            command_buffers_unused: (0..frames).map(|_| Vec::new()).collect::<Vec<Vec<_>>>(),

            _canvas: canvas.clone(),
        })
    }

    /// Wait until the next frame can start rendering.
    ///
    /// After ensuring that the next frame can be rendered, this also
    /// frees the resources that can now be freed up.
    #[profiling::function]
    pub fn wait_frame(&mut self, canvas: &Canvas) -> Result<FrameIndex, Error> {
        let (image_index, _) = unsafe {
            profiling::scope!("acquire next image");
            let fence = self.frame_start_fence.inner;
            canvas
                .swapchain
                .device
                .acquire_next_image(canvas.swapchain.inner, u64::MAX, vk::Semaphore::null(), fence)
                .map_err(Error::VulkanAcquireImage)
        }?;
        let frame_index = FrameIndex::new(image_index as usize);
        let fi = frame_index.index;

        let fences = [self.frame_start_fence.inner, self.frame_end_fence[fi].inner];
        unsafe {
            profiling::scope!("wait for the image");
            self.device
                .wait_for_fences(&fences, true, u64::MAX)
                .map_err(Error::VulkanFenceWait)?;
            self.device.reset_fences(&fences).map_err(Error::VulkanFenceReset)?;
        }
        self.temp_arena[fi].reset()?;

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
        let fi = frame_index.index;
        camera.update(descriptors, canvas, &mut self.temp_arena[fi], frame_index)?;

        let command_buffer = self.record_command_buffer(frame_index, descriptors, canvas, scene, debug_value)?;

        let signal_semaphores = [self.ready_for_present[fi].inner];
        let command_buffers = [command_buffer];
        let submit_infos = [vk::SubmitInfo::builder()
            .signal_semaphores(&signal_semaphores)
            .command_buffers(&command_buffers)
            .build()];
        unsafe {
            profiling::scope!("queue render");
            self.device
                .queue_submit(self.device.graphics_queue, &submit_infos, self.frame_end_fence[fi].inner)
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
            canvas.swapchain.device.queue_present(self.device.surface_queue, &present_info)
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
        descriptors: &mut Descriptors,
        canvas: &Canvas,
        scene: &Scene,
        debug_value: u32,
    ) -> Result<vk::CommandBuffer, Error> {
        let fi = frame_index.index;
        let temp_arena = &mut self.temp_arena[fi];
        let framebuffer = &canvas.framebuffers[frame_index.index];

        descriptors.write_descriptors(frame_index);

        let command_pool_rc = &self.command_pool[fi];
        let command_pool = command_pool_rc.inner;
        unsafe {
            profiling::scope!("reset command pool");
            self.device
                .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())
                .map_err(Error::VulkanResetCommandPool)
        }?;
        let unused_cbs = &mut self.command_buffers_unused[fi];
        let in_use_cbs = &mut self.command_buffers_in_use[fi];
        unused_cbs.append(in_use_cbs);

        let command_buffer = if let Some(command_buffer) = unused_cbs.pop() {
            debug_utils::name_vulkan_object(&self.device, command_buffer.inner, format_args!("frame rendering cmds"));
            command_buffer
        } else {
            profiling::scope!("allocate command buffer");
            unsafe {
                let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                    .command_pool(command_pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1);
                let command_buffers = self
                    .device
                    .allocate_command_buffers(&command_buffer_allocate_info)
                    .map_err(Error::VulkanCommandBuffersAllocation)?;
                debug_utils::name_vulkan_object(&self.device, command_buffers[0], format_args!("frame rendering cmds"));
                CommandBuffer {
                    inner: command_buffers[0],
                    device: self.device.clone(),
                    command_pool: command_pool_rc.clone(),
                }
            }
        };
        let command_buffer = {
            let inner = command_buffer.inner;
            in_use_cbs.push(command_buffer);
            inner
        };

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
            .framebuffer(framebuffer.inner)
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

        for (pipeline, meshes) in scene.pipeline_map.iter_with_pipeline() {
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

            for (i, (&(mesh, material), transforms)) in meshes.iter().enumerate() {
                profiling::scope!("mesh");
                let transform_buffer = {
                    profiling::scope!("create transform buffer");
                    let buffer_size = (transforms.len() * mem::size_of::<Mat4>()) as vk::DeviceSize;
                    let buffer_create_info = vk::BufferCreateInfo::builder()
                        .size(buffer_size)
                        .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE);
                    temp_arena.create_buffer(
                        *buffer_create_info,
                        format_args!("{}. transform buffer of pipeline {pipeline:?}", i + 1),
                    )?
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
