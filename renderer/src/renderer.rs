use alloc::rc::Rc;
use core::mem;
use core::time::Duration;
use std::thread;

use arrayvec::ArrayVec;
use ash::{vk, Instance};
use bytemuck::{Pod, Zeroable};
use enum_map::Enum;
use hashbrown::HashMap;

use crate::arena::buffers::{BufferUsage, ForBuffers};
use crate::arena::{MemoryProps, VulkanArena, VulkanArenaError};
use crate::physical_device::PhysicalDevice;
use crate::vertex_library::{VertexLibrary, VERTEX_LIBRARY_INDEX_TYPE};
use crate::vulkan_raii::{Buffer, CommandBuffer, CommandPool, Device, Fence, Semaphore};
use crate::JointsOffset;

pub(crate) mod descriptors;
pub(crate) mod framebuffers;
pub(crate) mod pipeline_parameters;
pub(crate) mod pipelines;
pub(crate) mod scene;
pub(crate) mod swapchain;

use descriptors::Descriptors;
use framebuffers::Framebuffers;
use pipeline_parameters::render_passes::{Attachment, RenderPass};
use pipeline_parameters::vertex_buffers::VertexBinding;
use pipeline_parameters::{uniforms, PipelineIndex, PipelineMap};
use pipelines::Pipelines;
use scene::Scene;
use swapchain::{Swapchain, SwapchainError};

/// Get from [`Renderer::wait_frame`].
pub struct FrameIndex {
    index: usize,
}

impl FrameIndex {
    fn new(index: usize) -> FrameIndex {
        FrameIndex { index }
    }
}

/// Wrapper around [`vk::DrawIndexedIndirectCommand`] which implements [`Pod`]
/// and [`Zeroable`].
#[derive(Clone, Copy)]
#[repr(C)]
struct DrawIndexedIndirectCommand(vk::DrawIndexedIndirectCommand);
unsafe impl Zeroable for DrawIndexedIndirectCommand {}
unsafe impl Pod for DrawIndexedIndirectCommand {}

pub struct Renderer {
    device: Device,
    // TODO: Create an "queue submission" interface for uploaders and renderer to use (one per frame in flight?)
    // they should at least hold the command pool, have a way to submit the
    // commands (with it internally tracking the submission with a fence), and
    // expose a "done yet?" function (using the fence)
    frame_start_fence: Fence,
    ready_for_present: Semaphore,
    frame_end_fence: Fence,
    temp_arena: VulkanArena<ForBuffers>,
    command_pool: Rc<CommandPool>,
    command_buffer: Option<CommandBuffer>,
    uniform_joints_offsets: uniforms::JointsOffsets,
    uniform_material_indices: uniforms::MaterialIds,
}

impl Renderer {
    pub fn new(instance: &Instance, device: &Device, physical_device: &PhysicalDevice) -> Renderer {
        profiling::scope!("renderer creation (per-frame-stuff)");

        let ready_for_present = {
            let semaphore = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                .expect("system should have enough memory to create vulkan semaphores");
            crate::name_vulkan_object(device, semaphore, format_args!("render finish semaphore"));
            Semaphore { inner: semaphore, device: device.clone() }
        };

        let frame_start_fence = {
            let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
                .expect("system should have enough memory to create vulkan fences");
            crate::name_vulkan_object(device, fence, format_args!("wait_frame fence"));
            Fence { inner: fence, device: device.clone() }
        };

        let frame_end_fence = {
            let fence_create_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            let fence =
                unsafe { device.create_fence(&fence_create_info, None) }.expect("system should have enough memory to create vulkan fences");
            crate::name_vulkan_object(device, fence, format_args!("frame end fence"));
            Fence { inner: fence, device: device.clone() }
        };

        let command_pool = {
            let command_pool_create_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(physical_device.graphics_queue_family.index)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);
            let command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None) }
                .expect("system should have enough memory to create vulkan command pools");
            crate::name_vulkan_object(device, command_pool, format_args!("rendering cmds"));
            Rc::new(CommandPool { inner: command_pool, device: device.clone() })
        };

        let temp_arena =
            VulkanArena::new(instance, device, physical_device, 10_000_000, MemoryProps::for_buffers(), format_args!("frame local arena"))
                .expect("system should have enough memory for the renderer's temp arena");

        Renderer {
            device: device.clone(),
            frame_start_fence,
            ready_for_present,
            frame_end_fence,
            temp_arena,
            command_pool,
            command_buffer: None,
            uniform_joints_offsets: uniforms::JointsOffsets::zeroed(),
            uniform_material_indices: uniforms::MaterialIds::zeroed(),
        }
    }

    /// Wait until the next frame can start rendering.
    ///
    /// After ensuring that the next frame can be rendered, this also
    /// frees the resources that can now be freed up.
    #[profiling::function]
    pub fn wait_frame(&mut self, swapchain: &Swapchain) -> Result<FrameIndex, SwapchainError> {
        let (image_index, _) = unsafe {
            profiling::scope!("acquire next image");
            let fence = self.frame_start_fence.inner;
            loop {
                match swapchain.device().acquire_next_image(swapchain.inner(), u64::MAX, vk::Semaphore::null(), fence) {
                    Ok(result) => break result,
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => return Err(SwapchainError::OutOfDate),
                    Err(vk::Result::TIMEOUT | vk::Result::NOT_READY) => {
                        // This shouldn't ever happen, given the u64::MAX timeout, but this is better than panicking.
                        thread::sleep(Duration::from_millis(500));
                        continue;
                    }
                    Err(err) => panic!("acquiring vulkan swapchain images should not fail: {err}"),
                }
            }
        };

        let fences = [self.frame_start_fence.inner, self.frame_end_fence.inner];
        unsafe {
            profiling::scope!("wait for the image and for the previous frame to finish");
            self.device.wait_for_fences(&fences, true, u64::MAX).expect("waiting for vulkan fences should not fail");
            self.device.reset_fences(&fences).expect("resetting vulkan fences should not fail");
        }

        // TODO: This reset *could* technically not be valid, if the buffers are still in use.
        self.temp_arena.reset().expect("renderer's temp arena should not have any hanging buffers at this point");

        // We know they aren't, since we only have one frame in flight and we've
        // waited on the previous frame's fence, but as the user might want to
        // drop some resources during rendering, "resources in use" would be a
        // good thing to track.

        // The idea: in render_frame, and maybe somewhere in Descriptors, clone
        // each Rc<Buffer> and Rc<Image> being used into some
        // ResourcesInUseDuringAFrame struct. When queueing the command buffer
        // using these resources to be executed, save the fence from that in the
        // struct as well. Then, the struct should be stored somewhere
        // relatively static, which would occasionally check the fence, and if
        // rendering has finished, it drops all the Rc's. Maybe take a &'static
        // GpuResourceWatcher, on which you'd periodically call like,
        // ::release_resources()?

        // This is just additional overhead to what we have now, but it would
        // make the Rc<Buffer/Image> architecture so much safer. With a similar
        // thing for Uploader (it could use the GpuResourceWatcher as well!), we
        // might not even invalidly destroy everything on panic, like we
        // currently do!

        Ok(FrameIndex::new(image_index as usize))
    }

    /// Starts rendering the frame. Returns a VulkanArenaError if the internal
    /// temporary memory arena fills up.
    #[profiling::function]
    pub fn render_frame(
        &mut self,
        frame_index: &FrameIndex,
        descriptors: &mut Descriptors,
        pipelines: &Pipelines,
        framebuffers: &Framebuffers,
        scene: &mut Scene,
        debug_value: u32,
    ) {
        fn create_uniform_buffer<T: bytemuck::Pod>(
            temp_arena: &mut VulkanArena<ForBuffers>,
            buffer: &[T],
            name: &str,
        ) -> Result<Buffer, VulkanArenaError> {
            profiling::scope!(name);
            let buffer_bytes: &[u8] = bytemuck::cast_slice(buffer);
            let buffer_create_info = vk::BufferCreateInfo::default()
                .size(buffer_bytes.len() as u64)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let usage = BufferUsage::UNIFORM;
            temp_arena.create_buffer(buffer_create_info, usage, buffer_bytes, None, None, format_args!("uniform ({name})"))
        }

        // Prepare the data (CPU-side work):

        let vk::Extent2D { width, height } = framebuffers.extent;
        // Contains interleaved 4x3 and 3x3 matrices (regular transforms and their inverse transposes)
        let mut transforms: Vec<f32> = Vec::new();
        let mut draws: PipelineMap<HashMap<&VertexLibrary, Vec<DrawIndexedIndirectCommand>>> = PipelineMap::from_fn(|_| HashMap::new());

        scene.draws.sort();
        let mut prev_tag = None;
        let mut prev_joints = None;
        for draw in &scene.draws {
            let pipeline = draw.tag.pipeline;
            let index_count = draw.tag.mesh.index_count;
            let first_index = draw.tag.mesh.first_index;
            let vertex_offset = draw.tag.mesh.vertex_offset;

            let indirect_draw_set = draws[pipeline].entry(draw.tag.vertex_library).or_default();

            let first_instance = (transforms.len() / (4 * 3 + 3 * 3)) as u32;
            let normal_transform = draw.transform.matrix3.inverse().transpose();
            transforms.extend_from_slice(&draw.transform.to_cols_array());
            transforms.extend_from_slice(&normal_transform.to_cols_array());

            if Some(draw.tag) == prev_tag && Some(draw.joints) == prev_joints {
                indirect_draw_set.last_mut().unwrap().0.instance_count += 1;
            } else {
                indirect_draw_set.push(DrawIndexedIndirectCommand(vk::DrawIndexedIndirectCommand {
                    index_count,
                    instance_count: 1,
                    first_index,
                    vertex_offset,
                    first_instance,
                }));
                self.uniform_material_indices.material_id[first_instance as usize] = draw.tag.material.material_id;
                if let Some(JointsOffset(joints_offset)) = draw.joints {
                    self.uniform_joints_offsets.joints_offset[first_instance as usize] = joints_offset;
                }
                prev_tag = Some(draw.tag);
                prev_joints = Some(draw.joints);
            }
        }

        let transforms_buffer = {
            profiling::scope!("create transform buffer");
            let transforms_bytes = bytemuck::cast_slice::<f32, u8>(&transforms);
            let buffer_create_info = vk::BufferCreateInfo::default()
                .size(transforms_bytes.len() as vk::DeviceSize)
                .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            self.temp_arena
                .create_buffer(buffer_create_info, BufferUsage::VERTEX, transforms_bytes, None, None, format_args!("transforms"))
                .expect("renderer's temp arena should have enough memory for the transforms buffer")
        };

        // Create and update descriptors (buffer allocations and then desc writes):

        let global_transforms = &[scene.camera.create_proj_view_transforms(width as f32, height as f32, scene.world_space)];
        let global_transforms_buffer = create_uniform_buffer(&mut self.temp_arena, global_transforms, "view+proj matrices")
            .expect("renderer's temp arena should have enough memory for the view+proj matrices buffer");

        let render_settings = &[uniforms::RenderSettings { debug_value }];
        let render_settings_buffer = create_uniform_buffer(&mut self.temp_arena, render_settings, "render settings")
            .expect("renderer's temp arena should have enough memory for the render settings buffer");

        let skinned_mesh_joints = &mut scene.skinned_mesh_joints_buffer;
        let skinned_mesh_joints_buffer = create_uniform_buffer(&mut self.temp_arena, skinned_mesh_joints, "joint transforms")
            .expect("renderer's temp arena should have enough memory for the joint transforms buffer");

        let materials_temp_uniform = descriptors
            .create_temp_uniforms(&mut self.temp_arena)
            .expect("renderer's temp arena should have enough memory for the materials buffer");

        let draw_call_vert_params = &[self.uniform_joints_offsets];
        let draw_call_vert_params_buffer =
            create_uniform_buffer(&mut self.temp_arena, draw_call_vert_params, "draw call params (for vertex shader)")
                .expect("renderer's temp arena should have enough memory for the draw call params buffer");

        let draw_call_frag_params = &[self.uniform_material_indices];
        let draw_call_frag_params_buffer =
            create_uniform_buffer(&mut self.temp_arena, draw_call_frag_params, "draw call params (for fragment shader)")
                .expect("renderer's temp arena should have enough memory for the draw call params buffer");

        descriptors.write_descriptors(
            &global_transforms_buffer,
            &render_settings_buffer,
            &skinned_mesh_joints_buffer,
            &draw_call_vert_params_buffer,
            &draw_call_frag_params_buffer,
            &materials_temp_uniform,
            &framebuffers.hdr_image,
        );

        self.temp_arena.add_buffer(global_transforms_buffer);
        self.temp_arena.add_buffer(render_settings_buffer);
        self.temp_arena.add_buffer(skinned_mesh_joints_buffer);
        self.temp_arena.add_buffer(materials_temp_uniform.buffer);
        self.temp_arena.add_buffer(draw_call_vert_params_buffer);
        self.temp_arena.add_buffer(draw_call_frag_params_buffer);

        // Draw (record the actual draw calls):

        let command_buffer = self.record_command_buffer(frame_index, descriptors, pipelines, framebuffers, &draws, &transforms_buffer);
        self.temp_arena.add_buffer(transforms_buffer);

        let signal_semaphores = [
            vk::SemaphoreSubmitInfo::default().semaphore(self.ready_for_present.inner).stage_mask(vk::PipelineStageFlags2::NONE), // this signals vkQueuePresent, which does not need synchronization nor have a stage
        ];
        let command_buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)];
        let submit_infos = [vk::SubmitInfo2::default().signal_semaphore_infos(&signal_semaphores).command_buffer_infos(&command_buffers)];
        unsafe {
            profiling::scope!("queue render");
            self.device
                .queue_submit2(self.device.graphics_queue, &submit_infos, self.frame_end_fence.inner)
                .expect("vulkan queue submission should not fail");
        }
    }

    #[profiling::function]
    pub fn present_frame(&mut self, frame_index: FrameIndex, swapchain: &Swapchain) -> Result<(), SwapchainError> {
        let wait_semaphores = [self.ready_for_present.inner];
        let swapchains = [swapchain.inner()];
        let image_indices = [frame_index.index as u32];
        let present_info =
            vk::PresentInfoKHR::default().wait_semaphores(&wait_semaphores).swapchains(&swapchains).image_indices(&image_indices);
        let present_result = unsafe {
            profiling::scope!("queue present");
            swapchain.device().queue_present(self.device.surface_queue, &present_info)
        };

        match present_result {
            Ok(false) => Ok(()),
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Err(SwapchainError::OutOfDate),
            Err(err) => panic!("all vulkan queue present errors should've been handled: {err}"),
        }
    }

    #[profiling::function]
    fn record_command_buffer(
        &mut self,
        frame_index: &FrameIndex,
        descriptors: &mut Descriptors,
        pipelines: &Pipelines,
        framebuffers: &Framebuffers,
        draws: &PipelineMap<HashMap<&VertexLibrary, Vec<DrawIndexedIndirectCommand>>>,
        transforms_buffer: &Buffer,
    ) -> vk::CommandBuffer {
        let command_pool = self.command_pool.inner;
        unsafe {
            profiling::scope!("reset command pool");
            self.device
                .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())
                .expect("beginning vulkan command buffer recording should not fail");
        };

        let command_buffer = if let Some(command_buffer) = &self.command_buffer {
            command_buffer.inner
        } else {
            profiling::scope!("allocate command buffer");
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffers = unsafe { self.device.allocate_command_buffers(&command_buffer_allocate_info) }
                .expect("system should have enough memory to allocate vulkan command buffers");
            self.command_buffer =
                Some(CommandBuffer { inner: command_buffers[0], device: self.device.clone(), command_pool: self.command_pool.clone() });
            command_buffers[0]
        };

        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            profiling::scope!("begin command buffer");
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .expect("beginning vulkan command buffer recording should not fail");
        }
        crate::name_vulkan_object(&self.device, command_buffer, format_args!("one frame's rendering cmds"));

        // Prepare geometry render pass:

        let fb_geom_pass_barrier = RenderPass::Geometry.barriers(&framebuffers.attachment_images(frame_index.index));
        let dep_info = vk::DependencyInfo::default().image_memory_barriers(&fb_geom_pass_barrier);
        unsafe { self.device.cmd_pipeline_barrier2(command_buffer, &dep_info) };

        let color_attachments = RenderPass::Geometry.color_attachment_infos(&framebuffers.attachment_image_views(frame_index.index), &[]);
        let depth_attachment = RenderPass::Geometry.depth_attachment_info(&framebuffers.attachment_image_views(frame_index.index));
        let render_area = vk::Rect2D::default().extent(framebuffers.extent);
        let rendering_info = vk::RenderingInfoKHR::default()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment);
        unsafe {
            profiling::scope!("begin main rendering");
            self.device.cmd_begin_rendering(command_buffer, &rendering_info);
        }

        let shared_pipeline = PipelineIndex::SHARED_DESCRIPTOR_PIPELINE;
        let shared_descriptor_set = descriptors.descriptor_sets(shared_pipeline)[0];
        unsafe {
            profiling::scope!("bind shared descriptor set");
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                descriptors.pipeline_layouts[shared_pipeline].inner,
                0,
                &[shared_descriptor_set],
                &[],
            );
        }

        for pl_idx in [
            PipelineIndex::PbrOpaque,
            PipelineIndex::PbrSkinnedOpaque,
            PipelineIndex::PbrSkinnedAlphaToCoverage,
            PipelineIndex::PbrAlphaToCoverage,
            PipelineIndex::PbrBlended,
            PipelineIndex::PbrSkinnedBlended,
        ] {
            profiling::scope!("pipeline");
            let draws = &draws[pl_idx];
            if draws.is_empty() {
                continue;
            }
            let pipeline = pipelines.pipelines[pl_idx].inner;
            let layout = descriptors.pipeline_layouts[pl_idx].inner;
            let descriptor_sets = descriptors.descriptor_sets(pl_idx);
            let bind_point = vk::PipelineBindPoint::GRAPHICS;
            unsafe { self.device.cmd_bind_pipeline(command_buffer, bind_point, pipeline) };
            unsafe {
                self.device.cmd_bind_descriptor_sets(command_buffer, bind_point, layout, 1, &descriptor_sets[1..], &[]);
            }
            let vertex_layout = pl_idx.vertex_layout();
            for (vertex_library, draws) in draws {
                const VERTEX_BUFFERS: usize = VertexBinding::LENGTH;
                let mut vertex_offsets = ArrayVec::<vk::DeviceSize, VERTEX_BUFFERS>::new();
                vertex_offsets.push(0);
                vertex_offsets.extend(
                    vertex_layout.required_inputs().iter().map(|&b| vertex_library.vertex_buffer_offsets[vertex_layout][b].unwrap()),
                );
                let mut vertex_buffers = ArrayVec::<vk::Buffer, VERTEX_BUFFERS>::new();
                vertex_buffers.push(transforms_buffer.inner);
                for _ in 1..vertex_offsets.len() {
                    vertex_buffers.push(vertex_library.vertex_buffer.inner);
                }

                let index_buffer = vertex_library.index_buffer.inner;
                let index_buffer_offset = vertex_library.index_buffer_offsets[vertex_layout];

                unsafe {
                    self.device.cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &vertex_offsets);
                    self.device.cmd_bind_index_buffer(command_buffer, index_buffer, index_buffer_offset, VERTEX_LIBRARY_INDEX_TYPE);
                }

                let indirect_draws_buffer = {
                    profiling::scope!("create indirect draws buffer");
                    let draw_cmds_bytes = bytemuck::cast_slice(draws);
                    let buffer_create_info = vk::BufferCreateInfo::default()
                        .size(draw_cmds_bytes.len() as vk::DeviceSize)
                        .usage(vk::BufferUsageFlags::INDIRECT_BUFFER)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE);
                    self.temp_arena
                        .create_buffer(
                            buffer_create_info,
                            BufferUsage::INDIRECT_DRAW,
                            draw_cmds_bytes,
                            None,
                            None,
                            format_args!("indirect draw command buffer ({pl_idx:?})"),
                        )
                        .expect("renderer's temp arena should have enough memory for the indirect draws buffer")
                };
                let draw_count = draws.len() as u32;
                let stride = mem::size_of::<DrawIndexedIndirectCommand>() as u32;
                unsafe {
                    profiling::scope!("draw indexed indirect");
                    self.device.cmd_draw_indexed_indirect(command_buffer, indirect_draws_buffer.inner, 0, draw_count, stride);
                }
                self.temp_arena.add_buffer(indirect_draws_buffer);
            }
        }

        unsafe {
            profiling::scope!("end main rendering");
            self.device.cmd_end_rendering(command_buffer);
        }

        // End of geometry render pass.

        // Prepare post-processing render pass:

        let fb_pp_pass_barrier = RenderPass::PostProcess.barriers(&framebuffers.attachment_images(frame_index.index));
        let dep_info = vk::DependencyInfo::default().image_memory_barriers(&fb_pp_pass_barrier);
        unsafe { self.device.cmd_pipeline_barrier2(command_buffer, &dep_info) };

        let swapchain_to_write_layout = framebuffers.swapchain_write_barrier(frame_index.index);
        let dep_info = vk::DependencyInfo::default().image_memory_barriers(&swapchain_to_write_layout);
        unsafe { self.device.cmd_pipeline_barrier2(command_buffer, &dep_info) };

        let mut resolve_targets = ArrayVec::<(Attachment, vk::ImageView), 1>::new();
        if framebuffers.multisampled_final_image.is_some() {
            resolve_targets.push((Attachment::PostProcess, framebuffers.swapchain_images[frame_index.index].inner));
        }
        let color_attachments =
            RenderPass::PostProcess.color_attachment_infos(&framebuffers.attachment_image_views(frame_index.index), &resolve_targets);
        let depth_attachment = RenderPass::PostProcess.depth_attachment_info(&framebuffers.attachment_image_views(frame_index.index));
        let render_area = vk::Rect2D::default().extent(framebuffers.extent);
        let rendering_info = vk::RenderingInfoKHR::default()
            .render_area(render_area)
            .layer_count(1)
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment);
        unsafe {
            profiling::scope!("begin post-processing rendering");
            self.device.cmd_begin_rendering(command_buffer, &rendering_info);
        }

        {
            profiling::scope!("record tonemapping subpass");
            let pl_index = PipelineIndex::RenderResolutionPostProcess;
            let bind_point = vk::PipelineBindPoint::GRAPHICS;
            unsafe {
                self.device.cmd_bind_pipeline(command_buffer, bind_point, pipelines.pipelines[pl_index].inner);
            }
            let layout = descriptors.pipeline_layouts[pl_index].inner;
            let descriptors = &descriptors.descriptor_sets(pl_index)[1..];
            unsafe {
                self.device.cmd_bind_descriptor_sets(command_buffer, bind_point, layout, 1, descriptors, &[]);
            }
            unsafe { self.device.cmd_draw(command_buffer, 3, 1, 0, 0) };
        }

        unsafe {
            profiling::scope!("end rendering");
            self.device.cmd_end_rendering(command_buffer);
        }

        let swapchain_to_present_layout = framebuffers.swapchain_present_barrier(frame_index.index);
        let dep_info = vk::DependencyInfo::default().image_memory_barriers(&swapchain_to_present_layout);
        unsafe { self.device.cmd_pipeline_barrier2(command_buffer, &dep_info) };

        // End of post-processing render pass.

        unsafe {
            profiling::scope!("end command buffer");
            self.device.end_command_buffer(command_buffer).expect("ending vulkan command buffer recording should not fail");
        }

        command_buffer
    }
}
