use alloc::rc::Rc;
use core::mem;

use arrayvec::ArrayVec;
use ash::{vk, Instance};
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use hashbrown::HashMap;

use crate::arena::buffers::ForBuffers;
use crate::arena::{MemoryProps, VulkanArena, VulkanArenaError};
use crate::physical_device::PhysicalDevice;
use crate::vertex_library::{VertexLibrary, VERTEX_LIBRARY_INDEX_TYPE};
use crate::vulkan_raii::{Buffer, CommandBuffer, CommandPool, Device, Fence, Semaphore};

pub(crate) mod descriptors;
pub(crate) mod framebuffers;
pub(crate) mod pipeline_parameters;
pub(crate) mod pipelines;
pub(crate) mod scene;
pub(crate) mod swapchain;

use descriptors::{DescriptorError, Descriptors};
use framebuffers::Framebuffers;
use pipeline_parameters::constants::MAX_BONE_COUNT;
use pipeline_parameters::render_passes::{Attachment, RenderPass};
use pipeline_parameters::{DrawCallParametersSoa, PipelineIndex, PipelineMap, RenderSettings};
use pipelines::Pipelines;
use scene::Scene;
use swapchain::Swapchain;

use self::pipeline_parameters::VERTEX_BINDING_COUNT;

#[derive(thiserror::Error, Debug)]
pub enum RendererError {
    #[error("failed to create semaphore for renderer signalling present")]
    PresentReadySemaphoreCreation(#[source] vk::Result),
    #[error("failed to create semaphore for renderer signalling frame start")]
    FrameStartSemaphoreCreation(#[source] vk::Result),
    #[error("failed to create fence for renderer signalling frame end")]
    FrameEndFenceCreation(#[source] vk::Result),
    #[error("failed to create command pools for renderer")]
    CommandPoolCreation(#[source] vk::Result),
    #[error("failed to create frame-local vulkan arenas")]
    FrameLocalArenaCreation(#[source] VulkanArenaError),
    #[error("failed to acquire next frame's swapchain image (window issues?)")]
    AcquireImage(#[source] vk::Result),
    #[error("failed to wait for the frame end fence")]
    FrameEndFenceWait(#[source] vk::Result),
    #[error("failed to rest the frame-local vulkan arena")]
    FrameLocalArenaReset(#[source] VulkanArenaError),
    #[error("failed to create uniform buffer for camera transforms")]
    CameraTransformUniformCreation(#[source] VulkanArenaError),
    #[error("failed to create uniform buffer for render settings")]
    RenderSettingsUniformCreation(#[source] VulkanArenaError),
    #[error("failed to create uniform buffer for skinned meshes' joint transforms")]
    JointTransformUniformCreation(#[source] VulkanArenaError),
    #[error("failed to create per-frame material uniforms")]
    MaterialsUniformCreation(#[source] DescriptorError),
    #[error("failed to create draw call parameters uniform")]
    DrawCallParamsUniformCreation(#[source] VulkanArenaError),
    #[error("failed to submit rendering command buffers to the graphics queue (device lost or out of memory?)")]
    RenderQueueSubmit(#[source] vk::Result),
    #[error("present was successful, but may display oddly; swapchain is out of date")]
    SwapchainOutOfDate,
    #[error("failed to present to the surface queue (window issues?)")]
    RenderQueuePresent(#[source] vk::Result),
    #[error("failed to reset command pools for rendering")]
    CommandPoolReset(#[source] vk::Result),
    #[error("failed to allocate command buffers for rendering")]
    CommandBufferAllocation(#[source] vk::Result),
    #[error("failed to begin command buffer for rendering")]
    CommandBufferBegin(#[source] vk::Result),
    #[error("failed to create transform buffer")]
    TransformBufferCreation(#[source] VulkanArenaError),
    #[error("failed to create indirect draw command buffer")]
    IndirectCmdBufferCreation(#[source] VulkanArenaError),
    #[error("failed to end rendering command buffer")]
    CommandBufferEnd(#[source] vk::Result),
}

/// Get from [`Renderer::wait_frame`].
pub struct FrameIndex {
    index: usize,
}

impl FrameIndex {
    fn new(index: usize) -> FrameIndex {
        FrameIndex { index }
    }
}

struct DrawCallParameters {
    data: DrawCallParametersSoa,
    instance_count: vk::DeviceSize,
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
    frame_start_fence: Fence,
    ready_for_present: Semaphore,
    frame_end_fence: Fence,
    temp_arena: VulkanArena<ForBuffers>,
    draw_call_parameters: DrawCallParameters,
    command_pool: Rc<CommandPool>,
    command_buffer: Option<CommandBuffer>,
}

impl Renderer {
    pub fn new(instance: &Instance, device: &Device, physical_device: &PhysicalDevice) -> Result<Renderer, RendererError> {
        profiling::scope!("renderer creation (per-frame-stuff)");

        let ready_for_present = {
            let semaphore = unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                .map_err(RendererError::PresentReadySemaphoreCreation)?;
            crate::name_vulkan_object(device, semaphore, format_args!("render finish semaphore"));
            Semaphore {
                inner: semaphore,
                device: device.clone(),
            }
        };

        let frame_end_fence = {
            let fence_create_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            let fence = unsafe { device.create_fence(&fence_create_info, None) }.map_err(RendererError::FrameEndFenceCreation)?;
            crate::name_vulkan_object(device, fence, format_args!("frame end fence"));
            Fence {
                inner: fence,
                device: device.clone(),
            }
        };

        let command_pool = {
            let command_pool_create_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(physical_device.graphics_queue_family.index)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);
            let command_pool =
                unsafe { device.create_command_pool(&command_pool_create_info, None) }.map_err(RendererError::CommandPoolCreation)?;
            crate::name_vulkan_object(device, command_pool, format_args!("rendering cmds"));
            Rc::new(CommandPool {
                inner: command_pool,
                device: device.clone(),
            })
        };

        let temp_arena = VulkanArena::new(
            instance,
            device,
            physical_device,
            10_000_000,
            MemoryProps::for_buffers(),
            format_args!("frame local arena"),
        )
        .map_err(RendererError::FrameLocalArenaCreation)?;

        let frame_start_fence = unsafe {
            device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(RendererError::FrameStartSemaphoreCreation)
        }?;
        crate::name_vulkan_object(device, frame_start_fence, format_args!("wait_frame fence"));
        let frame_start_fence = Fence {
            inner: frame_start_fence,
            device: device.clone(),
        };

        let draw_call_parameters = DrawCallParameters {
            data: DrawCallParametersSoa::zeroed(),
            instance_count: 0,
        };

        Ok(Renderer {
            device: device.clone(),
            frame_start_fence,
            ready_for_present,
            frame_end_fence,
            temp_arena,
            draw_call_parameters,
            command_pool,
            command_buffer: None,
        })
    }

    /// Wait until the next frame can start rendering.
    ///
    /// After ensuring that the next frame can be rendered, this also
    /// frees the resources that can now be freed up.
    #[profiling::function]
    pub fn wait_frame(&mut self, swapchain: &Swapchain) -> Result<FrameIndex, RendererError> {
        let (image_index, _) = unsafe {
            profiling::scope!("acquire next image");
            let fence = self.frame_start_fence.inner;
            swapchain
                .device()
                .acquire_next_image(swapchain.inner(), u64::MAX, vk::Semaphore::null(), fence)
                .map_err(|err| match err {
                    vk::Result::ERROR_OUT_OF_DATE_KHR => RendererError::SwapchainOutOfDate,
                    err => RendererError::AcquireImage(err),
                })
        }?;

        let fences = [self.frame_start_fence.inner, self.frame_end_fence.inner];
        unsafe {
            profiling::scope!("wait for the image and for the previous frame to finish");
            self.device
                .wait_for_fences(&fences, true, u64::MAX)
                .map_err(RendererError::FrameEndFenceWait)?;
            self.device.reset_fences(&fences).map_err(RendererError::FrameEndFenceWait)?;
        }
        self.temp_arena.reset().map_err(RendererError::FrameLocalArenaReset)?;

        Ok(FrameIndex::new(image_index as usize))
    }

    #[profiling::function]
    pub fn render_frame(
        &mut self,
        frame_index: &FrameIndex,
        descriptors: &mut Descriptors,
        pipelines: &Pipelines,
        framebuffers: &Framebuffers,
        mut scene: Scene,
        debug_value: u32,
    ) -> Result<(), RendererError> {
        fn create_uniform_buffer<T: bytemuck::Pod>(
            temp_arena: &mut VulkanArena<ForBuffers>,
            buffer: &[T],
            name: &str,
        ) -> Result<Buffer, VulkanArenaError> {
            let buffer_bytes: &[u8] = bytemuck::cast_slice(buffer);
            let buffer_create_info = vk::BufferCreateInfo::default()
                .size(buffer_bytes.len() as u64)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            temp_arena.create_buffer(buffer_create_info, buffer_bytes, None, None, format_args!("uniform ({name})"))
        }

        // Prepare the data (CPU-side work):

        let vk::Extent2D { width, height } = framebuffers.extent;
        self.draw_call_parameters.instance_count = 0;
        let mut transforms = Vec::new();
        let mut draws: PipelineMap<HashMap<&VertexLibrary, Vec<DrawIndexedIndirectCommand>>> =
            PipelineMap::from_infallible(|_| HashMap::new());
        scene.static_draws.sort();
        let mut prev_tag = None;
        for static_draw in scene.static_draws {
            let pipeline = static_draw.tag.pipeline;
            let index_count = static_draw.tag.mesh.index_count;
            let first_index = static_draw.tag.mesh.first_index;
            let vertex_offset = static_draw.tag.mesh.vertex_offset;
            let material_index = static_draw.tag.material.array_index(pipeline).unwrap();

            let indirect_draw_set = draws[pipeline].entry(static_draw.tag.vertex_library).or_default();

            let first_instance = transforms.len() as u32;
            transforms.push(static_draw.transform);
            self.draw_call_parameters.instance_count += 1;

            if Some(static_draw.tag) == prev_tag {
                indirect_draw_set.last_mut().unwrap().0.instance_count += 1;
            } else {
                indirect_draw_set.push(DrawIndexedIndirectCommand(vk::DrawIndexedIndirectCommand {
                    index_count,
                    instance_count: 1,
                    first_index,
                    vertex_offset,
                    first_instance,
                }));
                self.draw_call_parameters.data.material_index[first_instance as usize] = material_index;
                prev_tag = Some(static_draw.tag);
            }
        }

        let transforms_buffer = {
            profiling::scope!("create transform buffer");
            let transforms_bytes = bytemuck::cast_slice::<Mat4, u8>(&transforms);
            let buffer_create_info = vk::BufferCreateInfo::default()
                .size(transforms_bytes.len() as vk::DeviceSize)
                .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            self.temp_arena
                .create_buffer(buffer_create_info, transforms_bytes, None, None, format_args!("transforms"))
                .map_err(RendererError::TransformBufferCreation)?
        };

        // Create and update descriptors (buffer allocations and then desc writes):

        let global_transforms = &[scene
            .camera
            .create_proj_view_transforms(width as f32, height as f32, scene.world_space)];
        let global_transforms_buffer = create_uniform_buffer(&mut self.temp_arena, global_transforms, "view+proj matrices")
            .map_err(RendererError::CameraTransformUniformCreation)?;

        let render_settings = &[RenderSettings { debug_value }];
        let render_settings_buffer = create_uniform_buffer(&mut self.temp_arena, render_settings, "render settings")
            .map_err(RendererError::RenderSettingsUniformCreation)?;

        let mut skinned_mesh_joints = scene.skinned_mesh_joints_buffer;
        // The joint buffer needs to have backing memory for the entire uniform
        // buffer's length at all offsets, so without this padding, the last
        // (few) skeletons would overflow the buffer's end.
        let empty_full_length_skeleton = &[Mat4::ZERO; MAX_BONE_COUNT as usize];
        skinned_mesh_joints.extend_from_slice(bytemuck::cast_slice(empty_full_length_skeleton));
        let skinned_mesh_joints_buffer = create_uniform_buffer(&mut self.temp_arena, &skinned_mesh_joints, "joint transforms")
            .map_err(RendererError::JointTransformUniformCreation)?;

        let materials_temp_uniform = descriptors
            .create_materials_temp_uniform(&mut self.temp_arena)
            .map_err(RendererError::MaterialsUniformCreation)?;

        let mut draw_call_params_update_ranges = [(0, 0); 1];
        let draw_call_params = if self.draw_call_parameters.instance_count > 0 {
            let draw_call_parameters = &[self.draw_call_parameters.data];
            let draw_call_params_buffer = create_uniform_buffer(&mut self.temp_arena, draw_call_parameters, "draw call params")
                .map_err(RendererError::DrawCallParamsUniformCreation)?;
            draw_call_params_update_ranges[0] = (
                DrawCallParametersSoa::MATERIAL_INDEX_OFFSET,
                self.draw_call_parameters.instance_count * DrawCallParametersSoa::MATERIAL_INDEX_ELEMENT_SIZE,
            );
            let params = (draw_call_params_buffer.inner, &draw_call_params_update_ranges[..]);
            self.temp_arena.add_buffer(draw_call_params_buffer);
            Some(params)
        } else {
            None
        };

        descriptors.write_descriptors(
            &global_transforms_buffer,
            &render_settings_buffer,
            &skinned_mesh_joints_buffer,
            &materials_temp_uniform,
            draw_call_params,
            &framebuffers.hdr_image,
        );

        self.temp_arena.add_buffer(global_transforms_buffer);
        self.temp_arena.add_buffer(render_settings_buffer);
        self.temp_arena.add_buffer(skinned_mesh_joints_buffer);
        self.temp_arena.add_buffer(materials_temp_uniform.buffer);

        // Draw (record the actual draw calls):

        let command_buffer = self.record_command_buffer(frame_index, descriptors, pipelines, framebuffers, &draws, &transforms_buffer)?;
        self.temp_arena.add_buffer(transforms_buffer);

        let signal_semaphores = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.ready_for_present.inner)
                .stage_mask(vk::PipelineStageFlags2::NONE), // this signals vkQueuePresent, which does not need synchronization nor have a stage
        ];
        let command_buffers = [vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)];
        let submit_infos = [vk::SubmitInfo2::default()
            .signal_semaphore_infos(&signal_semaphores)
            .command_buffer_infos(&command_buffers)];
        unsafe {
            profiling::scope!("queue render");
            self.device
                .sync2
                .queue_submit2(self.device.graphics_queue, &submit_infos, self.frame_end_fence.inner)
                .map_err(RendererError::RenderQueueSubmit)
        }
    }

    #[profiling::function]
    pub fn present_frame(&mut self, frame_index: FrameIndex, swapchain: &Swapchain) -> Result<(), RendererError> {
        let wait_semaphores = [self.ready_for_present.inner];
        let swapchains = [swapchain.inner()];
        let image_indices = [frame_index.index as u32];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&wait_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        let present_result = unsafe {
            profiling::scope!("queue present");
            swapchain.device().queue_present(self.device.surface_queue, &present_info)
        };

        match present_result {
            Ok(false) => Ok(()),
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Err(RendererError::SwapchainOutOfDate),
            Err(err) => Err(RendererError::RenderQueuePresent(err)),
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
    ) -> Result<vk::CommandBuffer, RendererError> {
        let command_pool = self.command_pool.inner;
        unsafe {
            profiling::scope!("reset command pool");
            self.device
                .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())
                .map_err(RendererError::CommandPoolReset)
        }?;

        let command_buffer = if let Some(command_buffer) = &self.command_buffer {
            command_buffer.inner
        } else {
            profiling::scope!("allocate command buffer");
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffers = unsafe { self.device.allocate_command_buffers(&command_buffer_allocate_info) }
                .map_err(RendererError::CommandBufferAllocation)?;
            self.command_buffer = Some(CommandBuffer {
                inner: command_buffers[0],
                device: self.device.clone(),
                command_pool: self.command_pool.clone(),
            });
            command_buffers[0]
        };

        let begin_info = vk::CommandBufferBeginInfo::default();
        unsafe {
            profiling::scope!("begin command buffer");
            self.device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(RendererError::CommandBufferBegin)?;
        }
        crate::name_vulkan_object(&self.device, command_buffer, format_args!("one frame's rendering cmds"));

        // Prepare geometry render pass:

        let fb_geom_pass_barrier = RenderPass::Geometry.barriers(&framebuffers.attachment_images(frame_index.index));
        let dep_info = vk::DependencyInfo::default().image_memory_barriers(&fb_geom_pass_barrier);
        unsafe { self.device.sync2.cmd_pipeline_barrier2(command_buffer, &dep_info) };

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
            self.device.dynamic_rendering.cmd_begin_rendering(command_buffer, &rendering_info);
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

        for (static_pl, _skinned_pl) in [
            (PipelineIndex::PbrOpaque, PipelineIndex::PbrSkinnedOpaque),
            (PipelineIndex::PbrAlphaToCoverage, PipelineIndex::PbrSkinnedAlphaToCoverage),
            (PipelineIndex::PbrBlended, PipelineIndex::PbrSkinnedBlended),
        ] {
            profiling::scope!("pipeline");
            let pipeline = pipelines.pipelines[static_pl].inner;
            let layout = descriptors.pipeline_layouts[static_pl].inner;
            let descriptor_sets = descriptors.descriptor_sets(static_pl);
            let bind_point = vk::PipelineBindPoint::GRAPHICS;
            unsafe { self.device.cmd_bind_pipeline(command_buffer, bind_point, pipeline) };
            unsafe {
                self.device
                    .cmd_bind_descriptor_sets(command_buffer, bind_point, layout, 1, &descriptor_sets[1..], &[]);
            }
            for (vertex_library, draws) in &draws[static_pl] {
                const VERTEX_BUFFERS: usize = VERTEX_BINDING_COUNT + 1;
                let mut vertex_offsets = ArrayVec::<vk::DeviceSize, VERTEX_BUFFERS>::new();
                vertex_offsets.push(0);
                vertex_offsets
                    .try_extend_from_slice(&vertex_library.vertex_buffer_offsets[static_pl])
                    .unwrap();
                let mut vertex_buffers = ArrayVec::<vk::Buffer, VERTEX_BUFFERS>::new();
                vertex_buffers.push(transforms_buffer.inner);
                for _ in 1..vertex_offsets.len() {
                    vertex_buffers.push(vertex_library.vertex_buffer.inner);
                }

                unsafe {
                    self.device
                        .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &vertex_offsets);
                    self.device
                        .cmd_bind_index_buffer(command_buffer, vertex_library.index_buffer.inner, 0, VERTEX_LIBRARY_INDEX_TYPE);
                }

                let indirect_draws_buffer = {
                    profiling::scope!("create indirect draws buffer");
                    let transforms_bytes = bytemuck::cast_slice(draws);
                    let buffer_create_info = vk::BufferCreateInfo::default()
                        .size(transforms_bytes.len() as vk::DeviceSize)
                        .usage(vk::BufferUsageFlags::INDIRECT_BUFFER)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE);
                    self.temp_arena
                        .create_buffer(
                            buffer_create_info,
                            transforms_bytes,
                            None,
                            None,
                            format_args!("indirect draw command buffer ({static_pl:?})"),
                        )
                        .map_err(RendererError::IndirectCmdBufferCreation)?
                };
                unsafe {
                    self.device.cmd_draw_indexed_indirect(
                        command_buffer,
                        indirect_draws_buffer.inner,
                        0,
                        draws.len() as u32,
                        mem::size_of::<DrawIndexedIndirectCommand>() as u32,
                    );
                }
                self.temp_arena.add_buffer(indirect_draws_buffer);
            }
            // TODO: Draw skinned meshes again
        }

        unsafe {
            profiling::scope!("end main rendering");
            self.device.dynamic_rendering.cmd_end_rendering(command_buffer);
        }

        // End of geometry render pass.

        // Prepare post-processing render pass:

        let fb_pp_pass_barrier = RenderPass::PostProcess.barriers(&framebuffers.attachment_images(frame_index.index));
        let dep_info = vk::DependencyInfo::default().image_memory_barriers(&fb_pp_pass_barrier);
        unsafe { self.device.sync2.cmd_pipeline_barrier2(command_buffer, &dep_info) };

        let swapchain_to_write_layout = framebuffers.swapchain_write_barrier(frame_index.index);
        let dep_info = vk::DependencyInfo::default().image_memory_barriers(&swapchain_to_write_layout);
        unsafe { self.device.sync2.cmd_pipeline_barrier2(command_buffer, &dep_info) };

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
            self.device.dynamic_rendering.cmd_begin_rendering(command_buffer, &rendering_info);
        }

        {
            profiling::scope!("record tonemapping subpass");
            let pl_index = PipelineIndex::RenderResolutionPostProcess;
            let bind_point = vk::PipelineBindPoint::GRAPHICS;
            unsafe {
                self.device
                    .cmd_bind_pipeline(command_buffer, bind_point, pipelines.pipelines[pl_index].inner);
            }
            let layout = descriptors.pipeline_layouts[pl_index].inner;
            let descriptors = &descriptors.descriptor_sets(pl_index)[1..];
            unsafe {
                self.device
                    .cmd_bind_descriptor_sets(command_buffer, bind_point, layout, 1, descriptors, &[]);
            }
            unsafe { self.device.cmd_draw(command_buffer, 3, 1, 0, 0) };
        }

        unsafe {
            profiling::scope!("end rendering");
            self.device.dynamic_rendering.cmd_end_rendering(command_buffer);
        }

        let swapchain_to_present_layout = framebuffers.swapchain_present_barrier(frame_index.index);
        let dep_info = vk::DependencyInfo::default().image_memory_barriers(&swapchain_to_present_layout);
        unsafe { self.device.sync2.cmd_pipeline_barrier2(command_buffer, &dep_info) };

        // End of post-processing render pass.

        unsafe {
            profiling::scope!("end command buffer");
            self.device
                .end_command_buffer(command_buffer)
                .map_err(RendererError::CommandBufferEnd)?;
        }

        Ok(command_buffer)
    }
}
