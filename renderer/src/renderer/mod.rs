use crate::arena::{ForBuffers, MemoryProps, VulkanArena, VulkanArenaError};
use crate::physical_device::PhysicalDevice;
use crate::vulkan_raii::{Buffer, CommandBuffer, CommandPool, Device, Fence, Semaphore};
use alloc::rc::Rc;
use arrayvec::ArrayVec;
use ash::{vk, Instance};
use core::fmt::Arguments;
use glam::Mat4;

pub(crate) mod camera;
pub(crate) mod coordinate_system;
pub(crate) mod descriptors;
pub(crate) mod framebuffers;
pub(crate) mod mesh;
pub(crate) mod pipelines;
pub(crate) mod scene;
pub(crate) mod swapchain;

use camera::Camera;
use descriptors::{DescriptorError, Descriptors};
use framebuffers::Framebuffers;
use mesh::Mesh;
use pipelines::pipeline_parameters::{MaterialPushConstants, PipelineIndex, PipelineMap, RenderSettings, MAX_BONE_COUNT};
use pipelines::Pipelines;
use scene::{Scene, SkinnedModel, StaticMeshMap};
use swapchain::Swapchain;

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
    #[error("failed to submit rendering command buffers to the graphics queue (device lost or out of memory?)")]
    RenderQueueSubmit(#[source] vk::Result),
    #[error("present was successful, but may display oddly; swapchain is out of date")]
    SwapchainOutOfDate(#[source] vk::Result),
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

pub struct Renderer {
    device: Device,
    frame_start_fence: Fence,
    ready_for_present: Semaphore,
    frame_end_fence: Fence,
    temp_arena: VulkanArena<ForBuffers>,
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

        Ok(Renderer {
            device: device.clone(),
            frame_start_fence,
            ready_for_present,
            frame_end_fence,
            temp_arena,
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
                    err @ vk::Result::ERROR_OUT_OF_DATE_KHR => RendererError::SwapchainOutOfDate(err),
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
        scene: Scene,
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

        let vk::Extent2D { width, height } = framebuffers.extent;

        let global_transforms = &[scene
            .camera
            .create_global_transforms(width as f32, height as f32, scene.world_space)];
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
        descriptors.write_descriptors(
            &global_transforms_buffer,
            &render_settings_buffer,
            &skinned_mesh_joints_buffer,
            &materials_temp_uniform,
            &framebuffers.inner[frame_index.index],
        );

        self.temp_arena.add_buffer(global_transforms_buffer);
        self.temp_arena.add_buffer(render_settings_buffer);
        self.temp_arena.add_buffer(skinned_mesh_joints_buffer);

        let command_buffer = self.record_command_buffer(
            frame_index,
            descriptors,
            pipelines,
            framebuffers,
            &scene.static_meshes,
            &scene.skinned_meshes,
        )?;

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
            Err(err @ vk::Result::ERROR_OUT_OF_DATE_KHR) => Err(RendererError::SwapchainOutOfDate(err)),
            Err(err) => Err(RendererError::RenderQueuePresent(err)),
            _ => Ok(()),
        }
    }

    #[profiling::function]
    fn record_command_buffer(
        &mut self,
        frame_index: &FrameIndex,
        descriptors: &mut Descriptors,
        pipelines: &Pipelines,
        framebuffers: &Framebuffers,
        static_meshes: &PipelineMap<StaticMeshMap>,
        skinned_meshes: &PipelineMap<Vec<SkinnedModel>>,
    ) -> Result<vk::CommandBuffer, RendererError> {
        let framebuffer = &framebuffers.inner[frame_index.index];

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
            crate::name_vulkan_object(&self.device, command_buffers[0], format_args!("frame rendering cmds"));
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

        let render_area = vk::Rect2D::default().extent(framebuffers.extent);
        let mut depth_clear_value = vk::ClearValue::default();
        depth_clear_value.depth_stencil.depth = 0.0;
        let clear_colors = [vk::ClearValue::default(), depth_clear_value, vk::ClearValue::default()];
        let render_pass_begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(pipelines.render_pass.inner)
            .framebuffer(framebuffer.inner)
            .render_area(render_area)
            .clear_values(&clear_colors);
        let subpass_begin_info = vk::SubpassBeginInfo::default().contents(vk::SubpassContents::INLINE);
        unsafe {
            profiling::scope!("begin render pass");
            self.device
                .cmd_begin_render_pass2(command_buffer, &render_pass_begin_info, &subpass_begin_info);
        }

        // Bind the shared descriptor set
        {
            let pipeline = PipelineIndex::SHARED_DESCRIPTOR_PIPELINE;
            let shared_descriptor_set = descriptors.descriptor_sets(pipeline)[0];
            unsafe {
                profiling::scope!("bind shared descriptor set");
                self.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    descriptors.pipeline_layouts[pipeline].inner,
                    0,
                    &[shared_descriptor_set],
                    &[],
                );
            }
        }

        for (static_pl, skinned_pl) in [
            (PipelineIndex::PbrOpaque, PipelineIndex::PbrSkinnedOpaque),
            (PipelineIndex::PbrAlphaToCoverage, PipelineIndex::PbrSkinnedAlphaToCoverage),
            (PipelineIndex::PbrBlended, PipelineIndex::PbrSkinnedBlended),
        ] {
            profiling::scope!("pipeline");
            let static_meshes = &static_meshes[static_pl];
            if !static_meshes.is_empty() {
                let pipeline = pipelines.pipelines[static_pl].inner;
                let layout = descriptors.pipeline_layouts[static_pl].inner;
                let descriptor_sets = descriptors.descriptor_sets(static_pl);
                self.record_static_pipeline(
                    command_buffer,
                    (pipeline, layout),
                    descriptor_sets,
                    static_meshes,
                    static_pl,
                    format_args!("instanced statics {static_pl:?}"),
                )?;
            }
            let skinned_meshes = &skinned_meshes[skinned_pl];
            if !skinned_meshes.is_empty() {
                let pipeline = pipelines.pipelines[skinned_pl].inner;
                let layout = descriptors.pipeline_layouts[skinned_pl].inner;
                let descriptor_sets = descriptors.descriptor_sets(skinned_pl);
                self.record_skinned_pipeline(
                    command_buffer,
                    (pipeline, layout),
                    descriptor_sets,
                    skinned_meshes,
                    skinned_pl,
                    format_args!("skinned model {static_pl:?}"),
                )?;
            }
        }

        unsafe {
            profiling::scope!("vk::cmd_next_subpass");
            self.device.cmd_next_subpass(command_buffer, vk::SubpassContents::INLINE);
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
            let descriptor_sets = descriptors.descriptor_sets(pl_index);
            unsafe {
                self.device
                    .cmd_bind_descriptor_sets(command_buffer, bind_point, layout, 1, &descriptor_sets[1..], &[]);
            }
            unsafe { self.device.cmd_draw(command_buffer, 3, 1, 0, 0) };
        }

        unsafe {
            profiling::scope!("end render pass");
            self.device.cmd_end_render_pass2(command_buffer, &vk::SubpassEndInfo::default());
        }

        unsafe {
            profiling::scope!("end command buffer");
            self.device
                .end_command_buffer(command_buffer)
                .map_err(RendererError::CommandBufferEnd)?;
        }

        Ok(command_buffer)
    }

    fn record_static_pipeline(
        &mut self,
        command_buffer: vk::CommandBuffer,
        (pipeline, layout): (vk::Pipeline, vk::PipelineLayout),
        descriptor_sets: &[vk::DescriptorSet],
        meshes: &StaticMeshMap,
        pl_index: PipelineIndex,
        pl_name: Arguments,
    ) -> Result<(), RendererError> {
        let bind_point = vk::PipelineBindPoint::GRAPHICS;
        unsafe { self.device.cmd_bind_pipeline(command_buffer, bind_point, pipeline) };
        unsafe {
            self.device
                .cmd_bind_descriptor_sets(command_buffer, bind_point, layout, 1, &descriptor_sets[1..], &[]);
        }

        for (i, (&(mesh, material), transforms)) in meshes.iter().enumerate() {
            profiling::scope!("static instanced meshes");
            let transform_buffer = {
                profiling::scope!("create transform buffer");
                let transforms_bytes = bytemuck::cast_slice(transforms);
                let buffer_create_info = vk::BufferCreateInfo::default()
                    .size(transforms_bytes.len() as vk::DeviceSize)
                    .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE);
                self.temp_arena
                    .create_buffer(
                        buffer_create_info,
                        transforms_bytes,
                        None,
                        None,
                        format_args!("{}. transform buffer ({pl_name})", i + 1),
                    )
                    .map_err(RendererError::TransformBufferCreation)?
            };

            let push_constants = [MaterialPushConstants {
                // Safety: The material is from a Scene which gets its PipelineIndexes from the materials themselves.
                texture_index: material.array_index(pl_index).unwrap(),
            }];
            let push_constants = bytemuck::bytes_of(&push_constants);
            unsafe {
                profiling::scope!("push constants");
                self.device
                    .cmd_push_constants(command_buffer, layout, vk::ShaderStageFlags::FRAGMENT, 0, push_constants);
            }

            let mut vertex_buffers = ArrayVec::<vk::Buffer, { mesh::VERTEX_BUFFERS + 1 }>::new();
            vertex_buffers.push(transform_buffer.inner);
            for vertex_buffer in &mesh.vertex_buffers {
                vertex_buffers.push(vertex_buffer.inner);
            }
            let mut vertex_offsets = ArrayVec::<vk::DeviceSize, { mesh::VERTEX_BUFFERS + 1 }>::new();
            vertex_offsets.push(0);
            vertex_offsets.try_extend_from_slice(&mesh.vertices_offsets).unwrap();

            unsafe {
                profiling::scope!("draw");
                self.device
                    .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &vertex_offsets);
                self.device
                    .cmd_bind_index_buffer(command_buffer, mesh.index_buffer.inner, mesh.index_buffer_offset, mesh.index_type);
                self.device
                    .cmd_draw_indexed(command_buffer, mesh.index_count, transforms.len() as u32, 0, 0, 0);
            }

            self.temp_arena.add_buffer(transform_buffer);
        }
        Ok(())
    }

    fn record_skinned_pipeline(
        &mut self,
        command_buffer: vk::CommandBuffer,
        (pipeline, layout): (vk::Pipeline, vk::PipelineLayout),
        descriptor_sets: &[vk::DescriptorSet],
        meshes: &[SkinnedModel],
        pl_index: PipelineIndex,
        pl_name: Arguments,
    ) -> Result<(), RendererError> {
        let bind_point = vk::PipelineBindPoint::GRAPHICS;
        unsafe { self.device.cmd_bind_pipeline(command_buffer, bind_point, pipeline) };

        for model in meshes {
            unsafe {
                self.device.cmd_bind_descriptor_sets(
                    command_buffer,
                    bind_point,
                    layout,
                    1,
                    &descriptor_sets[1..],
                    &[model.joints_offset.0],
                );
            }
            let transform_buffer = {
                profiling::scope!("create transform buffer for skinned model");
                let transforms = &[model.transform];
                let transforms_bytes = bytemuck::cast_slice(transforms);
                let buffer_create_info = vk::BufferCreateInfo::default()
                    .size(transforms_bytes.len() as vk::DeviceSize)
                    .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE);
                self.temp_arena
                    .create_buffer(
                        buffer_create_info,
                        transforms_bytes,
                        None,
                        None,
                        format_args!("transform buffer for skinned model ({pl_name})"),
                    )
                    .map_err(RendererError::TransformBufferCreation)?
            };

            for (mesh, material) in &model.meshes {
                profiling::scope!("skinned mesh");

                let push_constants = [MaterialPushConstants {
                    // Safety: The material is from a Scene which gets its PipelineIndexes from the materials themselves.
                    texture_index: material.array_index(pl_index).unwrap(),
                }];
                let push_constants = bytemuck::bytes_of(&push_constants);
                unsafe {
                    profiling::scope!("push constants");
                    self.device
                        .cmd_push_constants(command_buffer, layout, vk::ShaderStageFlags::FRAGMENT, 0, push_constants);
                }

                let mut vertex_buffers = ArrayVec::<vk::Buffer, { mesh::VERTEX_BUFFERS + 1 }>::new();
                vertex_buffers.push(transform_buffer.inner);
                for vertex_buffer in &mesh.vertex_buffers {
                    vertex_buffers.push(vertex_buffer.inner);
                }
                let mut vertex_offsets = ArrayVec::<vk::DeviceSize, { mesh::VERTEX_BUFFERS + 1 }>::new();
                vertex_offsets.push(0);
                vertex_offsets.try_extend_from_slice(&mesh.vertices_offsets).unwrap();

                unsafe {
                    profiling::scope!("draw");
                    self.device
                        .cmd_bind_vertex_buffers(command_buffer, 0, &vertex_buffers, &vertex_offsets);
                    self.device
                        .cmd_bind_index_buffer(command_buffer, mesh.index_buffer.inner, mesh.index_buffer_offset, mesh.index_type);
                    self.device.cmd_draw_indexed(command_buffer, mesh.index_count, 1, 0, 0, 0);
                }
            }

            self.temp_arena.add_buffer(transform_buffer);
        }
        Ok(())
    }
}
