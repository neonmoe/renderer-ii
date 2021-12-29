use crate::descriptors::Descriptors;
use crate::pipeline::{Pipeline, PushConstantStruct, MAX_TEXTURE_COUNT};
use crate::{Arena, Camera, Canvas, Driver, Error, Scene};
use ash::extensions::{ext, khr};
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk::Handle;
use ash::{vk, Device, Instance};
use glam::Mat4;
use std::ffi::CStr;
use std::mem;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::sync::mpsc::{self, Receiver, Sender};

/// Get from [Gpu::wait_frame].
#[derive(Clone, Copy)]
pub struct FrameIndex {
    index: u32,
}

impl FrameIndex {
    fn new(index: u32) -> FrameIndex {
        FrameIndex { index }
    }

    pub fn get_arena<'a>(self, arenas: &'a [Arena]) -> &'a Arena<'a> {
        &arenas[self.index as usize]
    }
}

/// A unique id for every distinct GPU.
pub struct GpuId([u8; 16]);

/// Describes a GPU that can be used as a [Gpu]. Queried in
/// [Gpu::new].
pub struct GpuInfo {
    pub in_use: bool,
    pub name: String,
    pub id: GpuId,
}

struct WaitSemaphore(vk::Semaphore, vk::PipelineStageFlags);

/// Synchronization objects and buffers used during a single frame,
/// which is only cleaned up after enough frames have passed that the
/// specific FrameLocal struct is reused. This allows for processing a
/// frame while the previous one is still being rendered.
struct FrameLocal {
    in_use: AtomicBool,
    finished_command_buffers_sp: vk::Semaphore,
    frame_end_fence: vk::Fence,
    temp_command_buffers: (Sender<vk::CommandBuffer>, Receiver<vk::CommandBuffer>),
    temp_semaphores: (Sender<vk::Semaphore>, Receiver<vk::Semaphore>),
    temp_buffers: (Sender<vk::Buffer>, Receiver<vk::Buffer>),
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
pub struct Gpu<'a> {
    /// Held by [Gpu] to ensure that the devices are dropped before
    /// the instance.
    pub driver: &'a Driver,

    pub(crate) surface_ext: khr::Surface,
    pub(crate) swapchain_ext: khr::Swapchain,

    pub physical_device: vk::PhysicalDevice,
    pub device: Device,
    pub(crate) command_pool: vk::CommandPool,

    pub(crate) graphics_family_index: u32,
    pub(crate) surface_family_index: u32,
    pub(crate) graphics_queue: vk::Queue,
    surface_queue: vk::Queue,
    frame_start_fence: vk::Fence,

    pub(crate) descriptors: Descriptors,
    pub(crate) texture_indices: Vec<AtomicBool>,

    frame_locals: Vec<FrameLocal>,
    render_wait_semaphores: (Sender<WaitSemaphore>, Receiver<WaitSemaphore>),
}

impl Drop for Gpu<'_> {
    #[profiling::function]
    fn drop(&mut self) {
        let _ = self.wait_idle();

        {
            profiling::scope!("destroy descriptors");
            self.descriptors.clean_up(&self.device);
        }

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
                self.device.destroy_semaphore(frame_local.finished_command_buffers_sp, None);
                self.device.destroy_fence(frame_local.frame_end_fence, None);
            }
        }

        {
            profiling::scope!("destroy command pools");
            unsafe { self.device.destroy_command_pool(self.command_pool, None) };
        }

        {
            profiling::scope!("destroy vulkan device");
            unsafe { self.device.destroy_device(None) };
        }
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
    pub fn new(driver: &Driver, preferred_physical_device: Option<[u8; 16]>) -> Result<(Gpu<'_>, Vec<GpuInfo>), Error> {
        profiling::scope!("new_gpu");
        let surface_ext = khr::Surface::new(&driver.entry, &driver.instance);
        let queue_family_supports_surface = |pd: vk::PhysicalDevice, index: u32| {
            let support = unsafe { surface_ext.get_physical_device_surface_support(pd, index, driver.surface) };
            matches!(support, Ok(true))
        };

        let all_physical_devices =
            unsafe { driver.instance.enumerate_physical_devices() }.map_err(Error::VulkanEnumeratePhysicalDevices)?;
        let mut physical_devices = all_physical_devices
            .into_iter()
            .filter_map(|physical_device| {
                if !is_extension_supported(&driver.instance, physical_device, "VK_KHR_swapchain") {
                    return None;
                }

                let queue_families = unsafe { driver.instance.get_physical_device_queue_family_properties(physical_device) };
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

                let properties = unsafe { driver.instance.get_physical_device_properties(physical_device) };
                let format_properties = unsafe {
                    driver
                        .instance
                        .get_physical_device_format_properties(physical_device, vk::Format::R8G8B8A8_SRGB)
                };
                match format_properties.optimal_tiling_features {
                    features if !features.contains(vk::FormatFeatureFlags::BLIT_SRC) => {
                        log::warn!(
                            "physical device '{}' does not have BLIT_SRC for optimal tiling 32-bit srgb images",
                            get_device_name(&properties)
                        );
                        return None;
                    }
                    features if !features.contains(vk::FormatFeatureFlags::BLIT_DST) => {
                        log::warn!(
                            "physical device '{}' does not have BLIT_DST for optimal tiling 32-bit srgb images",
                            get_device_name(&properties)
                        );
                        return None;
                    }
                    features if !features.contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR) => {
                        log::warn!(
                            "physical device '{}' does not have SAMPLED_IMAGE_FILTER_LINEAR for optimal tiling 32-bit srgb images",
                            get_device_name(&properties)
                        );
                        return None;
                    }
                    _ => {}
                }

                let features = unsafe { driver.instance.get_physical_device_features(physical_device) };
                if let (Some(graphics_family_index), Some(surface_family_index)) = (graphics_family_index, surface_family_index) {
                    Some((physical_device, properties, features, graphics_family_index, surface_family_index))
                } else {
                    None
                }
            })
            .collect::<Vec<(
                vk::PhysicalDevice,
                vk::PhysicalDeviceProperties,
                vk::PhysicalDeviceFeatures,
                u32,
                u32,
            )>>();

        let (physical_device, physical_device_properties, physical_device_features, graphics_family_index, surface_family_index) =
            if let Some(uuid) = preferred_physical_device {
                physical_devices
                    .iter()
                    .find_map(|tuple| {
                        let (_, properties, _, _, _) = tuple;
                        if properties.pipeline_cache_uuid == uuid {
                            Some(*tuple)
                        } else {
                            None
                        }
                    })
                    .ok_or(Error::VulkanPhysicalDeviceMissing)?
            } else {
                physical_devices.sort_by(|(_, a_props, _, a_gfx, a_surf), (_, b_props, _, b_gfx, b_surf)| {
                    let type_score = |properties: vk::PhysicalDeviceProperties| match properties.device_type {
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
                    let a_score = type_score(*a_props) + queue_score(a_gfx, a_surf);
                    let b_score = type_score(*b_props) + queue_score(b_gfx, b_surf);
                    // Highest score first.
                    b_score.cmp(&a_score)
                });
                physical_devices.get(0).copied().ok_or(Error::VulkanPhysicalDeviceMissing)?
            };

        let physical_devices = physical_devices
            .into_iter()
            .map(|(_, properties, _, _, _)| {
                let name = get_device_name(&properties);
                let pd_type = match properties.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => " (Discrete GPU)",
                    vk::PhysicalDeviceType::INTEGRATED_GPU => " (Integrated GPU)",
                    vk::PhysicalDeviceType::VIRTUAL_GPU => " (vGPU)",
                    vk::PhysicalDeviceType::CPU => " (CPU)",
                    _ => "",
                };
                let in_use = properties.pipeline_cache_uuid == physical_device_properties.pipeline_cache_uuid;
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
        let mut extensions = vec![cstr!("VK_KHR_swapchain").as_ptr()];
        log::debug!("Device extension: VK_KHR_swapchain");
        if is_extension_supported(&driver.instance, physical_device, "VK_EXT_memory_budget") {
            extensions.push(cstr!("VK_EXT_memory_budget").as_ptr());
            log::debug!("Device extension: VK_EXT_memory_budget");
        }

        let mut physical_device_descriptor_indexing_features =
            vk::PhysicalDeviceDescriptorIndexingFeatures::builder().descriptor_binding_partially_bound(true);
        let device_create_info = vk::DeviceCreateInfo::builder()
            .push_next(&mut physical_device_descriptor_indexing_features)
            .queue_create_infos(&queue_create_infos)
            .enabled_features(&physical_device_features)
            .enabled_extension_names(&extensions);
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
                let _ = debug_utils_ext.debug_utils_set_object_name(device.handle(), &graphics_name_info);
                let _ = debug_utils_ext.debug_utils_set_object_name(device.handle(), &surface_name_info);
            }
        }

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
                    finished_command_buffers_sp,
                    frame_end_fence,
                    temp_command_buffers: mpsc::channel(),
                    temp_semaphores: mpsc::channel(),
                    temp_buffers: mpsc::channel(),
                })
            })
            .collect::<Result<Vec<FrameLocal>, Error>>()?;

        let frame_start_fence = unsafe {
            device
                .create_fence(&vk::FenceCreateInfo::default(), None)
                .map_err(Error::VulkanSemaphoreCreation)
        }?;

        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(graphics_family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool =
            unsafe { device.create_command_pool(&command_pool_create_info, None) }.map_err(Error::VulkanCommandPoolCreation)?;

        let descriptors = Descriptors::new(&device, &physical_device_properties, &physical_device_features, frame_in_use_count)?;

        let mut texture_indices = Vec::with_capacity(MAX_TEXTURE_COUNT as usize);
        for _ in 0..MAX_TEXTURE_COUNT {
            texture_indices.push(AtomicBool::new(false));
        }

        Ok((
            Gpu {
                driver,

                surface_ext,
                swapchain_ext,

                physical_device,
                device,
                command_pool,

                graphics_family_index,
                surface_family_index,
                graphics_queue,
                surface_queue,
                frame_start_fence,

                descriptors,
                texture_indices,

                frame_locals,
                render_wait_semaphores: mpsc::channel(),
            },
            physical_devices,
        ))
    }

    pub fn temp_arena_count(&self) -> usize {
        self.frame_locals.len()
    }

    pub(crate) fn image_index(&self, frame_index: FrameIndex) -> usize {
        frame_index.index as usize
    }

    fn frame_local(&self, frame_index: FrameIndex) -> &FrameLocal {
        &self.frame_locals[self.image_index(frame_index)]
    }

    pub(crate) fn add_temp_buffer(&self, frame_index: FrameIndex, buffer: vk::Buffer) {
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
            unsafe { self.device.destroy_buffer(buffer, None) };
        }
    }

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
            self.swapchain_ext
                .acquire_next_image(canvas.swapchain, u64::MAX, vk::Semaphore::null(), self.frame_start_fence)
                .map_err(Error::VulkanAcquireImage)
        }?;
        let frame_index = FrameIndex::new(image_index);
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

    // TODO: Re-do the texture index reservation system
    // Currently it basically requires Rc's to communicate that a texture has been released.
    // It does not work after the memory management refactor.
    #[profiling::function]
    pub(crate) fn reserve_texture_index(
        &self,
        base_color: Option<vk::ImageView>,
        metallic_roughness: Option<vk::ImageView>,
        normal: Option<vk::ImageView>,
        occlusion: Option<vk::ImageView>,
        emission: Option<vk::ImageView>,
    ) -> Result<TextureIndex, Error> {
        let index = self
            .texture_indices
            .iter()
            .position(|occupied| {
                occupied
                    .compare_exchange_weak(false, true, Ordering::Relaxed, Ordering::Relaxed)
                    .is_ok()
            })
            .ok_or(Error::TextureIndexReserve)? as u32;
        let image_views = &[base_color, metallic_roughness, normal, occlusion, emission];
        self.descriptors
            .set_uniform_images(&self.device, Pipeline::Default, 1, 1, image_views, index..index + 1);
        Ok(TextureIndex(index))
    }

    #[profiling::function]
    pub(crate) fn release_texture_index(&self, index: u32) {
        self.texture_indices[index as usize].store(false, Ordering::Relaxed);
    }

    /// Queue up all the rendering commands.
    ///
    /// The rendered frame will appear on the screen some time in the
    /// future. Use [Gpu::wait_frame] to block until that
    /// happens.
    #[profiling::function]
    pub fn render_frame(
        &self,
        temp_arenas: &[Arena],
        frame_index: FrameIndex,
        canvas: &Canvas,
        camera: &Camera,
        scene: &Scene,
        debug_value: u32,
    ) -> Result<(), Error> {
        let frame_local = self.frame_local(frame_index);
        frame_local.in_use.store(true, Ordering::Relaxed);
        camera.update(canvas, temp_arenas, frame_index)?;

        let image_index = self.image_index(frame_index);
        let command_buffer = canvas.command_buffers[image_index];
        let framebuffer = canvas.framebuffers[image_index];
        self.record_commmand_buffer(temp_arenas, frame_index, command_buffer, framebuffer, canvas, scene, debug_value)?;

        let render_wait_semaphores = self.render_wait_semaphores.1.try_iter().collect::<Vec<WaitSemaphore>>();
        let mut wait_semaphores = Vec::with_capacity(render_wait_semaphores.len());
        let mut wait_stages = Vec::with_capacity(render_wait_semaphores.len());
        for WaitSemaphore(semaphore, wait_stage) in render_wait_semaphores {
            wait_semaphores.push(semaphore);
            wait_stages.push(wait_stage);
        }

        let signal_semaphores = [frame_local.finished_command_buffers_sp];
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

        let swapchains = [canvas.swapchain];
        let image_indices = [image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        let present_result = unsafe {
            profiling::scope!("queue present");
            self.swapchain_ext.queue_present(self.surface_queue, &present_info)
        };

        match present_result {
            Err(err @ vk::Result::ERROR_OUT_OF_DATE_KHR) => return Err(Error::VulkanSwapchainOutOfDate(err)),
            Err(err) => return Err(Error::VulkanQueuePresent(err)),
            _ => {}
        }

        Ok(())
    }

    #[profiling::function]
    fn record_commmand_buffer(
        &self,
        temp_arenas: &[Arena],
        frame_index: FrameIndex,
        command_buffer: vk::CommandBuffer,
        framebuffer: vk::Framebuffer,
        canvas: &Canvas,
        scene: &Scene,
        debug_value: u32,
    ) -> Result<(), Error> {
        let temp_arena = frame_index.get_arena(temp_arenas);

        unsafe {
            profiling::scope!("reset command buffer");
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(Error::VulkanResetCommandBuffer)?;
        }

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
            .render_pass(canvas.final_render_pass)
            .framebuffer(framebuffer)
            .render_area(render_area)
            .clear_values(&clear_colors);
        unsafe {
            profiling::scope!("begin render pass");
            self.device
                .cmd_begin_render_pass(command_buffer, &render_pass_begin_info, vk::SubpassContents::INLINE);
        }

        // Bind the shared descriptor set (#0)
        let shared_descriptor_set = self.descriptors.descriptor_sets(self, frame_index, 0)[0];
        unsafe {
            profiling::scope!("bind shared descriptor set");
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.descriptors.pipeline_layouts[0],
                0,
                &[shared_descriptor_set],
                &[],
            );
        }

        for (pipeline, meshes) in &scene.pipeline_map {
            profiling::scope!("pipeline");
            let pipeline_idx = *pipeline as usize;
            if meshes.is_empty() {
                continue;
            }

            unsafe {
                self.device
                    .cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, canvas.pipelines[pipeline_idx])
            };
            let layout = self.descriptors.pipeline_layouts[pipeline_idx];
            let descriptor_sets = self.descriptors.descriptor_sets(self, frame_index, pipeline_idx);
            if descriptor_sets.len() > 1 {
                unsafe {
                    self.device.cmd_bind_descriptor_sets(
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        layout,
                        1,
                        &descriptor_sets[1..],
                        &[],
                    )
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
                    if let Err(err) = unsafe { transform_buffer.write(temp_arena, transforms.as_ptr() as *const u8, 0, vk::WHOLE_SIZE) } {
                        transform_buffer.clean_up(temp_arena);
                        return Err(err);
                    }
                }

                self.add_temp_buffer(frame_index, transform_buffer.buffer);

                let push_constants = [PushConstantStruct {
                    texture_index: material.texture_index.0,
                    debug_value,
                }];
                let push_constants = bytemuck::bytes_of(&push_constants);
                unsafe {
                    profiling::scope!("push constants");
                    self.device
                        .cmd_push_constants(command_buffer, layout, vk::ShaderStageFlags::FRAGMENT, 0, push_constants);
                }

                let mut vertex_buffers = vec![mesh.buffer(); mesh.vertices_offsets.len() + 1];
                vertex_buffers[0] = transform_buffer.buffer;
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

#[profiling::function]
fn is_extension_supported(instance: &Instance, physical_device: vk::PhysicalDevice, target_extension_name: &str) -> bool {
    match unsafe { instance.enumerate_device_extension_properties(physical_device) } {
        Err(_) => false,
        Ok(extensions) => extensions.iter().any(|extension_properties| {
            let extension_name_slice = &extension_properties.extension_name[..];
            let extension_name = unsafe { CStr::from_ptr(extension_name_slice.as_ptr()) }.to_string_lossy();
            extension_name == target_extension_name
        }),
    }
}

fn get_device_name(properties: &vk::PhysicalDeviceProperties) -> std::borrow::Cow<'_, str> {
    unsafe { CStr::from_ptr((&properties.device_name[..]).as_ptr()) }.to_string_lossy()
}
