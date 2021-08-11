use crate::buffer_ops::BufferUpload;
use crate::descriptors::Descriptors;
use crate::pipeline::Pipeline;
use crate::{Camera, Canvas, Driver, Error, Mesh, Texture};
use ash::extensions::{ext, khr};
use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::vk::Handle;
use ash::{vk, Device, Instance};
use std::cell::Cell;
use std::ffi::CStr;
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
    pub(crate) allocator: vk_mem::Allocator,
    pub(crate) command_pool: vk::CommandPool,

    pub(crate) staging_cpu_buffer_pool: vk_mem::AllocatorPool,
    pub(crate) main_gpu_buffer_pool: vk_mem::AllocatorPool,
    pub(crate) main_gpu_texture_pool: vk_mem::AllocatorPool,
    pub(crate) temp_gpu_buffer_pools: Vec<vk_mem::AllocatorPool>,

    pub(crate) graphics_family_index: u32,
    pub(crate) surface_family_index: u32,
    pub(crate) graphics_queue: vk::Queue,
    surface_queue: vk::Queue,

    pub(crate) descriptors: Descriptors,

    frame_locals: Vec<FrameLocal>,
    frame_index: Cell<u32>,
    frame_count: Cell<u32>,
    buffer_upload_semaphores: (Sender<WaitSemaphore>, Receiver<WaitSemaphore>),
}

struct BufferAllocation(vk::Buffer, vk_mem::Allocation);

/// Synchronization objects and buffers used during a single frame,
/// which is only cleaned up after enough frames have passed that the
/// specific FrameLocal struct is reused. This allows for processing a
/// frame while the previous one is still being rendered.
struct FrameLocal {
    acquired_image_sp: vk::Semaphore,
    finished_command_buffers_sp: vk::Semaphore,
    finished_queue_fence: vk::Fence,

    buffer_uploads: (Sender<BufferUpload>, Receiver<BufferUpload>),
    temporary_buffers: (Sender<BufferAllocation>, Receiver<BufferAllocation>),
}

impl Drop for Gpu<'_> {
    #[profiling::function]
    fn drop(&mut self) {
        let _ = self.wait_idle();

        self.descriptors.clean_up(&self.device);

        for frame_local in &self.frame_locals {
            profiling::scope!("destroy frame locals");
            let _ = unsafe {
                self.device
                    .wait_for_fences(&[frame_local.finished_queue_fence], true, u64::MAX)
                    .map_err(Error::VulkanFenceWait)
            };

            let _ = self.cleanup_buffer_uploads(frame_local);
            let _ = self.cleanup_temp_buffers(frame_local);

            let &FrameLocal {
                acquired_image_sp,
                finished_command_buffers_sp,
                finished_queue_fence,
                ..
            } = frame_local;
            unsafe {
                self.device.destroy_semaphore(acquired_image_sp, None);
                self.device
                    .destroy_semaphore(finished_command_buffers_sp, None);
                self.device.destroy_fence(finished_queue_fence, None);
            }
        }

        {
            profiling::scope!("destroy vma allocator");
            let _ = self.allocator.destroy_pool(&self.staging_cpu_buffer_pool);
            let _ = self.allocator.destroy_pool(&self.main_gpu_buffer_pool);
            let _ = self.allocator.destroy_pool(&self.main_gpu_texture_pool);
            for temp_gpu_buffer_pool in &self.temp_gpu_buffer_pools {
                let _ = self.allocator.destroy_pool(temp_gpu_buffer_pool);
            }
            self.allocator.destroy();
        }

        {
            profiling::scope!("destroy vma command pools");
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
    #[profiling::function]
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
                let properties = unsafe {
                    driver
                        .instance
                        .get_physical_device_properties(physical_device)
                };
                let features = unsafe {
                    driver
                        .instance
                        .get_physical_device_features(physical_device)
                };
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
                    Some((
                        physical_device,
                        properties,
                        features,
                        graphics_family_index,
                        surface_family_index,
                    ))
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

        let (
            physical_device,
            physical_device_properties,
            physical_device_features,
            graphics_family_index,
            surface_family_index,
        ) = if let Some(uuid) = preferred_physical_device {
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
            physical_devices.sort_by(
                |(_, a_props, _, a_gfx, a_surf), (_, b_props, _, b_gfx, b_surf)| {
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
                    let a_score = type_score(*a_props) + queue_score(a_gfx, a_surf);
                    let b_score = type_score(*b_props) + queue_score(b_gfx, b_surf);
                    // Highest score first.
                    b_score.cmp(&a_score)
                },
            );
            physical_devices
                .get(0)
                .copied()
                .ok_or(Error::VulkanPhysicalDeviceMissing)?
        };

        let physical_devices = physical_devices
            .into_iter()
            .map(|(_, properties, _, _, _)| {
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
        let mut extensions = vec![cstr!("VK_KHR_swapchain").as_ptr()];
        log::debug!("Device extension: VK_KHR_swapchain");
        if is_extension_supported(&driver.instance, physical_device, "VK_EXT_memory_budget") {
            extensions.push(cstr!("VK_EXT_memory_budget").as_ptr());
            log::debug!("Device extension: VK_EXT_memory_budget");
        }

        let device_create_info = vk::DeviceCreateInfo::builder()
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
                let _ = debug_utils_ext
                    .debug_utils_set_object_name(device.handle(), &graphics_name_info);
                let _ = debug_utils_ext
                    .debug_utils_set_object_name(device.handle(), &surface_name_info);
            }
        }

        let frame_in_use_count = 3;
        let frame_locals = (0..frame_in_use_count)
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
                Ok(FrameLocal {
                    acquired_image_sp,
                    finished_command_buffers_sp,
                    finished_queue_fence,
                    buffer_uploads: mpsc::channel(),
                    temporary_buffers: mpsc::channel(),
                })
            })
            .collect::<Result<Vec<FrameLocal>, Error>>()?;

        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(graphics_family_index)
            .flags(
                vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            );
        let command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None) }
            .map_err(Error::VulkanCommandPoolCreation)?;

        let descriptors = Descriptors::new(
            &device,
            &physical_device_properties,
            &physical_device_features,
            frame_in_use_count,
        )?;

        let allocator_create_info = vk_mem::AllocatorCreateInfo {
            physical_device,
            device: device.clone(),
            instance: driver.instance.clone(),
            flags: vk_mem::AllocatorCreateFlags::EXTERNALLY_SYNCHRONIZED,
            preferred_large_heap_block_size: 128 * 1024 * 1024,
            frame_in_use_count,
            heap_size_limits: None,
        };
        let allocator =
            vk_mem::Allocator::new(&allocator_create_info).map_err(Error::VmaAllocatorCreation)?;

        let mesh_usage = vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::VERTEX_BUFFER;
        let uniform_usage = vk::BufferUsageFlags::UNIFORM_BUFFER;
        let pool_defaults = vk_mem::AllocatorPoolCreateInfo {
            flags: vk_mem::AllocatorPoolCreateFlags::IGNORE_BUFFER_IMAGE_GRANULARITY,
            frame_in_use_count,
            ..Default::default()
        };

        let cpu_buffer_info = vk::BufferCreateInfo::builder()
            .size(1024)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let cpu_alloc_info = vk_mem::AllocationCreateInfo {
            required_flags: vk::MemoryPropertyFlags::HOST_VISIBLE,
            ..Default::default()
        };
        let staging_cpu_buffer_pool = allocator
            .create_pool(&vk_mem::AllocatorPoolCreateInfo {
                memory_type_index: allocator
                    .find_memory_type_index_for_buffer_info(&cpu_buffer_info, &cpu_alloc_info)
                    .map_err(Error::VmaFindMemoryType)?,
                ..pool_defaults
            })
            .map_err(Error::VmaPoolCreation)?;

        let gpu_buffer_info = vk::BufferCreateInfo::builder()
            .size(1024)
            .usage(vk::BufferUsageFlags::TRANSFER_DST | uniform_usage | mesh_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let gpu_image_info = vk::ImageCreateInfo::builder()
            .image_type(vk::ImageType::TYPE_2D)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .format(vk::Format::R8G8B8A8_SRGB)
            .tiling(vk::ImageTiling::OPTIMAL)
            .extent(vk::Extent3D {
                width: 1024,
                height: 1024,
                depth: 1,
            })
            .mip_levels(9)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1);
        let gpu_alloc_info = vk_mem::AllocationCreateInfo {
            required_flags: vk::MemoryPropertyFlags::DEVICE_LOCAL,
            ..Default::default()
        };
        let main_gpu_buffer_pool = allocator
            .create_pool(&vk_mem::AllocatorPoolCreateInfo {
                memory_type_index: allocator
                    .find_memory_type_index_for_buffer_info(&gpu_buffer_info, &gpu_alloc_info)
                    .map_err(Error::VmaFindMemoryType)?,
                ..pool_defaults
            })
            .map_err(Error::VmaPoolCreation)?;
        let main_gpu_texture_pool = allocator
            .create_pool(&vk_mem::AllocatorPoolCreateInfo {
                memory_type_index: allocator
                    .find_memory_type_index_for_image_info(&gpu_image_info, &gpu_alloc_info)
                    .map_err(Error::VmaFindMemoryType)?,
                ..pool_defaults
            })
            .map_err(Error::VmaPoolCreation)?;
        let temp_gpu_buffer_pools = (0..frame_in_use_count)
            .map(|_| {
                allocator
                    .create_pool(&vk_mem::AllocatorPoolCreateInfo {
                        memory_type_index: allocator
                            .find_memory_type_index_for_buffer_info(
                                &gpu_buffer_info,
                                &gpu_alloc_info,
                            )
                            .map_err(Error::VmaFindMemoryType)?,
                        flags: pool_defaults.flags
                            | vk_mem::AllocatorPoolCreateFlags::LINEAR_ALGORITHM,
                        ..pool_defaults
                    })
                    .map_err(Error::VmaPoolCreation)
            })
            .collect::<Result<Vec<_>, Error>>()?;

        Ok((
            Gpu {
                driver,

                surface_ext,
                swapchain_ext,

                physical_device,
                device,
                allocator,
                command_pool,

                staging_cpu_buffer_pool,
                main_gpu_buffer_pool,
                main_gpu_texture_pool,
                temp_gpu_buffer_pools,

                graphics_family_index,
                surface_family_index,
                graphics_queue,
                surface_queue,

                descriptors,

                frame_locals,
                frame_index: Cell::new(0),
                frame_count: Cell::new(frame_in_use_count),
                buffer_upload_semaphores: mpsc::channel(),
            },
            physical_devices,
        ))
    }

    #[profiling::function]
    pub(crate) fn add_buffer_upload(&self, frame_index: FrameIndex, buffer_upload: BufferUpload) {
        let upload_wait = WaitSemaphore(buffer_upload.finished_upload, buffer_upload.wait_stage);
        let _ = self.buffer_upload_semaphores.0.send(upload_wait);
        let frame_local = &self.frame_locals[self.frame_mod(frame_index)];
        let _ = frame_local.buffer_uploads.0.send(buffer_upload);
    }

    #[profiling::function]
    pub(crate) fn add_temporary_buffer(
        &self,
        frame_index: FrameIndex,
        buffer: vk::Buffer,
        allocation: vk_mem::Allocation,
    ) {
        let frame_local = &self.frame_locals[self.frame_mod(frame_index)];
        let buffer_allocation = BufferAllocation(buffer, allocation);
        let _ = frame_local.temporary_buffers.0.send(buffer_allocation);
    }

    pub(crate) fn set_frame_count(&self, new_frame_count: u32) {
        self.frame_count.set(new_frame_count);
    }

    /// Returns the frame index % frame count. Generally everything
    /// frame-specific is a vec that can be indexed into with this.
    pub(crate) fn frame_mod(&self, frame_index: FrameIndex) -> usize {
        (frame_index.index % self.frame_count.get()) as usize
    }

    #[profiling::function]
    fn cleanup_buffer_uploads(&self, frame_local: &FrameLocal) -> Result<(), Error> {
        for buffer_upload in frame_local.buffer_uploads.1.try_iter() {
            if let Some(buffer) = buffer_upload.staging_buffer {
                unsafe { self.device.destroy_buffer(buffer, None) };
            }
            if let Some(allocation) = &buffer_upload.staging_allocation {
                self.allocator
                    .free_memory(allocation)
                    .map_err(Error::VmaBufferDestruction)?;
            }
            let cmdbufs = [buffer_upload.upload_cmdbuf];
            unsafe {
                self.device
                    .destroy_semaphore(buffer_upload.finished_upload, None);
                self.device
                    .free_command_buffers(self.command_pool, &cmdbufs);
            }
        }
        Ok(())
    }

    #[profiling::function]
    fn cleanup_temp_buffers(&self, frame_local: &FrameLocal) -> Result<(), Error> {
        for BufferAllocation(buffer, allocation) in frame_local.temporary_buffers.1.try_iter() {
            self.allocator
                .destroy_buffer(buffer, &allocation)
                .map_err(Error::VmaBufferDestruction)?;
        }
        Ok(())
    }

    /// Returns the total bytes used by resources, and the total
    /// allocated bytes.
    #[profiling::function]
    pub fn vram_usage(&self) -> Result<(vk::DeviceSize, vk::DeviceSize), Error> {
        let stats = self
            .allocator
            .calculate_stats()
            .map_err(Error::VmaCalculateStats)?;
        Ok((
            stats.total.usedBytes,
            stats.total.usedBytes + stats.total.unusedBytes,
        ))
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
    pub fn wait_frame(&self) -> Result<FrameIndex, Error> {
        let frame_index = FrameIndex::new(self.frame_index.get().wrapping_add(1));
        let frame_local = &self.frame_locals[self.frame_mod(frame_index)];

        {
            profiling::scope!("wait for frame fence");
            unsafe {
                self.device
                    .wait_for_fences(&[frame_local.finished_queue_fence], true, u64::MAX)
                    .map_err(Error::VulkanFenceWait)
            }?;
        }

        self.cleanup_temp_buffers(frame_local)?;
        self.cleanup_buffer_uploads(frame_local)?;

        Ok(frame_index)
    }

    /// Updates the texture(s) for the pipeline.
    ///
    /// The amount of textures to pass depends on the pipeline.
    pub fn set_pipeline_textures(
        &self,
        frame_index: FrameIndex,
        pipeline: Pipeline,
        textures: &[&Texture<'_>],
    ) {
        self.descriptors
            .set_uniform_images(&self, frame_index, pipeline, 1, 0, textures);
    }

    /// Queue up all the rendering commands.
    ///
    /// The rendered frame will appear on the screen some time in the
    /// future. Use [Gpu::wait_frame] to block until that
    /// happens.
    #[profiling::function]
    pub fn render_frame(
        &self,
        frame_index: FrameIndex,
        canvas: &Canvas,
        camera: &Camera,
        meshes: &[Mesh],
    ) -> Result<(), Error> {
        self.frame_index.set(frame_index.index);
        let _ = self.allocator.set_current_frame_index(frame_index.index);

        camera.update(canvas, frame_index)?;

        let FrameLocal {
            acquired_image_sp,
            finished_command_buffers_sp,
            finished_queue_fence,
            ..
        } = self.frame_locals[self.frame_mod(frame_index)];

        let (image_index, _) = unsafe {
            profiling::scope!("acquire next image");
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

        let command_buffer = canvas.command_buffers[image_index as usize];
        let framebuffer = canvas.swapchain_framebuffers[image_index as usize];
        self.record_commmand_buffer(frame_index, command_buffer, framebuffer, canvas, meshes)?;

        let mut wait_semaphores = vec![acquired_image_sp];
        let mut wait_stages = vec![vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        for WaitSemaphore(semaphore, wait_stage) in self.buffer_upload_semaphores.1.try_iter() {
            wait_semaphores.push(semaphore);
            wait_stages.push(wait_stage);
        }

        let signal_semaphores = [finished_command_buffers_sp];
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
                .queue_submit(self.graphics_queue, &submit_infos, finished_queue_fence)
                .map_err(Error::VulkanQueueSubmit)
        }?;

        let swapchains = [canvas.swapchain];
        let image_indices = [image_index];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(&signal_semaphores)
            .swapchains(&swapchains)
            .image_indices(&image_indices);
        let present_result = unsafe {
            profiling::scope!("queue present");
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

    #[profiling::function]
    fn record_commmand_buffer(
        &self,
        frame_index: FrameIndex,
        command_buffer: vk::CommandBuffer,
        framebuffer: vk::Framebuffer,
        canvas: &Canvas,
        meshes: &[Mesh],
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

        let render_area = vk::Rect2D::builder().extent(canvas.extent).build();
        let clear_colors = [vk::ClearValue::default()];
        let render_pass_begin_info = vk::RenderPassBeginInfo::builder()
            .render_pass(canvas.final_render_pass)
            .framebuffer(framebuffer)
            .render_area(render_area)
            .clear_values(&clear_colors);
        unsafe {
            self.device.cmd_begin_render_pass(
                command_buffer,
                &render_pass_begin_info,
                vk::SubpassContents::INLINE,
            )
        };

        // Bind the shared descriptor set (#0)
        let shared_descriptor_set = self.descriptors.descriptor_sets(&self, frame_index, 0)[0];
        unsafe {
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.descriptors.pipeline_layouts[0],
                0,
                &[shared_descriptor_set],
                &[],
            )
        };

        let mut meshes_per_pipeline = (0..(Pipeline::Count as usize))
            .map(|_| Vec::with_capacity(meshes.len()))
            .collect::<Vec<Vec<&Mesh<'_>>>>();
        for mesh in meshes {
            let bucket = unsafe { meshes_per_pipeline.get_unchecked_mut(mesh.pipeline as usize) };
            bucket.push(mesh);
        }

        for (pipeline_idx, meshes) in meshes_per_pipeline.into_iter().enumerate() {
            profiling::scope!("pipeline");
            if meshes.is_empty() {
                continue;
            }

            unsafe {
                self.device.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    canvas.pipelines[pipeline_idx],
                )
            };
            let layout = self.descriptors.pipeline_layouts[pipeline_idx];
            let descriptor_sets =
                self.descriptors
                    .descriptor_sets(&self, frame_index, pipeline_idx);
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

            for mesh in meshes {
                profiling::scope!("mesh");
                let mesh_buffer = if let Ok(buffer) = mesh.mesh_buffer.buffer(frame_index) {
                    buffer
                } else {
                    continue;
                };
                unsafe {
                    self.device
                        .cmd_bind_vertex_buffers(command_buffer, 0, &[mesh_buffer], &[0]);
                    self.device.cmd_bind_index_buffer(
                        command_buffer,
                        mesh_buffer,
                        mesh.indices_offset,
                        mesh.index_type,
                    );
                    self.device
                        .cmd_draw_indexed(command_buffer, mesh.index_count, 1, 0, 0, 0);
                }
            }
        }

        unsafe {
            self.device.cmd_end_render_pass(command_buffer);
        }

        unsafe { self.device.end_command_buffer(command_buffer) }
            .map_err(Error::VulkanEndCommandBuffer)?;
        Ok(())
    }
}

#[profiling::function]
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
