use crate::arena::VulkanArenaError;
use crate::debug_utils;
use crate::vulkan_raii::{Buffer, CommandPool, Device, Fence, Semaphore};
use crate::{ForBuffers, PhysicalDevice, VulkanArena};
use ash::vk;
use ash::Instance;
use core::fmt::Arguments;
use core::time::Duration;

/// General errors related to the creation and usage of [Uploader]. The actual
/// uploads have their own error type, [UploadError].
#[derive(thiserror::Error, Debug)]
pub enum UploaderError {
    #[error("failed to wait for uploader's vulkan fences")]
    FenceWait(#[source] vk::Result),
    #[error("tried to reset uploader while some uploads are still in progress (or the device has been lost)")]
    NotResettable,
    #[error("failed to create staging arena")]
    StagingArenaCreation(#[source] VulkanArenaError),
    #[error("failed to create uploader transfer command pool (out of memory?)")]
    TransferCommandPoolCreation(#[source] vk::Result),
    #[error("failed to create uploader graphics command pool (out of memory?)")]
    GraphicsCommandPoolCreation(#[source] vk::Result),
    #[error("failed to reset staging arena")]
    StagingArenaReset(#[source] VulkanArenaError),
    #[error("failed to reset uploader transfer command pool")]
    TransferCommandPoolReset(#[source] vk::Result),
    #[error("failed to reset uploader graphics command pool")]
    GraphicsCommandPoolReset(#[source] vk::Result),
}

/// Errors that may be generated when starting an upload with [Uploader].
#[derive(thiserror::Error, Debug)]
pub enum UploadError {
    #[error("failed to create uploader transfer command buffer (out of memory?)")]
    TransferCommandBufferCreation(#[source] vk::Result),
    #[error("failed to create uploader graphics command buffer (out of memory?)")]
    GraphicsCommandBufferCreation(#[source] vk::Result),
    #[error("failed to reset uploader vulkan fence")]
    FenceReset(#[source] vk::Result),
    #[error("failed to create vulkan fence for uploader (out of memory?)")]
    FenceCreation(#[source] vk::Result),
    #[error("failed to create vulkan semaphore for uploader (out of memory?)")]
    SemaphoreCreation(#[source] vk::Result),
    #[error("failed to begin transfer command buffer for upload")]
    TransferCommandBufferBegin(#[source] vk::Result),
    #[error("failed to end transfer command buffer for upload")]
    TransferCommandBufferEnd(#[source] vk::Result),
    #[error("failed to begin graphics command buffer for upload")]
    GraphicsCommandBufferBegin(#[source] vk::Result),
    #[error("failed to end graphics command buffer for upload")]
    GraphicsCommandBufferEnd(#[source] vk::Result),
    #[error("failed to submit uploader transfer commands")]
    TransferQueueSubmit(#[source] vk::Result),
    #[error("failed to submit uploader graphics commands")]
    GraphicsQueueSubmit(#[source] vk::Result),
}

pub struct Uploader {
    pub staging_arena: VulkanArena<ForBuffers>,
    pub graphics_queue_family: u32,
    pub transfer_queue_family: u32,
    device: Device,
    transfer_queue: vk::Queue,
    transfer_command_pool: CommandPool,
    graphics_queue: vk::Queue,
    graphics_command_pool: CommandPool,
    staging_buffers: Vec<Buffer>,
    upload_fences: Vec<Fence>,
    free_fences: Vec<Fence>,
    transfer_semaphores: Vec<Semaphore>,
    free_semaphores: Vec<Semaphore>,
    debug_identifier: &'static str,
}

impl Uploader {
    pub fn new(
        instance: &Instance,
        device: &Device,
        graphics_queue: vk::Queue,
        transfer_queue: vk::Queue,
        physical_device: &PhysicalDevice,
        staging_buffer_size: vk::DeviceSize,
        debug_identifier: &'static str,
    ) -> Result<Uploader, UploaderError> {
        profiling::scope!("uploader creation");

        let transfer_command_pool = {
            profiling::scope!("transfer command pool creation");
            let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(physical_device.transfer_queue_family.index)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);
            let command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None) }
                .map_err(UploaderError::TransferCommandPoolCreation)?;
            debug_utils::name_vulkan_object(device, command_pool, format_args!("upload cmds (T) for {}", debug_identifier));
            CommandPool {
                inner: command_pool,
                device: device.clone(),
            }
        };

        let graphics_command_pool = {
            profiling::scope!("graphics command pool creation");
            let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(physical_device.graphics_queue_family.index)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);
            let command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None) }
                .map_err(UploaderError::GraphicsCommandPoolCreation)?;
            debug_utils::name_vulkan_object(device, command_pool, format_args!("upload cmds (G) for {}", debug_identifier));
            CommandPool {
                inner: command_pool,
                device: device.clone(),
            }
        };

        let staging_memory = VulkanArena::new(
            instance,
            device,
            physical_device,
            staging_buffer_size,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            format_args!("uploader staging memory"),
        )
        .map_err(UploaderError::StagingArenaCreation)?;

        Ok(Uploader {
            staging_arena: staging_memory,
            graphics_queue_family: physical_device.graphics_queue_family.index,
            transfer_queue_family: physical_device.transfer_queue_family.index,
            device: device.clone(),
            transfer_queue,
            transfer_command_pool,
            graphics_queue,
            graphics_command_pool,
            staging_buffers: Vec::new(),
            upload_fences: Vec::new(),
            free_fences: Vec::new(),
            transfer_semaphores: Vec::new(),
            free_semaphores: Vec::new(),
            debug_identifier,
        })
    }

    pub fn get_upload_statuses(&self) -> impl Iterator<Item = bool> + '_ {
        profiling::scope!("uploader fence status enumeration");
        self.upload_fences
            .iter()
            .map(move |fence| unsafe { self.device.get_fence_status(fence.inner) }.unwrap_or(false))
    }

    /// Waits `timeout` for the uploads to finish, then returns true if the
    /// uploads are done, false if some are still in progress. Passing in None
    /// will attempt to wait until the uploads are done.
    ///
    /// A zero duration can be passed in to simply check the status of the
    /// uploads as a whole, as opposed to the status of every individual upload
    /// operation with [Uploader::get_upload_statuses], which may be more
    /// inefficient.
    pub fn wait<D: Into<Option<Duration>>>(&self, timeout: D) -> Result<bool, UploaderError> {
        let timeout = if let Some(timeout) = timeout.into() {
            timeout.as_nanos() as u64
        } else {
            u64::MAX
        };
        profiling::scope!("waiting on uploader fences");
        let fences = self.upload_fences.iter().map(|fence| fence.inner).collect::<Vec<_>>();
        if fences.is_empty() {
            Ok(true)
        } else {
            match unsafe { self.device.wait_for_fences(&fences, true, timeout) } {
                Ok(_) => Ok(true),
                Err(vk::Result::TIMEOUT) => Ok(false),
                Err(err) => Err(UploaderError::FenceWait(err)),
            }
        }
    }

    pub(crate) fn start_upload<F, G>(
        &mut self,
        staging_buffer: Buffer,
        debug_identifier: Arguments,
        queue_transfer_commands: F,
        queue_graphics_commands: G,
    ) -> Result<(), UploadError>
    where
        F: Fn(&ash::Device, &Buffer, vk::CommandBuffer),
        G: Fn(&ash::Device, vk::CommandBuffer),
    {
        self._start_upload(staging_buffer, debug_identifier, &queue_transfer_commands, &queue_graphics_commands)
    }

    fn _start_upload(
        &mut self,
        staging_buffer: Buffer,
        debug_identifier: Arguments,
        queue_transfer_commands: &dyn Fn(&ash::Device, &Buffer, vk::CommandBuffer),
        queue_graphics_commands: &dyn Fn(&ash::Device, vk::CommandBuffer),
    ) -> Result<(), UploadError> {
        profiling::scope!("start upload");
        let [transfer_cmdbuf, graphics_cmdbuf] = {
            profiling::scope!("allocate command buffers");
            let transfer = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.transfer_command_pool.inner)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let transfer_buffers =
                unsafe { self.device.allocate_command_buffers(&transfer) }.map_err(UploadError::TransferCommandBufferCreation)?;
            let graphics = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.graphics_command_pool.inner)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let graphics_buffers =
                unsafe { self.device.allocate_command_buffers(&graphics) }.map_err(UploadError::GraphicsCommandBufferCreation)?;
            [transfer_buffers[0], graphics_buffers[0]]
        };
        debug_utils::name_vulkan_object(
            &self.device,
            transfer_cmdbuf,
            format_args!("upload cmds (T) for {}: {}", self.debug_identifier, debug_identifier),
        );
        debug_utils::name_vulkan_object(
            &self.device,
            graphics_cmdbuf,
            format_args!("upload cmds (G) for {}: {}", self.debug_identifier, debug_identifier),
        );

        let upload_fence = if let Some(fence) = self.free_fences.pop() {
            let fences = [fence.inner];
            unsafe { self.device.reset_fences(&fences) }.map_err(UploadError::FenceReset)?;
            fence
        } else {
            profiling::scope!("create fence");
            let fence = unsafe { self.device.create_fence(&vk::FenceCreateInfo::default(), None) }.map_err(UploadError::FenceCreation)?;
            Fence {
                inner: fence,
                device: self.device.clone(),
            }
        };
        debug_utils::name_vulkan_object(
            &self.device,
            upload_fence.inner,
            format_args!("upload fence for {}: {}", self.debug_identifier, debug_identifier),
        );

        let transfer_signal_semaphore = if let Some(semaphore) = self.free_semaphores.pop() {
            // Cannot be reset manually, but semaphores get unsignaled
            // after they are waited on, so all of the free semaphores
            // should be ok to use.
            semaphore
        } else {
            profiling::scope!("create semaphore");
            let semaphore = unsafe { self.device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None) }
                .map_err(UploadError::SemaphoreCreation)?;
            Semaphore {
                inner: semaphore,
                device: self.device.clone(),
            }
        };
        debug_utils::name_vulkan_object(
            &self.device,
            transfer_signal_semaphore.inner,
            format_args!("T->G signal for {}: {}", self.debug_identifier, debug_identifier),
        );

        {
            profiling::scope!("record commands for transfer queue");
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe { self.device.begin_command_buffer(transfer_cmdbuf, &command_buffer_begin_info) }
                .map_err(UploadError::TransferCommandBufferBegin)?;
            queue_transfer_commands(&self.device, &staging_buffer, transfer_cmdbuf);
            unsafe { self.device.end_command_buffer(transfer_cmdbuf) }.map_err(UploadError::TransferCommandBufferEnd)?;
        }

        {
            profiling::scope!("record commands for graphics queue");
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe { self.device.begin_command_buffer(graphics_cmdbuf, &command_buffer_begin_info) }
                .map_err(UploadError::GraphicsCommandBufferBegin)?;
            queue_graphics_commands(&self.device, graphics_cmdbuf);
            unsafe { self.device.end_command_buffer(graphics_cmdbuf) }.map_err(UploadError::GraphicsCommandBufferEnd)?;
        }

        {
            profiling::scope!("submit transfer command buffer");
            let command_buffers = [transfer_cmdbuf];
            let signal_semaphores = [transfer_signal_semaphore.inner];
            let submit_infos = [vk::SubmitInfo::builder()
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores)
                .build()];
            unsafe {
                self.device
                    .queue_submit(self.transfer_queue, &submit_infos, vk::Fence::null())
                    .map_err(UploadError::TransferQueueSubmit)
            }?;
        }

        {
            profiling::scope!("submit graphics command buffer");
            let command_buffers = [graphics_cmdbuf];
            let dst_stage_mask = [vk::PipelineStageFlags::TRANSFER];
            let wait_semaphores = [transfer_signal_semaphore.inner];
            let submit_infos = [vk::SubmitInfo::builder()
                .command_buffers(&command_buffers)
                .wait_dst_stage_mask(&dst_stage_mask)
                .wait_semaphores(&wait_semaphores)
                .build()];
            unsafe {
                self.device
                    .queue_submit(self.graphics_queue, &submit_infos, upload_fence.inner)
                    .map_err(UploadError::GraphicsQueueSubmit)
            }?;
        }

        self.staging_buffers.push(staging_buffer);
        self.upload_fences.push(upload_fence);
        self.transfer_semaphores.push(transfer_signal_semaphore);
        Ok(())
    }

    pub fn reset(&mut self) -> Result<(), UploaderError> {
        profiling::scope!("uploader reset");
        if self.get_upload_statuses().any(|uploaded| !uploaded) {
            return Err(UploaderError::NotResettable);
        }
        self.staging_buffers.clear();
        self.staging_arena.reset().map_err(UploaderError::StagingArenaReset)?;
        self.free_fences.append(&mut self.upload_fences);
        self.free_semaphores.append(&mut self.transfer_semaphores);
        unsafe {
            self.device
                .reset_command_pool(self.transfer_command_pool.inner, vk::CommandPoolResetFlags::empty())
                .map_err(UploaderError::TransferCommandPoolReset)
        }?;
        unsafe {
            self.device
                .reset_command_pool(self.graphics_command_pool.inner, vk::CommandPoolResetFlags::empty())
                .map_err(UploaderError::GraphicsCommandPoolReset)
        }?;
        Ok(())
    }
}
