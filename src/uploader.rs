use crate::debug_utils;
use crate::vulkan_raii::{Buffer, CommandPool, Device, Fence, Semaphore};
use crate::{Error, PhysicalDevice, VulkanArena};
use ash::vk;
use ash::Instance;
use std::fmt::Arguments;
use std::rc::Rc;
use std::time::Duration;

pub struct Uploader {
    pub staging_arena: VulkanArena,
    pub graphics_queue_family: u32,
    pub transfer_queue_family: u32,
    device: Rc<Device>,
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
        device: &Rc<Device>,
        graphics_queue: vk::Queue,
        transfer_queue: vk::Queue,
        physical_device: &PhysicalDevice,
        staging_buffer_size: vk::DeviceSize,
        debug_identifier: &'static str,
    ) -> Result<Uploader, Error> {
        let transfer_command_pool = {
            let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(physical_device.transfer_queue_family.index)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);
            let command_pool =
                unsafe { device.create_command_pool(&command_pool_create_info, None) }.map_err(Error::VulkanCommandPoolCreation)?;
            debug_utils::name_vulkan_object(device, command_pool, format_args!("upload cmds (T) for {}", debug_identifier));
            CommandPool {
                inner: command_pool,
                device: device.clone(),
            }
        };

        let graphics_command_pool = {
            let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(physical_device.graphics_queue_family.index)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);
            let command_pool =
                unsafe { device.create_command_pool(&command_pool_create_info, None) }.map_err(Error::VulkanCommandPoolCreation)?;
            debug_utils::name_vulkan_object(device, command_pool, format_args!("upload cmds (G) for {}", debug_identifier));
            CommandPool {
                inner: command_pool,
                device: device.clone(),
            }
        };

        let staging_memory = VulkanArena::new(
            instance,
            device,
            physical_device.inner,
            staging_buffer_size,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            format_args!("uploader staging memory"),
        )?;

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
        self.upload_fences
            .iter()
            .map(move |fence| unsafe { self.device.get_fence_status(fence.inner) }.unwrap_or(false))
    }

    /// Waits `timeout` for the uploads to finish, then returns true
    /// if the uploads are done, false if some are still in progress.
    ///
    /// A zero duration can be passed in to simply check the status of
    /// the uploads as a whole, as opposed to the status of every
    /// individual upload operation with
    /// [Uploader::get_upload_statuses], which may be more
    /// inefficient.
    pub fn wait(&self, timeout: Duration) -> Result<bool, Error> {
        let fences = self.upload_fences.iter().map(|fence| fence.inner).collect::<Vec<_>>();
        match unsafe { self.device.wait_for_fences(&fences, true, timeout.as_nanos() as u64) } {
            Ok(_) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(err) => Err(Error::VulkanFenceWait(err)),
        }
    }

    pub(crate) fn start_upload<F, G>(
        &mut self,
        // Not currently used, but very probably will be used at a
        // later point, for intra-frame-upload-waiting.
        #[allow(unused_variables)] wait_stage: vk::PipelineStageFlags,
        staging_buffer: Buffer,
        debug_identifier: Arguments,
        queue_transfer_commands: F,
        queue_graphics_commands: G,
    ) -> Result<(), Error>
    where
        F: FnOnce(&ash::Device, &Buffer, vk::CommandBuffer),
        G: FnOnce(&ash::Device, vk::CommandBuffer),
    {
        profiling::scope!("start upload");
        let [transfer_cmdbuf, graphics_cmdbuf] = {
            profiling::scope!("allocate command buffers");
            let transfer = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.transfer_command_pool.inner)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let transfer_buffers =
                unsafe { self.device.allocate_command_buffers(&transfer) }.map_err(Error::VulkanCommandBuffersAllocation)?;
            let graphics = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.graphics_command_pool.inner)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let graphics_buffers =
                unsafe { self.device.allocate_command_buffers(&graphics) }.map_err(Error::VulkanCommandBuffersAllocation)?;
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
            unsafe { self.device.reset_fences(&fences) }.map_err(Error::VulkanFenceReset)?;
            fence
        } else {
            profiling::scope!("create fence");
            let fence = unsafe { self.device.create_fence(&vk::FenceCreateInfo::default(), None) }.map_err(Error::VulkanFenceCreation)?;
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
                .map_err(Error::VulkanSemaphoreCreation)?;
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
                .map_err(Error::VulkanBeginCommandBuffer)?;
            queue_transfer_commands(&self.device, &staging_buffer, transfer_cmdbuf);
            unsafe { self.device.end_command_buffer(transfer_cmdbuf) }.map_err(Error::VulkanEndCommandBuffer)?;
        }

        {
            profiling::scope!("record commands for graphics queue");
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe { self.device.begin_command_buffer(graphics_cmdbuf, &command_buffer_begin_info) }
                .map_err(Error::VulkanBeginCommandBuffer)?;
            queue_graphics_commands(&self.device, graphics_cmdbuf);
            unsafe { self.device.end_command_buffer(graphics_cmdbuf) }.map_err(Error::VulkanEndCommandBuffer)?;
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
                    .map_err(Error::VulkanQueueSubmit)
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
                    .map_err(Error::VulkanQueueSubmit)
            }?;
        }

        self.staging_buffers.push(staging_buffer);
        self.upload_fences.push(upload_fence);
        self.transfer_semaphores.push(transfer_signal_semaphore);
        Ok(())
    }

    pub fn reset(&mut self) -> Result<(), Error> {
        if self.get_upload_statuses().any(|uploaded| !uploaded) {
            return Err(Error::UploaderNotResettable);
        }
        self.staging_buffers.clear();
        self.staging_arena.reset()?;
        self.free_fences.append(&mut self.upload_fences);
        self.free_semaphores.append(&mut self.transfer_semaphores);
        unsafe {
            self.device
                .reset_command_pool(self.transfer_command_pool.inner, vk::CommandPoolResetFlags::empty())
                .map_err(Error::VulkanResetCommandPool)
        }?;
        unsafe {
            self.device
                .reset_command_pool(self.graphics_command_pool.inner, vk::CommandPoolResetFlags::empty())
                .map_err(Error::VulkanResetCommandPool)
        }?;
        Ok(())
    }
}
