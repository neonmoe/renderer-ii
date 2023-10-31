use core::fmt::Arguments;
use core::time::Duration;

use ash::vk;

use crate::physical_device::PhysicalDevice;
use crate::vulkan_raii::{Buffer, CommandPool, Device, Fence, Semaphore};

pub struct Uploader {
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
        device: &Device,
        graphics_queue: vk::Queue,
        transfer_queue: vk::Queue,
        physical_device: &PhysicalDevice,
        debug_identifier: &'static str,
    ) -> Uploader {
        profiling::scope!("uploader creation");

        let transfer_command_pool = {
            profiling::scope!("transfer command pool creation");
            let command_pool_create_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(physical_device.transfer_queue_family.index)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);
            let command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None) }
                .expect("system should have enough memory to create vulkan command pools");
            crate::name_vulkan_object(device, command_pool, format_args!("upload cmds (T) for {debug_identifier}"));
            CommandPool {
                inner: command_pool,
                device: device.clone(),
            }
        };

        let graphics_command_pool = {
            profiling::scope!("graphics command pool creation");
            let command_pool_create_info = vk::CommandPoolCreateInfo::default()
                .queue_family_index(physical_device.graphics_queue_family.index)
                .flags(vk::CommandPoolCreateFlags::TRANSIENT);
            let command_pool = unsafe { device.create_command_pool(&command_pool_create_info, None) }
                .expect("system should have enough memory to create vulkan command pools");
            crate::name_vulkan_object(device, command_pool, format_args!("upload cmds (G) for {debug_identifier}"));
            CommandPool {
                inner: command_pool,
                device: device.clone(),
            }
        };

        Uploader {
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
        }
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
    /// operation with [`Uploader::get_upload_statuses`], which may be more
    /// inefficient.
    pub fn wait(&self, timeout: Option<Duration>) -> bool {
        let timeout = if let Some(timeout) = timeout {
            timeout.as_nanos() as u64
        } else {
            u64::MAX
        };
        profiling::scope!("waiting on uploader fences");
        let fences = self.upload_fences.iter().map(|fence| fence.inner).collect::<Vec<_>>();
        if fences.is_empty() {
            true
        } else {
            match unsafe { self.device.wait_for_fences(&fences, true, timeout) } {
                Ok(()) => true,
                Err(vk::Result::TIMEOUT) => false,
                Err(err) => panic!("waiting for uploader vulkan fences should not fail: {err}"),
            }
        }
    }

    pub(crate) fn start_upload<F, G>(
        &mut self,
        staging_buffer: Buffer,
        debug_identifier: Arguments,
        queue_transfer_commands: F,
        queue_graphics_commands: G,
    ) where
        F: Fn(&Device, &Buffer, vk::CommandBuffer),
        G: Fn(&Device, vk::CommandBuffer),
    {
        self._start_upload(staging_buffer, debug_identifier, &queue_transfer_commands, &queue_graphics_commands);
    }

    fn _start_upload(
        &mut self,
        staging_buffer: Buffer,
        debug_identifier: Arguments,
        queue_transfer_commands: &dyn Fn(&Device, &Buffer, vk::CommandBuffer),
        queue_graphics_commands: &dyn Fn(&Device, vk::CommandBuffer),
    ) {
        profiling::scope!("start upload");
        // TODO: Create command buffers on init, use the same ones for all uploads? One per thread?
        let [transfer_cmdbuf, graphics_cmdbuf] = {
            profiling::scope!("allocate command buffers");
            let transfer = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.transfer_command_pool.inner)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let transfer_buffers = unsafe { self.device.allocate_command_buffers(&transfer) }
                .expect("system should have enough memory to allocate vulkan command buffers");
            let graphics = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.graphics_command_pool.inner)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let graphics_buffers = unsafe { self.device.allocate_command_buffers(&graphics) }
                .expect("system should have enough memory to allocate vulkan command buffers");
            [transfer_buffers[0], graphics_buffers[0]]
        };

        let upload_fence = if let Some(fence) = self.free_fences.pop() {
            let fences = [fence.inner];
            unsafe { self.device.reset_fences(&fences) }.expect("resetting uploader vulkan fences should not fail");
            fence
        } else {
            profiling::scope!("create fence");
            let fence = unsafe { self.device.create_fence(&vk::FenceCreateInfo::default(), None) }
                .expect("system should have enough memory to create a vulkan fence");
            Fence {
                inner: fence,
                device: self.device.clone(),
            }
        };
        crate::name_vulkan_object(
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
                .expect("system should have enough memory to create a vulkan semaphore");
            Semaphore {
                inner: semaphore,
                device: self.device.clone(),
            }
        };
        crate::name_vulkan_object(
            &self.device,
            transfer_signal_semaphore.inner,
            format_args!("T->G signal for {}: {}", self.debug_identifier, debug_identifier),
        );

        {
            profiling::scope!("record commands for transfer queue");
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe { self.device.begin_command_buffer(transfer_cmdbuf, &command_buffer_begin_info) }
                .expect("beginning vulkan command buffer recording should not fail");
            crate::name_vulkan_object(
                &self.device,
                transfer_cmdbuf,
                format_args!("upload cmds (T) for {}: {}", self.debug_identifier, debug_identifier),
            );
            queue_transfer_commands(&self.device, &staging_buffer, transfer_cmdbuf);
            unsafe { self.device.end_command_buffer(transfer_cmdbuf) }.expect("ending vulkan command buffer recording should not fail");
        }

        {
            profiling::scope!("record commands for graphics queue");
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe { self.device.begin_command_buffer(graphics_cmdbuf, &command_buffer_begin_info) }
                .expect("beginning vulkan command buffer recording should not fail");
            crate::name_vulkan_object(
                &self.device,
                graphics_cmdbuf,
                format_args!("upload cmds (G) for {}: {}", self.debug_identifier, debug_identifier),
            );
            queue_graphics_commands(&self.device, graphics_cmdbuf);
            unsafe { self.device.end_command_buffer(graphics_cmdbuf) }.expect("ending vulkan command buffer recording should not fail");
        }

        {
            profiling::scope!("submit transfer command buffer");
            let command_buffers = [transfer_cmdbuf];
            let signal_semaphores = [transfer_signal_semaphore.inner];
            let submit_infos = [vk::SubmitInfo::default()
                .command_buffers(&command_buffers)
                .signal_semaphores(&signal_semaphores)];
            unsafe {
                self.device
                    .queue_submit(self.transfer_queue, &submit_infos, vk::Fence::null())
                    .expect("vulkan queue submission should not fail");
            }
        }

        {
            profiling::scope!("submit graphics command buffer");
            let command_buffers = [graphics_cmdbuf];
            let dst_stage_mask = [vk::PipelineStageFlags::TRANSFER];
            let wait_semaphores = [transfer_signal_semaphore.inner];
            let submit_infos = [vk::SubmitInfo::default()
                .command_buffers(&command_buffers)
                .wait_dst_stage_mask(&dst_stage_mask)
                .wait_semaphores(&wait_semaphores)];
            unsafe {
                self.device
                    .queue_submit(self.graphics_queue, &submit_infos, upload_fence.inner)
                    .expect("vulkan queue submission should not fail");
            }
        }

        self.staging_buffers.push(staging_buffer);
        self.upload_fences.push(upload_fence);
        self.transfer_semaphores.push(transfer_signal_semaphore);
    }

    /// Resets the uploader if all uploads have finished, and returns true.
    /// Returns false if there are still uploads in progress.
    pub fn reset(&mut self) -> bool {
        profiling::scope!("uploader reset");
        if self.get_upload_statuses().any(|uploaded| !uploaded) {
            return false;
        }
        self.staging_buffers.clear();
        self.free_fences.append(&mut self.upload_fences);
        self.free_semaphores.append(&mut self.transfer_semaphores);
        unsafe {
            self.device
                .reset_command_pool(self.transfer_command_pool.inner, vk::CommandPoolResetFlags::empty())
                .expect("resetting a vulkan command pool should not fail");
            self.device
                .reset_command_pool(self.graphics_command_pool.inner, vk::CommandPoolResetFlags::empty())
                .expect("resetting a vulkan command pool should not fail");
        }
        true
    }
}
