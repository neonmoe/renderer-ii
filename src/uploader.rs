use crate::vulkan_raii::{Buffer, CommandPool, Device, Fence};
use crate::{Error, PhysicalDevice, VulkanArena};
use ash::version::DeviceV1_0;
use ash::vk;
use ash::Instance;
use std::rc::Rc;
use std::time::Duration;

pub struct Uploader {
    pub staging_arena: VulkanArena,
    pub queue_family_indices: [u32; 2],
    device: Rc<Device>,
    transfer_queue: vk::Queue,
    command_pool: CommandPool,
    staging_buffers: Vec<Buffer>,
    upload_fences: Vec<Fence>,
    free_fences: Vec<Fence>,
}

impl Uploader {
    pub fn new(
        instance: &Instance,
        device: &Rc<Device>,
        transfer_queue: vk::Queue,
        upload_queue_family_indices: [u32; 2],
        physical_device: &PhysicalDevice,
        staging_buffer_size: vk::DeviceSize,
    ) -> Result<Uploader, Error> {
        let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(physical_device.graphics_family_index)
            .flags(vk::CommandPoolCreateFlags::TRANSIENT);
        let command_pool =
            unsafe { device.create_command_pool(&command_pool_create_info, None) }.map_err(Error::VulkanCommandPoolCreation)?;
        let command_pool = CommandPool {
            inner: command_pool,
            device: device.clone(),
        };

        let staging_memory = VulkanArena::new(
            instance,
            device,
            physical_device.inner,
            staging_buffer_size,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
            "uploader staging memory",
        )?;

        Ok(Uploader {
            staging_arena: staging_memory,
            queue_family_indices: upload_queue_family_indices,
            device: device.clone(),
            transfer_queue,
            command_pool,
            staging_buffers: Vec::new(),
            upload_fences: Vec::new(),
            free_fences: Vec::new(),
        })
    }

    /// Returns true if the uploads should use concurrent sharing mode.
    pub(crate) fn dedicated_transfers(&self) -> bool {
        self.queue_family_indices[0] != self.queue_family_indices[1]
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

    pub(crate) fn start_upload<F>(
        &mut self,
        // Not currently used, but very probably will be used at a
        // later point, for intra-frame-upload-waiting.
        #[allow(unused_variables)] wait_stage: vk::PipelineStageFlags,
        staging_buffer: Buffer,
        record_upload_commands: F,
    ) -> Result<(), Error>
    where
        F: FnOnce(&ash::Device, &Buffer, vk::CommandBuffer),
    {
        profiling::scope!("start upload");
        let command_buffers = {
            profiling::scope!("allocate command buffer");
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.command_pool.inner)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            unsafe { self.device.allocate_command_buffers(&command_buffer_allocate_info) }.map_err(Error::VulkanCommandBuffersAllocation)?
        };
        let temp_command_buffer = command_buffers[0];

        let upload_fence = if let Some(fence) = self.free_fences.pop() {
            let fences = [fence.inner];
            unsafe { self.device.reset_fences(&fences) }.map_err(Error::VulkanFenceReset)?;
            fence
        } else {
            profiling::scope!("create fence");
            let fence =
                unsafe { self.device.create_fence(&vk::FenceCreateInfo::default(), None) }.map_err(Error::VulkanSemaphoreCreation)?;
            Fence {
                inner: fence,
                device: self.device.clone(),
            }
        };

        {
            profiling::scope!("begin command buffer recording");
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            unsafe { self.device.begin_command_buffer(temp_command_buffer, &command_buffer_begin_info) }
                .map_err(Error::VulkanBeginCommandBuffer)?;
        }

        {
            profiling::scope!("record commands");
            record_upload_commands(&self.device, &staging_buffer, temp_command_buffer);
        }

        {
            profiling::scope!("end command buffer recording");
            unsafe { self.device.end_command_buffer(temp_command_buffer) }.map_err(Error::VulkanEndCommandBuffer)?;
        }

        {
            profiling::scope!("submit command buffer");
            let command_buffers = [temp_command_buffer];
            let submit_infos = [vk::SubmitInfo::builder().command_buffers(&command_buffers).build()];
            unsafe {
                self.device
                    .queue_submit(self.transfer_queue, &submit_infos, upload_fence.inner)
                    .map_err(Error::VulkanQueueSubmit)
            }?;
        }

        self.staging_buffers.push(staging_buffer);
        self.upload_fences.push(upload_fence);
        Ok(())
    }

    pub fn reset(&mut self) -> Result<(), Error> {
        if self.get_upload_statuses().any(|uploaded| !uploaded) {
            return Err(Error::UploaderNotResettable);
        }
        self.staging_buffers.clear();
        self.staging_arena.reset()?;
        self.free_fences.append(&mut self.upload_fences);
        unsafe {
            self.device
                .reset_command_pool(self.command_pool.inner, vk::CommandPoolResetFlags::empty())
                .map_err(Error::VulkanResetCommandPool)
        }?;
        Ok(())
    }
}
