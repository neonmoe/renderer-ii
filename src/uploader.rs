use crate::vulkan_raii::{CommandPool, Device, Semaphore};
use crate::{Error, PhysicalDevice, VulkanArena};
use ash::version::DeviceV1_0;
use ash::vk;
use ash::Instance;
use std::rc::Rc;

struct WaitSemaphore(Semaphore, vk::PipelineStageFlags);

pub struct Uploader {
    pub staging_memory: VulkanArena,
    pub transfer_queue: vk::Queue,
    pub graphics_queue: vk::Queue,
    device: Rc<Device>,
    command_pool: CommandPool,
    wait_semaphores: Vec<WaitSemaphore>,
    free_semaphores: Vec<Semaphore>,
}

impl Uploader {
    pub fn new(
        instance: &Instance,
        device: &Rc<Device>,
        transfer_queue: vk::Queue,
        graphics_queue: vk::Queue,
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
            staging_memory,
            transfer_queue,
            graphics_queue,
            device: device.clone(),
            command_pool,
            wait_semaphores: Vec::new(),
            free_semaphores: Vec::new(),
        })
    }

    pub(crate) fn start_command_buffer<F>(&mut self, wait_stage: vk::PipelineStageFlags, f: F) -> Result<(), Error>
    where
        F: FnOnce(vk::CommandBuffer),
    {
        let command_buffers = {
            profiling::scope!("allocate command buffer");
            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.command_pool.inner)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            unsafe { self.device.allocate_command_buffers(&command_buffer_allocate_info) }.map_err(Error::VulkanCommandBuffersAllocation)?
        };
        let temp_command_buffer = command_buffers[0];

        let signal_semaphore = if let Some(semaphore) = self.free_semaphores.pop() {
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
            let signal_semaphores = [signal_semaphore.inner];
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

        self.wait_semaphores.push(WaitSemaphore(signal_semaphore, wait_stage));
        Ok(())
    }

    pub fn reset(&mut self) -> Result<(), Error> {
        self.free_semaphores
            .extend(self.wait_semaphores.drain(..).map(|WaitSemaphore(semaphore, _)| semaphore));
        unsafe {
            self.device
                .reset_command_pool(self.command_pool.inner, vk::CommandPoolResetFlags::empty())
                .map_err(Error::VulkanResetCommandPool)
        }
    }
}
