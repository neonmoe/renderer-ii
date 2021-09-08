use crate::buffer_ops;
use crate::{Canvas, Error, FrameIndex, Pipeline};
use ash::vk;
use std::mem;
use ultraviolet::{projection, Mat4, Vec3};

struct GlobalTransforms {
    _projection: Mat4,
    _view: Mat4,
}

impl GlobalTransforms {
    fn new(canvas: &Canvas) -> GlobalTransforms {
        GlobalTransforms {
            _projection: projection::perspective_vk(
                74f32.to_radians(),
                canvas.extent.width as f32 / canvas.extent.height as f32,
                0.1,
                100.0,
            ),
            _view: Mat4::look_at(
                Vec3::new(0.0, 0.0, 1.5),
                Vec3::zero(),
                Vec3::new(0.0, 1.0, 0.0),
            ),
        }
    }
}

pub struct Camera {}

impl Camera {
    #[profiling::function]
    pub fn new<'a>() -> Camera {
        Camera {}
    }

    /// Updates Vulkan buffers with the current state of the
    /// [Camera] and [Canvas].
    #[profiling::function]
    pub(crate) fn update(&self, canvas: &Canvas, frame_index: FrameIndex) -> Result<(), Error> {
        let gpu = &canvas.gpu;
        let (buffer, allocation, alloc_info) = {
            profiling::scope!("create uniform buffer");
            let buffer_size = mem::size_of::<GlobalTransforms>() as vk::DeviceSize;
            let buffer_create_info = vk::BufferCreateInfo::builder()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let allocation_create_info = vk_mem::AllocationCreateInfo {
                flags: vk_mem::AllocationCreateFlags::MAPPED,
                pool: Some(gpu.get_temp_buffer_pool(frame_index)),
                ..Default::default()
            };
            gpu.allocator
                .create_buffer(&buffer_create_info, &allocation_create_info)
                .map_err(Error::VmaBufferAllocation)?
        };
        gpu.add_temporary_buffer(frame_index, buffer, allocation);
        buffer_ops::copy_to_allocation(
            &[GlobalTransforms::new(canvas)],
            gpu,
            &allocation,
            &alloc_info,
        )?;
        gpu.descriptors
            .set_uniform_buffer(gpu, frame_index, Pipeline::Default, 0, 0, buffer);
        Ok(())
    }
}
