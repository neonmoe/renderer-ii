use crate::{Canvas, Descriptors, Error, FrameIndex, PipelineIndex, VulkanArena};
use ash::vk;
use glam::{Mat4, Quat, Vec3};
use std::mem;

#[repr(C)]
struct GlobalTransforms {
    _projection: Mat4,
    _view: Mat4,
}

impl GlobalTransforms {
    fn new(canvas: &Canvas) -> GlobalTransforms {
        let fov = 74f32.to_radians();
        let aspect_ratio = canvas.extent.width as f32 / canvas.extent.height as f32;
        // Lower values seem to cause Z-fighting in the sponza scene.
        // Might be better to use two projection matrixes for e.g. 0.1->5, 5->inf.
        let near = 0.5;
        let camera_transform = Mat4::from_rotation_translation(Quat::from_rotation_y(1.45), Vec3::new(3.0, 1.6, 0.5));
        GlobalTransforms {
            _projection: reverse_z_rh_infinite_projection(fov, aspect_ratio, near),
            _view: camera_transform.inverse(),
        }
    }
}

fn reverse_z_rh_infinite_projection(fov: f32, aspect_ratio: f32, near: f32) -> Mat4 {
    let sy = 1.0 / (fov / 2.0).tan();
    let sx = sy / aspect_ratio;
    Mat4::from_cols_array(&[sx, 0.0, 0.0, 0.0, 0.0, -sy, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, near, 0.0])
}

#[derive(Default)]
pub struct Camera {}

impl Camera {
    /// Updates Vulkan buffers with the current state of the
    /// [Camera] and [Canvas].
    #[profiling::function]
    pub(crate) fn update(
        &self,
        descriptors: &Descriptors,
        canvas: &Canvas,
        temp_arena: &mut VulkanArena,
        frame_index: FrameIndex,
    ) -> Result<(), Error> {
        let temp_buffer = {
            profiling::scope!("create uniform buffer");
            let buffer_size = mem::size_of::<GlobalTransforms>() as vk::DeviceSize;
            let buffer_create_info = vk::BufferCreateInfo::builder()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            temp_arena.create_buffer(*buffer_create_info, format_args!("uniform (view+proj matrices)"))?
        };

        {
            profiling::scope!("write uniform buffer");
            let src = &[GlobalTransforms::new(canvas)];
            unsafe { temp_buffer.write(src.as_ptr() as *const u8, 0, vk::WHOLE_SIZE) }?;
        }

        let pipeline = PipelineIndex::SHARED_DESCRIPTOR_PIPELINE;
        descriptors.set_uniform_buffer(frame_index, pipeline, 0, 0, temp_buffer.buffer.inner);
        temp_arena.add_buffer(temp_buffer.buffer);
        Ok(())
    }
}
