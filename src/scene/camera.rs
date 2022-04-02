use crate::{Descriptors, Error, ForBuffers, FrameIndex, PipelineIndex, VulkanArena};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Quat, Vec3};
use std::mem;

#[repr(C)]
#[derive(Clone, Copy)]
struct GlobalTransforms {
    _projection: Mat4,
    _view: Mat4,
}

// Mat4's are Pods, therefore they are Zeroable, therefore this is too.
unsafe impl Zeroable for GlobalTransforms {}

// repr(c) + Mat4's are Pods since glam has the bytemuck feature enabled.
unsafe impl Pod for GlobalTransforms {}

impl GlobalTransforms {
    fn new(width: f32, height: f32) -> GlobalTransforms {
        let fov = 74f32.to_radians();
        let aspect_ratio = width / height;
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
    #[profiling::function]
    pub(crate) fn upload(
        &self,
        descriptors: &Descriptors,
        temp_arena: &mut VulkanArena<ForBuffers>,
        frame_index: FrameIndex,
        width: f32,
        height: f32,
    ) -> Result<(), Error> {
        let src = &[GlobalTransforms::new(width, height)];
        let temp_buffer = {
            profiling::scope!("create uniform buffer");
            let buffer_size = mem::size_of::<GlobalTransforms>() as vk::DeviceSize;
            let buffer_create_info = vk::BufferCreateInfo::builder()
                .size(buffer_size)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            temp_arena.create_buffer(
                *buffer_create_info,
                bytemuck::cast_slice(src),
                None,
                format_args!("uniform (view+proj matrices)"),
            )?
        };

        let pipeline = PipelineIndex::SHARED_DESCRIPTOR_PIPELINE;
        descriptors.set_uniform_buffer(frame_index, pipeline, 0, 0, temp_buffer.inner);
        temp_arena.add_buffer(temp_buffer);
        Ok(())
    }
}
