use crate::buffer::Buffer;
use crate::{Canvas, Error, Gpu, Pipeline};
use ash::vk;
use ultraviolet::Mat4;

struct GlobalTransforms {
    _projection: Mat4,
    _view: Mat4,
}

impl GlobalTransforms {
    fn new(_canvas: &Canvas) -> GlobalTransforms {
        GlobalTransforms {
            _projection: Mat4::identity(),
            _view: Mat4::identity(),
        }
    }
}

pub struct Camera<'a> {
    pub(crate) transforms_buffer: Buffer<'a>,
}

impl Camera<'_> {
    pub fn new<'a>(gpu: &'a Gpu, canvas: &Canvas) -> Result<Camera<'a>, Error> {
        let transforms_buffer = Buffer::new(
            gpu,
            &[GlobalTransforms::new(canvas)],
            true,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
        )?;
        gpu.descriptors.set_uniform_buffer(
            &gpu.device,
            Pipeline::PlainVertexColor,
            0,
            0,
            transforms_buffer.buffer,
            0,
            vk::WHOLE_SIZE,
        );
        Ok(Camera { transforms_buffer })
    }

    /// Updates Vulkan buffers with the current state of the
    /// [Camera] and [Canvas].
    pub(crate) fn update(&self, canvas: &Canvas) -> Result<(), Error> {
        self.transforms_buffer
            .update_data(&[GlobalTransforms::new(canvas)])
    }
}
