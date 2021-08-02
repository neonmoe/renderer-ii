use crate::buffer::Buffer;
use crate::{Canvas, Error, Gpu, Pipeline};
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
                Vec3::new(0.75, 0.0, 1.5),
                Vec3::zero(),
                Vec3::new(0.0, 1.0, 0.0),
            ),
        }
    }
}

pub struct Camera<'a> {
    pub(crate) transforms_buffer: Buffer<'a>,
}

impl Camera<'_> {
    pub fn new<'a>(gpu: &'a Gpu, canvas: &Canvas) -> Result<Camera<'a>, Error> {
        let transforms_buffer = Buffer::new(gpu, &[GlobalTransforms::new(canvas)], true)?;
        Ok(Camera { transforms_buffer })
    }

    /// Updates Vulkan buffers with the current state of the
    /// [Camera] and [Canvas].
    pub(crate) fn update(&self, canvas: &Canvas, frame_index: u32) -> Result<(), Error> {
        self.transforms_buffer
            .update_data(&canvas.gpu, &[GlobalTransforms::new(canvas)])?;
        let buffer = self.transforms_buffer.buffer(frame_index)?;
        canvas.gpu.descriptors.set_uniform_buffer(
            &canvas.gpu,
            Pipeline::PlainVertexColor,
            frame_index,
            0,
            0,
            buffer,
        );
        Ok(())
    }
}
