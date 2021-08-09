use crate::buffer::Buffer;
use crate::{Canvas, Error, FrameIndex, Gpu, Pipeline};
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
    #[profiling::function]
    pub fn new<'a>(
        gpu: &'a Gpu,
        canvas: &Canvas,
        frame_index: FrameIndex,
    ) -> Result<Camera<'a>, Error> {
        let transforms_buffer =
            Buffer::new(gpu, frame_index, &[GlobalTransforms::new(canvas)], true)?;
        Ok(Camera { transforms_buffer })
    }

    /// Updates Vulkan buffers with the current state of the
    /// [Camera] and [Canvas].
    #[profiling::function]
    pub(crate) fn update(&self, canvas: &Canvas, frame_index: FrameIndex) -> Result<(), Error> {
        self.transforms_buffer
            .update_data(&canvas.gpu, &[GlobalTransforms::new(canvas)])?;
        canvas.gpu.descriptors.set_uniform_buffer(
            &canvas.gpu,
            frame_index,
            Pipeline::PlainVertexColor,
            0,
            0,
            self.transforms_buffer.buffer(frame_index)?,
        );
        Ok(())
    }
}
