use crate::{Error, FrameIndex, Gpu};
use ash::vk;
use std::hash::{Hash, Hasher};

pub struct Buffer {
    pub(crate) buffer: vk::Buffer,
}

impl PartialEq for Buffer {
    fn eq(&self, other: &Self) -> bool {
        self.buffer == other.buffer
    }
}

impl Eq for Buffer {}

impl Hash for Buffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.buffer.hash(state);
    }
}

impl Buffer {
    /// Creates a new buffer. Ensure that the vertices match the
    /// pipeline. If not `editable`, call [Gpu::wait_buffer_uploads]
    /// after your buffer creation code, before they're rendered.
    ///
    /// Currently the buffers are always created as INDEX | VERTEX |
    /// UNIFORM buffers.
    pub fn new<T>(gpu: &Gpu, frame_index: FrameIndex, data: &[T]) -> Result<Buffer, Error> {
        profiling::scope!("new_buffer");
        todo!()
    }
}
