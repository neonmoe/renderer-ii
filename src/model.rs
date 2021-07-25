use crate::buffer::Buffer;
use crate::{Error, Gpu};

/// The interface to load and render 3D models.
pub struct Model<'a> {
    pub(crate) buffer: Buffer<'a>,
}

impl Model<'_> {
    pub fn new<'a, V>(
        gpu: &'a Gpu<'_>,
        vertices: &[V],
        editable: bool,
    ) -> Result<Model<'a>, Error> {
        let buffer = Buffer::new(gpu, vertices, editable)?;
        Ok(Model { buffer })
    }

    pub fn update_vertices<V>(&mut self, new_vertices: &[V]) -> Result<(), Error> {
        self.buffer.update_vertices(new_vertices)
    }
}
