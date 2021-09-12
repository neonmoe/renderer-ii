use crate::{Mesh, Pipeline};
use std::collections::HashMap;
use ultraviolet::Mat4;

type MeshMap<'m> = HashMap<&'m Mesh<'m>, Vec<Mat4>>;
type PipelineMap<'m> = HashMap<Pipeline, MeshMap<'m>>;

/// A container for the meshes rendered during a particular frame, and
/// the transforms those meshes are rendered with.
pub struct Scene<'a> {
    pub pipeline_map: PipelineMap<'a>,
}

impl<'a> Default for Scene<'a> {
    fn default() -> Scene<'a> {
        Scene::new()
    }
}

impl<'a> Scene<'a> {
    pub fn new() -> Scene<'a> {
        Scene {
            pipeline_map: HashMap::new(),
        }
    }

    pub fn queue(&mut self, mesh: &'a Mesh, transform: Mat4) {
        let mesh_map = self.pipeline_map.entry(mesh.pipeline).or_insert_with(HashMap::new);
        let mesh_vec = mesh_map.entry(mesh).or_insert_with(Vec::new);
        mesh_vec.push(transform);
    }
}
