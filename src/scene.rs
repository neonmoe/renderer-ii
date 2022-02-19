use crate::mesh::Mesh;
use crate::{Material, Pipeline};
use glam::Mat4;
use std::collections::HashMap;

type MeshMap<'a> = HashMap<(&'a Mesh, &'a Material), Vec<Mat4>>;
// TODO: Use PipelineMap here.
type PipelineMap<'a> = HashMap<Pipeline, MeshMap<'a>>;

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

    #[profiling::function]
    pub fn queue(&mut self, mesh: &'a Mesh, material: &'a Material, transform: Mat4) {
        let mesh_map = self.pipeline_map.entry(mesh.pipeline).or_insert_with(HashMap::new);
        let mesh_vec = mesh_map.entry((mesh, material)).or_insert_with(Vec::new);
        mesh_vec.push(transform);
    }
}
