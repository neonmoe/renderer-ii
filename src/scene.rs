use crate::mesh::Mesh;
use crate::pipeline_parameters::PipelineMap;
use crate::Material;
use glam::Mat4;
use std::collections::HashMap;

type MeshMap<'a> = HashMap<(&'a Mesh, &'a Material), Vec<Mat4>>;

/// A container for the meshes rendered during a particular frame, and
/// the transforms those meshes are rendered with.
pub struct Scene<'a> {
    pub pipeline_map: PipelineMap<MeshMap<'a>>,
}

impl<'a> Default for Scene<'a> {
    fn default() -> Scene<'a> {
        Scene::new()
    }
}

impl<'a> Scene<'a> {
    pub fn new() -> Scene<'a> {
        Scene {
            pipeline_map: PipelineMap::new::<(), _>(|_| Ok(HashMap::new())).unwrap(),
        }
    }

    #[profiling::function]
    pub fn queue(&mut self, mesh: &'a Mesh, material: &'a Material, transform: Mat4) {
        let mesh_map = self.pipeline_map.get_mut(mesh.pipeline);
        let mesh_vec = mesh_map.entry((mesh, material)).or_insert_with(Vec::new);
        mesh_vec.push(transform);
    }
}
