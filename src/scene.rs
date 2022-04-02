use crate::mesh::Mesh;
use crate::pipeline_parameters::PipelineMap;
use crate::Material;
use glam::Mat4;
use std::collections::HashMap;

mod camera;
pub use camera::Camera;

type MeshMap<'a> = HashMap<(&'a Mesh, &'a Material), Vec<Mat4>>;

/// A container for the materials and meshes to render during a particular
/// frame, and transforms for each instance.
pub struct Scene<'a> {
    pub camera: Camera,
    pub pipeline_map: PipelineMap<MeshMap<'a>>,
}

impl Default for Scene<'_> {
    fn default() -> Self {
        Scene {
            camera: Camera::default(),
            pipeline_map: PipelineMap::new::<(), _>(|_| Ok(HashMap::new())).unwrap(),
        }
    }
}

impl<'a> Scene<'a> {
    #[profiling::function]
    pub fn queue(&mut self, mesh: &'a Mesh, material: &'a Material, transform: Mat4) {
        let mesh_map = self.pipeline_map.get_mut(mesh.pipeline);
        let mesh_vec = mesh_map.entry((mesh, material)).or_insert_with(Vec::new);
        mesh_vec.push(transform);
    }
}
