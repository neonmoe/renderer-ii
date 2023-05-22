use crate::descriptors::Material;
use crate::mesh::Mesh;
use crate::physical_device::PhysicalDevice;
use crate::pipeline_parameters::{PipelineIndex, PipelineMap};
use glam::Mat4;
use hashbrown::HashMap;

pub(crate) mod camera;
pub use camera::Camera;

pub(crate) struct SkinnedModel<'a> {
    pub(crate) skin: usize,
    pub(crate) pipeline: PipelineIndex,
    pub(crate) meshes: Vec<(&'a Mesh, &'a Material)>,
    pub(crate) transform: Mat4,
    pub(crate) joints_offset: u32,
}

pub(crate) type StaticMeshMap<'a> = HashMap<(&'a Mesh, &'a Material), Vec<Mat4>>;

/// A container for the materials and meshes to render during a particular
/// frame, and transforms for each instance.
pub struct Scene<'a> {
    pub camera: Camera,
    pub(crate) static_meshes: PipelineMap<StaticMeshMap<'a>>,
    pub(crate) skinned_meshes: PipelineMap<Vec<SkinnedModel<'a>>>,
    pub(crate) skinned_mesh_joints_buffer: Vec<u8>,
    pub(crate) joints_alignment: u32,
}

impl<'a> Scene<'a> {
    /// Creates a new scene for queueing meshes to render.
    ///
    /// Needs the PhysicalDevice which is currently being used for rendering to
    /// prepare the mesh data with proper alignment.
    pub fn new(physical_device: &PhysicalDevice) -> Self {
        Scene {
            camera: Camera::default(),
            static_meshes: PipelineMap::new::<(), _>(|_| Ok(HashMap::with_capacity(0))).unwrap(),
            skinned_meshes: PipelineMap::new::<(), _>(|_| Ok(Vec::with_capacity(0))).unwrap(),
            skinned_mesh_joints_buffer: Vec::new(),
            joints_alignment: physical_device.properties.limits.min_uniform_buffer_offset_alignment as u32,
        }
    }

    pub fn queue_mesh(&mut self, mesh: &'a Mesh, material: &'a Material, transform: Mat4) {
        profiling::scope!("static mesh");
        let mesh_map = &mut self.static_meshes[material.pipeline(false)];
        let mesh_vec = mesh_map.entry((mesh, material)).or_insert_with(Vec::new);
        mesh_vec.push(transform);
    }
}
