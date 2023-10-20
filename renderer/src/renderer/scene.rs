use glam::Mat4;
use hashbrown::HashMap;

use crate::physical_device::PhysicalDevice;
use crate::renderer::descriptors::material::Material;
use crate::renderer::pipeline_parameters::{PipelineIndex, PipelineMap};

pub(crate) mod camera;
pub(crate) mod coordinate_system;
pub(crate) mod mesh;

use camera::Camera;
use coordinate_system::CoordinateSystem;
use mesh::Mesh;

pub struct JointOffset(pub(crate) u32);

pub struct SkinnedModel<'a> {
    pub skin: usize,
    pub pipeline: PipelineIndex,
    pub meshes: Vec<(&'a Mesh, &'a Material)>,
    pub transform: Mat4,
    pub joints_offset: JointOffset,
}

pub type StaticMeshMap<'a> = HashMap<(&'a Mesh, &'a Material), Vec<Mat4>>;

/// A container for the materials and meshes to render during a particular
/// frame, and transforms for each instance.
pub struct Scene<'a> {
    pub world_space: CoordinateSystem,
    pub camera: Camera,
    pub static_meshes: PipelineMap<StaticMeshMap<'a>>,
    pub skinned_meshes: PipelineMap<Vec<SkinnedModel<'a>>>,
    pub(crate) skinned_mesh_joints_buffer: Vec<u8>,
    pub(crate) joints_alignment: u32,
}

impl<'a> Scene<'a> {
    /// Creates a new scene for queueing meshes to render.
    ///
    /// Needs the `PhysicalDevice` which is currently being used for rendering to
    /// prepare the mesh data with proper alignment.
    pub fn new(physical_device: &PhysicalDevice) -> Self {
        Scene {
            world_space: CoordinateSystem::VULKAN,
            camera: Camera::default(),
            static_meshes: PipelineMap::from_infallible(|_| HashMap::with_capacity(0)),
            skinned_meshes: PipelineMap::from_infallible(|_| Vec::with_capacity(0)),
            skinned_mesh_joints_buffer: Vec::new(),
            joints_alignment: physical_device.properties.limits.min_uniform_buffer_offset_alignment as u32,
        }
    }

    pub fn queue_mesh(&mut self, mesh: &'a Mesh, material: &'a Material, transform: Mat4) {
        profiling::scope!("static mesh");
        let mesh_map = &mut self.static_meshes[material.pipeline(false)];
        let mesh_vec = mesh_map.entry((mesh, material)).or_default();
        mesh_vec.push(transform);
    }

    pub fn allocate_joint_offset(&mut self, size: usize) -> (JointOffset, &mut [u8]) {
        let current_allocated = self.skinned_mesh_joints_buffer.len();
        let aligned_offset = current_allocated.next_multiple_of(self.joints_alignment as usize);
        let new_allocated = aligned_offset + size;
        self.skinned_mesh_joints_buffer.resize(new_allocated, 0);
        (
            JointOffset(aligned_offset as u32),
            &mut self.skinned_mesh_joints_buffer[aligned_offset..new_allocated],
        )
    }
}
