use core::cmp::Ordering;

use glam::Mat4;

use crate::physical_device::PhysicalDevice;
use crate::renderer::descriptors::material::Material;
use crate::renderer::pipeline_parameters::{PipelineIndex, PipelineMap};

pub(crate) mod camera;
pub(crate) mod coordinate_system;
pub(crate) mod draw_call_tag;
pub(crate) mod mesh;

use camera::Camera;
use coordinate_system::CoordinateSystem;
use draw_call_tag::DrawCallTag;
use mesh::Mesh;

use crate::renderer::pipeline_parameters::constants::MAX_DRAWS;

pub struct JointOffset(pub(crate) u32);

pub struct SkinnedModel<'a> {
    pub skin: usize,
    pub pipeline: PipelineIndex,
    pub meshes: Vec<(&'a Mesh, &'a Material)>,
    pub transform: Mat4,
    pub joints_offset: JointOffset,
}

pub struct StaticDraw<'a> {
    pub tag: DrawCallTag<'a>,
    pub transform: Mat4,
}

impl PartialEq for StaticDraw<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.tag == other.tag
    }
}

impl Eq for StaticDraw<'_> {}

impl Ord for StaticDraw<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.tag.cmp(&other.tag)
    }
}

impl PartialOrd for StaticDraw<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub type StaticDraws<'a> = Vec<StaticDraw<'a>>;

/// A container for the materials and meshes to render during a particular
/// frame, and transforms for each instance.
pub struct Scene<'a> {
    pub world_space: CoordinateSystem,
    pub camera: Camera,
    pub total_draws: u32,
    pub static_draws: StaticDraws<'a>,
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
            total_draws: 0,
            static_draws: Vec::new(),
            skinned_meshes: PipelineMap::from_infallible(|_| Vec::new()),
            skinned_mesh_joints_buffer: Vec::new(),
            joints_alignment: physical_device.properties.limits.min_uniform_buffer_offset_alignment as u32,
        }
    }

    /// Returns true if the mesh could be added to the queue. The only reason it cannot, is if the
    /// Scene has reached maximum supported draws ([`MAX_DRAWS`]).
    pub fn queue_mesh(&mut self, mesh: &'a Mesh, material: &'a Material, transform: Mat4) -> bool {
        profiling::scope!("static mesh");
        if self.total_draws < MAX_DRAWS {
            // TODO: Check mesh's vertex layout and material's pipeline compatibility
            self.static_draws.push(StaticDraw {
                tag: DrawCallTag {
                    pipeline: material.pipeline(false),
                    vertex_library: &mesh.library,
                    mesh,
                    material,
                },
                transform,
            });
            self.total_draws += 1;
            true
        } else {
            false
        }
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
