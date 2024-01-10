use core::cmp::Ordering;
use core::mem;

use glam::Mat4;

use crate::renderer::descriptors::material::Material;
use crate::renderer::pipeline_parameters::vertex_buffers::VertexLayout;

pub(crate) mod camera;
pub(crate) mod coordinate_system;
pub(crate) mod draw_call_tag;
pub(crate) mod mesh;

use camera::Camera;
use coordinate_system::CoordinateSystem;
use draw_call_tag::DrawCallTag;
use mesh::Mesh;

use crate::renderer::pipeline_parameters::constants::{MAX_DRAW_CALLS, MAX_JOINT_COUNT};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct JointsOffset(pub(crate) u32);

pub struct DrawParameters<'a> {
    pub tag: DrawCallTag<'a>,
    pub transform: Mat4,
    pub joints: Option<JointsOffset>,
}
impl PartialEq for DrawParameters<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.tag == other.tag && self.joints == other.joints
    }
}
impl Eq for DrawParameters<'_> {}
impl Ord for DrawParameters<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.joints.cmp(&other.joints) {
            Ordering::Equal => self.tag.cmp(&other.tag),
            ordering => ordering,
        }
    }
}
impl PartialOrd for DrawParameters<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// A container for the materials and meshes to render during a particular
/// frame, and transforms for each instance.
pub struct Scene<'a> {
    pub world_space: CoordinateSystem,
    pub camera: Camera,
    pub draws: Vec<DrawParameters<'a>>,
    pub(crate) skinned_mesh_joints_buffer: Vec<u8>,
}

impl Default for Scene<'_> {
    fn default() -> Self {
        Scene {
            world_space: CoordinateSystem::VULKAN,
            camera: Camera::default(),
            draws: Vec::new(),
            skinned_mesh_joints_buffer: Vec::new(),
        }
    }
}

impl<'a> Scene<'a> {
    pub fn clear(&mut self) {
        self.draws.clear();
        self.skinned_mesh_joints_buffer.clear();
    }

    /// Returns true if the mesh could be added to the queue. The only reason it
    /// cannot, is if the Scene has reached maximum supported draws
    /// ([`MAX_DRAWS`]). If `mesh` has a vertex layout of
    /// `VertexLayout::SkinnedMesh`, `joints` must be defined, and vice-versa.
    pub fn queue_mesh(&mut self, mesh: &'a Mesh, material: &'a Material, joints: Option<JointsOffset>, transform: Mat4) -> bool {
        profiling::scope!("queue mesh");
        if self.draws.len() < MAX_DRAW_CALLS as usize {
            // TODO: Check mesh's vertex layout and material's pipeline compatibility
            assert_eq!(mesh.vertex_layout == VertexLayout::SkinnedMesh, joints.is_some(), "skinned meshes must have joints defined");
            let pipeline = material.pipeline(mesh.vertex_layout);
            self.draws.push(DrawParameters {
                tag: DrawCallTag { pipeline, vertex_library: &mesh.library, mesh, material },
                transform,
                joints,
            });
            true
        } else {
            false
        }
    }

    /// Allocates `count` mat4's (each being 16 bytes) off of the `mat4[]
    /// uf_skeleton.bones` array, or None if there's no more space.
    pub fn allocate_joint_offset(&mut self, count: usize) -> Option<(JointsOffset, &mut [u8])> {
        let joint_size = mem::size_of::<Mat4>();
        let offset = self.skinned_mesh_joints_buffer.len();
        let next_offset = offset + count * joint_size;
        if next_offset <= MAX_JOINT_COUNT as usize * joint_size {
            self.skinned_mesh_joints_buffer.resize(next_offset, 0);
            let joints_offset = JointsOffset((offset / joint_size) as u32);
            Some((joints_offset, &mut self.skinned_mesh_joints_buffer[offset..next_offset]))
        } else {
            None
        }
    }
}
