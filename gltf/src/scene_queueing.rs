use std::mem::size_of;

use glam::Mat4;
use renderer::Scene;

use crate::{Animation, AnimationError, Gltf};

impl Gltf {
    pub fn queue<'a>(&'a self, scene: &mut Scene<'a>, transform: Mat4) {
        profiling::scope!("queue model for rendering");
        for mesh in self.mesh_iter() {
            scene.queue_mesh(mesh.mesh, mesh.material, None, transform * mesh.transform);
        }
    }

    /// Returns true if all the meshes could be queued for rendering. If false,
    /// the Scene's draw budget has been spent, and not all meshes will be
    /// drawn.
    pub fn queue_animated<'a>(
        &'a self,
        scene: &mut Scene<'a>,
        transform: Mat4,
        playing_animations: &[(f32, &Animation)],
    ) -> Result<bool, AnimationError> {
        profiling::scope!("queue animated model for rendering");
        let mut all_drawn = true;
        let mut joints_offsets_per_skin = Vec::new();
        let animated_node_transforms = self.get_node_transforms(playing_animations)?;
        for mesh in self.mesh_iter() {
            if let Some(skin_index) = mesh.skin {
                profiling::scope!("skinned mesh");
                let joints_offset = if let Some(&(_, offset)) = joints_offsets_per_skin.iter().find(|(skin, _)| *skin == skin_index) {
                    offset
                } else {
                    let skin = &self.skins[skin_index];
                    let joint_size = size_of::<Mat4>();
                    let (joints_offset, joints_buffer) = scene.allocate_joint_offset(skin.joints.len()).expect("too many bones in scene");
                    for (i, joint) in skin.joints.iter().enumerate() {
                        let animated_transform = animated_node_transforms[joint.node_index].unwrap_or(Mat4::IDENTITY);
                        let joint_transform = animated_transform * joint.inverse_bind_matrix;
                        let offset = i * joint_size;
                        let dst = &mut joints_buffer[offset..offset + joint_size];
                        dst.copy_from_slice(bytemuck::cast_slice(&[joint_transform]));
                    }
                    joints_offsets_per_skin.push((skin_index, joints_offset));
                    joints_offset
                };
                all_drawn &= scene.queue_mesh(mesh.mesh, mesh.material, Some(joints_offset), transform);
            } else {
                profiling::scope!("non-skinned mesh");
                let animated_transform = animated_node_transforms[mesh.node_index].unwrap_or(Mat4::IDENTITY);
                all_drawn &= scene.queue_mesh(mesh.mesh, mesh.material, None, transform * animated_transform);
            }
        }
        Ok(all_drawn)
    }
}
