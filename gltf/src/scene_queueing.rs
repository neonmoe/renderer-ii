use std::mem::size_of;

use crate::{Animation, AnimationError, Gltf};
use glam::Mat4;
use renderer::{Scene, SkinnedModel};

impl Gltf {
    pub fn queue<'a>(&'a self, scene: &mut Scene<'a>, transform: Mat4) {
        profiling::scope!("queue model for rendering");
        for mesh in self.mesh_iter() {
            scene.queue_mesh(mesh.mesh, mesh.material, transform * mesh.transform);
        }
    }

    pub fn queue_animated<'a>(
        &'a self,
        scene: &mut Scene<'a>,
        transform: Mat4,
        playing_animations: &[(f32, &Animation)],
    ) -> Result<(), AnimationError> {
        profiling::scope!("queue animated model for rendering");
        let mut skinned_models: Vec<SkinnedModel<'_>> = Vec::with_capacity(0);
        let animated_node_transforms = self.get_node_transforms(playing_animations)?;
        for mesh in self.mesh_iter() {
            if let Some(skin_index) = mesh.skin {
                profiling::scope!("skinned mesh");
                let pipeline = mesh.material.pipeline(true);
                if let Some(skinned_model) = skinned_models
                    .iter_mut()
                    .find(|model| model.skin == skin_index && model.pipeline == pipeline)
                {
                    skinned_model.meshes.push((mesh.mesh, mesh.material));
                } else {
                    let skin = &self.skins[skin_index];
                    let joint_size = size_of::<Mat4>();
                    let joints_buffer_size = skin.joints.len() * joint_size;
                    // TODO: Reuse skeletons (aka joint_offset) between identical skins
                    let (joints_offset, joints_buffer) = scene.allocate_joint_offset(joints_buffer_size);
                    for (i, joint) in skin.joints.iter().enumerate() {
                        let animated_transform = animated_node_transforms[joint.node_index].unwrap_or(Mat4::IDENTITY);
                        let joint_transform = animated_transform * joint.inverse_bind_matrix;
                        let offset = i * joint_size;
                        let dst = &mut joints_buffer[offset..offset + joint_size];
                        dst.copy_from_slice(bytemuck::cast_slice(&[joint_transform]));
                    }
                    skinned_models.push(SkinnedModel {
                        pipeline,
                        skin: skin_index,
                        meshes: vec![(mesh.mesh, mesh.material)],
                        transform,
                        joints_offset,
                    });
                }
            } else {
                profiling::scope!("non-skinned mesh");
                let animated_transform = animated_node_transforms[mesh.node_index].unwrap_or(Mat4::IDENTITY);
                let mesh_map = &mut scene.static_meshes[mesh.material.pipeline(false)];
                let mesh_vec = mesh_map.entry((mesh.mesh, mesh.material)).or_insert_with(Vec::new);
                mesh_vec.push(transform * animated_transform);
            }
        }
        for skinned_model in skinned_models {
            let mesh_vec = &mut scene.skinned_meshes[skinned_model.pipeline];
            mesh_vec.push(skinned_model);
        }
        Ok(())
    }
}
