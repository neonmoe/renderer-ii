use crate::gltf::Keyframes;
use crate::mesh::Mesh;
use crate::pipeline_parameters::PipelineMap;
use crate::{Animation, Gltf, Material, PipelineIndex};
use glam::{Mat4, Quat, Vec3};
use std::collections::HashMap;

mod camera;
pub use camera::Camera;

#[derive(thiserror::Error, Debug)]
pub enum QueueingError {
    #[error("invalid timestamp {time} for animation {animation:?}")]
    InvalidAnimationTimestamp { animation: Option<String>, time: f32 },
}

pub(crate) struct SkinnedModel<'a> {
    skin: usize,
    pipeline: PipelineIndex,
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
}

impl Default for Scene<'_> {
    fn default() -> Self {
        Scene {
            camera: Camera::default(),
            static_meshes: PipelineMap::new::<(), _>(|_| Ok(HashMap::with_capacity(0))).unwrap(),
            skinned_meshes: PipelineMap::new::<(), _>(|_| Ok(Vec::with_capacity(0))).unwrap(),
            skinned_mesh_joints_buffer: Vec::new(),
        }
    }
}

impl<'a> Scene<'a> {
    pub fn queue_mesh(&mut self, mesh: &'a Mesh, material: &'a Material, transform: Mat4) {
        profiling::scope!("static mesh");
        let mesh_map = &mut self.static_meshes[material.pipeline(false)];
        let mesh_vec = mesh_map.entry((mesh, material)).or_insert_with(Vec::new);
        mesh_vec.push(transform);
    }

    pub fn queue(&mut self, model: &'a Gltf, transform: Mat4) {
        profiling::scope!("queue model for rendering");
        for mesh in model.mesh_iter() {
            self.queue_mesh(mesh.mesh, mesh.material, transform * mesh.transform);
        }
    }

    pub fn queue_animated(
        &mut self,
        model: &'a Gltf,
        transform: Mat4,
        playing_animations: &[(f32, &Animation)],
    ) -> Result<(), QueueingError> {
        profiling::scope!("queue animated model for rendering");
        let mut skinned_models: Vec<SkinnedModel<'_>> = Vec::with_capacity(0);
        for mesh in model.mesh_iter() {
            if let Some(skin_index) = mesh.skin {
                profiling::scope!("skinned mesh");
                let pipeline = mesh.material.pipeline(true);
                if let Some(skinned_model) = skinned_models
                    .iter_mut()
                    .find(|model| model.skin == skin_index && model.pipeline == pipeline)
                {
                    skinned_model.meshes.push((mesh.mesh, mesh.material));
                } else {
                    let skin = &model.skins[skin_index];
                    // TODO: Align each skeleton to VkPhysicalDeviceLimits::minUniformBufferOffsetAlignment
                    let joints_offset = self.skinned_mesh_joints_buffer.len() as u32;
                    for joint in &skin.joints {
                        let animated_transform = get_animation_transform(joint.node_index, joint.resting_transform, playing_animations)?;
                        let joint_transform = animated_transform * joint.inverse_bind_matrix;
                        self.skinned_mesh_joints_buffer
                            .extend_from_slice(bytemuck::cast_slice(&[joint_transform]));
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
                let animated_transform = get_animation_transform(mesh.node_index, mesh.transform, playing_animations)?;
                let mesh_map = &mut self.static_meshes[mesh.material.pipeline(false)];
                let mesh_vec = mesh_map.entry((mesh.mesh, mesh.material)).or_insert_with(Vec::new);
                mesh_vec.push(transform * animated_transform);
            }
        }
        for skinned_model in skinned_models {
            let mesh_vec = &mut self.skinned_meshes[skinned_model.pipeline];
            mesh_vec.push(skinned_model);
        }
        Ok(())
    }
}

// TODO: Move this to gltf, use node.transform instead of neutral_transform, and run this for all nodes when animating.
// The animate-function should just return a Vec<Option<Mat4>> which matches all
// the nodes' transforms. Then just do lookups to get the animated transform. No
// need for joint resting transforms, and parent influence can be done for both
// skinned and static animations in the same place.
fn get_animation_transform(
    node_index: usize,
    neutral_transform: Mat4,
    playing_animations: &[(f32, &Animation)],
) -> Result<Mat4, QueueingError> {
    let mut animated_transform = neutral_transform;
    for (time, animation) in playing_animations {
        let time = *time;
        let animation_channels = if let Some(channels) = &animation.nodes_channels[node_index] {
            channels
        } else {
            continue;
        };
        let (mut scale, mut rotation, mut translation) = (Vec3::ONE, Quat::IDENTITY, Vec3::ZERO);
        for channel in animation_channels {
            match &channel.keyframes {
                Keyframes::Translation(frames) => {
                    translation =
                        channel
                            .interpolation
                            .interpolate_vec3(frames, time)
                            .ok_or_else(|| QueueingError::InvalidAnimationTimestamp {
                                animation: animation.name.clone(),
                                time,
                            })?;
                }
                Keyframes::Rotation(frames) => {
                    rotation =
                        channel
                            .interpolation
                            .interpolate_quat(frames, time)
                            .ok_or_else(|| QueueingError::InvalidAnimationTimestamp {
                                animation: animation.name.clone(),
                                time,
                            })?;
                }
                Keyframes::Scale(frames) => {
                    scale =
                        channel
                            .interpolation
                            .interpolate_vec3(frames, time)
                            .ok_or_else(|| QueueingError::InvalidAnimationTimestamp {
                                animation: animation.name.clone(),
                                time,
                            })?;
                }
                Keyframes::Weight(_) => todo!(),
            }
        }
        animated_transform = Mat4::from_scale_rotation_translation(scale, rotation, translation) * animated_transform;
    }
    Ok(animated_transform)
}
