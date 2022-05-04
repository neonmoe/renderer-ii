use crate::gltf::Keyframes;
use crate::mesh::Mesh;
use crate::pipeline_parameters::PipelineMap;
use crate::{Animation, Gltf, Material, PipelineIndex};
use glam::Mat4;
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
    pub(crate) joints: Vec<Mat4>,
}

type StaticMeshMap<'a> = HashMap<(&'a Mesh, &'a Material), Vec<Mat4>>;

/// A container for the materials and meshes to render during a particular
/// frame, and transforms for each instance.
pub struct Scene<'a> {
    pub camera: Camera,
    pub(crate) static_meshes: PipelineMap<StaticMeshMap<'a>>,
    pub(crate) skinned_meshes: PipelineMap<Vec<SkinnedModel<'a>>>,
}

impl Default for Scene<'_> {
    fn default() -> Self {
        Scene {
            camera: Camera::default(),
            static_meshes: PipelineMap::new::<(), _>(|_| Ok(HashMap::default())).unwrap(),
            skinned_meshes: PipelineMap::new::<(), _>(|_| Ok(Vec::new())).unwrap(),
        }
    }
}

impl<'a> Scene<'a> {
    pub fn queue(&mut self, model: &'a Gltf, transform: Mat4) {
        profiling::scope!("queue model for rendering");
        for mesh in model.mesh_iter() {
            profiling::scope!("static mesh");
            let mesh_map = &mut self.static_meshes[mesh.mesh.pipeline];
            let mesh_vec = mesh_map.entry((mesh.mesh, mesh.material)).or_insert_with(Vec::new);
            mesh_vec.push(transform * mesh.transform);
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
                if let Some(skinned_model) = skinned_models
                    .iter_mut()
                    .find(|model| model.skin == skin_index && model.pipeline == mesh.mesh.pipeline)
                {
                    skinned_model.meshes.push((mesh.mesh, mesh.material));
                } else {
                    let skin = &model.skins[skin_index];
                    let mut joints = Vec::with_capacity(skin.joints.len());
                    for joint in &skin.joints {
                        let inverse_bind_matrix = joint.inverse_bind_matrix;
                        let animated_transform = get_animation_transform(joint.node_index, joint.resting_transform, playing_animations)?;
                        let joint_transform = animated_transform * inverse_bind_matrix;
                        joints.push(joint_transform);
                    }
                    skinned_models.push(SkinnedModel {
                        pipeline: mesh.mesh.pipeline,
                        skin: skin_index,
                        meshes: vec![(mesh.mesh, mesh.material)],
                        joints,
                        transform,
                    });
                }
            } else {
                profiling::scope!("non-skinned mesh");
                let animated_transform = get_animation_transform(mesh.node_index, mesh.transform, playing_animations)?;
                let mesh_map = &mut self.static_meshes[mesh.mesh.pipeline];
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
        let (mut scale, mut rotation, mut translation) = animated_transform.to_scale_rotation_translation();
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
        animated_transform = Mat4::from_scale_rotation_translation(scale, rotation, translation);
    }
    Ok(animated_transform)
}
