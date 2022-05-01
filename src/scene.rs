use crate::gltf::Keyframes;
use crate::mesh::Mesh;
use crate::pipeline_parameters::PipelineMap;
use crate::{Animation, Gltf, Material};
use glam::Mat4;
use std::collections::HashMap;

mod camera;
pub use camera::Camera;

#[derive(thiserror::Error, Debug)]
pub enum QueueingError {
    #[error("invalid timestamp {time} for animation {animation:?}")]
    InvalidAnimationTimestamp { animation: Option<String>, time: f32 },
}

pub(crate) struct SkinnedMesh<'a> {
    mesh: &'a Mesh,
    material: &'a Material,
    joints: Vec<Mat4>,
    transform: Mat4,
}

type StaticMeshMap<'a> = HashMap<(&'a Mesh, &'a Material), Vec<Mat4>>;

/// A container for the materials and meshes to render during a particular
/// frame, and transforms for each instance.
pub struct Scene<'a> {
    pub camera: Camera,
    pub(crate) static_meshes: PipelineMap<StaticMeshMap<'a>>,
    pub(crate) skinned_meshes: PipelineMap<Vec<SkinnedMesh<'a>>>,
}

impl Default for Scene<'_> {
    fn default() -> Self {
        Scene {
            camera: Camera::default(),
            static_meshes: PipelineMap::new::<(), _>(|_| Ok(HashMap::new())).unwrap(),
            skinned_meshes: PipelineMap::new::<(), _>(|_| Ok(Vec::new())).unwrap(),
        }
    }
}

impl<'a> Scene<'a> {
    pub fn queue(&mut self, model: &'a Gltf, transform: Mat4) {
        for mesh in model.mesh_iter() {
            profiling::scope!("queue mesh for rendering");
            let mesh_map = self.static_meshes.get_mut(mesh.mesh.pipeline);
            let mesh_vec = mesh_map.entry((mesh.mesh, mesh.material)).or_insert_with(Vec::new);
            mesh_vec.push(transform * mesh.transform);
        }
    }

    pub fn queue_animated(
        &mut self,
        model: &'a Gltf,
        transform: Mat4,
        played_animations: &[(f32, &Animation)],
    ) -> Result<(), QueueingError> {
        profiling::scope!("queue mesh with animation for rendering");
        for mesh in model.mesh_iter() {
            profiling::scope!("queue mesh for rendering");
            let mut animated_transform = mesh.transform;
            for (time, animation) in played_animations {
                let time = *time;
                let animation_channels = if let Some(channels) = &animation.nodes_channels[mesh.node_index] {
                    channels
                } else {
                    continue;
                };
                let (mut scale, mut rotation, mut translation) = animated_transform.to_scale_rotation_translation();
                for channel in animation_channels {
                    match &channel.keyframes {
                        Keyframes::Translation(frames) => {
                            translation = channel.interpolation.interpolate_vec3(frames, time).ok_or_else(|| {
                                QueueingError::InvalidAnimationTimestamp {
                                    animation: animation.name.clone(),
                                    time,
                                }
                            })?;
                        }
                        Keyframes::Rotation(frames) => {
                            rotation = channel.interpolation.interpolate_quat(frames, time).ok_or_else(|| {
                                QueueingError::InvalidAnimationTimestamp {
                                    animation: animation.name.clone(),
                                    time,
                                }
                            })?;
                        }
                        Keyframes::Scale(frames) => {
                            scale = channel.interpolation.interpolate_vec3(frames, time).ok_or_else(|| {
                                QueueingError::InvalidAnimationTimestamp {
                                    animation: animation.name.clone(),
                                    time,
                                }
                            })?;
                        }
                        Keyframes::Weight(_) => {}
                    }
                }
                animated_transform = Mat4::from_scale_rotation_translation(scale, rotation, translation);
            }

            if let Some(skin) = mesh.skin {
                let mesh_vec = self.skinned_meshes.get_mut(mesh.mesh.pipeline);
                mesh_vec.push(SkinnedMesh {
                    mesh: mesh.mesh,
                    material: mesh.material,
                    joints: vec![],
                    transform: transform * animated_transform,
                });
            } else {
                let mesh_map = self.static_meshes.get_mut(mesh.mesh.pipeline);
                let mesh_vec = mesh_map.entry((mesh.mesh, mesh.material)).or_insert_with(Vec::new);
                mesh_vec.push(transform * animated_transform);
            }
        }
        Ok(())
    }
}
