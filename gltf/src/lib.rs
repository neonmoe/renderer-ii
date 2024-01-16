extern crate alloc;

use alloc::rc::Rc;
use core::mem;
use core::ops::Range;
use std::fs::File;
use std::path::{Path, PathBuf};

use arrayvec::ArrayString;
use glam::{Affine3A, Mat4, Quat, Vec3};
use hashbrown::HashMap;
use memmap2::{Mmap, MmapOptions};
use renderer::image_loading::{ntex, TextureKind};
use renderer::{
    ForImages, Material, Mesh, PipelineIndex, VertexBinding, VertexBindingMap, VertexLibraryMeasurer, VulkanArenaError, VulkanArenaMeasurer,
};

mod gltf_json;
mod mesh_iter;
mod pending_gltf;
mod scene_queueing;

use gltf_json::AnimationInterpolation;
pub use pending_gltf::*;

const MAX_VERTEX_BUFFERS: usize = 6;
const GLTF_BYTE: i32 = 5120;
const GLTF_UNSIGNED_BYTE: i32 = 5121;
const GLTF_SHORT: i32 = 5122;
const GLTF_UNSIGNED_SHORT: i32 = 5123;
const GLTF_UNSIGNED_INT: i32 = 5125;
const GLTF_FLOAT: i32 = 5126;

#[derive(thiserror::Error, Debug)]
pub enum GltfLoadingError {
    #[error("not a glb file")]
    InvalidGlbHeader,
    #[error("glb header length mismatch")]
    InvalidGlbLength,
    #[error("glb chunk length mismatch")]
    InvalidGlbChunkLength,
    #[error("invalid glb chunk type")]
    InvalidGlbChunkType,
    #[error("too many glb json chunks")]
    TooManyGlbJsonChunks,
    #[error("too many glb binary chunks")]
    TooManyGlbBinaryChunks,
    #[error("glb json is not valid utf-8")]
    InvalidGlbJson(#[source] core::str::Utf8Error),
    #[error("glb json chunk missing")]
    MissingGlbJson,
    #[error("failed to deserialize gltf json")]
    JsonDeserialization(serde_json::Error),
    #[error("unsupported gltf minimum version ({0}), 2.0 is supported")]
    UnsupportedGltfVersion(ArrayString<32>),
    #[error("gltf has buffer without an uri but no glb BIN buffer")]
    GlbBinMissing,
    #[error("gltf refers to external data ({0}) but no directory was given in from_gltf/from_glb")]
    MissingDirectory(String),
    #[error("could not open resource file for gltf at {1}")]
    OpenFile(#[source] std::io::Error, std::path::PathBuf),
    #[error("could not map resource file into memory for gltf at {1}")]
    MapFile(#[source] std::io::Error, std::path::PathBuf),
    #[error("failed to create a buffer for gltf from a file {1}")]
    BufferCreationFromFile(#[source] VulkanArenaError, std::path::PathBuf),
    #[error("failed to create a buffer for gltf from the internal glb buffer")]
    BufferCreationFromGlb(#[source] VulkanArenaError),
    #[error("failed to create a buffer for gltf from the material parameters")]
    BufferCreationFromMaterialParameters(#[source] VulkanArenaError),
    #[error("failed to decode ntex {1}")]
    NtexDecoding(#[source] ntex::NtexDecodeError, String),
    #[error("failed to create material, ran out of material slots")]
    NotEnoughMaterialSlots,
    #[error("failed to load image {1}")]
    ImageLoading(#[source] VulkanArenaError, String),
    #[error("gltf node has multiple parents, which is not allowed by the 2.0 spec")]
    InvalidNodeGraph,
    #[error("gltf has an out-of-bounds index ({0})")]
    Oob(&'static str),
    #[error("gltf does not conform to the 2.0 spec: {0}")]
    Spec(&'static str),
    #[error("unimplemented gltf feature: {0}")]
    Misc(&'static str),
}

#[derive(thiserror::Error, Debug)]
pub enum AnimationError {
    #[error("invalid timestamp {time} for animation {animation:?}")]
    InvalidAnimationTimestamp { animation: Option<String>, time: f32 },
}

pub type NodeAnimChannels = Vec<AnimationChannel>;
#[derive(Clone)]
pub struct Animation {
    pub name: Option<String>,
    pub start_time: f32,
    pub end_time: f32,
    pub nodes_channels: Vec<Option<NodeAnimChannels>>,
}

#[derive(Clone)]
pub struct AnimationChannel {
    pub interpolation: AnimationInterpolation,
    pub keyframes: Keyframes,
}

#[derive(Clone)]
pub enum Keyframes {
    Translation(Vec<(f32, Vec3)>),
    Rotation(Vec<(f32, Quat)>),
    Scale(Vec<(f32, Vec3)>),
    Weight(Vec<(f32, f32)>),
}

#[derive(Clone)]
pub struct Node {
    pub name: Option<ArrayString<64>>,
    pub transform: Affine3A,
    pub children: Vec<usize>,
    /// The tuple consists of (min coord, max coord).
    pub bounding_box: Option<(Vec3, Vec3)>,
    mesh: Option<usize>,
    skin: Option<usize>,
}

#[derive(Clone)]
pub(crate) struct Joint {
    pub(crate) inverse_bind_matrix: Mat4,
    pub(crate) node_index: usize,
}

#[derive(Clone)]
pub(crate) struct Skin {
    pub(crate) joints: Vec<Joint>,
}

#[derive(Clone)]
pub struct Gltf {
    pub animations: Vec<Animation>,
    pub nodes: Vec<Node>,
    root_nodes: Vec<usize>,
    meshes: Vec<Vec<(Rc<Mesh>, usize)>>,
    materials: Vec<Rc<Material>>,
    pub(crate) skins: Vec<Skin>,
}

impl Gltf {
    /// Loads the glTF scene from a .glb file and measures textures and meshes, but does not write to VRAM yet.
    ///
    /// Any external files referenced in the glTF are searched relative to
    /// `resource_path`.
    pub fn preload_glb<'a>(
        glb_file_contents: &'a [u8],
        resource_path: PathBuf,
        (texture_measurer, mesh_measurer): (&mut VulkanArenaMeasurer<ForImages>, &mut VertexLibraryMeasurer),
    ) -> Result<PendingGltf<'a>, GltfLoadingError> {
        profiling::scope!("preloading glb from disk");
        let (json, buffer) = read_glb_json_and_buffer(glb_file_contents)?;
        let gltf: gltf_json::GltfJson = serde_json::from_str(json).map_err(GltfLoadingError::JsonDeserialization)?;
        create_gltf(gltf, resource_path, Some(buffer), texture_measurer, mesh_measurer)
    }

    /// Loads the glTF scene from a .gltf file and measures textures and meshes, but does not write to VRAM yet.
    ///
    /// Any external files referenced in the glTF are searched relative to
    /// `resource_path`.
    pub fn preload_gltf<'a>(
        gltf_file_contents: &'a str,
        resource_path: PathBuf,
        (texture_measurer, mesh_measurer): (&mut VulkanArenaMeasurer<ForImages>, &mut VertexLibraryMeasurer),
    ) -> Result<PendingGltf<'a>, GltfLoadingError> {
        profiling::scope!("preloading gltf from disk");
        let gltf: gltf_json::GltfJson = serde_json::from_str(gltf_file_contents).map_err(GltfLoadingError::JsonDeserialization)?;
        create_gltf(gltf, resource_path, None, texture_measurer, mesh_measurer)
    }

    pub fn get_animation(&self, name: &str) -> Option<&Animation> {
        self.animations.iter().find(|animation| if let Some(name_) = &animation.name { name == name_ } else { false })
    }

    pub fn get_node_transforms(&self, playing_animations: &[(f32, &Animation)]) -> Result<Vec<Option<Affine3A>>, AnimationError> {
        let mut transforms = vec![None; self.nodes.len()];
        let mut nodes_with_parent_transform = self.root_nodes.iter().map(|&node| (node, Affine3A::IDENTITY)).collect::<Vec<_>>();
        while let Some((node_index, parent_transform)) = nodes_with_parent_transform.pop() {
            let current_transform = parent_transform * self.get_animated_transform(node_index, playing_animations)?;
            assert_eq!(transforms[node_index], None);
            transforms[node_index] = Some(current_transform);
            for &child_index in &self.nodes[node_index].children {
                nodes_with_parent_transform.push((child_index, current_transform));
            }
        }
        Ok(transforms)
    }

    fn mesh_iter(&self) -> mesh_iter::MeshIter<'_> {
        mesh_iter::MeshIter::new(self, self.root_nodes.clone())
    }

    fn get_animated_transform(&self, node_index: usize, playing_animations: &[(f32, &Animation)]) -> Result<Affine3A, AnimationError> {
        let mut animated_transform = self.nodes[node_index].transform;
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
                        translation = channel
                            .interpolation
                            .interpolate_vec3(frames, time)
                            .ok_or_else(|| AnimationError::InvalidAnimationTimestamp { animation: animation.name.clone(), time })?;
                    }
                    Keyframes::Rotation(frames) => {
                        rotation = channel
                            .interpolation
                            .interpolate_quat(frames, time)
                            .ok_or_else(|| AnimationError::InvalidAnimationTimestamp { animation: animation.name.clone(), time })?;
                    }
                    Keyframes::Scale(frames) => {
                        scale = channel
                            .interpolation
                            .interpolate_vec3(frames, time)
                            .ok_or_else(|| AnimationError::InvalidAnimationTimestamp { animation: animation.name.clone(), time })?;
                    }
                    Keyframes::Weight(_) => todo!(),
                }
            }
            animated_transform = Affine3A::from_scale_rotation_translation(scale, rotation, translation);
        }
        Ok(animated_transform)
    }
}

#[profiling::function]
pub(crate) fn read_glb_json_and_buffer(glb: &[u8]) -> Result<(&str, &[u8]), GltfLoadingError> {
    fn read_u32(bytes: &[u8]) -> u32 {
        if let [a, b, c, d] = *bytes {
            u32::from_le_bytes([a, b, c, d])
        } else {
            unreachable!();
        }
    }

    const MAGIC_GLTF: u32 = 0x46546C67;
    if glb.len() < 12 || read_u32(&glb[0..4]) != MAGIC_GLTF {
        return Err(GltfLoadingError::InvalidGlbHeader);
    }
    let version = read_u32(&glb[4..8]);
    let length = read_u32(&glb[8..12]) as usize;
    if version != 2 {
        log::warn!(".glb file is not version 2, but trying to read anyway");
    }
    if length != glb.len() {
        return Err(GltfLoadingError::InvalidGlbLength);
    }

    let mut next_chunk = &glb[12..];
    let mut json: Option<&str> = None;
    let mut buffer: Option<&[u8]> = None;
    while next_chunk.len() >= 8 {
        let chunk_length = read_u32(&next_chunk[0..4]) as usize;
        if chunk_length > next_chunk.len() - 8 {
            return Err(GltfLoadingError::InvalidGlbChunkLength);
        }
        let chunk_bytes = &next_chunk[8..chunk_length + 8];

        const MAGIC_JSON: u32 = 0x4E4F534A;
        const MAGIC_BIN: u32 = 0x004E4942;
        let chunk_type = read_u32(&next_chunk[4..8]);
        match chunk_type {
            MAGIC_JSON => {
                if json.is_some() {
                    return Err(GltfLoadingError::TooManyGlbJsonChunks);
                }
                json = Some(core::str::from_utf8(chunk_bytes).map_err(GltfLoadingError::InvalidGlbJson)?);
            }
            MAGIC_BIN => {
                if buffer.is_some() {
                    return Err(GltfLoadingError::TooManyGlbBinaryChunks);
                }
                buffer = Some(chunk_bytes);
            }
            _ => return Err(GltfLoadingError::InvalidGlbChunkType),
        }

        next_chunk = &next_chunk[chunk_length + 8..];
    }

    let buffer = buffer.ok_or(GltfLoadingError::Misc("glb buffer is required"))?;
    let json = json.ok_or(GltfLoadingError::MissingGlbJson)?;
    Ok((json, buffer))
}

#[profiling::function]
fn create_gltf<'a>(
    gltf: gltf_json::GltfJson,
    resource_path: PathBuf,
    bin_buffer: Option<&'a [u8]>,
    texture_measurer: &mut VulkanArenaMeasurer<ForImages>,
    mesh_measurer: &mut VertexLibraryMeasurer,
) -> Result<PendingGltf<'a>, GltfLoadingError> {
    if let Some(min_version) = &gltf.asset.min_version {
        let min_version_f32 = str::parse::<f32>(min_version);
        if min_version_f32 != Ok(2.0) {
            return Err(GltfLoadingError::UnsupportedGltfVersion(*min_version));
        }
    } else if let Ok(version) = str::parse::<f32>(&gltf.asset.version) {
        if !(2.0..3.0).contains(&version) {
            return Err(GltfLoadingError::UnsupportedGltfVersion(gltf.asset.version));
        }
    } else {
        log::warn!("Could not parse glTF version {}, assuming 2.0.", gltf.asset.version);
    }

    let scene_index = gltf.scene.ok_or(GltfLoadingError::Misc("gltf does not have a scene"))?;
    let scenes = gltf.scenes.as_ref().ok_or(GltfLoadingError::Misc("scenes missing"))?;
    let scene = scenes.get(scene_index).ok_or(GltfLoadingError::Oob("scene"))?;
    let root_nodes = scene.nodes.clone().ok_or(GltfLoadingError::Misc("no nodes in scene"))?;

    let mut memmap_holder = None;

    let mut meshes = Vec::with_capacity(gltf.meshes.len());
    for mesh in &gltf.meshes {
        profiling::scope!("reading primitives");
        let mut primitives = Vec::with_capacity(mesh.primitives.len());
        for primitive in &mesh.primitives {
            let params = create_primitive(&gltf, primitive)?;
            let vertex_buffer_lengths: VertexBindingMap<usize> =
                VertexBindingMap::from_fn(|binding| params.vertex_buffers[binding].as_ref().map(|buf| buf.length));
            let index_count = params.index_buffer.length / params.index_buffer.stride;
            match params.index_buffer.stride {
                4 => mesh_measurer.add_mesh_by_len::<u32>(params.pipeline.vertex_layout(), &vertex_buffer_lengths, index_count),
                2 => mesh_measurer.add_mesh_by_len::<u16>(params.pipeline.vertex_layout(), &vertex_buffer_lengths, index_count),
                stride => panic!("create_primitive returned invalid stride {stride} for index buffer"),
            }
            let material_index = primitive.material.ok_or(GltfLoadingError::Misc("material missing"))?;
            primitives.push((params, material_index));
        }
        meshes.push(MeshParameters { name: mesh.name.clone(), primitives });
    }

    let mut nodes = Vec::with_capacity(gltf.nodes.len());
    for node in &gltf.nodes {
        profiling::scope!("reading node transform and bounding box");
        let transform = if let Some(cols_array) = node.matrix {
            Affine3A::from_mat4(Mat4::from_cols_array(&cols_array))
        } else {
            let translation = match node.translation {
                Some([x, y, z]) => Vec3::new(x, y, z),
                _ => Vec3::ZERO,
            };
            let rotation = match node.rotation {
                Some([x, y, z, w]) => Quat::from_xyzw(x, y, z, w),
                _ => Quat::IDENTITY,
            };
            let scale = match node.scale {
                Some([x, y, z]) => Vec3::new(x, y, z),
                _ => Vec3::new(1.0, 1.0, 1.0),
            };
            Affine3A::from_scale_rotation_translation(scale, rotation, translation)
        };

        let mut bounding_box = None;
        if let Some(mesh) = node.mesh {
            for primitive in &gltf.meshes[mesh].primitives {
                if let Some(&positions_accessor) = primitive.attributes.get("POSITION") {
                    let accessor = &gltf.accessors[positions_accessor];
                    if accessor.min.len() == 3 && accessor.max.len() == 3 {
                        let min = Vec3::from_slice(&accessor.min);
                        let max = Vec3::from_slice(&accessor.max);
                        bounding_box = Some((min, max));
                    }
                }
            }
        }

        nodes.push(Node {
            name: node.name,
            mesh: node.mesh,
            skin: node.skin,
            children: node.children.clone().unwrap_or_default(),
            transform,
            bounding_box,
        });
    }

    let image_texture_kinds = get_gltf_texture_kinds(&gltf)?;

    let mut images = Vec::with_capacity(gltf.images.len());
    for (i, image) in gltf.images.iter().enumerate() {
        profiling::scope!("uploading image", &format!("file: {:?}", image.uri));
        let Some(uri) = &image.uri else {
            return Err(GltfLoadingError::Misc("image missing an uri"));
        };
        let name = uri.clone();
        let mut path = resource_path.join(uri);
        path.set_extension("ntex");
        let bytes = map_file(&mut memmap_holder, &path, Some(0..1024))?;
        let data = ImageData::File(PathBuf::from(uri));
        let kind = image_texture_kinds.get(&i).copied().unwrap_or(TextureKind::LinearColor);
        let image_header = ntex::decode_header(bytes).map_err(|err| GltfLoadingError::NtexDecoding(err, name.clone()))?;
        let image_create_info = image_header.get_create_info(kind);
        texture_measurer.add_image(image_create_info);
        images.push(ImageParameters { data, name, image_create_info });
    }

    let mut animations = Vec::with_capacity(gltf.animations.len());
    for animation in &gltf.animations {
        profiling::scope!("reading animation data");
        let mut nodes_channels = vec![None; gltf.nodes.len()];
        let mut start_time: Option<f32> = None;
        let mut end_time: Option<f32> = None;
        for channel in &animation.channels {
            let sampler = animation.samplers.get(channel.sampler).ok_or(GltfLoadingError::Oob("animation samplers"))?;
            let channels_for_node = nodes_channels
                .get_mut(channel.target.node)
                .ok_or(GltfLoadingError::Oob("animation target node"))?
                .get_or_insert_with(Vec::new);
            let (timestamps, _) = get_slice_and_component_type_from_accessor(
                &mut memmap_holder,
                &resource_path,
                bin_buffer,
                &gltf,
                sampler.input,
                GLTF_FLOAT,
                "SCALAR",
            )?;
            let timestamps: &[f32] = bytemuck::cast_slice(timestamps);

            let timestamp_accessor = gltf.accessors.get(sampler.input).ok_or(GltfLoadingError::Oob("animation sampler input accessor"))?;
            if let Some(&min) = timestamp_accessor.min.first() {
                start_time = Some(if let Some(min_) = start_time { min_.min(min) } else { min });
            }
            if let Some(&max) = timestamp_accessor.max.first() {
                end_time = Some(if let Some(max_) = end_time { max_.max(max) } else { max });
            }

            let keyframes = match channel.target.path {
                gltf_json::AnimatedProperty::Translation | gltf_json::AnimatedProperty::Scale => {
                    let mut keyframes = timestamps.iter().map(|f| (*f, Vec3::ZERO)).collect::<Vec<(f32, Vec3)>>();
                    let (translations, _) = get_slice_and_component_type_from_accessor(
                        &mut memmap_holder,
                        &resource_path,
                        bin_buffer,
                        &gltf,
                        sampler.output,
                        GLTF_FLOAT,
                        "VEC3",
                    )?;
                    let translations: &[Vec3] = bytemuck::cast_slice(translations);
                    for (from, (_, to)) in translations.iter().zip(keyframes.iter_mut()) {
                        *to = *from;
                    }
                    if let gltf_json::AnimatedProperty::Translation = channel.target.path {
                        Keyframes::Translation(keyframes)
                    } else {
                        Keyframes::Scale(keyframes)
                    }
                }
                gltf_json::AnimatedProperty::Rotation => {
                    let mut keyframes = timestamps.iter().map(|f| (*f, Quat::IDENTITY)).collect::<Vec<(f32, Quat)>>();
                    let (rotations, ctype) = get_slice_and_component_type_from_accessor(
                        &mut memmap_holder,
                        &resource_path,
                        bin_buffer,
                        &gltf,
                        sampler.output,
                        None,
                        "VEC4",
                    )?;
                    let component_stride = stride_for(ctype, "SCALAR");
                    for (quat_bytes, (_, to)) in rotations.chunks_exact(component_stride * 4).zip(keyframes.iter_mut()) {
                        let x = parse_float(ctype, &quat_bytes[0..component_stride])?;
                        let y = parse_float(ctype, &quat_bytes[component_stride..component_stride * 2])?;
                        let z = parse_float(ctype, &quat_bytes[component_stride * 2..component_stride * 3])?;
                        let w = parse_float(ctype, &quat_bytes[component_stride * 3..component_stride * 4])?;
                        *to = Quat::from_xyzw(x, y, z, w);
                    }
                    Keyframes::Rotation(keyframes)
                }
                gltf_json::AnimatedProperty::Weights => {
                    let mut keyframes = timestamps.iter().map(|f| (*f, 0.0)).collect::<Vec<(f32, f32)>>();
                    let (weights, ctype) = get_slice_and_component_type_from_accessor(
                        &mut memmap_holder,
                        &resource_path,
                        bin_buffer,
                        &gltf,
                        sampler.output,
                        None,
                        "SCALAR",
                    )?;
                    let stride = stride_for(ctype, "SCALAR");
                    for (weight_bytes, (_, to)) in weights.chunks_exact(stride).zip(keyframes.iter_mut()) {
                        *to = parse_float(ctype, weight_bytes)?;
                    }
                    Keyframes::Weight(keyframes)
                }
            };
            channels_for_node.push(AnimationChannel { interpolation: sampler.interpolation, keyframes });
        }
        animations.push(Animation {
            name: animation.name.clone(),
            start_time: start_time.ok_or(GltfLoadingError::Spec("animation channel input accessor must have a min"))?,
            end_time: end_time.ok_or(GltfLoadingError::Spec("animation channel input accessor must have a max"))?,
            nodes_channels,
        });
    }

    let mut skins = Vec::new();
    for skin in &gltf.skins {
        profiling::scope!("reading skin (bones)");
        let mut joints = Vec::new();
        if let Some(inverse_bind_matrices) = skin.inverse_bind_matrices {
            let (inverse_bind_matrices, _) = get_slice_and_component_type_from_accessor(
                &mut memmap_holder,
                &resource_path,
                bin_buffer,
                &gltf,
                inverse_bind_matrices,
                GLTF_FLOAT,
                "MAT4",
            )?;
            if skin.joints.len() * mem::size_of::<Mat4>() != inverse_bind_matrices.len() {
                return Err(GltfLoadingError::Spec("skin has a different amount of joints and inverse bind matrices"));
            }
            // Re-allocation needed so that the [u8] is Mat4 aligned.
            // Don't know why the format doesn't enforce this.
            let inverse_bind_matrices = Vec::from(inverse_bind_matrices);
            let inverse_bind_matrices: &[Mat4] = bytemuck::cast_slice(&inverse_bind_matrices);
            for (&node_index, &inverse_bind_matrix) in skin.joints.iter().zip(inverse_bind_matrices) {
                joints.push(Joint { inverse_bind_matrix, node_index });
            }
        } else {
            for &node_index in &skin.joints {
                joints.push(Joint { inverse_bind_matrix: Mat4::IDENTITY, node_index });
            }
        }
        skins.push(Skin { joints });
    }

    for node in &nodes {
        if let Some(skin) = node.skin {
            if skin >= skins.len() {
                return Err(GltfLoadingError::Oob("node has an out-of-bounds skin index"));
            }
        }
    }

    // Check that the nodes form a proper node graph
    let mut visited_nodes = vec![false; nodes.len()];
    let mut queue = Vec::with_capacity(nodes.len());
    queue.extend_from_slice(&root_nodes);
    while let Some(node) = queue.pop() {
        if visited_nodes[node] {
            return Err(GltfLoadingError::InvalidNodeGraph);
        } else {
            visited_nodes[node] = true;
            for child in &nodes[node].children {
                queue.push(*child);
            }
        }
    }

    // Apply parent transforms to their children
    let mut parent_nodes = root_nodes.clone();
    while let Some(parent_node) = parent_nodes.pop() {
        let parent_node_transform = nodes[parent_node].transform;
        let mut children = nodes[parent_node].children.clone();
        for &child in &children {
            nodes[child].transform = parent_node_transform * nodes[child].transform;
        }
        parent_nodes.append(&mut children);
    }

    Ok(PendingGltf {
        gltf_base: Gltf { animations, nodes, root_nodes, skins, meshes: Vec::new(), materials: Vec::new() },
        json: gltf,
        bin_buffer,
        resource_path,
        image_texture_kinds,
        meshes,
        images,
    })
}

fn map_file<'a>(memmap_holder: &'a mut Option<Mmap>, path: &Path, range: Option<Range<usize>>) -> Result<&'a [u8], GltfLoadingError> {
    let file = File::open(path).map_err(|err| GltfLoadingError::OpenFile(err, path.to_owned()))?;
    let mut memmap_options = MmapOptions::new();
    if let Some(range) = range {
        memmap_options.offset(range.start as u64);
        memmap_options.len(range.count());
    }
    let memmap = unsafe { memmap_options.map(&file) }.map_err(|err| GltfLoadingError::MapFile(err, path.to_owned()))?;
    #[cfg(target_family = "unix")]
    {
        let _ = memmap.advise(memmap2::Advice::Sequential);
        let _ = memmap.advise(memmap2::Advice::WillNeed);
    }
    *memmap_holder = Some(memmap);
    Ok(memmap_holder.as_deref().unwrap())
}

#[profiling::function]
fn create_primitive(gltf: &gltf_json::GltfJson, primitive: &gltf_json::Primitive) -> Result<PrimitiveParameters, GltfLoadingError> {
    let index_accessor = primitive.indices.ok_or(GltfLoadingError::Misc("missing indices"))?;
    let (index_buffer, index_ctype) = get_buffer_view_from_accessor(gltf, index_accessor, None, "SCALAR")?;
    let large_indices = if index_ctype == GLTF_UNSIGNED_SHORT {
        false
    } else if index_ctype == GLTF_UNSIGNED_INT {
        true
    } else {
        return Err(GltfLoadingError::Spec("index ctype is not UNSIGNED_SHORT or UNSIGNED_INT"));
    };

    let mut vertex_buffers = VertexBindingMap::default();

    let pos_accessor = *primitive.attributes.get("POSITION").ok_or(GltfLoadingError::Misc("missing position attributes"))?;
    let (pos_buffer, _) = get_buffer_view_from_accessor(gltf, pos_accessor, Some(GLTF_FLOAT), "VEC3")?;
    vertex_buffers[VertexBinding::Position] = Some(pos_buffer);

    let tex_accessor = *primitive.attributes.get("TEXCOORD_0").ok_or(GltfLoadingError::Misc("missing UV0 attributes"))?;
    let (tex_buffer, _) = get_buffer_view_from_accessor(gltf, tex_accessor, Some(GLTF_FLOAT), "VEC2")?;
    vertex_buffers[VertexBinding::Texcoord0] = Some(tex_buffer);

    let normal_accessor = *primitive.attributes.get("NORMAL").ok_or(GltfLoadingError::Misc("missing normal attributes"))?;
    let (normal_buffer, _) = get_buffer_view_from_accessor(gltf, normal_accessor, Some(GLTF_FLOAT), "VEC3")?;
    vertex_buffers[VertexBinding::NormalOrColor] = Some(normal_buffer);

    let tangent_accessor = *primitive.attributes.get("TANGENT").ok_or(GltfLoadingError::Misc("missing tangent attributes"))?;
    let (tangent_buffer, _) = get_buffer_view_from_accessor(gltf, tangent_accessor, Some(GLTF_FLOAT), "VEC4")?;
    vertex_buffers[VertexBinding::Tangent] = Some(tangent_buffer);

    let mut pipeline = PipelineIndex::PbrOpaque;
    if let (Some(&joints_accessor), Some(&weights_accessor)) = (primitive.attributes.get("JOINTS_0"), primitive.attributes.get("WEIGHTS_0"))
    {
        let (joints_buffer, _) = get_buffer_view_from_accessor(gltf, joints_accessor, Some(GLTF_UNSIGNED_BYTE), "VEC4")?;
        vertex_buffers[VertexBinding::Joints0] = Some(joints_buffer);
        let (weights_buffer, _) = get_buffer_view_from_accessor(gltf, weights_accessor, Some(GLTF_FLOAT), "VEC4")?;
        vertex_buffers[VertexBinding::Weights0] = Some(weights_buffer);
        pipeline = PipelineIndex::PbrSkinnedOpaque;
    }

    Ok(PrimitiveParameters { pipeline, vertex_buffers, index_buffer, large_indices })
}

#[profiling::function]
fn get_slice_and_component_type_from_accessor<'buffer, C: Into<Option<i32>>>(
    memmap_holder: &'buffer mut Option<Mmap>,
    resource_path: &Path,
    bin_buffer: Option<&'buffer [u8]>,
    gltf: &gltf_json::GltfJson,
    accessor: usize,
    ctype: C,
    atype: &str,
) -> Result<(&'buffer [u8], i32), GltfLoadingError> {
    let ctype = ctype.into();
    let (view, ctype) = get_buffer_view_from_accessor(gltf, accessor, ctype, atype)?;
    let buffer = gltf.buffers.get(view.buffer).ok_or(GltfLoadingError::Oob("buffer"))?;
    if view.offset + view.length > buffer.byte_length {
        return Err(GltfLoadingError::Oob("buffer offset + length"));
    }
    let buffer = if let Some(uri) = &buffer.uri {
        let path = resource_path.join(uri);
        map_file(memmap_holder, &path, Some(view.offset..view.offset + view.length))?
    } else if let Some(bin_buffer) = bin_buffer.as_ref() {
        &bin_buffer[view.offset..view.offset + view.length]
    } else {
        return Err(GltfLoadingError::Misc("buffer has no uri but there's no glb buffer"));
    };
    Ok((buffer, ctype))
}

fn get_buffer_view_from_accessor(
    gltf: &gltf_json::GltfJson,
    accessor: usize,
    ctype: Option<i32>,
    atype: &str,
) -> Result<(BufferView, i32), GltfLoadingError> {
    let accessor = gltf.accessors.get(accessor).ok_or(GltfLoadingError::Oob("accessor"))?;
    let ctype = ctype.unwrap_or(accessor.component_type);
    if accessor.component_type != ctype {
        return Err(GltfLoadingError::Misc("unexpected component type"));
    }
    if accessor.attribute_type != atype {
        return Err(GltfLoadingError::Misc("unexpected attribute type"));
    }
    let view = match accessor.buffer_view {
        Some(view) => view,
        None => return Err(GltfLoadingError::Misc("no buffer view")),
    };
    let view = gltf.buffer_views.get(view).ok_or(GltfLoadingError::Oob("buffer view"))?;
    let offset = view.byte_offset + accessor.byte_offset;
    let length = view.byte_length;
    let stride = stride_for(ctype, atype);
    match view.byte_stride {
        Some(x) if x != stride => return Err(GltfLoadingError::Misc("wrong stride")),
        _ => {}
    }
    Ok((BufferView { buffer: view.buffer, offset, length: length.min(accessor.count * stride), stride }, ctype))
}

fn stride_for(component_type: i32, attribute_type: &str) -> usize {
    let bytes_per_component = match component_type {
        GLTF_UNSIGNED_INT | GLTF_FLOAT => 4,
        GLTF_SHORT | GLTF_UNSIGNED_SHORT => 2,
        GLTF_BYTE | GLTF_UNSIGNED_BYTE => 1,
        _ => unreachable!(),
    };
    let components = match attribute_type {
        "MAT4" => 16,
        "MAT3" => 9,
        "MAT2" => 4,
        "VEC4" => 4,
        "VEC3" => 3,
        "VEC2" => 2,
        "SCALAR" => 1,
        _ => unreachable!(),
    };
    bytes_per_component * components
}

pub(crate) fn get_gltf_texture_kinds(gltf: &gltf_json::GltfJson) -> Result<HashMap<usize, TextureKind>, GltfLoadingError> {
    profiling::scope!("image kind map creation");
    let mut image_texture_kinds = HashMap::new();
    for material in &gltf.materials {
        if let Some(pbr) = &material.pbr_metallic_roughness {
            if let Some(base_color) = &pbr.base_color_texture {
                let texture = gltf.textures.get(base_color.index).ok_or(GltfLoadingError::Oob("texture"))?;
                if let Some(image_index) = texture.source {
                    image_texture_kinds.insert(image_index, TextureKind::SrgbColor);
                }
            }
            if let Some(metallic_roughness) = &pbr.metallic_roughness_texture {
                let texture = gltf.textures.get(metallic_roughness.index).ok_or(GltfLoadingError::Oob("texture"))?;
                if let Some(image_index) = texture.source {
                    image_texture_kinds.insert(image_index, TextureKind::LinearColor);
                }
            }
        }
        if let Some(normal) = &material.normal_texture {
            let texture = gltf.textures.get(normal.index).ok_or(GltfLoadingError::Oob("texture"))?;
            if let Some(image_index) = texture.source {
                image_texture_kinds.insert(image_index, TextureKind::NormalMap);
            }
        }
        if let Some(emissive) = &material.emissive_texture {
            let texture = gltf.textures.get(emissive.index).ok_or(GltfLoadingError::Oob("texture"))?;
            if let Some(image_index) = texture.source {
                image_texture_kinds.insert(image_index, TextureKind::SrgbColor);
            }
        }
        if let Some(occlusion) = &material.occlusion_texture {
            let texture = gltf.textures.get(occlusion.index).ok_or(GltfLoadingError::Oob("texture"))?;
            if let Some(image_index) = texture.source {
                image_texture_kinds.insert(image_index, TextureKind::LinearColor);
            }
        }
    }
    Ok(image_texture_kinds)
}

fn parse_float(ctype: i32, x: &[u8]) -> Result<f32, GltfLoadingError> {
    let f = match ctype {
        GLTF_BYTE => (x[0] as f32 / 127.0).max(-1.0),
        GLTF_UNSIGNED_BYTE => x[0] as f32 / 255.0,
        GLTF_SHORT => (u16::from_le_bytes([x[0], x[1]]) as f32 / 32767.0).max(-1.0),
        GLTF_UNSIGNED_SHORT => u16::from_le_bytes([x[0], x[1]]) as f32 / 65535.0,
        GLTF_FLOAT => f32::from_le_bytes([x[0], x[1], x[2], x[3]]),
        _ => return Err(GltfLoadingError::Spec("component type of accessor can't be recognized")),
    };
    Ok(f)
}
