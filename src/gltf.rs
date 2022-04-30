use crate::arena::{VulkanArena, VulkanArenaError};
use crate::descriptors::{DescriptorError, GltfFactors, PipelineSpecificData};
use crate::gltf::gltf_json::AnimationInterpolation;
use crate::image_loading::{self, ImageLoadingError, TextureKind};
use crate::mesh::Mesh;
use crate::vk;
use crate::vulkan_raii::{Buffer, Device, ImageView};
use crate::{Descriptors, ForBuffers, ForImages, Material, PipelineIndex, Uploader};
use glam::{Mat4, Quat, Vec3, Vec4};
use memmap2::{Advice, Mmap, MmapOptions};
use std::collections::HashMap;
use std::fs::{self, File};
use std::ops::Range;
use std::path::Path;
use std::rc::Rc;

pub(crate) mod gltf_json;
mod mesh_iter;
pub use mesh_iter::MeshIter;

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
    InvalidGlbJson(#[source] std::str::Utf8Error),
    #[error("glb json chunk missing")]
    MissingGlbJson,
    #[error("failed to deserialize gltf json")]
    JsonDeserialization(#[source] serde_json::Error),
    #[error("unsupported gltf minimum version ({0}), 2.0 is supported")]
    UnsupportedGltfVersion(String),
    #[error("gltf has buffer without an uri but no glb BIN buffer")]
    GlbBinMissing,
    #[error("given gltf/glb file cannot be read")]
    MissingFile(#[source] std::io::Error),
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
    #[error("failed to load image {1}")]
    ImageLoading(#[source] ImageLoadingError, String),
    #[error("failed to create material")]
    MaterialCreation(#[source] DescriptorError),
    #[error("gltf node has multiple parents, which is not allowed by the 2.0 spec")]
    InvalidNodeGraph,
    #[error("gltf has an out-of-bounds index ({0})")]
    Oob(&'static str),
    #[error("gltf does not conform to the 2.0 spec: {0}")]
    Spec(&'static str),
    #[error("unimplemented gltf feature: {0}")]
    Misc(&'static str),
}

type NodeAnimChannels = Vec<AnimationChannel>;
pub struct Animation {
    pub name: Option<String>,
    nodes_channels: Vec<Option<NodeAnimChannels>>,
}

#[derive(Clone)]
struct AnimationChannel {
    interpolation: AnimationInterpolation,
    keyframes: Keyframes,
}

#[derive(Clone)]
enum Keyframes {
    Translation(Vec<(f32, Vec3)>),
    Rotation(Vec<(f32, Quat)>),
    Scale(Vec<(f32, Vec3)>),
}

struct Node {
    mesh: Option<usize>,
    children: Option<Vec<usize>>,
    transform: Mat4,
}

pub struct Gltf {
    pub animations: Vec<Animation>,
    nodes: Vec<Node>,
    root_nodes: Vec<usize>,
    meshes: Vec<Vec<(Mesh, usize)>>,
    materials: Vec<Rc<Material>>,
}

impl Gltf {
    /// Loads the glTF scene from a .glb file.
    ///
    /// Any external files referenced in the glTF are searched relative to
    /// `resource_path`.
    #[profiling::function]
    pub fn from_glb(
        device: &Device,
        uploader: &mut Uploader,
        descriptors: &mut Descriptors,
        arenas: (&mut VulkanArena<ForBuffers>, &mut VulkanArena<ForImages>),
        glb_path: &Path,
        resource_path: &Path,
    ) -> Result<Gltf, GltfLoadingError> {
        let glb = fs::read(glb_path).map_err(GltfLoadingError::MissingFile)?;
        let (json, buffer) = read_glb_json_and_buffer(&glb)?;
        let gltf: gltf_json::GltfJson = serde_json::from_str(json).map_err(GltfLoadingError::JsonDeserialization)?;
        create_gltf(device, uploader, descriptors, arenas, gltf, (glb_path, resource_path), Some(buffer))
    }

    /// Loads the glTF scene from a .gltf file.
    ///
    /// Any external files referenced in the glTF are searched relative to
    /// `resource_path`.
    #[profiling::function]
    pub fn from_gltf(
        device: &Device,
        uploader: &mut Uploader,
        descriptors: &mut Descriptors,
        arenas: (&mut VulkanArena<ForBuffers>, &mut VulkanArena<ForImages>),
        gltf_path: &Path,
        resource_path: &Path,
    ) -> Result<Gltf, GltfLoadingError> {
        let gltf = fs::read_to_string(gltf_path).map_err(GltfLoadingError::MissingFile)?;
        let gltf: gltf_json::GltfJson = serde_json::from_str(&gltf).map_err(GltfLoadingError::JsonDeserialization)?;
        create_gltf(device, uploader, descriptors, arenas, gltf, (gltf_path, resource_path), None)
    }

    pub fn mesh_iter(&self) -> MeshIter<'_> {
        MeshIter::new(self, self.root_nodes.clone())
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
                json = Some(std::str::from_utf8(chunk_bytes).map_err(GltfLoadingError::InvalidGlbJson)?);
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
fn create_gltf(
    device: &Device,
    uploader: &mut Uploader,
    descriptors: &mut Descriptors,
    (buffer_arena, image_arena): (&mut VulkanArena<ForBuffers>, &mut VulkanArena<ForImages>),
    gltf: gltf_json::GltfJson,
    (gltf_path, resource_path): (&Path, &Path),
    bin_buffer: Option<&[u8]>,
) -> Result<Gltf, GltfLoadingError> {
    if let Some(min_version) = &gltf.asset.min_version {
        let min_version_f32 = str::parse::<f32>(min_version);
        if min_version_f32 != Ok(2.0) {
            return Err(GltfLoadingError::UnsupportedGltfVersion(min_version.clone()));
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

    let mut buffers = Vec::with_capacity(gltf.buffers.len());
    for buffer in &gltf.buffers {
        let buffer_create_info = get_mesh_buffer_create_info(buffer.byte_length as vk::DeviceSize);
        if let Some(uri) = buffer.uri.as_ref() {
            let path = resource_path.join(uri);
            let data = map_file(&mut memmap_holder, &path, None)?;
            let data = &data[0..buffer.byte_length];
            let buffer = buffer_arena
                .create_buffer(
                    buffer_create_info,
                    data,
                    Some(uploader),
                    format_args!("{} ({})", uri, gltf_path.display()),
                )
                .map_err(|err| GltfLoadingError::BufferCreationFromFile(err, path))?;
            buffers.push(Rc::new(buffer));
        } else {
            match bin_buffer {
                Some(data) => {
                    let data = &data[0..buffer.byte_length];
                    let buffer = buffer_arena
                        .create_buffer(
                            buffer_create_info,
                            data,
                            Some(uploader),
                            format_args!("glb buffer ({})", gltf_path.display()),
                        )
                        .map_err(GltfLoadingError::BufferCreationFromGlb)?;
                    buffers.push(Rc::new(buffer));
                }
                None => return Err(GltfLoadingError::GlbBinMissing),
            }
        }
    }

    let mut meshes = Vec::with_capacity(gltf.meshes.len());
    for mesh in &gltf.meshes {
        let mut primitives = Vec::with_capacity(mesh.primitives.len());
        for primitive in &mesh.primitives {
            let mesh = create_primitive(&gltf, &buffers, primitive)?;
            let material_index = primitive.material.ok_or(GltfLoadingError::Misc("material missing"))?;
            primitives.push((mesh, material_index));
        }
        meshes.push(primitives);
    }

    let mut nodes = Vec::with_capacity(gltf.nodes.len());
    for node in &gltf.nodes {
        let transform = if let Some(cols_array) = node.matrix {
            Mat4::from_cols_array(&cols_array)
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
            Mat4::from_scale_rotation_translation(scale, rotation, translation)
        };
        nodes.push(Node {
            mesh: node.mesh,
            children: node.children.clone(),
            transform,
        });
    }

    let image_texture_kinds = get_gltf_texture_kinds(&gltf)?;

    let mut images = Vec::with_capacity(gltf.images.len());
    for (i, image) in gltf.images.iter().enumerate() {
        let bytes = load_image_bytes(&mut memmap_holder, resource_path, bin_buffer, image, &gltf)?;
        let kind = image_texture_kinds.get(&i).copied().unwrap_or(TextureKind::LinearColor);
        let name = image.uri.as_deref().unwrap_or("glb binary buffer");
        images.push(Rc::new(
            image_loading::load_ntex(device, uploader, image_arena, bytes, kind, name)
                .map_err(|err| GltfLoadingError::ImageLoading(err, name.to_string()))?,
        ));
    }

    let material_factors = get_material_factors(&gltf)?;
    let factors_slice = bytemuck::cast_slice(&material_factors);
    let buffer_create_info = get_material_factors_buffer_create_info(factors_slice.len() as vk::DeviceSize);
    let factors_buffer = buffer_arena
        .create_buffer(
            buffer_create_info,
            factors_slice,
            Some(uploader),
            format_args!("material parameters ({})", gltf_path.display()),
        )
        .map_err(GltfLoadingError::BufferCreationFromMaterialParameters)?;
    let factors_buffer = Rc::new(factors_buffer);

    let mut materials = Vec::with_capacity(gltf.materials.len());
    for (i, mat) in gltf.materials.iter().enumerate() {
        let mktex = |images: &[Rc<ImageView>], texture_info: &gltf_json::TextureInfo| {
            if texture_info.texcoord.is_some() && texture_info.texcoord != Some(0) {
                return Some(Err(GltfLoadingError::Misc("non-0 texCoord used for texture")));
            }
            let texture = match gltf.textures.get(texture_info.index) {
                Some(tex) => tex,
                None => return Some(Err(GltfLoadingError::Oob("texture"))),
            };
            let image_index = texture.source?;
            let image_view = match images.get(image_index) {
                Some(image_view) => image_view,
                None => return Some(Err(GltfLoadingError::Oob("image"))),
            };
            Some(Ok(image_view.clone()))
        };

        macro_rules! handle_optional_result {
            ($expression:expr) => {
                match $expression {
                    Some(Ok(ok)) => Some(ok),
                    Some(Err(err)) => return Err(err),
                    None => None,
                }
            };
        }

        let pbr = mat.pbr_metallic_roughness.as_ref().ok_or(GltfLoadingError::Misc("pbr missing"))?;
        let base_color = handle_optional_result!(pbr.base_color_texture.as_ref().and_then(|tex| mktex(&images, tex)));
        let metallic_roughness = handle_optional_result!(pbr.metallic_roughness_texture.as_ref().and_then(|tex| mktex(&images, tex)));
        let normal = handle_optional_result!(mat.normal_texture.as_ref().and_then(|tex| mktex(&images, tex)));
        let occlusion = handle_optional_result!(mat.occlusion_texture.as_ref().and_then(|tex| mktex(&images, tex)));
        let emissive = handle_optional_result!(mat.emissive_texture.as_ref().and_then(|tex| mktex(&images, tex)));

        let factors_size = std::mem::size_of::<GltfFactors>() as u64;
        let factors = (factors_buffer.clone(), factors_size * i as u64, factors_size);

        let pipeline_specific_data = PipelineSpecificData::Gltf {
            base_color,
            metallic_roughness,
            normal,
            occlusion,
            emissive,
            factors,
        };
        let pipeline = match mat.alpha_mode {
            gltf_json::AlphaMode::Opaque => PipelineIndex::Opaque,
            gltf_json::AlphaMode::Mask => PipelineIndex::Clipped,
            gltf_json::AlphaMode::Blend => PipelineIndex::Blended,
        };
        let name = mat.name.clone().unwrap_or_else(|| String::from("unnamed material"));
        materials.push(Material::new(descriptors, pipeline, pipeline_specific_data, name).map_err(GltfLoadingError::MaterialCreation)?);
    }

    // TODO: Animations:
    // TODO: Sending the inverse bind matrices to the GPU
    // TODO: Sending the transforms of the nodes in the joints-array of skins, animated with the current Animation, to the GPU
    // The above stuff could probably all be batched into one array of buffers,
    // or simply a continuous buffer and referring to the sub parts with UBO
    // offsets. Not sure, but all the animation data should definitely be
    // written all at once.
    // TODO: Test by pretending that every vertex has max weight for joint 0 and zero for the rest.
    // TODO: Sending joints and weights to the GPU (should be easy, just more attributes)
    // The hard part however, is deciding how to deal with some meshes having them and some not.

    let mut animations = Vec::with_capacity(gltf.animations.len());
    for animation in &gltf.animations {
        let mut nodes_channels = vec![None; gltf.nodes.len()];
        for channel in &animation.channels {
            let sampler = animation
                .samplers
                .get(channel.sampler)
                .ok_or(GltfLoadingError::Oob("animation samplers"))?;
            let channels_for_node = nodes_channels
                .get_mut(channel.target.node)
                .ok_or(GltfLoadingError::Oob("animation target node"))?
                .get_or_insert_with(Vec::new);
            let (timestamps, _) = get_slice_and_component_type_from_accessor(
                &mut memmap_holder,
                resource_path,
                bin_buffer,
                &gltf,
                sampler.input,
                GLTF_FLOAT,
                "SCALAR",
            )?;
            let timestamps: &[f32] = bytemuck::cast_slice(timestamps);

            let keyframes = match channel.target.path {
                gltf_json::AnimatedProperty::Translation | gltf_json::AnimatedProperty::Scale => {
                    let mut keyframes = timestamps.iter().map(|f| (*f, Vec3::ZERO)).collect::<Vec<(f32, Vec3)>>();
                    let (translations, _) = get_slice_and_component_type_from_accessor(
                        &mut memmap_holder,
                        resource_path,
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
                        resource_path,
                        bin_buffer,
                        &gltf,
                        sampler.output,
                        None,
                        "VEC4",
                    )?;
                    let component_stride = stride_for(ctype, "SCALAR");
                    for (quat_bytes, (_, to)) in rotations.chunks_exact(component_stride * 4).zip(keyframes.iter_mut()) {
                        let x = &quat_bytes[0..component_stride];
                        let y = &quat_bytes[component_stride..component_stride * 2];
                        let z = &quat_bytes[component_stride * 2..component_stride * 3];
                        let w = &quat_bytes[component_stride * 3..component_stride * 4];
                        *to = match ctype {
                            GLTF_BYTE => Quat::from_xyzw(
                                (x[0] as f32 / 127.0).max(-1.0),
                                (y[0] as f32 / 127.0).max(-1.0),
                                (z[0] as f32 / 127.0).max(-1.0),
                                (w[0] as f32 / 127.0).max(-1.0),
                            ),
                            GLTF_UNSIGNED_BYTE => {
                                Quat::from_xyzw(x[0] as f32 / 255.0, y[0] as f32 / 255.0, z[0] as f32 / 255.0, w[0] as f32 / 255.0)
                            }
                            GLTF_SHORT => Quat::from_xyzw(
                                (u16::from_le_bytes([x[0], x[1]]) as f32 / 32767.0).max(-1.0),
                                (u16::from_le_bytes([y[0], y[1]]) as f32 / 32767.0).max(-1.0),
                                (u16::from_le_bytes([z[0], z[1]]) as f32 / 32767.0).max(-1.0),
                                (u16::from_le_bytes([w[0], w[1]]) as f32 / 32767.0).max(-1.0),
                            ),
                            GLTF_UNSIGNED_SHORT => Quat::from_xyzw(
                                u16::from_le_bytes([x[0], x[1]]) as f32 / 65535.0,
                                u16::from_le_bytes([y[0], y[1]]) as f32 / 65535.0,
                                u16::from_le_bytes([z[0], z[1]]) as f32 / 65535.0,
                                u16::from_le_bytes([w[0], w[1]]) as f32 / 65535.0,
                            ),
                            GLTF_FLOAT => Quat::from_xyzw(
                                f32::from_le_bytes([x[0], x[1], x[2], x[3]]),
                                f32::from_le_bytes([y[0], y[1], y[2], y[3]]),
                                f32::from_le_bytes([z[0], z[1], z[2], z[3]]),
                                f32::from_le_bytes([w[0], w[1], w[2], w[3]]),
                            ),
                            _ => return Err(GltfLoadingError::Spec("component type of accessor can't be recognized")),
                        };
                    }
                    Keyframes::Rotation(keyframes)
                }
            };
            channels_for_node.push(AnimationChannel {
                interpolation: sampler.interpolation,
                keyframes,
            });
        }
        animations.push(Animation {
            name: animation.name.clone(),
            nodes_channels,
        });
    }

    let mut visited_nodes = vec![false; nodes.len()];
    let mut queue = Vec::with_capacity(nodes.len());
    queue.extend_from_slice(&root_nodes);
    while let Some(node) = queue.pop() {
        if visited_nodes[node] {
            return Err(GltfLoadingError::InvalidNodeGraph);
        } else {
            visited_nodes[node] = true;
            if let Some(children) = &nodes[node].children {
                for child in children {
                    queue.push(*child);
                }
            }
        }
    }

    Ok(Gltf {
        animations,
        nodes,
        root_nodes,
        materials,
        meshes,
    })
}

fn map_file<'a>(memmap_holder: &'a mut Option<Mmap>, path: &Path, range: Option<Range<usize>>) -> Result<&'a [u8], GltfLoadingError> {
    let file = File::open(&path).map_err(|err| GltfLoadingError::OpenFile(err, path.to_owned()))?;
    let mut memmap_options = MmapOptions::new();
    if let Some(range) = range {
        memmap_options.offset(range.start as u64);
        memmap_options.len(range.count());
    }
    let memmap = unsafe { memmap_options.map(&file) }.map_err(|err| GltfLoadingError::MapFile(err, path.to_owned()))?;
    let _ = memmap.advise(Advice::Sequential);
    let _ = memmap.advise(Advice::WillNeed);
    *memmap_holder = Some(memmap);
    Ok(memmap_holder.as_deref().unwrap())
}

pub(crate) fn load_image_bytes<'a>(
    memmap_holder: &'a mut Option<Mmap>,
    resource_path: &Path,
    bin_buffer: Option<&'a [u8]>,
    image: &gltf_json::Image,
    gltf: &gltf_json::GltfJson,
) -> Result<&'a [u8], GltfLoadingError> {
    profiling::scope!("map image file into memory");
    if let (Some(mime_type), Some(buffer_view)) = (&image.mime_type, &image.buffer_view) {
        if mime_type.as_str() != "image/prs.ntex" {
            return Err(GltfLoadingError::Spec("mime type of texture is not image/prs.ntex"));
        }
        let buffer_view = gltf
            .buffer_views
            .get(*buffer_view)
            .ok_or(GltfLoadingError::Oob("texture buffer view"))?;
        let buffer = gltf
            .buffers
            .get(buffer_view.buffer)
            .ok_or(GltfLoadingError::Oob("texture buffer"))?;
        let buffer_offset = buffer_view.byte_offset.unwrap_or(0);
        let buffer_size = buffer_view.byte_length;
        let buffer_bytes = if let Some(uri) = buffer.uri.as_ref() {
            let path = resource_path.join(uri);
            map_file(memmap_holder, &path, Some(buffer_offset..buffer_offset + buffer_size))?
        } else {
            bin_buffer.ok_or(GltfLoadingError::GlbBinMissing)?
        };
        if buffer_offset + buffer_size >= buffer_bytes.len() {
            return Err(GltfLoadingError::Oob("texture buffer view bytes"));
        }
        Ok(&buffer_bytes[buffer_offset..buffer_offset + buffer_size])
    } else if let Some(uri) = &image.uri {
        if !uri.ends_with(".ntex") {
            return Err(GltfLoadingError::Misc("image uri does not end in .ntex"));
        };
        let path = resource_path.join(uri);
        map_file(memmap_holder, &path, None)
    } else {
        Err(GltfLoadingError::Spec("image does not have an uri nor a mimetype + buffer view"))
    }
}

#[profiling::function]
fn create_primitive(
    gltf: &gltf_json::GltfJson,
    buffers: &[Rc<Buffer>],
    primitive: &gltf_json::Primitive,
) -> Result<Mesh, GltfLoadingError> {
    let pipeline = if let Some(material_index) = primitive.material {
        let material = gltf.materials.get(material_index).ok_or(GltfLoadingError::Oob("material"))?;
        match material.alpha_mode {
            gltf_json::AlphaMode::Opaque => PipelineIndex::Opaque,
            gltf_json::AlphaMode::Mask => PipelineIndex::Clipped,
            gltf_json::AlphaMode::Blend => PipelineIndex::Blended,
        }
    } else {
        PipelineIndex::Opaque
    };

    let index_accessor = primitive.indices.ok_or(GltfLoadingError::Misc("missing indices"))?;
    let (index_buffer, index_buffer_offset, index_buffer_size) =
        get_buffer_from_accessor(buffers, gltf, index_accessor, GLTF_UNSIGNED_SHORT, "SCALAR")?;

    let pos_accessor = *primitive
        .attributes
        .get("POSITION")
        .ok_or(GltfLoadingError::Misc("missing position attributes"))?;
    let (pos_buffer, pos_offset, _) = get_buffer_from_accessor(buffers, gltf, pos_accessor, GLTF_FLOAT, "VEC3")?;

    let tex_accessor = *primitive
        .attributes
        .get("TEXCOORD_0")
        .ok_or(GltfLoadingError::Misc("missing UV0 attributes"))?;
    let (tex_buffer, tex_offset, _) = get_buffer_from_accessor(buffers, gltf, tex_accessor, GLTF_FLOAT, "VEC2")?;

    let normal_accessor = *primitive
        .attributes
        .get("NORMAL")
        .ok_or(GltfLoadingError::Misc("missing normal attributes"))?;
    let (normal_buffer, normal_offset, _) = get_buffer_from_accessor(buffers, gltf, normal_accessor, GLTF_FLOAT, "VEC3")?;

    let tangent_accessor = *primitive
        .attributes
        .get("TANGENT")
        .ok_or(GltfLoadingError::Misc("missing tangent attributes"))?;
    let (tangent_buffer, tangent_offset, _) = get_buffer_from_accessor(buffers, gltf, tangent_accessor, GLTF_FLOAT, "VEC4")?;

    Ok(Mesh::new::<u16>(
        pipeline,
        vec![pos_buffer, tex_buffer, normal_buffer, tangent_buffer],
        vec![pos_offset, tex_offset, normal_offset, tangent_offset],
        index_buffer,
        index_buffer_offset,
        index_buffer_size,
    ))
}

#[profiling::function]
fn get_buffer_from_accessor<'buffer>(
    buffers: &'buffer [Rc<Buffer>],
    gltf: &gltf_json::GltfJson,
    accessor: usize,
    ctype: i32,
    atype: &str,
) -> Result<(Rc<Buffer>, vk::DeviceSize, vk::DeviceSize), GltfLoadingError> {
    let (buffer, offset, length, _) = get_buffer_view_from_accessor(gltf, accessor, Some(ctype), atype)?;
    let buffer = buffers.get(buffer).ok_or(GltfLoadingError::Oob("buffer"))?;
    if (offset + length) as vk::DeviceSize > buffer.size {
        return Err(GltfLoadingError::Oob("buffer offset + length"));
    }
    Ok((buffer.clone(), offset as vk::DeviceSize, length as vk::DeviceSize))
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
    let (buffer, offset, length, ctype) = get_buffer_view_from_accessor(gltf, accessor, ctype, atype)?;
    let buffer = gltf.buffers.get(buffer).ok_or(GltfLoadingError::Oob("buffer"))?;
    if offset + length > buffer.byte_length {
        return Err(GltfLoadingError::Oob("buffer offset + length"));
    }
    let buffer = if let Some(uri) = &buffer.uri {
        let path = resource_path.join(uri);
        map_file(memmap_holder, &path, Some(offset..offset + length))?
    } else if let Some(bin_buffer) = bin_buffer.as_ref() {
        &bin_buffer[offset..offset + length]
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
) -> Result<(usize, usize, usize, i32), GltfLoadingError> {
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
    let offset = view.byte_offset.unwrap_or(0) + accessor.byte_offset.unwrap_or(0);
    let length = view.byte_length;
    let stride = stride_for(ctype, atype);
    match view.byte_stride {
        Some(x) if x != stride => return Err(GltfLoadingError::Misc("wrong stride")),
        _ => {}
    }
    if accessor.count != length / stride {
        return Err(GltfLoadingError::Oob("count != byte length / stride"));
    }
    Ok((view.buffer, offset, length, ctype))
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

pub(crate) fn get_mesh_buffer_create_info(size: vk::DeviceSize) -> vk::BufferCreateInfo {
    vk::BufferCreateInfo::builder()
        .size(size)
        .usage(
            vk::BufferUsageFlags::TRANSFER_DST
                | vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::UNIFORM_BUFFER,
        )
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .build()
}

pub(crate) fn get_material_factors_buffer_create_info(size: vk::DeviceSize) -> vk::BufferCreateInfo {
    vk::BufferCreateInfo::builder()
        .size(size)
        .usage(vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .build()
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
                let texture = gltf
                    .textures
                    .get(metallic_roughness.index)
                    .ok_or(GltfLoadingError::Oob("texture"))?;
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

pub(crate) fn get_material_factors(gltf: &gltf_json::GltfJson) -> Result<Vec<GltfFactors>, GltfLoadingError> {
    let mut material_factors = Vec::with_capacity(gltf.materials.len());
    for mat in &gltf.materials {
        let pbr = mat.pbr_metallic_roughness.as_ref().ok_or(GltfLoadingError::Misc("pbr missing"))?;
        let metallic_factor = pbr.metallic_factor.unwrap_or(1.0);
        let roughness_factor = pbr.roughness_factor.unwrap_or(1.0);
        let alpha_cutoff = if mat.alpha_mode == gltf_json::AlphaMode::Mask {
            mat.alpha_cutoff.unwrap_or(0.5)
        } else {
            0.0
        };
        material_factors.push(GltfFactors {
            base_color: pbr
                .base_color_factor
                .as_ref()
                .map(|&[r, g, b, a]| Vec4::new(r, g, b, a))
                .unwrap_or(Vec4::ONE),
            emissive: mat
                .emissive_factor
                .as_ref()
                .map(|&[r, g, b]| Vec4::new(r, g, b, 0.0))
                .unwrap_or(Vec4::ZERO),
            metallic_roughness_alpha_cutoff: Vec4::new(metallic_factor, roughness_factor, alpha_cutoff, 0.0),
        });
    }
    Ok(material_factors)
}
