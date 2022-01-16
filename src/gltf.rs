use crate::arena::{ImageAllocation, ImageView, VulkanArena};
use crate::image_loading::{self, TextureKind};
use crate::mesh::Mesh;
use crate::{Error, FrameIndex, Gpu, Material, Pipeline};
use glam::{Mat4, Quat, Vec3};
use std::borrow::Cow;
use std::collections::HashMap;
use std::rc::Rc;

mod gltf_json;
mod mesh_iter;
mod resources;
pub use mesh_iter::MeshIter;
pub use resources::GltfResources;

const GLTF_BYTE: i32 = 5120;
const GLTF_UNSIGNED_BYTE: i32 = 5121;
const GLTF_SHORT: i32 = 5122;
const GLTF_UNSIGNED_SHORT: i32 = 5123;
const GLTF_UNSIGNED_INT: i32 = 5125;
const GLTF_FLOAT: i32 = 5126;

struct Node {
    mesh: Option<usize>,
    children: Option<Vec<usize>>,
    transform: Mat4,
}

pub struct Gltf<'arena> {
    nodes: Vec<Node>,
    root_nodes: Vec<usize>,
    meshes: Vec<Vec<(Mesh<'arena>, usize)>>,
    materials: Vec<Material<'arena>>,
}

impl Gltf<'_> {
    /// Loads the glTF scene from the contents of a .glb file.
    ///
    /// Any external files referenced in the glTF are searched
    /// relative to `directory`.
    #[profiling::function]
    pub fn from_glb<'a>(
        gpu: &'a Gpu,
        main_arena: &'a VulkanArena,
        temp_arenas: &[VulkanArena],
        frame_index: FrameIndex,
        glb: &[u8],
        resources: &mut GltfResources,
    ) -> Result<Gltf<'a>, Error> {
        fn read_u32(bytes: &[u8]) -> u32 {
            debug_assert!(bytes.len() == 4);
            if let [a, b, c, d] = *bytes {
                u32::from_le_bytes([a, b, c, d])
            } else {
                unreachable!();
            }
        }

        const MAGIC_GLTF: u32 = 0x46546C67;
        if glb.len() < 12 || read_u32(&glb[0..4]) != MAGIC_GLTF {
            return Err(Error::InvalidGlbHeader);
        }
        let version = read_u32(&glb[4..8]);
        let length = read_u32(&glb[8..12]) as usize;
        if version != 2 {
            log::warn!(".glb file is not version 2, but trying to read anyway");
        }
        if length != glb.len() {
            return Err(Error::InvalidGlbLength);
        }

        let mut next_chunk = &glb[12..];
        let mut json: Option<&str> = None;
        let mut buffer: Option<&[u8]> = None;
        while next_chunk.len() >= 8 {
            let chunk_length = read_u32(&next_chunk[0..4]) as usize;
            if chunk_length > next_chunk.len() - 8 {
                return Err(Error::InvalidGlbChunkLength);
            }
            let chunk_bytes = &next_chunk[8..chunk_length + 8];

            const MAGIC_JSON: u32 = 0x4E4F534A;
            const MAGIC_BIN: u32 = 0x004E4942;
            let chunk_type = read_u32(&next_chunk[4..8]);
            match chunk_type {
                MAGIC_JSON => {
                    if json.is_some() {
                        return Err(Error::TooManyGlbJsonChunks);
                    }
                    json = Some(std::str::from_utf8(chunk_bytes).map_err(Error::InvalidGlbJson)?);
                }
                MAGIC_BIN => {
                    if buffer.is_some() {
                        return Err(Error::TooManyGlbBinaryChunks);
                    }
                    buffer = Some(chunk_bytes);
                }
                _ => return Err(Error::InvalidGlbChunkType),
            }

            next_chunk = &next_chunk[chunk_length + 8..];
        }

        let buffer = buffer.ok_or(Error::GltfMisc("glb buffer is required"))?;

        let json = json.ok_or(Error::MissingGlbJson)?;
        let gltf: gltf_json::GltfJson = miniserde::json::from_str(json).map_err(Error::GltfJsonDeserialization)?;
        create_gltf(gpu, main_arena, temp_arenas, frame_index, gltf, resources, Some(buffer))
    }

    /// Loads the glTF scene from the contents of a .gltf file.
    ///
    /// Any external files referenced in the glTF are searched
    /// relative to `directory`.
    #[profiling::function]
    pub fn from_gltf<'a>(
        gpu: &'a Gpu,
        main_arena: &'a VulkanArena,
        temp_arenas: &[VulkanArena],
        frame_index: FrameIndex,
        gltf: &str,
        resources: &mut GltfResources,
    ) -> Result<Gltf<'a>, Error> {
        let gltf: gltf_json::GltfJson = miniserde::json::from_str(gltf).map_err(Error::GltfJsonDeserialization)?;
        create_gltf(gpu, main_arena, temp_arenas, frame_index, gltf, resources, None)
    }

    pub fn mesh_iter(&self) -> MeshIter<'_> {
        MeshIter::new(self, self.root_nodes.clone())
    }
}

#[profiling::function]
fn create_gltf<'a>(
    gpu: &'a Gpu,
    arena: &'a VulkanArena,
    temp_arenas: &[VulkanArena],
    frame_index: FrameIndex,
    gltf: gltf_json::GltfJson,
    resources: &mut GltfResources,
    bin_buffer: Option<&[u8]>,
) -> Result<Gltf<'a>, Error> {
    if let Some(min_version) = &gltf.asset.min_version {
        let min_version_f32 = str::parse::<f32>(min_version);
        if min_version_f32 != Ok(2.0) {
            return Err(Error::UnsupportedGltfVersion(min_version.clone()));
        }
    } else if let Ok(version) = str::parse::<f32>(&gltf.asset.version) {
        if !(2.0..3.0).contains(&version) {
            return Err(Error::UnsupportedGltfVersion(gltf.asset.version));
        }
    } else {
        log::warn!("Could not parse glTF version {}, assuming 2.0.", gltf.asset.version);
    }

    let scene_index = gltf.scene.ok_or(Error::GltfMisc("gltf does not have a scene"))?;
    let scenes = gltf.scenes.as_ref().ok_or(Error::GltfMisc("scenes missing"))?;
    let scene = scenes.get(scene_index).ok_or(Error::GltfOob("scene"))?;
    let root_nodes = scene.nodes.clone().ok_or(Error::GltfMisc("no nodes in scene"))?;

    let mut buffers = Vec::with_capacity(gltf.buffers.len());
    for buffer in &gltf.buffers {
        if let Some(uri) = buffer.uri.as_ref() {
            buffers.push(resources.get_or_load(uri)?);
        } else {
            match bin_buffer {
                Some(bin_buffer) => buffers.push(Rc::new(Cow::Borrowed(bin_buffer))),
                None => return Err(Error::GlbBinMissing),
            }
        }
    }

    let meshes = {
        profiling::scope!("meshes");
        gltf.meshes
            .iter()
            .map(|mesh| {
                mesh.primitives
                    .iter()
                    .map(|primitive| {
                        let mesh = create_primitive(gpu, arena, temp_arenas, frame_index, &gltf, &buffers, primitive)?;
                        let material_index = primitive.material.ok_or(Error::GltfMisc("material missing"))?;
                        Ok((mesh, material_index))
                    })
                    .collect::<Result<Vec<(Mesh, usize)>, Error>>()
            })
            .collect::<Result<Vec<Vec<(Mesh, usize)>>, Error>>()?
    };

    let nodes = {
        profiling::scope!("nodes");
        gltf.nodes
            .iter()
            .map(|node| {
                let transform = if let Some(&[x0, y0, z0, w0, x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3]) = node.matrix.as_deref() {
                    Mat4::from_cols_array(&[x0, y0, z0, w0, x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3])
                } else {
                    let translation = match node.translation.as_deref() {
                        Some(&[x, y, z]) => Vec3::new(x, y, z),
                        _ => Vec3::ZERO,
                    };
                    let rotation = match node.rotation.as_deref() {
                        Some(&[x, y, z, w]) => Quat::from_xyzw(x, y, z, w),
                        _ => Quat::IDENTITY,
                    };
                    let scale = match node.scale.as_deref() {
                        Some(&[x, y, z]) => Vec3::new(x, y, z),
                        _ => Vec3::new(1.0, 1.0, 1.0),
                    };
                    Mat4::from_scale_rotation_translation(scale, rotation, translation)
                };
                Node {
                    mesh: node.mesh,
                    children: node.children.clone(),
                    transform,
                }
            })
            .collect::<Vec<Node>>()
    };

    let image_texture_kinds = {
        profiling::scope!("image kind map creation");
        let mut image_texture_kinds = HashMap::new();
        for material in &gltf.materials {
            if let Some(pbr) = &material.pbr_metallic_roughness {
                if let Some(base_color) = &pbr.base_color_texture {
                    let texture = gltf.textures.get(base_color.index).ok_or(Error::GltfOob("texture"))?;
                    if let Some(image_index) = texture.source {
                        image_texture_kinds.insert(image_index, TextureKind::SrgbColor);
                    }
                }
                if let Some(metallic_roughness) = &pbr.metallic_roughness_texture {
                    let texture = gltf.textures.get(metallic_roughness.index).ok_or(Error::GltfOob("texture"))?;
                    if let Some(image_index) = texture.source {
                        image_texture_kinds.insert(image_index, TextureKind::LinearColor);
                    }
                }
            }
            if let Some(normal) = &material.normal_texture {
                let texture = gltf.textures.get(normal.index).ok_or(Error::GltfOob("texture"))?;
                if let Some(image_index) = texture.source {
                    image_texture_kinds.insert(image_index, TextureKind::NormalMap);
                }
            }
            if let Some(emissive) = &material.emissive_texture {
                let texture = gltf.textures.get(emissive.index).ok_or(Error::GltfOob("texture"))?;
                if let Some(image_index) = texture.source {
                    image_texture_kinds.insert(image_index, TextureKind::SrgbColor);
                }
            }
            if let Some(occlusion) = &material.occlusion_texture {
                let texture = gltf.textures.get(occlusion.index).ok_or(Error::GltfOob("texture"))?;
                if let Some(image_index) = texture.source {
                    image_texture_kinds.insert(image_index, TextureKind::LinearColor);
                }
            }
        }
        image_texture_kinds
    };

    let images = {
        profiling::scope!("images");
        gltf.images
            .iter()
            .enumerate()
            .map(|(i, image)| {
                let image_load;
                let bytes;
                if let (Some(mime_type), Some(buffer_view)) = (&image.mime_type, &image.buffer_view) {
                    image_load = match mime_type.as_str() {
                        "image/ktx" => image_loading::load_ktx,
                        _ => return Err(Error::GltfSpec("mime type of texture is not image/ktx")),
                    };
                    let buffer_view = gltf.buffer_views.get(*buffer_view).ok_or(Error::GltfOob("texture buffer view"))?;
                    let buffer = buffers.get(buffer_view.buffer).ok_or(Error::GltfOob("texture buffer"))?;
                    let offset = buffer_view.byte_offset.unwrap_or(0);
                    let length = buffer_view.byte_length;
                    if offset + length >= buffer.len() {
                        return Err(Error::GltfOob("texture buffer view bytes"));
                    }
                    bytes = Rc::new(Cow::Borrowed(&buffer[offset..offset + length]));
                } else if let Some(uri) = &image.uri {
                    let mime_type = if uri.ends_with(".ktx") {
                        "image/ktx"
                    } else {
                        return Err(Error::GltfMisc("image uri does not end in .ktx"));
                    };
                    image_load = match mime_type {
                        "image/ktx" => image_loading::load_ktx,
                        _ => return Err(Error::GltfSpec("mime type of texture is not image/ktx")),
                    };
                    bytes = resources.get_or_load(uri)?;
                } else {
                    return Err(Error::GltfSpec("image does not have an uri nor a mimetype + buffer view"));
                };

                let texture_kind = image_texture_kinds.get(&i).copied().unwrap_or(TextureKind::LinearColor);
                image_load(gpu, arena, temp_arenas, frame_index, &bytes, texture_kind)
            })
            .collect::<Result<Vec<_>, Error>>()?
    };

    let materials = {
        profiling::scope!("materials");
        gltf.materials
            .iter()
            .map(|mat| {
                let mktex = |images: &[(&'a ImageAllocation, &'a ImageView)],
                             texture_info: &gltf_json::TextureInfo|
                 -> Option<Result<&'a ImageView, Error>> {
                    let texture = match gltf.textures.get(texture_info.index) {
                        Some(tex) => tex,
                        None => return Some(Err(Error::GltfOob("texture"))),
                    };
                    let image_index = texture.source?;
                    let image_view = match images.get(image_index) {
                        Some((_, image_view)) => image_view,
                        None => return Some(Err(Error::GltfOob("image"))),
                    };
                    Some(Ok(image_view))
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

                let pbr = mat.pbr_metallic_roughness.as_ref().ok_or(Error::GltfMisc("pbr missing"))?;
                let base_color = handle_optional_result!(pbr.base_color_texture.as_ref().and_then(|tex| mktex(&images, tex)));
                let metallic_roughness =
                    handle_optional_result!(pbr.metallic_roughness_texture.as_ref().and_then(|tex| mktex(&images, tex)));
                let normal = handle_optional_result!(mat.normal_texture.as_ref().and_then(|tex| mktex(&images, tex)));
                let occlusion = handle_optional_result!(mat.occlusion_texture.as_ref().and_then(|tex| mktex(&images, tex)));
                let emissive = handle_optional_result!(mat.emissive_texture.as_ref().and_then(|tex| mktex(&images, tex)));
                Material::new(gpu, base_color, metallic_roughness, normal, occlusion, emissive)
            })
            .collect::<Result<Vec<Material>, Error>>()?
    };

    {
        profiling::scope!("node graph creation");
        let mut visited_nodes = vec![false; nodes.len()];
        let mut queue = Vec::with_capacity(nodes.len());
        queue.extend_from_slice(&root_nodes);
        while let Some(node) = queue.pop() {
            if visited_nodes[node] {
                return Err(Error::GltfInvalidNodeGraph);
            } else {
                visited_nodes[node] = true;
                if let Some(children) = &nodes[node].children {
                    for child in children {
                        queue.push(*child);
                    }
                }
            }
        }
    }

    Ok(Gltf {
        nodes,
        root_nodes,
        materials,
        meshes,
    })
}

#[profiling::function]
fn create_primitive<'a>(
    gpu: &Gpu,
    arena: &'a VulkanArena,
    temp_arenas: &[VulkanArena],
    frame_index: FrameIndex,
    gltf: &gltf_json::GltfJson,
    buffers: &[Rc<Cow<'_, [u8]>>],
    primitive: &gltf_json::Primitive,
) -> Result<Mesh<'a>, Error> {
    let index_accessor = primitive.indices.ok_or(Error::GltfMisc("missing indices"))?;
    let index_buffer = get_slice_from_accessor(gltf, buffers, index_accessor, GLTF_UNSIGNED_SHORT, "SCALAR")?;

    let pos_accessor = *primitive
        .attributes
        .get("POSITION")
        .ok_or(Error::GltfMisc("missing position attributes"))?;
    let pos_buffer = get_slice_from_accessor(gltf, buffers, pos_accessor, GLTF_FLOAT, "VEC3")?;

    let tex_accessor = *primitive
        .attributes
        .get("TEXCOORD_0")
        .ok_or(Error::GltfMisc("missing UV0 attributes"))?;
    let tex_buffer = get_slice_from_accessor(gltf, buffers, tex_accessor, GLTF_FLOAT, "VEC2")?;

    let normal_accessor = *primitive
        .attributes
        .get("NORMAL")
        .ok_or(Error::GltfMisc("missing normal attributes"))?;
    let normal_buffer = get_slice_from_accessor(gltf, buffers, normal_accessor, GLTF_FLOAT, "VEC3")?;

    let tangent_accessor = *primitive
        .attributes
        .get("TANGENT")
        .ok_or(Error::GltfMisc("missing tangent attributes"))?;
    let tangent_buffer = get_slice_from_accessor(gltf, buffers, tangent_accessor, GLTF_FLOAT, "VEC4")?;

    let mesh = Mesh::new::<u16>(
        gpu,
        arena,
        temp_arenas,
        frame_index,
        &[pos_buffer, tex_buffer, normal_buffer, tangent_buffer],
        index_buffer,
        Pipeline::Default,
    )?;

    Ok(mesh)
}

#[profiling::function]
fn get_slice_from_accessor<'buffer>(
    gltf: &gltf_json::GltfJson,
    buffers: &'buffer [Rc<Cow<'_, [u8]>>],
    accessor: usize,
    ctype: i32,
    atype: &str,
) -> Result<&'buffer [u8], Error> {
    let accessor = gltf.accessors.get(accessor).ok_or(Error::GltfOob("accessor"))?;
    if accessor.component_type != ctype {
        return Err(Error::GltfMisc("unexpected component type"));
    }
    if accessor.attribute_type != atype {
        return Err(Error::GltfMisc("unexpected attribute type"));
    }
    let view = match accessor.buffer_view {
        Some(view) => view,
        None => return Err(Error::GltfMisc("no buffer view")),
    };
    let view = gltf.buffer_views.get(view).ok_or(Error::GltfOob("buffer view"))?;
    let buffer = buffers.get(view.buffer).ok_or(Error::GltfOob("buffer"))?;
    let offset = view.byte_offset.unwrap_or(0) + accessor.byte_offset.unwrap_or(0);
    let length = view.byte_length;
    let stride = stride_for(ctype, atype);
    match view.byte_stride {
        Some(x) if x != stride => return Err(Error::GltfMisc("wrong stride")),
        _ => {}
    }
    if offset + length > buffer.len() {
        return Err(Error::GltfOob("buffer offset + length"));
    }
    if accessor.count != length / stride {
        return Err(Error::GltfOob("count != byte length / stride"));
    }
    Ok(&buffer[offset..offset + length])
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
