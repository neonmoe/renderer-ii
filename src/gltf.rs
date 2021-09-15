use crate::{Error, FrameIndex, Gpu, Mesh, Pipeline};
use std::borrow::Cow;
use std::fs;
use std::path::Path;
use ultraviolet::{Isometry3, Mat4, Rotor3, Vec3, Vec4};

struct Node {
    mesh: Option<usize>,
    children: Option<Vec<usize>>,
    transform: Mat4,
}

pub struct Gltf<'a> {
    nodes: Vec<Node>,
    meshes: Vec<Vec<Mesh<'a>>>,
    root_nodes: Vec<usize>,
}

pub struct MeshIter<'a> {
    gltf: &'a Gltf<'a>,
    node_queue: Vec<usize>,
}

impl<'a> Iterator for MeshIter<'a> {
    type Item = (&'a [Mesh<'a>], Mat4);

    fn next(&mut self) -> Option<Self::Item> {
        let node_index = self.node_queue.pop()?;
        let node = self.gltf.nodes.get(node_index)?;
        if let Some(children) = &node.children {
            for child in children {
                self.node_queue.push(*child);
            }
        }
        let mesh = self.gltf.meshes.get(node.mesh?)?;
        Some((mesh, node.transform))
    }
}

impl Gltf<'_> {
    /// Loads the glTF scene from the contents of a .glb file.
    ///
    /// Any external files referenced in the glTF are searched
    /// relative to `directory`.
    pub fn from_glb<'gpu>(gpu: &'gpu Gpu, frame_index: FrameIndex, glb: &[u8], directory: Option<&Path>) -> Result<Gltf<'gpu>, Error> {
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
        create_gltf(gpu, frame_index, gltf, directory, Some(buffer))
    }

    /// Loads the glTF scene from the contents of a .gltf file.
    ///
    /// Any external files referenced in the glTF are searched
    /// relative to `directory`.
    pub fn from_gltf<'gpu>(gpu: &'gpu Gpu, frame_index: FrameIndex, gltf: &str, directory: Option<&Path>) -> Result<Gltf<'gpu>, Error> {
        let gltf: gltf_json::GltfJson = miniserde::json::from_str(gltf).map_err(Error::GltfJsonDeserialization)?;
        create_gltf(gpu, frame_index, gltf, directory, None)
    }

    pub fn mesh_iter(&self) -> MeshIter<'_> {
        MeshIter {
            gltf: self,
            node_queue: self.root_nodes.clone(),
        }
    }
}

fn create_gltf<'gpu>(
    gpu: &'gpu Gpu,
    frame_index: FrameIndex,
    gltf: gltf_json::GltfJson,
    directory: Option<&Path>,
    bin_buffer: Option<&[u8]>,
) -> Result<Gltf<'gpu>, Error> {
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

    let buffers = gltf
        .buffers
        .iter()
        .map(|buffer| match &buffer.uri {
            Some(uri) => match directory {
                Some(base_path) => {
                    let path = base_path.join(&uri);
                    let buffer = fs::read(path).map_err(|err| Error::GltfBufferLoading(uri.clone(), err))?;
                    Ok(Cow::Owned(buffer))
                }
                None => Err(Error::GltfMissingDirectory(uri.clone())),
            },
            None => match bin_buffer {
                Some(buffer) => Ok(Cow::Borrowed(buffer)),
                None => Err(Error::GlbBinMissing),
            },
        })
        .collect::<Result<Vec<Cow<'_, [u8]>>, Error>>()?;

    let meshes = gltf
        .meshes
        .iter()
        .enumerate()
        .map(|(i, mesh)| {
            mesh.primitives
                .iter()
                .enumerate()
                .map(move |(j, primitive)| (i, j, primitive))
                .filter_map(
                    |(i, j, primitive)| match create_mesh_from_primitive(gpu, frame_index, &gltf, &buffers, primitive) {
                        Ok(mesh) => Some(mesh),
                        Err(err) => {
                            log::warn!("skipping mesh #{}, primitive #{}: {}", i, j, err);
                            None
                        }
                    },
                )
                .collect::<Vec<Mesh<'_>>>()
        })
        .collect::<Vec<Vec<Mesh<'_>>>>();

    let nodes = gltf
        .nodes
        .iter()
        .map(|node| {
            let transform = if let Some(&[x0, y0, z0, w0, x1, y1, z1, w1, x2, y2, z2, w2, x3, y3, z3, w3]) = node.matrix.as_deref() {
                // Both the source and destination here are
                // actually column major: each Vec4 is a column.
                Mat4::new(
                    Vec4::new(x0, y0, z0, w0),
                    Vec4::new(x1, y1, z1, w1),
                    Vec4::new(x2, y2, z2, w2),
                    Vec4::new(x3, y3, z3, w3),
                )
            } else {
                let translation = match node.translation.as_deref() {
                    Some(&[x, y, z]) => Vec3::new(x, y, z),
                    _ => Vec3::zero(),
                };
                let rotation = match node.rotation.as_deref() {
                    Some(&[x, y, z, w]) => Rotor3::from_quaternion_array([x, y, z, w]),
                    _ => Rotor3::identity(),
                };
                let scale = match node.scale.as_deref() {
                    Some(&[x, y, z]) => Vec3::new(x, y, z),
                    _ => Vec3::one(),
                };
                Isometry3::new(translation, rotation).into_homogeneous_matrix() * Mat4::from_nonuniform_scale(scale)
            };
            Node {
                mesh: node.mesh,
                children: node.children.clone(),
                transform,
            }
        })
        .collect::<Vec<Node>>();

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

    Ok(Gltf { meshes, nodes, root_nodes })
}

fn create_mesh_from_primitive<'gpu>(
    gpu: &'gpu Gpu,
    frame_index: FrameIndex,
    gltf: &gltf_json::GltfJson,
    buffers: &[Cow<'_, [u8]>],
    primitive: &gltf_json::Primitive,
) -> Result<Mesh<'gpu>, Error> {
    let index_accessor = primitive.indices.ok_or(Error::GltfMisc("missing indices"))?;
    let index_buffer = get_slice_from_accessor(gltf, buffers, index_accessor, GLTF_UNSIGNED_SHORT, "SCALAR")?;

    let pos_accessor = *primitive
        .attributes
        .get("POSITION")
        .ok_or(Error::GltfMisc("missing POSITION attribute"))?;
    let pos_buffer = get_slice_from_accessor(gltf, buffers, pos_accessor, GLTF_FLOAT, "VEC3")?;

    let tex_accessor = *primitive
        .attributes
        .get("TEXCOORD_0")
        .ok_or(Error::GltfMisc("missing TEXCOORD_0 attribute"))?;
    let tex_buffer = get_slice_from_accessor(gltf, buffers, tex_accessor, GLTF_FLOAT, "VEC2")?;

    let mesh = Mesh::new::<u16>(gpu, frame_index, &[pos_buffer, tex_buffer], index_buffer, Pipeline::Default)?;

    Ok(mesh)
}

fn get_slice_from_accessor<'buffer>(
    gltf: &gltf_json::GltfJson,
    buffers: &'buffer [Cow<'_, [u8]>],
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

#[allow(dead_code)]
mod gltf_json {
    use miniserde::Deserialize;
    use std::collections::HashMap;

    #[derive(Deserialize)]
    pub struct GltfJson {
        pub asset: Asset,
        pub scene: Option<usize>,
        pub scenes: Option<Vec<Scene>>,
        pub nodes: Vec<Node>,
        pub meshes: Vec<Mesh>,
        pub accessors: Vec<Accessor>,
        #[serde(rename = "bufferViews")]
        pub buffer_views: Vec<BufferView>,
        pub buffers: Vec<Buffer>,
    }

    #[derive(Deserialize)]
    pub struct Asset {
        pub version: String,
        pub min_version: Option<String>,
    }

    #[derive(Deserialize)]
    pub struct Scene {
        pub nodes: Option<Vec<usize>>,
    }

    #[derive(Deserialize)]
    pub struct Node {
        pub name: Option<String>,
        pub mesh: Option<usize>,
        pub children: Option<Vec<usize>>,
        pub matrix: Option<Vec<f32>>,
        pub translation: Option<Vec<f32>>,
        pub rotation: Option<Vec<f32>>,
        pub scale: Option<Vec<f32>>,
    }

    #[derive(Deserialize)]
    pub struct Mesh {
        pub primitives: Vec<Primitive>,
    }

    #[derive(Deserialize)]
    pub struct Primitive {
        pub attributes: HashMap<String, usize>,
        pub indices: Option<usize>,
    }

    #[derive(Deserialize)]
    pub struct Buffer {
        #[serde(rename = "byteLength")]
        pub byte_length: usize,
        pub uri: Option<String>,
    }

    #[derive(Deserialize)]
    pub struct BufferView {
        pub buffer: usize,
        #[serde(rename = "byteOffset")]
        pub byte_offset: Option<usize>,
        #[serde(rename = "byteLength")]
        pub byte_length: usize,
        #[serde(rename = "byteStride")]
        pub byte_stride: Option<usize>,
    }

    #[derive(Deserialize)]
    pub struct Accessor {
        #[serde(rename = "bufferView")]
        pub buffer_view: Option<usize>,
        #[serde(rename = "byteOffset")]
        pub byte_offset: Option<usize>,
        #[serde(rename = "componentType")]
        pub component_type: i32,
        pub normalized: Option<bool>,
        pub count: usize,
        #[serde(rename = "type")]
        pub attribute_type: String,
        pub min: Option<Vec<f32>>,
        pub max: Option<Vec<f32>>,
    }
}

const GLTF_BYTE: i32 = 5120;
const GLTF_UNSIGNED_BYTE: i32 = 5121;
const GLTF_SHORT: i32 = 5122;
const GLTF_UNSIGNED_SHORT: i32 = 5123;
const GLTF_UNSIGNED_INT: i32 = 5125;
const GLTF_FLOAT: i32 = 5126;

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
