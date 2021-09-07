use crate::buffer::Buffer;
use crate::{Error, Gpu};

pub struct Gltf<'a> {
    buffers: Vec<Buffer<'a>>,
}

impl Gltf<'_> {
    /// Loads the glTF scene from the contents of a .glb file.
    pub fn from_glb<'gpu>(gpu: &'gpu Gpu, glb: &[u8]) -> Result<Gltf<'gpu>, Error> {
        fn read_u32(bytes: &[u8]) -> u32 {
            debug_assert!(bytes.len() == 4);
            if let [a, b, c, d] = *bytes {
                u32::from_le_bytes([a, b, c, d])
            } else {
                unreachable!();
            }
        }

        if glb.len() < 12 || read_u32(&glb[0..4]) != 0x46546C67 {
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

            let chunk_type = read_u32(&next_chunk[4..8]);
            match chunk_type {
                0x4E4F534A => {
                    if json.is_some() {
                        return Err(Error::TooManyGlbJsonChunks);
                    }
                    json = Some(std::str::from_utf8(chunk_bytes).map_err(Error::InvalidGlbJson)?);
                }
                0x004E4942 => {
                    if buffer.is_some() {
                        return Err(Error::TooManyGlbBinaryChunks);
                    }
                    buffer = Some(chunk_bytes);
                }
                _ => return Err(Error::InvalidGlbChunkType),
            }

            next_chunk = &next_chunk[chunk_length + 8..];
        }

        let json = json.ok_or(Error::MissingGlbJson)?;
        let loaded: gltf_json::LoadedJson =
            miniserde::json::from_str(json).map_err(Error::GltfJsonDeserialization)?;
        if let Some(min_version) = loaded.asset.min_version {
            let min_version_f32 = str::parse::<f32>(&min_version);
            if min_version_f32 != Ok(2.0) {
                return Err(Error::UnsupportedGltfVersion(min_version));
            }
        } else if let Ok(version) = str::parse::<f32>(&loaded.asset.version) {
            if !(2.0..3.0).contains(&version) {
                return Err(Error::UnsupportedGltfVersion(loaded.asset.version));
            }
        } else {
            log::warn!(
                "Could not parse glTF version {}, assuming 2.0.",
                loaded.asset.version
            );
        }

        if loaded.buffers.len() > 1 {
            return Err(Error::GltfTooManyBuffers);
        }
        if !loaded.buffers.is_empty() && loaded.buffers[0].uri.is_some() {
            return Err(Error::GltfBufferHasUri);
        }

        Ok(Gltf {
            buffers: Vec::new(),
        })
    }
}

#[allow(dead_code)]
mod gltf_json {
    use miniserde::Deserialize;
    use std::collections::HashMap;

    #[derive(Deserialize)]
    pub struct LoadedJson {
        pub asset: Asset,
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
    pub struct Node {
        pub mesh: Option<usize>,

        // Not in use yet:
        pub children: Option<Vec<usize>>,
        pub matrix: Option<Vec<f32>>,
        pub translation: Option<Vec<f32>>,
        pub rotation: Option<Vec<f32>>,
        pub scale: Option<Vec<f32>>,
        pub skin: Option<usize>,
        pub weights: Option<Vec<f32>>,
    }

    #[derive(Deserialize)]
    pub struct Mesh {
        pub primitives: Vec<Primitive>,
    }

    #[derive(Deserialize)]
    pub struct Primitive {
        pub attributes: HashMap<String, usize>,
        pub indices: Option<usize>,
        // Not in use yet:
        pub mode: Option<i32>,
        pub material: Option<usize>,
        pub targets: Option<Vec<HashMap<String, usize>>>,
        pub weights: Option<Vec<f32>>,
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
