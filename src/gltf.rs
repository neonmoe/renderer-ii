use crate::Error;

pub struct Gltf {
    _temp: gltf_json::LoadedJson,
}

impl Gltf {
    /// Loads the glTF scene from the contents of a .gltf file.
    pub fn from_gltf(gltf_json: &str) -> Result<Gltf, Error> {
        let loaded: gltf_json::LoadedJson =
            miniserde::json::from_str(gltf_json).map_err(Error::GltfDeserialization)?;
        Ok(Gltf { _temp: loaded })
    }
}

#[allow(dead_code)]
mod gltf_json {
    use miniserde::Deserialize;
    use std::collections::HashMap;

    #[derive(Deserialize)]
    pub struct LoadedJson {
        nodes: Vec<Node>,
        meshes: Vec<Mesh>,
        accessors: Vec<Accessor>,
        #[serde(rename = "bufferViews")]
        buffer_views: Vec<BufferView>,
        buffers: Vec<Buffer>,
    }

    #[derive(Deserialize)]
    pub struct Node {
        mesh: Option<usize>,

        // Not in use yet:
        children: Option<Vec<usize>>,
        matrix: Option<Vec<f32>>,
        translation: Option<Vec<f32>>,
        rotation: Option<Vec<f32>>,
        scale: Option<Vec<f32>>,
        skin: Option<usize>,
        weights: Option<Vec<f32>>,
    }

    #[derive(Deserialize)]
    pub struct Mesh {
        primitives: Vec<Primitive>,
    }

    #[derive(Deserialize)]
    pub struct Primitive {
        attributes: HashMap<String, usize>,
        indices: Option<usize>,
        // Not in use yet:
        mode: Option<i32>,
        material: Option<usize>,
        targets: Option<Vec<HashMap<String, usize>>>,
        weights: Option<Vec<f32>>,
    }

    #[derive(Deserialize)]
    pub struct Buffer {
        #[serde(rename = "byteLength")]
        byte_length: usize,
        uri: Option<String>,
    }

    #[derive(Deserialize)]
    pub struct BufferView {
        buffer: usize,
        #[serde(rename = "byteOffset")]
        byte_offset: Option<usize>,
        #[serde(rename = "byteLength")]
        byte_length: usize,
        #[serde(rename = "byteStride")]
        byte_stride: Option<usize>,
    }

    #[derive(Deserialize)]
    pub struct Accessor {
        #[serde(rename = "bufferView")]
        buffer_view: Option<usize>,
        #[serde(rename = "byteOffset")]
        byte_offset: Option<usize>,
        #[serde(rename = "componentType")]
        component_type: i32,
        normalized: Option<bool>,
        count: usize,
        #[serde(rename = "type")]
        attribute_type: String,
        min: Option<Vec<f32>>,
        max: Option<Vec<f32>>,
    }
}
