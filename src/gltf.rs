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
    }

    #[derive(Deserialize)]
    pub struct Node {
        mesh: Option<usize>,

        // Not in use yet:
        children: Vec<usize>,
        matrix: Option<Vec<f32>>,
        translation: Option<Vec<f32>>,
        rotation: Option<Vec<f32>>,
        scale: Option<Vec<f32>>,
        camera: Option<usize>,
    }

    #[derive(Deserialize)]
    pub struct Mesh {
        primitives: Vec<Primitive>,
    }

    #[derive(Deserialize)]
    pub struct Primitive {
        indices: usize,
        attributes: HashMap<String, usize>,

        // Not in use yet:
        mode: u32,
        material: usize,
        targets: Vec<HashMap<String, usize>>,
        weights: Vec<f32>,
    }

    #[derive(Deserialize)]
    pub struct Buffer {
        byte_length: usize,
        uri: String,
    }

    #[derive(Deserialize)]
    pub struct BufferView {
        buffer: usize,
        byte_offset: usize,
        byte_length: usize,
        byte_stride: usize,
    }

    #[derive(Deserialize)]
    pub struct Accessor {
        buffer_view: usize,
        byte_offset: usize,
        #[serde(rename = "type")]
        attribute_type: String,
        count: usize,
        min: Vec<f32>,
        max: Vec<f32>,
    }
}
