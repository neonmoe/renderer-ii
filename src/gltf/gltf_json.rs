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
    pub textures: Vec<Texture>,
    pub images: Vec<Image>,
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

#[derive(Deserialize)]
pub struct Texture {
    pub source: Option<usize>,
}

#[derive(Deserialize)]
pub struct Image {
    pub uri: Option<String>,
    #[serde(rename = "mimeType")]
    pub mime_type: Option<String>,
    #[serde(rename = "bufferView")]
    pub buffer_view: Option<usize>,
}
