use arrayvec::ArrayString;
use glam::{Quat, Vec3};
use hashbrown::HashMap;
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct GltfJson {
    pub asset: Asset,
    pub scene: Option<usize>,
    pub scenes: Option<Vec<Scene>>,
    pub nodes: Vec<Node>,
    pub meshes: Vec<Mesh>,
    pub accessors: Vec<Accessor>,
    #[serde(rename = "bufferViews")]
    pub buffer_views: Vec<BufferView>,
    pub buffers: Vec<Buffer>,
    #[serde(default)]
    pub textures: Vec<Texture>,
    #[serde(default)]
    pub images: Vec<Image>,
    pub materials: Vec<Material>,
    #[serde(default)]
    #[allow(dead_code)]
    /// Deliberately ignored for now. When implemented, remove the dead_code
    /// allowing tags from here and the Sampler definition.
    pub samplers: Vec<Sampler>,
    #[serde(default)]
    pub animations: Vec<Animation>,
    #[serde(default)]
    pub skins: Vec<Skin>,
}

#[derive(Deserialize)]
pub(crate) struct Asset {
    pub version: ArrayString<32>,
    pub min_version: Option<ArrayString<32>>,
}

#[derive(Deserialize)]
pub(crate) struct Scene {
    pub nodes: Option<Vec<usize>>,
}

#[derive(Deserialize)]
pub(crate) struct Node {
    pub name: Option<ArrayString<64>>,
    pub mesh: Option<usize>,
    pub skin: Option<usize>,
    pub children: Option<Vec<usize>>,
    pub matrix: Option<[f32; 16]>,
    pub translation: Option<[f32; 3]>,
    pub rotation: Option<[f32; 4]>,
    pub scale: Option<[f32; 3]>,
}

#[derive(Deserialize)]
pub(crate) struct Mesh {
    pub primitives: Vec<Primitive>,
}

#[derive(Deserialize)]
pub(crate) struct Primitive {
    pub attributes: HashMap<ArrayString<32>, usize>,
    pub indices: Option<usize>,
    pub material: Option<usize>,
}

#[derive(Deserialize)]
pub(crate) struct Buffer {
    #[serde(rename = "byteLength")]
    pub byte_length: usize,
    pub uri: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct BufferView {
    pub buffer: usize,
    #[serde(rename = "byteOffset", default = "const_zero_usize")]
    pub byte_offset: usize,
    #[serde(rename = "byteLength")]
    pub byte_length: usize,
    #[serde(rename = "byteStride")]
    pub byte_stride: Option<usize>,
}

#[derive(Deserialize)]
pub(crate) struct Accessor {
    #[serde(rename = "bufferView")]
    pub buffer_view: Option<usize>,
    #[serde(rename = "byteOffset", default = "const_zero_usize")]
    pub byte_offset: usize,
    #[serde(rename = "componentType")]
    pub component_type: i32,
    pub count: usize,
    #[serde(rename = "type")]
    pub attribute_type: String,
    #[serde(default)]
    pub min: Vec<f32>,
    #[serde(default)]
    pub max: Vec<f32>,
}

#[derive(Deserialize)]
pub(crate) struct Texture {
    pub source: Option<usize>,
}

#[derive(Deserialize)]
pub(crate) struct Image {
    pub uri: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct Material {
    pub name: Option<ArrayString<64>>,
    #[serde(rename = "pbrMetallicRoughness")]
    pub pbr_metallic_roughness: Option<PbrMetallicRoughness>,
    #[serde(rename = "normalTexture")]
    pub normal_texture: Option<TextureInfo>,
    #[serde(rename = "occlusionTexture")]
    pub occlusion_texture: Option<TextureInfo>,
    #[serde(rename = "emissiveTexture")]
    pub emissive_texture: Option<TextureInfo>,
    #[serde(rename = "emissiveFactor", default = "const_zero_vec3")]
    pub emissive_factor: [f32; 3],
    #[serde(rename = "alphaMode", default)]
    pub alpha_mode: AlphaMode,
    #[serde(rename = "alphaCutoff", default = "const_half_f32")]
    pub alpha_cutoff: f32,
}

#[derive(Deserialize, PartialEq, Clone, Copy, Default)]
pub enum AlphaMode {
    #[default]
    #[serde(rename = "OPAQUE")]
    Opaque,
    #[serde(rename = "MASK")]
    Mask,
    #[serde(rename = "BLEND")]
    Blend,
}

#[derive(Deserialize)]
pub(crate) struct PbrMetallicRoughness {
    #[serde(rename = "baseColorFactor", default = "const_one_vec4")]
    pub base_color_factor: [f32; 4],
    #[serde(rename = "metallicFactor", default = "const_one_f32")]
    pub metallic_factor: f32,
    #[serde(rename = "roughnessFactor", default = "const_one_f32")]
    pub roughness_factor: f32,
    #[serde(rename = "baseColorTexture")]
    pub base_color_texture: Option<TextureInfo>,
    #[serde(rename = "metallicRoughnessTexture")]
    pub metallic_roughness_texture: Option<TextureInfo>,
}

#[derive(Deserialize)]
pub(crate) struct TextureInfo {
    pub index: usize,
    #[serde(rename = "texCoord", default = "const_zero_u32")]
    pub texcoord: u32,
    /// The `strength` value from material.occlusionTextureInfo
    #[serde(default = "const_one_f32")]
    pub strength: f32,
    /// The `scale` value from material.normalTextureInfo
    #[serde(default = "const_one_f32")]
    pub scale: f32,
}

#[allow(dead_code)]
#[derive(Deserialize)]
pub(crate) struct Sampler {
    #[serde(rename = "magFilter")]
    pub mag_filter: Option<u32>,
    #[serde(rename = "minFilter")]
    pub min_filter: Option<u32>,
    #[serde(rename = "wrapS", default = "const_10497_u32")]
    pub wrap_s: u32,
    #[serde(rename = "wrapT", default = "const_10497_u32")]
    pub wrap_t: u32,
    pub name: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct Animation {
    pub channels: Vec<AnimationChannel>,
    pub samplers: Vec<AnimationSampler>,
    pub name: Option<String>,
}

#[derive(Deserialize)]
pub(crate) struct AnimationChannel {
    pub sampler: usize,
    pub target: AnimationTarget,
}

#[derive(Deserialize)]
pub(crate) struct AnimationTarget {
    pub node: usize,
    pub path: AnimatedProperty,
}

#[derive(Deserialize)]
pub(crate) enum AnimatedProperty {
    #[serde(rename = "translation")]
    Translation,
    #[serde(rename = "rotation")]
    Rotation,
    #[serde(rename = "scale")]
    Scale,
    #[serde(rename = "weights")]
    Weights,
}

#[derive(Deserialize)]
pub(crate) struct AnimationSampler {
    #[serde(default)]
    pub interpolation: AnimationInterpolation,
    pub input: usize,
    pub output: usize,
}

#[derive(Deserialize, Clone, Copy, Default)]
pub enum AnimationInterpolation {
    #[default]
    #[serde(rename = "LINEAR")]
    Linear,
    #[serde(rename = "STEP")]
    Step,
    // TODO: Support cubic animation interpolation
    // The output values of the sampler include tangents as well, so the animation
    // data needs to be fiddled with a bit.
    //#[serde(rename = "CUBICSPLINE")]
    //CubicSpline,
}
impl AnimationInterpolation {
    pub fn interpolate_vec3(self, keyframes: &[(f32, Vec3)], time: f32) -> Option<Vec3> {
        if keyframes.is_empty() {
            None
        } else if keyframes.len() == 1 || time < keyframes[0].0 {
            Some(keyframes[0].1)
        } else {
            for window in keyframes.windows(2) {
                if let &[(t_k, v_k), (t_k_1, v_k_1)] = window {
                    if t_k <= time && time < t_k_1 {
                        let result = match self {
                            AnimationInterpolation::Linear => {
                                let t = (time - t_k) / (t_k_1 - t_k);
                                v_k.lerp(v_k_1, t)
                            }
                            AnimationInterpolation::Step => v_k,
                        };
                        return Some(result);
                    }
                }
            }
            None
        }
    }

    pub fn interpolate_quat(self, keyframes: &[(f32, Quat)], time: f32) -> Option<Quat> {
        if keyframes.is_empty() {
            None
        } else if keyframes.len() == 1 || time < keyframes[0].0 {
            Some(keyframes[0].1)
        } else {
            for window in keyframes.windows(2) {
                if let &[(t_k, v_k), (t_k_1, v_k_1)] = window {
                    if t_k <= time && time < t_k_1 {
                        let result = match self {
                            AnimationInterpolation::Linear => {
                                let t = (time - t_k) / (t_k_1 - t_k);
                                v_k.slerp(v_k_1, t)
                            }
                            AnimationInterpolation::Step => v_k,
                        };
                        return Some(result);
                    }
                }
            }
            None
        }
    }
}

#[derive(Deserialize)]
pub(crate) struct Skin {
    #[serde(rename = "inverseBindMatrices")]
    pub inverse_bind_matrices: Option<usize>,
    pub joints: Vec<usize>,
}

const fn const_one_f32() -> f32 {
    1.0
}

const fn const_zero_u32() -> u32 {
    0
}

const fn const_zero_usize() -> usize {
    0
}

const fn const_zero_vec3() -> [f32; 3] {
    [0.0; 3]
}

const fn const_half_f32() -> f32 {
    0.5
}

const fn const_one_vec4() -> [f32; 4] {
    [1.0; 4]
}

const fn const_10497_u32() -> u32 {
    10497
}
