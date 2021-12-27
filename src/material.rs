use crate::gpu::TextureIndex;
use crate::{Error, Gpu};
use ash::vk;
use std::hash::{Hash, Hasher};

pub struct Material {
    pub(crate) texture_index: TextureIndex,

    base_color: Option<vk::ImageView>,
    metallic_roughness: Option<vk::ImageView>,
    normal: Option<vk::ImageView>,
    occlusion: Option<vk::ImageView>,
    emissive: Option<vk::ImageView>,
}

impl PartialEq for Material {
    fn eq(&self, other: &Self) -> bool {
        self.texture_index == other.texture_index
    }
}

impl Eq for Material {}

impl Hash for Material {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.texture_index.hash(state);
    }
}

impl Material {
    pub fn new(
        gpu: &Gpu,
        base_color: Option<vk::ImageView>,
        metallic_roughness: Option<vk::ImageView>,
        normal: Option<vk::ImageView>,
        occlusion: Option<vk::ImageView>,
        emissive: Option<vk::ImageView>,
    ) -> Result<Material, Error> {
        let texture_index = gpu.reserve_texture_index(base_color, metallic_roughness, normal, occlusion, emissive)?;
        Ok(Material {
            texture_index,
            base_color,
            metallic_roughness,
            normal,
            occlusion,
            emissive,
        })
    }
}
