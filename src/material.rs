use crate::gpu::TextureIndex;
use crate::vulkan_raii::ImageView;
use crate::{Error, Gpu};
use std::hash::{Hash, Hasher};
use std::rc::Rc;

pub struct Material {
    pub(crate) texture_index: TextureIndex,

    pub base_color: Option<Rc<ImageView>>,
    pub metallic_roughness: Option<Rc<ImageView>>,
    pub normal: Option<Rc<ImageView>>,
    pub occlusion: Option<Rc<ImageView>>,
    pub emissive: Option<Rc<ImageView>>,
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
        base_color: Option<Rc<ImageView>>,
        metallic_roughness: Option<Rc<ImageView>>,
        normal: Option<Rc<ImageView>>,
        occlusion: Option<Rc<ImageView>>,
        emissive: Option<Rc<ImageView>>,
    ) -> Result<Material, Error> {
        let texture_index = gpu.reserve_texture_index(
            base_color.clone(),
            metallic_roughness.clone(),
            normal.clone(),
            occlusion.clone(),
            emissive.clone(),
        )?;
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
