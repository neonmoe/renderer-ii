use crate::gpu::TextureIndex;
use crate::{Error, Gpu, Texture};
use std::hash::{Hash, Hasher};

pub struct Material {
    pub(crate) texture_index: TextureIndex,

    _base_color: Option<Texture>,
    _metallic_roughness: Option<Texture>,
    _normal: Option<Texture>,
    _occlusion: Option<Texture>,
    _emissive: Option<Texture>,
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
        _base_color: Option<Texture>,
        _metallic_roughness: Option<Texture>,
        _normal: Option<Texture>,
        _occlusion: Option<Texture>,
        _emissive: Option<Texture>,
    ) -> Result<Material, Error> {
        let texture_index = gpu.reserve_texture_index(&_base_color, &_metallic_roughness, &_normal, &_occlusion, &_emissive)?;
        Ok(Material {
            texture_index,
            _base_color,
            _metallic_roughness,
            _normal,
            _occlusion,
            _emissive,
        })
    }
}
