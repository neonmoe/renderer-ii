use crate::gpu::TextureIndex;
use crate::{Error, Gpu, Texture};
use std::hash::{Hash, Hasher};

pub struct Material<'a> {
    gpu: &'a Gpu<'a>,
    pub(crate) texture_index: TextureIndex,

    _base_color: Option<Texture<'a>>,
    _metallic_roughness: Option<Texture<'a>>,
    _normal: Option<Texture<'a>>,
    _occlusion: Option<Texture<'a>>,
    _emissive: Option<Texture<'a>>,
}

impl PartialEq for Material<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.gpu.device.handle() == other.gpu.device.handle() && self.texture_index == other.texture_index
    }
}

impl Eq for Material<'_> {}

impl Hash for Material<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.gpu.device.handle().hash(state);
        self.texture_index.hash(state);
    }
}

impl Drop for Material<'_> {
    fn drop(&mut self) {
        self.gpu.release_texture_index(unsafe { self.texture_index.inner() });
    }
}

impl Material<'_> {
    pub fn new<'a>(
        gpu: &'a Gpu<'a>,
        _base_color: Option<Texture<'a>>,
        _metallic_roughness: Option<Texture<'a>>,
        _normal: Option<Texture<'a>>,
        _occlusion: Option<Texture<'a>>,
        _emissive: Option<Texture<'a>>,
    ) -> Result<Material<'a>, Error> {
        let texture_index = gpu.reserve_texture_index(
            _base_color.as_ref().map(|mat| mat.image_view),
            _metallic_roughness.as_ref().map(|mat| mat.image_view),
            _normal.as_ref().map(|mat| mat.image_view),
            _occlusion.as_ref().map(|mat| mat.image_view),
            _emissive.as_ref().map(|mat| mat.image_view),
        )?;
        Ok(Material {
            gpu,
            texture_index,
            _base_color,
            _metallic_roughness,
            _normal,
            _occlusion,
            _emissive,
        })
    }
}
