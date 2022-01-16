use crate::arena::ImageView;
use crate::gpu::TextureIndex;
use crate::{Error, Gpu};
use std::hash::{Hash, Hasher};

pub struct Material<'a> {
    pub(crate) texture_index: TextureIndex,

    pub base_color: Option<&'a ImageView>,
    pub metallic_roughness: Option<&'a ImageView>,
    pub normal: Option<&'a ImageView>,
    pub occlusion: Option<&'a ImageView>,
    pub emissive: Option<&'a ImageView>,
}

impl PartialEq for Material<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.texture_index == other.texture_index
    }
}

impl Eq for Material<'_> {}

impl Hash for Material<'_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.texture_index.hash(state);
    }
}

impl Material<'_> {
    pub fn new<'a>(
        gpu: &'a Gpu,
        base_color: Option<&'a ImageView>,
        metallic_roughness: Option<&'a ImageView>,
        normal: Option<&'a ImageView>,
        occlusion: Option<&'a ImageView>,
        emissive: Option<&'a ImageView>,
    ) -> Result<Material<'a>, Error> {
        let texture_index = gpu.reserve_texture_index(
            base_color.map(|iv| iv.image_view),
            metallic_roughness.map(|iv| iv.image_view),
            normal.map(|iv| iv.image_view),
            occlusion.map(|iv| iv.image_view),
            emissive.map(|iv| iv.image_view),
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
