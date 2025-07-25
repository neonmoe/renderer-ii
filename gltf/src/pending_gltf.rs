use alloc::rc::Rc;
use std::path::PathBuf;

use arrayvec::{ArrayString, ArrayVec};
use glam::{Vec3, Vec4};
use hashbrown::HashMap;
use memmap2::Mmap;
use renderer::image_loading::ntex::{self};
use renderer::image_loading::{self, TextureKind};
use renderer::{
    vk, AlphaMode, Descriptors, Device, ForBuffers, ForImages, ImageView, Material, PbrMaterialParameters, PipelineIndex, Uploader,
    VertexBinding, VertexBindingMap, VertexLibraryBuilder, VulkanArena,
};

use crate::gltf_json::{self, GltfJson};
use crate::{map_file, Gltf, GltfLoadingError, MAX_VERTEX_BUFFERS};

pub struct BufferView {
    pub buffer: usize,
    pub offset: usize,
    pub length: usize,
    pub stride: usize,
}

pub struct MeshParameters {
    pub name: Option<String>,
    pub primitives: Vec<(PrimitiveParameters, usize)>,
}

pub struct PrimitiveParameters {
    pub pipeline: PipelineIndex,
    pub vertex_buffers: VertexBindingMap<BufferView>,
    pub index_buffer: BufferView,
    /// If true, indices are u32, otherwise u16.
    pub large_indices: bool,
}

pub struct ImageParameters {
    pub data: ImageData,
    pub name: String,
    pub image_create_info: vk::ImageCreateInfo<'static>,
}

pub enum ImageData {
    File(PathBuf),
    // TODO: Buffer(BufferView),
}

pub struct PendingGltf<'a> {
    /// The final [`Gltf `] object except that its `meshes` and `materials`
    /// fields are empty.
    pub(crate) gltf_base: Gltf,
    pub(crate) json: GltfJson,
    pub(crate) bin_buffer: Option<&'a [u8]>,
    pub(crate) resource_path: PathBuf,
    pub(crate) image_texture_kinds: HashMap<usize, TextureKind>,
    pub(crate) meshes: Vec<MeshParameters>,
    pub(crate) images: Vec<ImageParameters>,
}

impl PendingGltf<'_> {
    /// Upload meshes and textures to the gpu, returning the Gltf. NOTE: The
    /// Gltf isn't actually appropriate to render before:
    ///
    /// - Calling [`VertexLibraryBuilder::upload`] and waiting for the uploader
    ///   given to that to finish uploading the meshes.
    /// - Waiting for `uploader` to finish uploading the textures.
    pub fn upload(
        self,
        device: &Device,
        staging_arena: &mut VulkanArena<ForBuffers>,
        uploader: &mut Uploader,
        descriptors: &mut Descriptors,
        image_arena: &mut VulkanArena<ForImages>,
        vertex_library_builder: &mut VertexLibraryBuilder,
    ) -> Result<Gltf, GltfLoadingError> {
        profiling::scope!("uploading gltf to vram");
        let mut gltf = self.gltf_base;
        let mut memmap_holder = None;

        let mut meshes = Vec::with_capacity(gltf.meshes.len());
        const TOTAL_BUFFERS: usize = MAX_VERTEX_BUFFERS + 1; // vertex buffers + index buffer
        let mut mesh_buffers_memmaps: ArrayVec<Option<Mmap>, TOTAL_BUFFERS> = (0..TOTAL_BUFFERS).map(|_| None).collect();
        for mesh_params in &self.meshes {
            profiling::scope!("uploading mesh", mesh_params.name.as_deref().unwrap_or("<no name>"));
            let mut primitives = Vec::with_capacity(mesh_params.primitives.len());
            for (params, material_index) in &mesh_params.primitives {
                let mut buffers = params
                    .vertex_buffers
                    .iter()
                    .filter_map(|(binding, buffer)| buffer.as_ref().map(|buf| (binding, buf)))
                    .chain([(VertexBinding::Position /* not used */, &params.index_buffer)].into_iter())
                    .zip(&mut mesh_buffers_memmaps)
                    .map(|((binding, buffer_params), memmap)| {
                        let offset = buffer_params.offset;
                        let length = buffer_params.length;
                        let buffer = self.json.buffers.get(buffer_params.buffer).ok_or(GltfLoadingError::Oob("buffer"))?;
                        let buffer = if let Some(uri) = &buffer.uri {
                            let path = self.resource_path.join(uri);
                            map_file(memmap, &path, Some(offset..offset + length))?
                        } else if let Some(bin_buffer) = self.bin_buffer.as_ref() {
                            &bin_buffer[offset..offset + length]
                        } else {
                            return Err(GltfLoadingError::Misc("buffer has no uri but there's no glb buffer"));
                        };
                        Ok((binding, buffer))
                    })
                    .collect::<Result<ArrayVec<(VertexBinding, &[u8]), TOTAL_BUFFERS>, GltfLoadingError>>()?;
                let (_, index_buffer) = buffers.pop().unwrap();
                let buffers = VertexBindingMap::from_iter(buffers.into_iter().map(|(binding, buf)| (binding, Some(buf))));
                let vertex_layout = params.pipeline.vertex_layout();
                let mesh = if params.large_indices {
                    vertex_library_builder.add_mesh(vertex_layout, &buffers, bytemuck::cast_slice::<u8, u32>(index_buffer))
                } else {
                    vertex_library_builder.add_mesh(vertex_layout, &buffers, bytemuck::cast_slice::<u8, u16>(index_buffer))
                };
                primitives.push((Rc::new(mesh), *material_index));
            }
            meshes.push(primitives);
        }

        let mut images = Vec::with_capacity(self.images.len());
        for (i, image) in self.images.into_iter().enumerate() {
            profiling::scope!("uploading image", &image.name);
            let image_bytes = match image.data {
                ImageData::File(uri) => {
                    let mut path = self.resource_path.join(uri);
                    path.set_extension("ntex");
                    map_file(&mut memmap_holder, &path, None)?
                }
            };
            let image_data = ntex::decode(image_bytes).map_err(|err| GltfLoadingError::NtexDecoding(err, image.name.clone()))?;
            let kind = self.image_texture_kinds.get(&i).copied().unwrap_or(TextureKind::LinearColor);
            images.push(Rc::new(
                image_loading::load_image(device, staging_arena, uploader, image_arena, &image_data, kind, &image.name)
                    .map_err(|err| GltfLoadingError::ImageLoading(err, image.name.clone()))?,
            ));
        }

        gltf.meshes = meshes;
        gltf.materials = create_materials(&self.json, descriptors, &images)?;
        Ok(gltf)
    }
}

fn create_materials(
    gltf: &GltfJson,
    descriptors: &mut Descriptors,
    images: &[Rc<ImageView>],
) -> Result<Vec<Rc<Material>>, GltfLoadingError> {
    let mut materials = Vec::with_capacity(gltf.materials.len());
    for mat in &gltf.materials {
        let mktex = |images: &[Rc<ImageView>], texture_info: &gltf_json::TextureInfo| {
            if texture_info.texcoord != 0 {
                return Some(Err(GltfLoadingError::Misc("non-0 texCoord used for texture")));
            }
            let texture = match gltf.textures.get(texture_info.index) {
                Some(tex) => tex,
                None => return Some(Err(GltfLoadingError::Oob("texture"))),
            };
            let image_index = texture.source?;
            let image_view = match images.get(image_index) {
                Some(image_view) => image_view,
                None => return Some(Err(GltfLoadingError::Oob("image"))),
            };
            Some(Ok(image_view.clone()))
        };

        macro_rules! handle_optional_result {
            ($expression:expr) => {
                match $expression {
                    Some(Ok(ok)) => Some(ok),
                    Some(Err(err)) => return Err(err),
                    None => None,
                }
            };
        }

        let pbr = mat.pbr_metallic_roughness.as_ref().ok_or(GltfLoadingError::Misc("pbr missing"))?;
        let base_color = handle_optional_result!(pbr.base_color_texture.as_ref().and_then(|tex| mktex(images, tex)));
        let metallic_roughness = handle_optional_result!(pbr.metallic_roughness_texture.as_ref().and_then(|tex| mktex(images, tex)));
        let normal = handle_optional_result!(mat.normal_texture.as_ref().and_then(|tex| mktex(images, tex)));
        let occlusion = handle_optional_result!(mat.occlusion_texture.as_ref().and_then(|tex| mktex(images, tex)));
        let emissive = handle_optional_result!(mat.emissive_texture.as_ref().and_then(|tex| mktex(images, tex)));

        let pbr = mat.pbr_metallic_roughness.as_ref().ok_or(GltfLoadingError::Misc("pbr missing"))?;
        let metallic_factor = pbr.metallic_factor;
        let roughness_factor = pbr.roughness_factor;
        let base_color_factor = Vec4::from_array(pbr.base_color_factor);
        let emissive_factor = Vec3::from(mat.emissive_factor);
        let normal_strength = mat.normal_texture.as_ref().map(|t| t.scale).unwrap_or(1.0);
        let occlusion_factor = mat.occlusion_texture.as_ref().map(|t| t.strength).unwrap_or(1.0);
        let alpha_cutoff = if mat.alpha_mode == gltf_json::AlphaMode::Mask { mat.alpha_cutoff } else { 0.0 };

        let params = PbrMaterialParameters {
            base_color,
            metallic_roughness,
            normal,
            occlusion,
            emissive,
            base_color_factor,
            emissive_factor,
            occlusion_factor,
            roughness_factor,
            metallic_factor,
            normal_strength,
            alpha_cutoff,
            alpha_mode: match mat.alpha_mode {
                gltf_json::AlphaMode::Opaque => AlphaMode::Opaque,
                gltf_json::AlphaMode::Mask => AlphaMode::AlphaToCoverage,
                gltf_json::AlphaMode::Blend => AlphaMode::Blended,
            },
        };
        let name = mat.name.unwrap_or_else(|| ArrayString::from("unnamed material").unwrap());
        materials.push(Material::for_pbr(descriptors, name, params).ok_or(GltfLoadingError::NotEnoughMaterialSlots)?);
    }
    Ok(materials)
}
