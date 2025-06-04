use alloc::rc::Rc;
use core::cmp::Ordering;

use arrayvec::ArrayString;
use glam::{UVec4, Vec3, Vec4};

use crate::renderer::descriptors::Descriptors;
use crate::renderer::pipeline_parameters::PipelineIndex;
use crate::renderer::pipeline_parameters::uniforms::{ImGuiDrawCmd, PbrFactors};
use crate::renderer::pipeline_parameters::vertex_buffers::VertexLayout;
use crate::vulkan_raii::ImageView;

#[derive(Clone, Copy)]
pub enum AlphaMode {
    Opaque,
    AlphaToCoverage,
    Blended,
}

#[derive(Clone)]
pub enum PipelineSpecificData {
    Pbr {
        _base_color: Option<Rc<ImageView>>,
        _metallic_roughness: Option<Rc<ImageView>>,
        _normal: Option<Rc<ImageView>>,
        _occlusion: Option<Rc<ImageView>>,
        _emissive: Option<Rc<ImageView>>,
        _factors: Rc<PbrFactors>,
        alpha_mode: AlphaMode,
    },
    ImGui {
        texture: Rc<ImageView>,
        cmd: Rc<ImGuiDrawCmd>,
    },
}

pub struct PbrMaterialParameters {
    pub base_color: Option<Rc<ImageView>>,
    pub metallic_roughness: Option<Rc<ImageView>>,
    pub normal: Option<Rc<ImageView>>,
    pub occlusion: Option<Rc<ImageView>>,
    pub emissive: Option<Rc<ImageView>>,
    pub base_color_factor: Vec4,
    pub emissive_factor: Vec3,
    pub occlusion_factor: f32,
    pub roughness_factor: f32,
    pub metallic_factor: f32,
    pub normal_strength: f32,
    pub alpha_cutoff: f32,
    pub alpha_mode: AlphaMode,
}

impl Default for PbrMaterialParameters {
    fn default() -> Self {
        Self {
            base_color: None,
            metallic_roughness: None,
            normal: None,
            occlusion: None,
            emissive: None,
            base_color_factor: Vec4::ONE,
            emissive_factor: Vec3::ONE,
            occlusion_factor: 1.0,
            roughness_factor: 1.0,
            metallic_factor: 1.0,
            normal_strength: 1.0,
            alpha_cutoff: 0.5,
            alpha_mode: AlphaMode::Opaque,
        }
    }
}

/// A unique index into one pipeline's textures and other material data.
pub struct Material {
    pub name: ArrayString<64>,
    /// The number passed into the per-draw-call uniform. In the case of the pbr
    /// pipelines, this is an index to PbrFactors, which in turn have indices to
    /// the textures array, while in the imgui pipeline, this is just the
    /// texture index directly.
    pub(crate) material_id: u32,
    /// The data referred to by the shader, so that keeping this [`Material`]
    /// around keeps the textures around.
    pub(crate) data: PipelineSpecificData,
}

impl Material {
    pub fn for_pbr(descriptors: &mut Descriptors, name: ArrayString<64>, params: PbrMaterialParameters) -> Option<Rc<Material>> {
        fn allocate_texture_slot(descriptors: &mut Descriptors, tex: &Option<Rc<ImageView>>, fallback: u32) -> Option<u32> {
            if let Some(tex) = tex { descriptors.texture_slots.try_allocate_slot(Rc::downgrade(tex)) } else { Some(fallback) }
        }
        let PbrMaterialParameters { base_color, metallic_roughness, normal, occlusion, emissive, .. } = params;
        let idx_base_col = allocate_texture_slot(descriptors, &base_color, descriptors.pbr_defaults.base_color.1)?;
        let idx_mtl_rgh = allocate_texture_slot(descriptors, &metallic_roughness, descriptors.pbr_defaults.metallic_roughness.1)?;
        let idx_normal = allocate_texture_slot(descriptors, &normal, descriptors.pbr_defaults.normal.1)?;
        let idx_occlusion = allocate_texture_slot(descriptors, &occlusion, descriptors.pbr_defaults.occlusion.1)?;
        let idx_emissive = allocate_texture_slot(descriptors, &emissive, descriptors.pbr_defaults.emissive.1)?;
        let emissive_and_occlusion =
            Vec4::from_array([params.emissive_factor.x, params.emissive_factor.y, params.emissive_factor.z, params.occlusion_factor]);
        let alpha_rgh_mtl_normal =
            Vec4::from_array([params.alpha_cutoff, params.roughness_factor, params.metallic_factor, params.normal_strength]);
        let factors = Rc::new(PbrFactors {
            base_color: params.base_color_factor,
            emissive_and_occlusion,
            alpha_rgh_mtl_normal,
            textures: UVec4::new((idx_base_col << 16) | idx_mtl_rgh, idx_normal, idx_occlusion, idx_emissive),
        });
        let material_id = descriptors.pbr_factors_slots.try_allocate_slot(Rc::downgrade(&factors))?;
        let data = PipelineSpecificData::Pbr {
            _base_color: base_color,
            _metallic_roughness: metallic_roughness,
            _normal: normal,
            _occlusion: occlusion,
            _emissive: emissive,
            _factors: factors,
            alpha_mode: params.alpha_mode,
        };
        Some(Rc::new(Material { name, material_id, data }))
    }

    pub fn for_imgui(
        descriptors: &mut Descriptors,
        name: ArrayString<64>,
        texture: Rc<ImageView>,
        clip_rect: [f32; 4],
        just_alpha: bool,
    ) -> Option<Rc<Material>> {
        let texture_index = descriptors.texture_slots.try_allocate_slot(Rc::downgrade(&texture))? | ((just_alpha as u32) << 16);
        let cmd = Rc::new(ImGuiDrawCmd { clip_rect, texture_index });
        let material_id = descriptors.imgui_cmd_slots.try_allocate_slot(Rc::downgrade(&cmd))?;
        let data = PipelineSpecificData::ImGui { texture, cmd };
        Some(Rc::new(Material { name, material_id, data }))
    }

    pub fn from_existing_imgui_texture(
        descriptors: &mut Descriptors,
        name: ArrayString<64>,
        material: &Material,
        clip_rect: [f32; 4],
    ) -> Option<Rc<Material>> {
        let PipelineSpecificData::ImGui { texture, cmd } = &material.data else {
            return None;
        };
        let cmd = Rc::new(ImGuiDrawCmd { clip_rect, texture_index: cmd.texture_index });
        let material_id = descriptors.imgui_cmd_slots.try_allocate_slot(Rc::downgrade(&cmd))?;
        let data = PipelineSpecificData::ImGui { texture: texture.clone(), cmd };
        Some(Rc::new(Material { name, material_id, data }))
    }

    pub fn pipeline(&self, vertex_layout: VertexLayout) -> PipelineIndex {
        let skinned = vertex_layout == VertexLayout::SkinnedMesh;
        let pipeline = match &self.data {
            PipelineSpecificData::ImGui { .. } => PipelineIndex::ImGui,
            PipelineSpecificData::Pbr { alpha_mode: AlphaMode::Opaque, .. } if skinned => PipelineIndex::PbrSkinnedOpaque,
            PipelineSpecificData::Pbr { alpha_mode: AlphaMode::Opaque, .. } => PipelineIndex::PbrOpaque,
            PipelineSpecificData::Pbr { alpha_mode: AlphaMode::AlphaToCoverage, .. } if skinned => PipelineIndex::PbrSkinnedAlphaToCoverage,
            PipelineSpecificData::Pbr { alpha_mode: AlphaMode::AlphaToCoverage, .. } => PipelineIndex::PbrAlphaToCoverage,
            PipelineSpecificData::Pbr { alpha_mode: AlphaMode::Blended, .. } if skinned => PipelineIndex::PbrSkinnedBlended,
            PipelineSpecificData::Pbr { alpha_mode: AlphaMode::Blended, .. } => PipelineIndex::PbrBlended,
        };
        assert_eq!(vertex_layout, pipeline.vertex_layout(), "the mesh's vertex layout must fit the material's pipeline");
        pipeline
    }
}

impl PartialEq for Material {
    fn eq(&self, other: &Self) -> bool {
        self.material_id == other.material_id && material_id_class(&self.data).eq(&material_id_class(&other.data))
    }
}

impl Eq for Material {}

impl Ord for Material {
    fn cmp(&self, other: &Self) -> Ordering {
        self.material_id.cmp(&other.material_id).then_with(|| material_id_class(&self.data).cmp(&material_id_class(&other.data)))
    }
}

impl PartialOrd for Material {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Returns a different number for each variant. The point of this function is
/// to differentiate two different materials with the same material id (which
/// can happen, since the material ids can overlap).
fn material_id_class(data: &PipelineSpecificData) -> u8 {
    match data {
        PipelineSpecificData::Pbr { .. } => 0,
        PipelineSpecificData::ImGui { .. } => 1,
    }
}
