use alloc::rc::{Rc, Weak};
use core::mem;

use arrayvec::{ArrayString, ArrayVec};
use ash::vk;
use bytemuck::Zeroable;
use enum_map::Enum;

use crate::arena::buffers::ForBuffers;
use crate::arena::{VulkanArena, VulkanArenaError};
use crate::physical_device::PhysicalDevice;
use crate::renderer::pipeline_parameters::constants::{
    MAX_TEXTURE_COUNT, UF_DRAW_CALL_FRAG_PARAMS_BINDING, UF_DRAW_CALL_VERT_PARAMS_BINDING, UF_RENDER_SETTINGS_BINDING,
    UF_TRANSFORMS_BINDING,
};
use crate::renderer::pipeline_parameters::{
    uniforms, DescriptorSetLayoutParams, PipelineIndex, PipelineMap, PipelineParameters, PBR_PIPELINES, PIPELINE_PARAMETERS,
    SKINNED_PIPELINES,
};
use crate::vulkan_raii::{Buffer, DescriptorPool, DescriptorSetLayouts, DescriptorSets, Device, ImageView, PipelineLayout, Sampler};

pub(crate) mod material;

use material::{AlphaMode, Material, PbrFactors, PipelineSpecificData};

pub(crate) const MAX_PIPELINES_PER_MATERIAL: usize = 2;

fn get_pipelines(data: &PipelineSpecificData) -> ArrayVec<PipelineIndex, MAX_PIPELINES_PER_MATERIAL> {
    match data {
        PipelineSpecificData::Pbr { alpha_mode, .. } => match alpha_mode {
            AlphaMode::Opaque => [PipelineIndex::PbrOpaque, PipelineIndex::PbrSkinnedOpaque].into(),
            AlphaMode::AlphaToCoverage => [PipelineIndex::PbrAlphaToCoverage, PipelineIndex::PbrSkinnedAlphaToCoverage].into(),
            AlphaMode::Blend => [PipelineIndex::PbrBlended, PipelineIndex::PbrSkinnedBlended].into(),
        },
    }
}

pub struct PbrDefaults {
    pub base_color: ImageView,
    pub metallic_roughness: ImageView,
    pub normal: ImageView,
    pub occlusion: ImageView,
    pub emissive: ImageView,
}

type MaterialSlot = Option<Weak<Material>>;

#[derive(Default)]
struct PendingWrite<'a> {
    write_descriptor_set: Option<vk::WriteDescriptorSet<'a>>,
    _buffer_info: Option<Box<vk::DescriptorBufferInfo>>,
    _image_info: Option<Box<vk::DescriptorImageInfo>>,
}

pub(crate) struct MaterialTempUniforms {
    pub buffer: Buffer,
    pbr_factors_offsets_and_sizes: PipelineMap<(vk::DeviceSize, vk::DeviceSize)>,
}

const MATERIAL_UPDATES: usize = PipelineIndex::LENGTH * MAX_TEXTURE_COUNT as usize;
/// Descriptor writes:
/// - HDR framebuffer attachment (one post-process descriptor set)
/// - Global transforms (one shared descriptor set)
/// - Render settings (one shared descriptor set)
/// - Joints (one for each of the skinned pipelines)
/// - Material textures and buffers (two for each slot of each pipeline)
const MAX_DESCRIPTOR_WRITES: usize = 3 + SKINNED_PIPELINES.len() + 2 * MATERIAL_UPDATES;

type PendingWritesVec<'a> = ArrayVec<PendingWrite<'a>, MAX_DESCRIPTOR_WRITES>;

pub struct Descriptors {
    pub(crate) pipeline_layouts: PipelineMap<PipelineLayout>,
    descriptor_sets: PipelineMap<DescriptorSets>,
    device: Device,
    pbr_defaults: PbrDefaults,
    // TODO(next?): Use a shared array of materials instead of one array for each descriptor set?
    material_slots_per_pipeline: PipelineMap<ArrayVec<MaterialSlot, { MAX_TEXTURE_COUNT as usize }>>,
    material_updated_per_pipeline: PipelineMap<ArrayVec<bool, { MAX_TEXTURE_COUNT as usize }>>,
    uniform_buffer_offset_alignment: vk::DeviceSize,
}

impl Material {
    /// Reserves a new Material, if there's space.
    pub fn new(descriptors: &mut Descriptors, data: PipelineSpecificData, name: ArrayString<64>) -> Option<Rc<Material>> {
        profiling::scope!("material slot reservation");
        let array_indices = get_pipelines(&data)
            .iter()
            .map(|&pipeline| {
                let i = descriptors.material_slots_per_pipeline[pipeline].iter_mut().position(|slot| slot.is_none())?;
                Some((pipeline, i as u32))
            })
            .collect::<Option<ArrayVec<(PipelineIndex, u32), MAX_PIPELINES_PER_MATERIAL>>>()?;
        let material = Rc::new(Material { name, array_indices, data });
        for &(pipeline, index) in &material.array_indices {
            descriptors.material_slots_per_pipeline[pipeline][index as usize] = Some(Rc::downgrade(&material));
            descriptors.material_updated_per_pipeline[pipeline][index as usize] = false;
        }
        Some(material)
    }
}

impl Descriptors {
    pub fn new(device: &Device, physical_device: &PhysicalDevice, pbr_defaults: PbrDefaults) -> Descriptors {
        profiling::scope!("creating descriptor sets");
        let sampler_create_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(vk::LOD_CLAMP_NONE)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(true)
            .max_anisotropy(physical_device.properties.limits.max_sampler_anisotropy);
        let sampler = unsafe { device.create_sampler(&sampler_create_info, None) }
            .expect("system should have enough memory to create vulkan samplers");
        crate::name_vulkan_object(device, sampler, format_args!("immutable default sampler"));
        let sampler = Rc::new(Sampler { inner: sampler, device: device.clone() });

        let create_descriptor_set_layouts = |pl: PipelineIndex, sets: &[&[DescriptorSetLayoutParams]]| -> DescriptorSetLayouts {
            let samplers_vk = [sampler.inner];
            let samplers_rc = [sampler.clone()].into();

            let descriptor_set_layouts = sets
                .iter()
                .enumerate()
                .map(|(i, bindings)| {
                    let binding_flags =
                        bindings.iter().map(|params| params.binding_flags).collect::<ArrayVec<vk::DescriptorBindingFlags, 8>>();
                    let bindings = bindings
                        .iter()
                        .map(|params| {
                            let mut binding = vk::DescriptorSetLayoutBinding::default()
                                .descriptor_type(params.descriptor_type)
                                .descriptor_count(params.descriptor_count)
                                .stage_flags(params.stage_flags)
                                .binding(params.binding);
                            if params.descriptor_type == vk::DescriptorType::SAMPLER {
                                binding = binding.immutable_samplers(&samplers_vk);
                            }
                            binding
                        })
                        .collect::<ArrayVec<vk::DescriptorSetLayoutBinding, 8>>();
                    let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
                    let create_info = vk::DescriptorSetLayoutCreateInfo::default().push_next(&mut binding_flags).bindings(&bindings);
                    let dsl = unsafe { device.create_descriptor_set_layout(&create_info, None) }
                        .expect("system should have enough memory to create vulkan descriptor set layouts");
                    crate::name_vulkan_object(device, dsl, format_args!("set {i} for pipeline {pl:?}"));
                    dsl
                })
                .collect::<ArrayVec<_, 8>>();

            DescriptorSetLayouts { inner: descriptor_set_layouts, device: device.clone(), immutable_samplers: samplers_rc }
        };

        let pipeline_layouts = PipelineMap::from_fn(|pipeline| {
            let PipelineParameters { descriptor_sets, .. } = &PIPELINE_PARAMETERS[pipeline];
            let descriptor_set_layouts = Rc::new(create_descriptor_set_layouts(pipeline, descriptor_sets));
            let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&descriptor_set_layouts.inner);
            let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }
                .expect("system should have enough memory to create vulkan pipeline layouts");
            crate::name_vulkan_object(device, pipeline_layout, format_args!("for pipeline {pipeline:?}"));
            PipelineLayout { inner: pipeline_layout, device: device.clone(), descriptor_set_layouts: descriptor_set_layouts.clone() }
        });

        let pool_sizes = PIPELINE_PARAMETERS
            .values()
            .flat_map(|params| params.descriptor_sets.iter())
            .flat_map(|set| set.iter())
            .map(|descriptor_layout| {
                vk::DescriptorPoolSize::default().ty(descriptor_layout.descriptor_type).descriptor_count(descriptor_layout.descriptor_count)
            })
            .collect::<ArrayVec<vk::DescriptorPoolSize, { PipelineIndex::LENGTH * 16 }>>();
        let descriptor_sets_per_frame = pipeline_layouts.values().flat_map(|pl| &pl.descriptor_set_layouts.inner).count() as u32;
        let descriptor_pool_create_info =
            vk::DescriptorPoolCreateInfo::default().max_sets(descriptor_sets_per_frame).pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .expect("system should have enough memory to create vulkan descriptor pools")
        };
        crate::name_vulkan_object(device, descriptor_pool, format_args!("the descriptor pool"));
        let descriptor_pool = Rc::new(DescriptorPool { inner: descriptor_pool, device: device.clone() });

        let mut descriptor_set_layouts_per_pipeline = pipeline_layouts.values().map(|pl| &pl.descriptor_set_layouts);
        let descriptor_sets = PipelineMap::from_fn(|pl| {
            let descriptor_set_layouts = descriptor_set_layouts_per_pipeline.next().unwrap();
            let create_info =
                vk::DescriptorSetAllocateInfo::default().descriptor_pool(descriptor_pool.inner).set_layouts(&descriptor_set_layouts.inner);
            let descriptor_sets = unsafe { device.allocate_descriptor_sets(&create_info) }
                .expect("system should have enough memory to allocate descriptor sets");
            for (i, descriptor_set) in descriptor_sets.iter().enumerate() {
                crate::name_vulkan_object(device, *descriptor_set, format_args!("set {i} for pipeline {pl:?}"));
            }
            DescriptorSets { inner: ArrayVec::from_iter(descriptor_sets), device: device.clone(), descriptor_pool: descriptor_pool.clone() }
        });
        drop(descriptor_set_layouts_per_pipeline);

        let material_slots_per_pipeline =
            PipelineMap::from_fn(|_| ArrayVec::from_iter([None].into_iter().cycle().take(MAX_TEXTURE_COUNT as usize)));
        let material_updated_per_pipeline = PipelineMap::from_fn(|_| [true; MAX_TEXTURE_COUNT as usize].into());

        let uniform_buffer_offset_alignment = physical_device.properties.limits.min_uniform_buffer_offset_alignment;

        Descriptors {
            pipeline_layouts,
            descriptor_sets,
            device: device.clone(),
            material_slots_per_pipeline,
            material_updated_per_pipeline,
            pbr_defaults,
            uniform_buffer_offset_alignment,
        }
    }

    pub(crate) fn create_materials_temp_uniform(
        &mut self,
        temp_arena: &mut VulkanArena<ForBuffers>,
    ) -> Result<MaterialTempUniforms, VulkanArenaError> {
        profiling::scope!("creating material temp buffer");

        let mut factors_bytes = Vec::with_capacity(PBR_PIPELINES.len() * mem::size_of::<uniforms::PbrFactors>());
        let mut pbr_factors_offsets_and_sizes = PipelineMap::from_fn(|_| (0, 0));
        for pipeline in PBR_PIPELINES {
            profiling::scope!("copying over pbr factors");

            let factors = self.material_slots_per_pipeline[pipeline]
                .iter()
                .map(|slot| {
                    if let Some(slot) = slot.as_ref().and_then(Weak::upgrade) {
                        let PipelineSpecificData::Pbr { factors, .. } = slot.data;
                        factors
                    } else {
                        PbrFactors::default()
                    }
                })
                .collect::<ArrayVec<_, { MAX_TEXTURE_COUNT as usize }>>();
            if factors.is_empty() {
                continue;
            }

            let mut factors_soa = uniforms::PbrFactors::zeroed();
            for (i, factors) in factors.iter().enumerate() {
                factors_soa.base_color[i] = factors.base_color;
                factors_soa.emissive_and_occlusion[i] = factors.emissive_and_occlusion;
                factors_soa.alpha_rgh_mtl_normal[i] = factors.alpha_rgh_mtl_normal;
            }
            let factors_soa = [factors_soa];
            let factors_soa = bytemuck::cast_slice(&factors_soa);

            let offset = (factors_bytes.len() as vk::DeviceSize).next_multiple_of(self.uniform_buffer_offset_alignment);
            pbr_factors_offsets_and_sizes[pipeline] = (offset, factors_soa.len() as vk::DeviceSize);
            factors_bytes.resize(offset as usize, 0);
            factors_bytes.extend_from_slice(factors_soa);
        }

        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(factors_bytes.len() as u64)
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer =
            temp_arena.create_buffer(buffer_create_info, &factors_bytes, None, None, format_args!("uniform (temp material buffer)"))?;

        Ok(MaterialTempUniforms { buffer, pbr_factors_offsets_and_sizes })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_descriptors(
        &mut self,
        global_transforms_buffer: &Buffer,
        render_settings_buffer: &Buffer,
        skinned_mesh_joints_buffer: &Buffer,
        draw_call_vert_params: &Buffer,
        draw_call_frag_params: &Buffer,
        material_buffers: &MaterialTempUniforms,
        hdr_attachment: &ImageView,
    ) {
        profiling::scope!("updating descriptors");

        const UPDATE_COUNT: usize = PipelineIndex::LENGTH * MAX_TEXTURE_COUNT as usize;
        let mut materials_needing_update = ArrayVec::<_, UPDATE_COUNT>::new();
        for (pipeline, material_slots) in &self.material_slots_per_pipeline {
            for (i, material_slot) in material_slots.iter().enumerate() {
                if let Some(material) = material_slot.as_ref().and_then(Weak::upgrade) {
                    let written = self.material_updated_per_pipeline[pipeline][i];
                    if !written {
                        materials_needing_update.push((pipeline, i as u32, material));
                    }
                }
            }
        }

        let mut pending_writes = PendingWritesVec::new();

        // 0 is the index of the HDR attachment.
        self.set_uniform_images(PipelineIndex::RenderResolutionPostProcess, &mut pending_writes, (1, 0, 0), &[hdr_attachment.inner]);

        let shared_pipeline = PipelineIndex::SHARED_DESCRIPTOR_PIPELINE;
        let global_transforms_buffer = (global_transforms_buffer.inner, 0, global_transforms_buffer.size);
        self.set_uniform_buffer(shared_pipeline, &mut pending_writes, (0, UF_TRANSFORMS_BINDING, 0), global_transforms_buffer);
        let render_settings_buffer = (render_settings_buffer.inner, 0, render_settings_buffer.size);
        self.set_uniform_buffer(shared_pipeline, &mut pending_writes, (0, UF_RENDER_SETTINGS_BINDING, 0), render_settings_buffer);
        let vert_params_buffer = (draw_call_vert_params.inner, 0, draw_call_vert_params.size);
        self.set_uniform_buffer(shared_pipeline, &mut pending_writes, (0, UF_DRAW_CALL_VERT_PARAMS_BINDING, 0), vert_params_buffer);
        let frag_params_buffer = (draw_call_frag_params.inner, 0, draw_call_frag_params.size);
        self.set_uniform_buffer(shared_pipeline, &mut pending_writes, (0, UF_DRAW_CALL_FRAG_PARAMS_BINDING, 0), frag_params_buffer);

        for pipeline in SKINNED_PIPELINES {
            let skinned_mesh_joints_buffer = (skinned_mesh_joints_buffer.inner, 0, skinned_mesh_joints_buffer.size);
            self.set_uniform_buffer(pipeline, &mut pending_writes, (2, 0, 0), skinned_mesh_joints_buffer);
        }

        for pipeline in PBR_PIPELINES {
            let pbr_factors_offset_and_size = material_buffers.pbr_factors_offsets_and_sizes[pipeline];
            let (offset, size) = pbr_factors_offset_and_size;
            let buffer = (material_buffers.buffer.inner, offset, size);
            self.set_uniform_buffer(pipeline, &mut pending_writes, (1, 6, 0), buffer);
        }

        for (pipeline, i, material) in &materials_needing_update {
            self.write_material(*pipeline, *i, material, &mut pending_writes);
            self.material_updated_per_pipeline[*pipeline][*i as usize] = true;
        }

        // NOTE: pending_writes owns the image/buffers that are pointed to by
        // the write_descriptor_sets, and they need to be dropped only after
        // update_descriptor_sets.
        let mut writes = ArrayVec::<vk::WriteDescriptorSet, MAX_DESCRIPTOR_WRITES>::new();
        for pending_write in &pending_writes {
            writes.push(pending_write.write_descriptor_set.unwrap());
        }
        {
            profiling::scope!("vk::update_descriptor_sets");
            unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        }
        drop(pending_writes);
        drop(materials_needing_update);
    }

    fn write_material(&self, pipeline: PipelineIndex, index: u32, material: &Material, pending_writes: &mut PendingWritesVec) {
        match &material.data {
            PipelineSpecificData::Pbr { base_color, metallic_roughness, normal, occlusion, emissive, .. } => {
                let images = [
                    base_color.as_ref().map_or(&self.pbr_defaults.base_color, Rc::as_ref).inner,
                    metallic_roughness.as_ref().map_or(&self.pbr_defaults.metallic_roughness, Rc::as_ref).inner,
                    normal.as_ref().map_or(&self.pbr_defaults.normal, Rc::as_ref).inner,
                    occlusion.as_ref().map_or(&self.pbr_defaults.occlusion, Rc::as_ref).inner,
                    emissive.as_ref().map_or(&self.pbr_defaults.emissive, Rc::as_ref).inner,
                ];
                self.set_uniform_images(pipeline, pending_writes, (1, 1, index), &images);
            }
        }
    }

    #[profiling::function]
    fn set_uniform_buffer(
        &self,
        pipeline: PipelineIndex,
        pending_writes: &mut PendingWritesVec,
        (set, binding, array_index): (u32, u32, u32),
        (buffer, offset, size): (vk::Buffer, vk::DeviceSize, vk::DeviceSize),
    ) {
        let set_idx = set as usize;
        let descriptor_set = self.descriptor_sets[pipeline].inner[set_idx];
        let descriptor_buffer_info = Box::new(vk::DescriptorBufferInfo::default().buffer(buffer).offset(offset).range(size));
        let params = &PIPELINE_PARAMETERS[pipeline].descriptor_sets[set_idx][binding as usize];
        let write_descriptor_set = vk::WriteDescriptorSet {
            dst_set: descriptor_set,
            dst_binding: binding,
            dst_array_element: array_index,
            descriptor_type: params.descriptor_type,
            p_buffer_info: &*descriptor_buffer_info,
            descriptor_count: 1,
            ..Default::default()
        };
        pending_writes.push(PendingWrite {
            write_descriptor_set: Some(write_descriptor_set),
            _buffer_info: Some(descriptor_buffer_info),
            ..Default::default()
        });
        if let Some(max_size) = params.descriptor_size {
            if size > max_size {
                log::warn!(
                    "Writing {size} bytes to uniform buffer while the expected maximum size is {max_size} bytes! \
                    (pipeline {pipeline:?} set {set} binding {binding} index {array_index})"
                );
            }
        } else {
            log::warn!(
                "Uniform buffer descriptor is missing the descriptor_size hint! \
                (pipeline {pipeline:?} set {set} binding {binding} index {array_index})"
            );
        }
    }

    /// Uploads the image_views to the given set, starting at
    /// first_binding.
    #[profiling::function]
    fn set_uniform_images(
        &self,
        pipeline: PipelineIndex,
        pending_writes: &mut PendingWritesVec,
        (set, first_binding, array_index): (u32, u32, u32),
        image_views: &[vk::ImageView],
    ) {
        let set_idx = set as usize;
        let descriptor_set = self.descriptor_sets[pipeline].inner[set_idx];
        for (i, image_view) in image_views.iter().enumerate() {
            let descriptor_image_info = Box::new(
                vk::DescriptorImageInfo::default().image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL).image_view(*image_view),
            );
            let binding = first_binding + i as u32;
            let params = &PIPELINE_PARAMETERS[pipeline].descriptor_sets[set_idx][binding as usize];
            let write_descriptor_set = vk::WriteDescriptorSet {
                dst_set: descriptor_set,
                dst_binding: binding,
                dst_array_element: array_index,
                descriptor_type: params.descriptor_type,
                p_image_info: &*descriptor_image_info,
                descriptor_count: 1,
                ..Default::default()
            };
            pending_writes.push(PendingWrite {
                write_descriptor_set: Some(write_descriptor_set),
                _image_info: Some(descriptor_image_info),
                ..Default::default()
            });
        }
    }

    #[profiling::function]
    pub(crate) fn descriptor_sets(&self, pipeline: PipelineIndex) -> &[vk::DescriptorSet] {
        &self.descriptor_sets[pipeline].inner
    }
}
