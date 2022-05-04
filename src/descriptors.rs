use crate::arena::{VulkanArena, VulkanArenaError};
use crate::image_loading::{ImageLoadingError, PbrDefaults};
use crate::memory_measurement::{VulkanArenaMeasurementError, VulkanArenaMeasurer};
use crate::pipeline_parameters::{
    DescriptorSetLayoutParams, PipelineIndex, PipelineMap, PipelineParameters, PushConstantStruct, MAX_TEXTURE_COUNT, PIPELINE_PARAMETERS,
};
use crate::uploader::{Uploader, UploaderError};
use crate::vulkan_raii::{
    Buffer, DescriptorPool, DescriptorSetLayouts, DescriptorSets, Device, Framebuffer, ImageView, PipelineLayout, Sampler,
};
use crate::{debug_utils, Instance, PhysicalDevice};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Vec4;
use std::hash::{Hash, Hasher};
use std::mem;
use std::rc::{Rc, Weak};
use std::time::Duration;

#[derive(thiserror::Error, Debug)]
pub enum DescriptorError {
    #[error("material indices have been exhausted (max: {MAX_TEXTURE_COUNT}), cannot create more materials before releasing old ones")]
    MaterialIndexReserve,
    #[error("failed to create immutable sampler")]
    ImmutableSamplerCreation(#[source] vk::Result),
    #[error("failed to create descriptor set layout")]
    DescriptorSetLayoutCreation(#[source] vk::Result),
    #[error("failed to create pipeline layout")]
    PipelineLayoutCreation(#[source] vk::Result),
    #[error("failed to create descriptor pool")]
    DescriptorPoolCreation(#[source] vk::Result),
    #[error("failed to allocate descriptor sets")]
    AllocateDescriptorSets(#[source] vk::Result),
    #[error("failed to measure memory requirements for the fallback textures for materials")]
    MeasureMaterialDefaultTextures(#[source] VulkanArenaMeasurementError),
    #[error("failed to create an uploader for the fallback textures for materials")]
    CreateMaterialDefaultTexturesUploader(#[source] UploaderError),
    #[error("failed to allocate memory for the fallback textures for materials")]
    AllocateMaterialDefaultTextures(#[source] VulkanArenaError),
    #[error("failed to create fallback textures for materials")]
    CreateMaterialDefaultTextures(#[source] ImageLoadingError),
    #[error("an error occurred while waiting for the upload of fallback textures for materials")]
    WaitForMaterialDefaultTexturesUpload(#[source] UploaderError),
}

/// A unique index into one pipeline's textures and other material data.
pub struct Material {
    pub name: String,
    pub array_index: u32,
    pub pipeline: PipelineIndex,
    data: PipelineSpecificData,
}

#[derive(Clone)]
pub enum PipelineSpecificData {
    Gltf {
        base_color: Option<Rc<ImageView>>,
        metallic_roughness: Option<Rc<ImageView>>,
        normal: Option<Rc<ImageView>>,
        occlusion: Option<Rc<ImageView>>,
        emissive: Option<Rc<ImageView>>,
        /// (Buffer, offset, size) that contains a [GltfFactors].
        factors: (Rc<Buffer>, vk::DeviceSize, vk::DeviceSize),
    },
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct GltfFactors {
    /// (r, g, b, a).
    pub base_color: Vec4,
    /// (r, g, b, _). Vec4 to make sure there's no padding/alignment issues.
    pub emissive: Vec4,
    /// (metallic, roughness, alpha_cutoff, _). Vec4 to make sure there's no padding.
    pub metallic_roughness_alpha_cutoff: Vec4,
}
// Mat4's are Pods, therefore they are Zeroable, therefore this is too.
unsafe impl Zeroable for GltfFactors {}
// repr(c), the contents are Pods, and there's no padding.
unsafe impl Pod for GltfFactors {}

impl PartialEq for Material {
    fn eq(&self, other: &Self) -> bool {
        self.pipeline == other.pipeline && self.array_index == other.array_index
    }
}

impl Eq for Material {}

impl Hash for Material {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pipeline.hash(state);
        self.array_index.hash(state);
    }
}

impl Material {
    pub fn new(
        descriptors: &mut Descriptors,
        pipeline: PipelineIndex,
        data: PipelineSpecificData,
        name: String,
    ) -> Result<Rc<Material>, DescriptorError> {
        profiling::scope!("material slot reservation");
        let material_slot_array = &mut descriptors.material_slots_per_pipeline[pipeline];
        if let Some((i, slot)) = material_slot_array.iter_mut().enumerate().find(|(_, slot)| slot.is_none()) {
            let material = Rc::new(Material {
                name,
                pipeline,
                array_index: i as u32,
                data,
            });
            descriptors.material_updated_per_pipeline[pipeline][i] = false;
            *slot = Some(Rc::downgrade(&material));
            Ok(material)
        } else {
            Err(DescriptorError::MaterialIndexReserve)
        }
    }
}

type MaterialSlot = Option<Weak<Material>>;

#[derive(Default)]
struct PendingWrite {
    write_descriptor_set: Option<vk::WriteDescriptorSet>,
    _buffer_info: Option<Box<vk::DescriptorBufferInfo>>,
    _image_info: Option<Box<vk::DescriptorImageInfo>>,
}

pub struct Descriptors {
    pub(crate) pipeline_layouts: PipelineMap<PipelineLayout>,
    descriptor_sets: PipelineMap<DescriptorSets>,
    device: Device,
    material_slots_per_pipeline: PipelineMap<Vec<MaterialSlot>>,
    material_updated_per_pipeline: PipelineMap<Vec<bool>>,
    pbr_defaults: PbrDefaults,
}

impl Descriptors {
    pub fn new(instance: &Instance, device: &Device, physical_device: &PhysicalDevice) -> Result<Descriptors, DescriptorError> {
        profiling::scope!("creating descriptor sets");
        let sampler_create_info = vk::SamplerCreateInfo::builder()
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
        let sampler = unsafe { device.create_sampler(&sampler_create_info, None) }.map_err(DescriptorError::ImmutableSamplerCreation)?;
        debug_utils::name_vulkan_object(device, sampler, format_args!("immutable default sampler"));
        let sampler = Rc::new(Sampler {
            inner: sampler,
            device: device.clone(),
        });

        let create_descriptor_set_layouts =
            |pl: PipelineIndex, sets: &[&[DescriptorSetLayoutParams]]| -> Result<DescriptorSetLayouts, DescriptorError> {
                let samplers_vk = [sampler.inner];
                let samplers_rc = vec![sampler.clone()];

                let descriptor_set_layouts = sets
                    .iter()
                    .enumerate()
                    .map(|(i, bindings)| {
                        let binding_flags = bindings
                            .iter()
                            .map(|params| params.binding_flags)
                            .collect::<Vec<vk::DescriptorBindingFlags>>();
                        let bindings = bindings
                            .iter()
                            .map(|params| {
                                let mut builder = vk::DescriptorSetLayoutBinding::builder()
                                    .descriptor_type(params.descriptor_type)
                                    .descriptor_count(params.descriptor_count)
                                    .stage_flags(params.stage_flags)
                                    .binding(params.binding);
                                if params.descriptor_type == vk::DescriptorType::SAMPLER {
                                    builder = builder.immutable_samplers(&samplers_vk);
                                }
                                builder.build()
                            })
                            .collect::<Vec<vk::DescriptorSetLayoutBinding>>();
                        let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder().binding_flags(&binding_flags);
                        let create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                            .push_next(&mut binding_flags)
                            .bindings(&bindings);
                        let dsl = unsafe { device.create_descriptor_set_layout(&create_info, None) }
                            .map_err(DescriptorError::DescriptorSetLayoutCreation)?;
                        debug_utils::name_vulkan_object(device, dsl, format_args!("set {i} for pipeline {pl:?}"));
                        Ok(dsl)
                    })
                    .collect::<Result<Vec<vk::DescriptorSetLayout>, DescriptorError>>()?;

                Ok(DescriptorSetLayouts {
                    inner: descriptor_set_layouts,
                    device: device.clone(),
                    immutable_samplers: samplers_rc,
                })
            };

        let pipeline_layouts = PipelineMap::new::<DescriptorError, _>(|pipeline| {
            let PipelineParameters { descriptor_sets, .. } = &PIPELINE_PARAMETERS[pipeline];
            let descriptor_set_layouts = Rc::new(create_descriptor_set_layouts(pipeline, descriptor_sets)?);
            let push_constant_ranges = [vk::PushConstantRange::builder()
                .offset(0)
                .size(mem::size_of::<PushConstantStruct>() as u32)
                .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                .build()];
            let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&descriptor_set_layouts.inner)
                .push_constant_ranges(&push_constant_ranges);
            let pipeline_layout = unsafe { device.create_pipeline_layout(&pipeline_layout_create_info, None) }
                .map_err(DescriptorError::PipelineLayoutCreation)?;
            debug_utils::name_vulkan_object(device, pipeline_layout, format_args!("for pipeline {pipeline:?}"));
            let pipeline_layout = PipelineLayout {
                inner: pipeline_layout,
                device: device.clone(),
                descriptor_set_layouts: descriptor_set_layouts.clone(),
            };
            Ok(pipeline_layout)
        })?;

        let pool_sizes = PIPELINE_PARAMETERS
            .iter()
            .flat_map(|params| params.descriptor_sets.iter())
            .flat_map(|set| set.iter())
            .map(|descriptor_layout| {
                vk::DescriptorPoolSize::builder()
                    .ty(descriptor_layout.descriptor_type)
                    .descriptor_count(descriptor_layout.descriptor_count)
                    .build()
            })
            .collect::<Vec<vk::DescriptorPoolSize>>();
        let descriptor_sets_per_frame = pipeline_layouts.iter().flat_map(|pl| &pl.descriptor_set_layouts.inner).count() as u32;
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(descriptor_sets_per_frame)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .map_err(DescriptorError::DescriptorPoolCreation)
        }?;
        debug_utils::name_vulkan_object(device, descriptor_pool, format_args!("the descriptor pool"));
        let descriptor_pool = Rc::new(DescriptorPool {
            inner: descriptor_pool,
            device: device.clone(),
        });

        let mut descriptor_set_layouts_per_pipeline = pipeline_layouts.iter().map(|pl| &pl.descriptor_set_layouts);
        let descriptor_sets = PipelineMap::new::<DescriptorError, _>(|pl| {
            let descriptor_set_layouts = descriptor_set_layouts_per_pipeline.next().unwrap();
            let create_info = vk::DescriptorSetAllocateInfo::builder()
                .descriptor_pool(descriptor_pool.inner)
                .set_layouts(&descriptor_set_layouts.inner);
            let descriptor_sets =
                unsafe { device.allocate_descriptor_sets(&create_info) }.map_err(DescriptorError::AllocateDescriptorSets)?;
            for (i, descriptor_set) in descriptor_sets.iter().enumerate() {
                debug_utils::name_vulkan_object(device, *descriptor_set, format_args!("set {i} for pipeline {pl:?}"));
            }
            Ok(DescriptorSets {
                inner: descriptor_sets,
                device: device.clone(),
                descriptor_pool: descriptor_pool.clone(),
            })
        })?;
        drop(descriptor_set_layouts_per_pipeline);

        let material_slots_per_pipeline = PipelineMap::new::<DescriptorError, _>(|_| Ok(vec![None; MAX_TEXTURE_COUNT as usize]))?;
        let material_status_per_pipeline = PipelineMap::new::<DescriptorError, _>(|_| Ok(vec![true; MAX_TEXTURE_COUNT as usize]))?;

        let mut pbr_defaults_measurer = VulkanArenaMeasurer::new(device);
        PbrDefaults::measure(&mut pbr_defaults_measurer).map_err(DescriptorError::MeasureMaterialDefaultTextures)?;
        let mut uploader = Uploader::new(
            &instance.inner,
            device,
            device.graphics_queue,
            device.transfer_queue,
            physical_device,
            pbr_defaults_measurer.measured_size,
            "fallback materials",
        )
        .map_err(DescriptorError::CreateMaterialDefaultTexturesUploader)?;
        // TODO(low): Do something about the pbr defaults arena having its own tiny allocation
        let mut pbr_defaults_arena = VulkanArena::new(
            &instance.inner,
            device,
            physical_device,
            pbr_defaults_measurer.measured_size,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            format_args!("fallback materials (textures)"),
        )
        .map_err(DescriptorError::AllocateMaterialDefaultTextures)?;
        let pbr_defaults =
            PbrDefaults::new(device, &mut uploader, &mut pbr_defaults_arena).map_err(DescriptorError::CreateMaterialDefaultTextures)?;
        while !uploader.wait(None).map_err(DescriptorError::WaitForMaterialDefaultTexturesUpload)? {
            log::warn!("Waiting for u64::MAX ns timed out?");
            std::thread::sleep(Duration::from_millis(10));
        }

        Ok(Descriptors {
            pipeline_layouts,
            descriptor_sets,
            device: device.clone(),
            material_slots_per_pipeline,
            material_updated_per_pipeline: material_status_per_pipeline,
            pbr_defaults,
        })
    }

    pub(crate) fn write_descriptors(&mut self, global_transforms_buffer: &Buffer, framebuffer: &Framebuffer) {
        profiling::scope!("updating descriptors");

        let mut materials_needing_update = Vec::new();
        for (pipeline, material_slots) in self.material_slots_per_pipeline.iter_with_pipeline() {
            for (i, material_slot) in material_slots.iter().enumerate() {
                if let Some(material) = material_slot.as_ref().and_then(Weak::upgrade) {
                    let written = self.material_updated_per_pipeline[pipeline][i];
                    if !written {
                        materials_needing_update.push((pipeline, i, material));
                    }
                }
            }
        }

        let mut pending_writes = Vec::with_capacity(materials_needing_update.len() * 6 + 1);

        // 0 is the index of the HDR attachment.
        let framebuffer_hdr_view = [framebuffer.attachments[0].inner];
        self.set_uniform_images(
            PipelineIndex::RenderResolutionPostProcess,
            &mut pending_writes,
            (0, 0, 0),
            &framebuffer_hdr_view,
        );

        let shared_pipeline = PipelineIndex::SHARED_DESCRIPTOR_PIPELINE;
        let global_transforms_buffer = (global_transforms_buffer.inner, 0, vk::WHOLE_SIZE);
        self.set_uniform_buffer(shared_pipeline, &mut pending_writes, (0, 0, 0), global_transforms_buffer);

        for (pipeline, i, material) in &materials_needing_update {
            self.write_material(*pipeline, material, &mut pending_writes);
            self.material_updated_per_pipeline[*pipeline][*i] = true;
        }

        let mut writes = Vec::with_capacity(pending_writes.len());
        for pending_write in &pending_writes {
            writes.push(pending_write.write_descriptor_set.unwrap());
        }
        {
            profiling::scope!("vk::update_descriptor_sets");
            unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        }
        drop(materials_needing_update);
    }

    fn write_material(&self, pipeline: PipelineIndex, material: &Material, pending_writes: &mut Vec<PendingWrite>) {
        match &material.data {
            PipelineSpecificData::Gltf {
                base_color,
                metallic_roughness,
                normal,
                occlusion,
                emissive,
                factors,
            } => {
                let images = [
                    base_color.as_ref().map(Rc::as_ref).unwrap_or(&self.pbr_defaults.base_color).inner,
                    metallic_roughness
                        .as_ref()
                        .map(Rc::as_ref)
                        .unwrap_or(&self.pbr_defaults.metallic_roughness)
                        .inner,
                    normal.as_ref().map(Rc::as_ref).unwrap_or(&self.pbr_defaults.normal).inner,
                    occlusion.as_ref().map(Rc::as_ref).unwrap_or(&self.pbr_defaults.occlusion).inner,
                    emissive.as_ref().map(Rc::as_ref).unwrap_or(&self.pbr_defaults.emissive).inner,
                ];
                let factors = (factors.0.inner, factors.1, factors.2);
                self.set_uniform_images(pipeline, pending_writes, (1, 1, material.array_index), &images);
                self.set_uniform_buffer(pipeline, pending_writes, (1, 6, material.array_index), factors);
            }
        }
    }

    #[profiling::function]
    fn set_uniform_buffer(
        &self,
        pipeline: PipelineIndex,
        pending_writes: &mut Vec<PendingWrite>,
        (set, binding, array_index): (u32, u32, u32),
        (buffer, offset, size): (vk::Buffer, vk::DeviceSize, vk::DeviceSize),
    ) {
        let set_idx = set as usize;
        let descriptor_set = self.descriptor_sets[pipeline].inner[set_idx];
        let descriptor_buffer_info = Box::new(
            vk::DescriptorBufferInfo::builder()
                .buffer(buffer)
                .offset(offset)
                .range(size)
                .build(),
        );
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
    }

    /// Uploads the image_views to the given set, starting at
    /// first_binding.
    #[profiling::function]
    fn set_uniform_images(
        &self,
        pipeline: PipelineIndex,
        pending_writes: &mut Vec<PendingWrite>,
        (set, first_binding, array_index): (u32, u32, u32),
        image_views: &[vk::ImageView],
    ) {
        let set_idx = set as usize;
        let descriptor_set = self.descriptor_sets[pipeline].inner[set_idx];
        for (i, image_view) in image_views.iter().enumerate() {
            let descriptor_image_info = Box::new(
                vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(*image_view)
                    .build(),
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
