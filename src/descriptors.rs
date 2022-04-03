use crate::image_loading::{self, TextureKind};
use crate::pipeline_parameters::{
    DescriptorSetLayoutParams, PipelineIndex, PipelineMap, PipelineParameters, PushConstantStruct, MAX_TEXTURE_COUNT, PIPELINE_PARAMETERS,
};
use crate::vulkan_raii::{Buffer, DescriptorPool, DescriptorSetLayouts, DescriptorSets, Device, ImageView, PipelineLayout, Sampler};
use crate::{debug_utils, Error, ForImages, FrameIndex, Uploader, VulkanArena};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::Vec4;
use std::hash::{Hash, Hasher};
use std::mem;
use std::rc::{Rc, Weak};

/// A unique index into one pipeline's textures and other material data.
pub struct Material {
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
    /// (r, g, b, _). Vec4 to make sure there's no padding/alignment issues.
    pub base_color: Vec4,
    /// (r, g, b, _). Vec4 to make sure there's no padding/alignment issues.
    pub emissive: Vec4,
    /// (metallic, roughness, _, _). Vec4 to make sure there's no padding.
    pub metallic_roughness: Vec4,
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
    pub fn new(descriptors: &mut Descriptors, pipeline: PipelineIndex, data: PipelineSpecificData) -> Result<Rc<Material>, Error> {
        profiling::scope!("material slot reservation");
        let material_slot_array = descriptors.material_slots_per_pipeline.get_mut(pipeline);
        if let Some((i, slot)) = material_slot_array.iter_mut().enumerate().find(|(_, slot)| slot.is_none()) {
            let material = Rc::new(Material {
                pipeline,
                array_index: i as u32,
                data,
            });
            for status in &mut descriptors.material_status_per_pipeline.get_mut(pipeline)[i] {
                *status = false;
            }
            *slot = Some(Rc::downgrade(&material));
            Ok(material)
        } else {
            Err(Error::MaterialIndexReserve)
        }
    }
}

type MaterialSlot = Option<Weak<Material>>;
type MaterialSlotArray = Vec<MaterialSlot>;

/// Synchronization status per frame index. True at index i means the uniforms
/// for the frame index i are up to date.
type MaterialStatus = Vec<bool>;
type MaterialStatusArray = Vec<MaterialStatus>;

pub struct PbrDefaults {
    base_color: ImageView,
    metallic_roughness: ImageView,
    normal: ImageView,
    occlusion: ImageView,
    emissive: ImageView,
}

impl PbrDefaults {
    pub fn new(device: &Rc<Device>, uploader: &mut Uploader, arena: &mut VulkanArena<ForImages>) -> Result<PbrDefaults, Error> {
        profiling::scope!("pbr default textures creation");

        const WHITE: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF];
        const BLACK: [u8; 4] = [0, 0, 0, 0xFF];
        const NORMAL_Z: [u8; 4] = [0, 0, 0xFF, 0];
        const M_AND_R: [u8; 4] = [0, 0x88, 0, 0];

        let mut create_pixel = |color, kind, name| image_loading::create_pixel(device, uploader, arena, color, kind, name);
        let base_color = create_pixel(WHITE, TextureKind::SrgbColor, "default pbr base color")?;
        let metallic_roughness = create_pixel(M_AND_R, TextureKind::LinearColor, "default pbr metallic/roughness")?;
        let normal = create_pixel(NORMAL_Z, TextureKind::NormalMap, "default pbr normals")?;
        let occlusion = create_pixel(WHITE, TextureKind::LinearColor, "default pbr occlusion")?;
        let emissive = create_pixel(BLACK, TextureKind::SrgbColor, "default pbr emissive")?;

        Ok(PbrDefaults {
            base_color,
            metallic_roughness,
            normal,
            occlusion,
            emissive,
        })
    }
}

#[derive(Default)]
struct PendingWrite {
    write_descriptor_set: Option<vk::WriteDescriptorSet>,
    _buffer_info: Option<Box<vk::DescriptorBufferInfo>>,
    _image_info: Option<Box<vk::DescriptorImageInfo>>,
}

pub struct Descriptors {
    pub frame_count: u32,
    pub(crate) pipeline_layouts: PipelineMap<PipelineLayout>,
    descriptor_sets: Vec<PipelineMap<DescriptorSets>>,
    device: Rc<Device>,
    material_slots_per_pipeline: PipelineMap<MaterialSlotArray>,
    material_status_per_pipeline: PipelineMap<MaterialStatusArray>,
    pbr_defaults: PbrDefaults,
    physical_device_properties: vk::PhysicalDeviceProperties,
}

impl Descriptors {
    pub fn new(
        device: &Rc<Device>,
        physical_device_properties: vk::PhysicalDeviceProperties,
        pbr_defaults: PbrDefaults,
        frame_count: u32,
    ) -> Result<Descriptors, Error> {
        profiling::scope!("creating descriptor sets");
        Descriptors::new_(device, physical_device_properties, pbr_defaults, None, frame_count)
    }

    pub fn from_existing(old_descriptors: Descriptors, frame_count: u32) -> Result<Descriptors, Error> {
        profiling::scope!("recreating descriptor sets");
        let Descriptors {
            pipeline_layouts,
            device,
            physical_device_properties,
            pbr_defaults,
            material_slots_per_pipeline,
            descriptor_sets,
            ..
        } = old_descriptors;
        drop(descriptor_sets);
        let mut new_descriptors = Descriptors::new_(
            &device,
            physical_device_properties,
            pbr_defaults,
            Some(pipeline_layouts),
            frame_count,
        )?;
        for (pipeline, material_slots) in material_slots_per_pipeline.iter_with_pipeline() {
            for (i, material_slot) in material_slots.iter().enumerate() {
                if material_slot.is_none() || material_slot.as_ref().unwrap().strong_count() == 0 {
                    continue;
                }
                for status in &mut new_descriptors.material_status_per_pipeline.get_mut(pipeline)[i] {
                    *status = false;
                }
            }
        }
        new_descriptors.material_slots_per_pipeline = material_slots_per_pipeline;
        Ok(new_descriptors)
    }

    fn new_(
        device: &Rc<Device>,
        physical_device_properties: vk::PhysicalDeviceProperties,
        pbr_defaults: PbrDefaults,
        pipeline_layouts: Option<PipelineMap<PipelineLayout>>,
        frame_count: u32,
    ) -> Result<Descriptors, Error> {
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
            .max_anisotropy(physical_device_properties.limits.max_sampler_anisotropy);
        let sampler = unsafe { device.create_sampler(&sampler_create_info, None) }.map_err(Error::VulkanSamplerCreation)?;
        debug_utils::name_vulkan_object(device, sampler, format_args!("immutable default sampler"));
        let sampler = Rc::new(Sampler {
            inner: sampler,
            device: device.clone(),
        });

        let create_descriptor_set_layouts =
            |pl: PipelineIndex, sets: &[&[DescriptorSetLayoutParams]]| -> Result<DescriptorSetLayouts, Error> {
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
                            .map_err(Error::VulkanDescriptorSetLayoutCreation)?;
                        debug_utils::name_vulkan_object(device, dsl, format_args!("set {i} for pipeline {pl:?}"));
                        Ok(dsl)
                    })
                    .collect::<Result<Vec<vk::DescriptorSetLayout>, Error>>()?;

                Ok(DescriptorSetLayouts {
                    inner: descriptor_set_layouts,
                    device: device.clone(),
                    immutable_samplers: samplers_rc,
                })
            };

        let pipeline_layouts = pipeline_layouts.map(Ok).unwrap_or_else(|| {
            PipelineMap::new::<Error, _>(|pipeline| {
                let PipelineParameters { descriptor_sets, .. } = PIPELINE_PARAMETERS.get(pipeline);
                let descriptor_set_layouts = Rc::new(create_descriptor_set_layouts(pipeline, descriptor_sets)?);
                let push_constant_ranges = [vk::PushConstantRange::builder()
                    .offset(0)
                    .size(mem::size_of::<PushConstantStruct>() as u32)
                    .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                    .build()];
                let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&descriptor_set_layouts.inner)
                    .push_constant_ranges(&push_constant_ranges);
                let pipeline_layout = unsafe {
                    device
                        .create_pipeline_layout(&pipeline_layout_create_info, None)
                        .map_err(Error::VulkanPipelineLayoutCreation)
                }?;
                debug_utils::name_vulkan_object(device, pipeline_layout, format_args!("for pipeline {pipeline:?}"));
                let pipeline_layout = PipelineLayout {
                    inner: pipeline_layout,
                    device: device.clone(),
                    descriptor_set_layouts: descriptor_set_layouts.clone(),
                };
                Ok(pipeline_layout)
            })
        })?;

        let pool_sizes = PIPELINE_PARAMETERS
            .iter()
            .flat_map(|params| params.descriptor_sets.iter())
            .flat_map(|set| set.iter())
            .map(|descriptor_layout| {
                vk::DescriptorPoolSize::builder()
                    .ty(descriptor_layout.descriptor_type)
                    .descriptor_count(descriptor_layout.descriptor_count * frame_count)
                    .build()
            })
            .collect::<Vec<vk::DescriptorPoolSize>>();
        let descriptor_sets_per_frame = pipeline_layouts.iter().flat_map(|pl| &pl.descriptor_set_layouts.inner).count() as u32;
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(frame_count * descriptor_sets_per_frame)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .map_err(Error::VulkanDescriptorPoolCreation)
        }?;
        debug_utils::name_vulkan_object(device, descriptor_pool, format_args!("main pool ({frame_count} frames)"));
        let descriptor_pool = Rc::new(DescriptorPool {
            inner: descriptor_pool,
            device: device.clone(),
        });

        let descriptor_sets = (1..frame_count + 1)
            .map(|nth| {
                profiling::scope!("descriptor set allocation for a frame");
                let mut descriptor_set_layouts_per_pipeline = pipeline_layouts.iter().map(|pl| &pl.descriptor_set_layouts);
                let pipeline_map = PipelineMap::new::<Error, _>(|pl| {
                    let descriptor_set_layouts = descriptor_set_layouts_per_pipeline.next().unwrap();
                    let create_info = vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool.inner)
                        .set_layouts(&descriptor_set_layouts.inner);
                    let descriptor_sets =
                        unsafe { device.allocate_descriptor_sets(&create_info) }.map_err(Error::VulkanAllocateDescriptorSets)?;
                    for (i, descriptor_set) in descriptor_sets.iter().enumerate() {
                        debug_utils::name_vulkan_object(
                            device,
                            *descriptor_set,
                            format_args!("set {i} for pipeline {pl:?} (frame {nth}/{frame_count})"),
                        );
                    }
                    Ok(DescriptorSets {
                        inner: descriptor_sets,
                        device: device.clone(),
                        descriptor_pool: descriptor_pool.clone(),
                    })
                })?;
                Ok(pipeline_map)
            })
            .collect::<Result<Vec<PipelineMap<DescriptorSets>>, Error>>()?;

        let material_slots_per_pipeline = PipelineMap::new::<Error, _>(|_| Ok(vec![None; MAX_TEXTURE_COUNT as usize]))?;
        let material_status_per_pipeline =
            PipelineMap::new::<Error, _>(|_| Ok(vec![vec![true; frame_count as usize]; MAX_TEXTURE_COUNT as usize]))?;

        Ok(Descriptors {
            frame_count,
            pipeline_layouts,
            descriptor_sets,
            device: device.clone(),
            material_slots_per_pipeline,
            material_status_per_pipeline,
            pbr_defaults,
            physical_device_properties,
        })
    }

    pub(crate) fn write_descriptors(&mut self, frame_index: FrameIndex, global_transforms_buffer: &Buffer) {
        profiling::scope!("updating descriptors");
        let mut materials_needing_update = Vec::new();
        for (pipeline, material_slots) in self.material_slots_per_pipeline.iter_with_pipeline() {
            for (i, material_slot) in material_slots.iter().enumerate() {
                if let Some(material) = material_slot.as_ref().and_then(Weak::upgrade) {
                    let written = self.material_status_per_pipeline.get(pipeline)[i][frame_index.index()];
                    if !written {
                        materials_needing_update.push((pipeline, i, material));
                    }
                }
            }
        }

        let mut pending_writes = Vec::with_capacity(materials_needing_update.len() * 6 + 1);

        let shared_pipeline = PipelineIndex::SHARED_DESCRIPTOR_PIPELINE;
        let global_transforms_buffer = (global_transforms_buffer.inner, 0, vk::WHOLE_SIZE);
        self.set_uniform_buffer(
            frame_index,
            shared_pipeline,
            &mut pending_writes,
            (0, 0, 0),
            global_transforms_buffer,
        );

        for (pipeline, i, material) in &materials_needing_update {
            self.write_material(frame_index, *pipeline, material, &mut pending_writes);
            self.material_status_per_pipeline.get_mut(*pipeline)[*i][frame_index.index()] = true;
        }

        let mut writes = Vec::with_capacity(pending_writes.len());
        for pending_write in &pending_writes {
            writes.push(pending_write.write_descriptor_set.unwrap());
        }
        {
            profiling::scope!("vk::update_descriptor_sets");
            unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        }
    }

    fn write_material(
        &self,
        frame_index: FrameIndex,
        pipeline: PipelineIndex,
        material: &Material,
        pending_writes: &mut Vec<PendingWrite>,
    ) {
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
                self.set_uniform_images(frame_index, pipeline, pending_writes, (1, 1, material.array_index), &images);
                self.set_uniform_buffer(frame_index, pipeline, pending_writes, (1, 6, material.array_index), factors);
            }
        }
    }

    #[profiling::function]
    fn set_uniform_buffer(
        &self,
        frame_index: FrameIndex,
        pipeline: PipelineIndex,
        pending_writes: &mut Vec<PendingWrite>,
        (set, binding, array_index): (u32, u32, u32),
        (buffer, offset, size): (vk::Buffer, vk::DeviceSize, vk::DeviceSize),
    ) {
        let frame_idx = frame_index.index();
        let set_idx = set as usize;
        let descriptor_set = self.descriptor_sets[frame_idx].get(pipeline).inner[set_idx];
        let descriptor_buffer_info = Box::new(
            vk::DescriptorBufferInfo::builder()
                .buffer(buffer)
                .offset(offset)
                .range(size)
                .build(),
        );
        let params = &PIPELINE_PARAMETERS.get(pipeline).descriptor_sets[set_idx][binding as usize];
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
        frame_index: FrameIndex,
        pipeline: PipelineIndex,
        pending_writes: &mut Vec<PendingWrite>,
        (set, first_binding, array_index): (u32, u32, u32),
        image_views: &[vk::ImageView],
    ) {
        let frame_idx = frame_index.index();
        let set_idx = set as usize;
        let descriptor_set = self.descriptor_sets[frame_idx].get(pipeline).inner[set_idx];
        for (i, image_view) in image_views.iter().enumerate() {
            let descriptor_image_info = Box::new(
                vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(*image_view)
                    .build(),
            );
            let binding = first_binding + i as u32;
            let params = &PIPELINE_PARAMETERS.get(pipeline).descriptor_sets[set_idx][binding as usize];
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
    pub(crate) fn descriptor_sets(&self, frame_index: FrameIndex, pipeline: PipelineIndex) -> &[vk::DescriptorSet] {
        &self.descriptor_sets[frame_index.index()].get(pipeline).inner
    }
}
