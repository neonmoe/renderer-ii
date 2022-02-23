use crate::image_loading::{self, TextureKind};
use crate::pipeline::{DescriptorSetLayoutParams, Pipeline, PipelineMap, PushConstantStruct, MAX_TEXTURE_COUNT, PIPELINE_PARAMETERS};
use crate::vulkan_raii::{DescriptorPool, DescriptorSetLayouts, DescriptorSets, Device, ImageView, PipelineLayout, Sampler};
use crate::{Error, FrameIndex, Uploader, VulkanArena};
use ash::version::DeviceV1_0;
use ash::vk;
use std::hash::{Hash, Hasher};
use std::mem;
use std::ops::Range;
use std::rc::{Rc, Weak};

/// A unique index into one pipeline's textures. A [Weak] of
/// this is held by [Descriptors], [Rc] by [Material](crate::Material).
pub struct Material {
    pub array_index: u32,
    pipeline: Pipeline,
    data: PipelineSpecificData,
}

pub enum PipelineSpecificData {
    Gltf {
        base_color: Option<Rc<ImageView>>,
        metallic_roughness: Option<Rc<ImageView>>,
        normal: Option<Rc<ImageView>>,
        occlusion: Option<Rc<ImageView>>,
        emissive: Option<Rc<ImageView>>,
    },
}

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
    pub fn new(descriptors: &mut Descriptors, pipeline: Pipeline, data: PipelineSpecificData) -> Result<Rc<Material>, Error> {
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

/// Synchronization status per frame index. True means the image views
/// for the frame index are up to date.
type MaterialStatus = Vec<bool>;
type MaterialStatusArray = Vec<MaterialStatus>;

pub struct PbrDefaults {
    default_base_color: ImageView,
    default_metallic_roughness: ImageView,
    default_normal: ImageView,
    default_occlusion: ImageView,
    default_emissive: ImageView,
}

impl PbrDefaults {
    pub fn new(device: &Rc<Device>, uploader: &mut Uploader, arena: &mut VulkanArena) -> Result<PbrDefaults, Error> {
        const WHITE: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF];
        const BLACK: [u8; 4] = [0, 0, 0, 0xFF];
        const NORMAL: [u8; 4] = [0, 0, 0xFF, 0];
        const M_AND_R: [u8; 4] = [0, 0x88, 0, 0];

        let mut create_pixel = |color, kind, name| image_loading::create_pixel(device, uploader, arena, color, kind, name);
        let default_base_color = create_pixel(WHITE, TextureKind::SrgbColor, "default pbr base color")?;
        let default_metallic_roughness = create_pixel(M_AND_R, TextureKind::LinearColor, "default pbr metallic/roughness")?;
        let default_normal = create_pixel(NORMAL, TextureKind::NormalMap, "default pbr normals")?;
        let default_occlusion = create_pixel(WHITE, TextureKind::LinearColor, "default pbr occlusion")?;
        let default_emissive = create_pixel(BLACK, TextureKind::SrgbColor, "default pbr emissive")?;

        Ok(PbrDefaults {
            default_base_color,
            default_metallic_roughness,
            default_normal,
            default_occlusion,
            default_emissive,
        })
    }
}

pub struct Descriptors {
    pub(crate) pipeline_layouts: PipelineMap<PipelineLayout>,
    descriptor_sets: Vec<PipelineMap<DescriptorSets>>,
    device: Rc<Device>,
    material_slots_per_pipeline: PipelineMap<MaterialSlotArray>,
    material_status_per_pipeline: PipelineMap<MaterialStatusArray>,
    pbr_defaults: PbrDefaults,
}

impl Descriptors {
    pub fn new(
        device: &Rc<Device>,
        physical_device_properties: &vk::PhysicalDeviceProperties,
        physical_device_features: &vk::PhysicalDeviceFeatures,
        frame_count: u32,
        pbr_defaults: PbrDefaults,
    ) -> Result<Descriptors, Error> {
        profiling::scope!("new_descriptors");
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
            .anisotropy_enable(physical_device_features.sampler_anisotropy == vk::TRUE)
            .max_anisotropy(physical_device_properties.limits.max_sampler_anisotropy);
        let sampler = unsafe { device.create_sampler(&sampler_create_info, None) }.map_err(Error::VulkanSamplerCreation)?;
        let sampler = Rc::new(Sampler {
            inner: sampler,
            device: device.clone(),
        });

        let create_descriptor_set_layouts = |sets: &[&[DescriptorSetLayoutParams]]| -> Result<DescriptorSetLayouts, Error> {
            let samplers_vk = [sampler.inner];
            let samplers_rc = vec![sampler.clone()];

            let descriptor_set_layouts = sets
                .iter()
                .map(|bindings| {
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
                    unsafe { device.create_descriptor_set_layout(&create_info, None) }.map_err(Error::VulkanDescriptorSetLayoutCreation)
                })
                .collect::<Result<Vec<vk::DescriptorSetLayout>, Error>>()?;

            Ok(DescriptorSetLayouts {
                inner: descriptor_set_layouts,
                device: device.clone(),
                immutable_samplers: samplers_rc,
            })
        };

        let mut descriptor_set_layouts_per_pipeline = Vec::new();
        let pipeline_layouts = PipelineMap::new::<Error, _>(|pipeline| {
            let pipeline = PIPELINE_PARAMETERS.get(pipeline);
            let descriptor_set_layouts = Rc::new(create_descriptor_set_layouts(pipeline.descriptor_sets)?);
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
            let pipeline_layout = PipelineLayout {
                inner: pipeline_layout,
                device: device.clone(),
                descriptor_set_layouts: descriptor_set_layouts.clone(),
            };
            descriptor_set_layouts_per_pipeline.push(descriptor_set_layouts);
            Ok(pipeline_layout)
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
        let descriptor_sets_per_frame = descriptor_set_layouts_per_pipeline.iter().map(|dsl| &dsl.inner).flatten().count() as u32;
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(frame_count * descriptor_sets_per_frame)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .map_err(Error::VulkanDescriptorPoolCreation)
        }?;
        let descriptor_pool = Rc::new(DescriptorPool {
            inner: descriptor_pool,
            device: device.clone(),
        });

        let descriptor_sets = (0..frame_count)
            .map(|_| {
                let mut descriptor_set_layouts_per_pipeline = descriptor_set_layouts_per_pipeline.iter();
                let pipeline_map = PipelineMap::new::<Error, _>(|_| {
                    let descriptor_set_layouts = descriptor_set_layouts_per_pipeline.next().unwrap();
                    let create_info = vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool.inner)
                        .set_layouts(&descriptor_set_layouts.inner);
                    let descriptor_sets =
                        unsafe { device.allocate_descriptor_sets(&create_info) }.map_err(Error::VulkanAllocateDescriptorSets)?;
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
            pipeline_layouts,
            descriptor_sets,
            device: device.clone(),
            material_slots_per_pipeline,
            material_status_per_pipeline,
            pbr_defaults,
        })
    }

    pub(crate) fn write_descriptors(&mut self, frame_index: FrameIndex) {
        for (pipeline, material_slots) in self.material_slots_per_pipeline.iter_with_pipeline() {
            for (i, material_slot) in material_slots.iter().enumerate() {
                if let Some(material) = material_slot.as_ref().and_then(Weak::upgrade) {
                    let written = self.material_status_per_pipeline.get(pipeline)[i][frame_index.index()];
                    if !written {
                        self.write_material(frame_index, pipeline, &material);
                        self.material_status_per_pipeline.get_mut(pipeline)[i][frame_index.index()] = true;
                    }
                }
            }
        }
    }

    fn write_material(&self, frame_index: FrameIndex, pipeline: Pipeline, material: &Material) {
        match &material.data {
            PipelineSpecificData::Gltf {
                base_color,
                metallic_roughness,
                normal,
                occlusion,
                emissive,
            } => {
                let index = material.array_index..material.array_index + 1;
                let images = [
                    base_color.as_ref().map(Rc::as_ref).unwrap_or(&self.pbr_defaults.default_base_color),
                    metallic_roughness
                        .as_ref()
                        .map(Rc::as_ref)
                        .unwrap_or(&self.pbr_defaults.default_metallic_roughness),
                    normal.as_ref().map(Rc::as_ref).unwrap_or(&self.pbr_defaults.default_normal),
                    occlusion.as_ref().map(Rc::as_ref).unwrap_or(&self.pbr_defaults.default_occlusion),
                    emissive.as_ref().map(Rc::as_ref).unwrap_or(&self.pbr_defaults.default_emissive),
                ];
                self.set_uniform_images(frame_index, pipeline, 1, 1, &images, index);
            }
        }
    }

    #[profiling::function]
    pub(crate) fn set_uniform_buffer(&self, frame_index: FrameIndex, pipeline: Pipeline, set: u32, binding: u32, buffer: vk::Buffer) {
        let frame_idx = frame_index.index();
        let set_idx = set as usize;
        let descriptor_set = self.descriptor_sets[frame_idx].get(pipeline).inner[set_idx];
        let descriptor_buffer_info = [vk::DescriptorBufferInfo::builder()
            .buffer(buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)
            .build()];
        let params = &PIPELINE_PARAMETERS.get(pipeline).descriptor_sets[set_idx][binding as usize];
        let write_descriptor_set = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(binding)
            .dst_array_element(0)
            .descriptor_type(params.descriptor_type)
            .buffer_info(&descriptor_buffer_info)
            .build();
        unsafe { self.device.update_descriptor_sets(&[write_descriptor_set], &[]) };
    }

    /// Uploads the image_views to the given set, starting at
    /// first_binding. The range parameter controls which indices of
    /// the texture array are filled.
    #[profiling::function]
    fn set_uniform_images(
        &self,
        frame_index: FrameIndex,
        pipeline: Pipeline,
        set: u32,
        first_binding: u32,
        image_views: &[&ImageView],
        range: Range<u32>,
    ) {
        let frame_idx = frame_index.index();
        let set_idx = set as usize;
        let descriptor_set = self.descriptor_sets[frame_idx].get(pipeline).inner[set_idx];
        for (i, image_view) in image_views.iter().enumerate() {
            let descriptor_image_info = vk::DescriptorImageInfo::builder()
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image_view(image_view.inner)
                .build();
            let descriptor_image_infos = vec![descriptor_image_info; range.len()];
            let binding = first_binding + i as u32;
            let params = &PIPELINE_PARAMETERS.get(pipeline).descriptor_sets[set_idx][binding as usize];
            let write_descriptor_sets = [vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_set)
                .dst_binding(binding)
                .dst_array_element(range.start)
                .descriptor_type(params.descriptor_type)
                .image_info(&descriptor_image_infos)
                .build()];
            unsafe { self.device.update_descriptor_sets(&write_descriptor_sets, &[]) };
        }
    }

    #[profiling::function]
    pub(crate) fn descriptor_sets(&self, frame_index: FrameIndex, pipeline: Pipeline) -> &[vk::DescriptorSet] {
        &self.descriptor_sets[frame_index.index()].get(pipeline).inner
    }
}
