use alloc::rc::{Rc, Weak};

use arrayvec::ArrayVec;
use ash::vk;
use bytemuck::Zeroable;
use enum_map::Enum;

use crate::arena::buffers::ForBuffers;
use crate::arena::{VulkanArena, VulkanArenaError};
use crate::physical_device::PhysicalDevice;
use crate::renderer::pipeline_parameters::constants::{
    MAX_TEXTURES, UF_DRAW_CALL_FRAG_PARAMS_BINDING, UF_DRAW_CALL_VERT_PARAMS_BINDING, UF_IMGUI_DRAW_CMD_PARAMS_BINDING,
    UF_PBR_FACTORS_BINDING, UF_RENDER_SETTINGS_BINDING, UF_TEXTURES_BINDING, UF_TRANSFORMS_BINDING,
};
use crate::renderer::pipeline_parameters::uniforms::{ImGuiDrawCmd, PbrFactors, StructureOfArraysUniform};
use crate::renderer::pipeline_parameters::{
    DescriptorSetLayoutParams, PipelineIndex, PipelineMap, PipelineParameters, PBR_PIPELINES, PIPELINE_PARAMETERS, SKINNED_PIPELINES,
};
use crate::vulkan_raii::{Buffer, DescriptorPool, DescriptorSetLayouts, DescriptorSets, Device, ImageView, PipelineLayout, Sampler};

pub(crate) mod material;

pub struct PbrDefaults {
    pub base_color: ImageView,
    pub metallic_roughness: ImageView,
    pub normal: ImageView,
    pub occlusion: ImageView,
    pub emissive: ImageView,
}

pub struct PbrDefaultTextureSlots {
    pub base_color: (Rc<ImageView>, u32),
    pub metallic_roughness: (Rc<ImageView>, u32),
    pub normal: (Rc<ImageView>, u32),
    pub occlusion: (Rc<ImageView>, u32),
    pub emissive: (Rc<ImageView>, u32),
}

#[derive(Default)]
struct PendingWrite<'a> {
    write_descriptor_set: Option<vk::WriteDescriptorSet<'a>>,
    _buffer_info: Option<Box<vk::DescriptorBufferInfo>>,
    _image_info: Option<Box<vk::DescriptorImageInfo>>,
}

pub(crate) struct TempUniforms {
    pub buffer: Buffer,
    pbr_factors_offset_and_size: (vk::DeviceSize, vk::DeviceSize),
    imgui_cmds_offset_and_size: (vk::DeviceSize, vk::DeviceSize),
}

pub(crate) struct ReusableSlots<T, const LEN: usize> {
    slots: ArrayVec<Option<Weak<T>>, LEN>,
    dirty: ArrayVec<bool, LEN>,
}

impl<T, const LEN: usize> ReusableSlots<T, LEN> {
    pub(crate) fn new() -> ReusableSlots<T, LEN> {
        // FIXME: Don't preallocate all the slots
        // Try_allocate_slot should still primarily try and find strong_count == 0 slots to replace, but otherwise, the Nones aren't needed anymore with the soa changes
        // Actually, Option<Weak<>> could probably just be Weak<> now.
        let slots = ArrayVec::from_iter([None].into_iter().cycle().take(LEN));
        let dirty = ArrayVec::from_iter([false; LEN]);
        ReusableSlots { slots, dirty }
    }

    pub(crate) fn try_allocate_slot(&mut self, data: Weak<T>) -> Option<u32> {
        let (i, slot) =
            self.slots.iter_mut().enumerate().find(|(_, slot)| if let Some(slot) = &slot { slot.strong_count() == 0 } else { true })?;
        let _ = slot.insert(data);
        self.dirty[i] = true;
        Some(i as u32)
    }
}

pub struct Descriptors {
    pub(crate) pipeline_layouts: PipelineMap<PipelineLayout>,
    descriptor_sets: PipelineMap<DescriptorSets>,
    device: Device,
    pbr_defaults: PbrDefaultTextureSlots,
    pub(crate) pbr_factors_slots: ReusableSlots<PbrFactors, { PbrFactors::MAX_COUNT }>,
    pub(crate) imgui_cmd_slots: ReusableSlots<ImGuiDrawCmd, { ImGuiDrawCmd::MAX_COUNT }>,
    pub(crate) texture_slots: ReusableSlots<ImageView, { MAX_TEXTURES as usize }>,
    uniform_buffer_offset_alignment: vk::DeviceSize,
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

        let mut texture_slots = ReusableSlots::new();

        let mut pbr_defaults = PbrDefaultTextureSlots {
            base_color: (Rc::new(pbr_defaults.base_color), 0),
            metallic_roughness: (Rc::new(pbr_defaults.metallic_roughness), 0),
            normal: (Rc::new(pbr_defaults.normal), 0),
            occlusion: (Rc::new(pbr_defaults.occlusion), 0),
            emissive: (Rc::new(pbr_defaults.emissive), 0),
        };
        for (texture, slot) in [
            &mut pbr_defaults.base_color,
            &mut pbr_defaults.metallic_roughness,
            &mut pbr_defaults.normal,
            &mut pbr_defaults.occlusion,
            &mut pbr_defaults.emissive,
        ] {
            *slot = texture_slots.try_allocate_slot(Rc::downgrade(texture)).unwrap();
        }

        let uniform_buffer_offset_alignment = physical_device.properties.limits.min_uniform_buffer_offset_alignment;

        Descriptors {
            pipeline_layouts,
            descriptor_sets,
            device: device.clone(),
            pbr_factors_slots: ReusableSlots::new(),
            imgui_cmd_slots: ReusableSlots::new(),
            texture_slots,
            pbr_defaults,
            uniform_buffer_offset_alignment,
        }
    }

    // TODO: Should this function be somewhere else? uniforms.rs?
    pub(crate) fn create_temp_uniforms(&mut self, temp_arena: &mut VulkanArena<ForBuffers>) -> Result<TempUniforms, VulkanArenaError> {
        profiling::scope!("creating temp uniform buffer");

        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(
                PbrFactors::DESCRIPTOR_SIZE.next_multiple_of(self.uniform_buffer_offset_alignment)
                    + ImGuiDrawCmd::DESCRIPTOR_SIZE.next_multiple_of(self.uniform_buffer_offset_alignment),
            )
            .usage(vk::BufferUsageFlags::UNIFORM_BUFFER)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let mut mapped_buffer = temp_arena.create_staging_buffer(buffer_create_info, format_args!("uniform (temp material buffer)"))?;
        let buffer = mapped_buffer.data_mut();
        let mut written = 0usize;

        let pbr_factors_offset_and_size;
        {
            profiling::scope!("copying over pbr factors");
            let factors = self
                .pbr_factors_slots
                .slots
                .iter()
                .map(|slot| if let Some(factors) = slot.as_ref().and_then(Weak::upgrade) { *factors } else { PbrFactors::zeroed() })
                .collect::<ArrayVec<PbrFactors, { PbrFactors::MAX_COUNT }>>();
            let offset = written.next_multiple_of(self.uniform_buffer_offset_alignment as usize);
            let len = factors.soa_size();
            pbr_factors_offset_and_size = (offset as vk::DeviceSize, len as vk::DeviceSize);
            factors.soa_write(&mut buffer[offset..offset + len]);
            written = offset + len;
        }

        let imgui_cmds_offset_and_size;
        {
            profiling::scope!("copying over imgui cmds");
            let draw_cmds = self
                .imgui_cmd_slots
                .slots
                .iter()
                .map(|slot| if let Some(draw_cmd) = slot.as_ref().and_then(Weak::upgrade) { *draw_cmd } else { ImGuiDrawCmd::zeroed() })
                .collect::<ArrayVec<ImGuiDrawCmd, { ImGuiDrawCmd::MAX_COUNT }>>();

            let offset = written.next_multiple_of(self.uniform_buffer_offset_alignment as usize);
            let len = draw_cmds.soa_size();
            imgui_cmds_offset_and_size = (offset as vk::DeviceSize, len as vk::DeviceSize);
            draw_cmds.soa_write(&mut buffer[offset..offset + len]);
            written = offset + len;
        }

        // TODO: Allocate and write the other dynamic uniforms here as well? (most write_descriptors params)
        // Maybe also an appropriate place where the zeroed areas could be
        // eliminated, just include (to, from) pairs of byte ranges for the each
        // uniform write.

        let _ = written; // to avoid clippy complaining that written is not used. It will be used later!

        Ok(TempUniforms { buffer: mapped_buffer.buffer, pbr_factors_offset_and_size, imgui_cmds_offset_and_size })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_descriptors(
        &mut self,
        global_transforms_buffer: &Buffer,
        render_settings_buffer: &Buffer,
        skinned_mesh_joints_buffer: &Buffer,
        draw_call_vert_params: &Buffer,
        draw_call_frag_params: &Buffer,
        material_buffers: &TempUniforms,
        hdr_attachment: &ImageView,
    ) {
        profiling::scope!("updating descriptors");

        let mut textures_needing_update = ArrayVec::<_, { MAX_TEXTURES as usize }>::new();
        for (i, texture_slot) in self.texture_slots.slots.iter().enumerate().filter(|(i, _)| self.texture_slots.dirty[*i]) {
            if let Some(texture) = texture_slot.as_ref().and_then(Weak::upgrade) {
                textures_needing_update.push((i as u32, texture));
            }
        }

        let mut pending_writes = Vec::new();

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

        for (i, texture) in &textures_needing_update {
            self.set_uniform_images(shared_pipeline, &mut pending_writes, (0, UF_TEXTURES_BINDING, *i), &[texture.inner]);
            self.texture_slots.dirty[*i as usize] = false;
        }

        for pipeline in SKINNED_PIPELINES {
            let skinned_mesh_joints_buffer = (skinned_mesh_joints_buffer.inner, 0, skinned_mesh_joints_buffer.size);
            self.set_uniform_buffer(pipeline, &mut pending_writes, (2, 0, 0), skinned_mesh_joints_buffer);
        }

        for pipeline in PBR_PIPELINES {
            let (offset, size) = material_buffers.pbr_factors_offset_and_size;
            let buffer = (material_buffers.buffer.inner, offset, size);
            self.set_uniform_buffer(pipeline, &mut pending_writes, (1, UF_PBR_FACTORS_BINDING, 0), buffer);
        }

        {
            let pipeline = PipelineIndex::ImGui;
            let (offset, size) = material_buffers.imgui_cmds_offset_and_size;
            let buffer = (material_buffers.buffer.inner, offset, size);
            self.set_uniform_buffer(pipeline, &mut pending_writes, (1, UF_IMGUI_DRAW_CMD_PARAMS_BINDING, 0), buffer);
        }

        // NOTE: pending_writes owns the image/buffers that are pointed to by
        // the write_descriptor_sets, and they need to be dropped only after
        // update_descriptor_sets.
        let mut writes = Vec::with_capacity(pending_writes.len());
        for pending_write in &pending_writes {
            writes.push(pending_write.write_descriptor_set.unwrap());
        }
        {
            profiling::scope!("vk::update_descriptor_sets");
            unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        }
        drop(pending_writes);
        drop(textures_needing_update);
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
        pending_writes: &mut Vec<PendingWrite>,
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
