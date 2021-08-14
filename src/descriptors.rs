use crate::pipeline::{DescriptorSetLayoutParams, Pipeline, PIPELINE_PARAMETERS};
use crate::{Error, FrameIndex, Gpu, Texture};
use ash::version::DeviceV1_0;
use ash::vk;
use ash::Device;

pub(crate) struct Descriptors {
    pub(crate) pipeline_layouts: Vec<vk::PipelineLayout>,
    descriptor_set_layouts_per_pipeline: Vec<Vec<vk::DescriptorSetLayout>>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<Vec<Vec<vk::DescriptorSet>>>,
    sampler: vk::Sampler,
}

impl Descriptors {
    #[profiling::function]
    pub fn clean_up(&mut self, device: &Device) {
        {
            profiling::scope!("destroy sampler");
            unsafe { device.destroy_sampler(self.sampler, None) };
        }

        {
            profiling::scope!("destroy descriptor pool");
            unsafe { device.destroy_descriptor_pool(self.descriptor_pool, None) };
        }

        for descriptor_set_layouts in &self.descriptor_set_layouts_per_pipeline {
            for &descriptor_set_layout in descriptor_set_layouts {
                profiling::scope!("destroy descriptor set layout");
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
            }
        }

        for &pipeline_layout in &self.pipeline_layouts {
            profiling::scope!("destroy pipeline layout");
            unsafe { device.destroy_pipeline_layout(pipeline_layout, None) };
        }
    }

    #[profiling::function]
    pub fn new(
        device: &Device,
        physical_device_properties: &vk::PhysicalDeviceProperties,
        physical_device_features: &vk::PhysicalDeviceFeatures,
        frame_count: u32,
    ) -> Result<Descriptors, Error> {
        let create_descriptor_set_layouts = |sets: &[&[DescriptorSetLayoutParams]]| {
            sets.iter()
                .map(|bindings| {
                    let bindings = bindings
                        .iter()
                        .map(|params| {
                            vk::DescriptorSetLayoutBinding::builder()
                                .descriptor_type(params.descriptor_type)
                                .descriptor_count(params.descriptor_count)
                                .stage_flags(params.stage_flags)
                                .build()
                        })
                        .collect::<Vec<vk::DescriptorSetLayoutBinding>>();
                    let create_info =
                        vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
                    unsafe { device.create_descriptor_set_layout(&create_info, None) }
                        .map_err(Error::VulkanDescriptorSetLayoutCreation)
                })
                .collect::<Result<Vec<vk::DescriptorSetLayout>, Error>>()
        };

        let layout_tuples = PIPELINE_PARAMETERS
            .iter()
            .map(|pipeline| {
                let descriptor_set_layouts =
                    create_descriptor_set_layouts(pipeline.descriptor_sets)?;
                let pipeline_layout_create_info =
                    vk::PipelineLayoutCreateInfo::builder().set_layouts(&descriptor_set_layouts);
                let pipeline_layout = unsafe {
                    device
                        .create_pipeline_layout(&pipeline_layout_create_info, None)
                        .map_err(Error::VulkanPipelineLayoutCreation)
                }?;
                Ok((pipeline_layout, descriptor_set_layouts))
            })
            .collect::<Result<Vec<(vk::PipelineLayout, Vec<vk::DescriptorSetLayout>)>, Error>>()?;
        let pipeline_layouts = layout_tuples
            .iter()
            .map(|(pl, _)| *pl)
            .collect::<Vec<vk::PipelineLayout>>();
        let descriptor_set_layouts_per_pipeline = layout_tuples
            .into_iter()
            .map(|(_, sets)| sets)
            .collect::<Vec<Vec<vk::DescriptorSetLayout>>>();

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
        let descriptor_sets_per_frame =
            descriptor_set_layouts_per_pipeline.iter().flatten().count() as u32;
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(frame_count * descriptor_sets_per_frame)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .map_err(Error::VulkanDescriptorPoolCreation)
        }?;

        let descriptor_sets = (0..frame_count)
            .map(|_| {
                descriptor_set_layouts_per_pipeline
                    .iter()
                    .map(|descriptor_set_layouts| {
                        let create_info = vk::DescriptorSetAllocateInfo::builder()
                            .descriptor_pool(descriptor_pool)
                            .set_layouts(descriptor_set_layouts);
                        unsafe { device.allocate_descriptor_sets(&create_info) }
                            .map_err(Error::VulkanAllocateDescriptorSets)
                    })
                    .collect::<Result<Vec<Vec<vk::DescriptorSet>>, Error>>()
            })
            .collect::<Result<Vec<Vec<Vec<vk::DescriptorSet>>>, Error>>()?;

        let sampler_create_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .anisotropy_enable(physical_device_features.sampler_anisotropy == vk::TRUE)
            .max_anisotropy(physical_device_properties.limits.max_sampler_anisotropy);
        let sampler = unsafe { device.create_sampler(&sampler_create_info, None) }
            .map_err(Error::VulkanSamplerCreation)?;

        Ok(Descriptors {
            pipeline_layouts,
            descriptor_set_layouts_per_pipeline,
            descriptor_pool,
            descriptor_sets,
            sampler,
        })
    }

    #[profiling::function]
    pub(crate) fn set_uniform_buffer(
        &self,
        gpu: &Gpu,
        frame_index: FrameIndex,
        pipeline: Pipeline,
        set: u32,
        binding: u32,
        buffer: vk::Buffer,
    ) {
        let frame_idx = gpu.image_index(frame_index);
        let pipeline_idx = pipeline as usize;
        let set_idx = set as usize;
        let descriptor_set = self.descriptor_sets[frame_idx][pipeline_idx][set_idx];
        let descriptor_buffer_info = [vk::DescriptorBufferInfo::builder()
            .buffer(buffer)
            .offset(0)
            .range(vk::WHOLE_SIZE)
            .build()];
        let params = &PIPELINE_PARAMETERS[pipeline_idx].descriptor_sets[set_idx][binding as usize];
        let write_descriptor_set = vk::WriteDescriptorSet::builder()
            .dst_set(descriptor_set)
            .dst_binding(binding)
            .dst_array_element(0)
            .descriptor_type(params.descriptor_type)
            .buffer_info(&descriptor_buffer_info)
            .build();
        unsafe {
            gpu.device
                .update_descriptor_sets(&[write_descriptor_set], &[])
        };
    }

    #[profiling::function]
    pub(crate) fn set_uniform_images(
        &self,
        gpu: &Gpu,
        frame_index: FrameIndex,
        pipeline: Pipeline,
        set: u32,
        first_binding: u32,
        textures: &[&Texture<'_>],
    ) {
        let frame_idx = gpu.image_index(frame_index);
        let pipeline_idx = pipeline as usize;
        let set_idx = set as usize;
        let descriptor_set = self.descriptor_sets[frame_idx][pipeline_idx][set_idx];
        let descriptor_image_infos = textures
            .iter()
            .map(|texture| {
                [vk::DescriptorImageInfo::builder()
                    .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                    .image_view(texture.image_view)
                    .sampler(self.sampler)
                    .build()]
            })
            .collect::<Vec<[vk::DescriptorImageInfo; 1]>>();
        let write_descriptor_sets = descriptor_image_infos
            .iter()
            .enumerate()
            .map(|(i, descriptor_image_info)| {
                let binding = first_binding + i as u32;
                let params =
                    &PIPELINE_PARAMETERS[pipeline_idx].descriptor_sets[set_idx][binding as usize];
                vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set)
                    .dst_binding(binding)
                    .dst_array_element(0)
                    .descriptor_type(params.descriptor_type)
                    .image_info(&descriptor_image_info[..])
                    .build()
            })
            .collect::<Vec<vk::WriteDescriptorSet>>();
        unsafe {
            gpu.device
                .update_descriptor_sets(&write_descriptor_sets, &[])
        };
    }

    #[profiling::function]
    pub(crate) fn descriptor_sets(
        &self,
        gpu: &Gpu,
        frame_index: FrameIndex,
        pipeline_idx: usize,
    ) -> &[vk::DescriptorSet] {
        &self.descriptor_sets[gpu.image_index(frame_index)][pipeline_idx]
    }
}
