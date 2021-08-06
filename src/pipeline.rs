use crate::{Error, Gpu};
use ash::version::DeviceV1_0;
use ash::vk;
use ash::Device;
use std::mem;
use ultraviolet::Vec3;

#[derive(Clone, Copy)]
pub enum Pipeline {
    PlainVertexColor,
    #[doc(hidden)]
    Count,
}

pub(crate) struct DescriptorSetLayoutParams {
    pub descriptor_type: vk::DescriptorType,
    pub descriptor_count: u32,
    pub stage_flags: vk::ShaderStageFlags,
    // This matches DescriptorSetLayoutBinding, except for the immutable samplers.
    // I don't see a practical use case for them, and this is simpler.
}

pub(crate) struct PipelineParameters {
    pub vertex_shader: &'static [u32],
    pub fragment_shader: &'static [u32],
    pub bindings: &'static [vk::VertexInputBindingDescription],
    pub attributes: &'static [vk::VertexInputAttributeDescription],
    pub descriptor_sets: &'static [&'static [DescriptorSetLayoutParams]],
}

/// A descriptor set that should be used as the first set for every
/// pipeline, so that global state (projection, view transforms) can
/// be bound once and never touched again during a frame.
///
/// In concrete terms, this maps to uniforms in shaders with the
/// layout `set = 0`, and the bindings are in order.
static SHARED_DESCRIPTOR_SET_0: &[DescriptorSetLayoutParams] = &[DescriptorSetLayoutParams {
    descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
    descriptor_count: 1,
    stage_flags: vk::ShaderStageFlags::VERTEX,
}];

pub(crate) static PIPELINE_PARAMETERS: [PipelineParameters; Pipeline::Count as usize] =
    [PipelineParameters {
        vertex_shader: shaders::include_spirv!("shaders/plain_color.vert"),
        fragment_shader: shaders::include_spirv!("shaders/plain_color.frag"),
        bindings: &[vk::VertexInputBindingDescription {
            binding: 0,
            stride: mem::size_of::<[Vec3; 2]>() as u32,
            input_rate: vk::VertexInputRate::VERTEX,
        }],
        attributes: &[
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 0,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0,
            },
            vk::VertexInputAttributeDescription {
                binding: 0,
                location: 1,
                format: vk::Format::R32G32B32_SFLOAT,
                offset: mem::size_of::<[Vec3; 1]>() as u32,
            },
        ],
        descriptor_sets: &[SHARED_DESCRIPTOR_SET_0],
    }];

pub(crate) struct Descriptors {
    pub(crate) pipeline_layouts: Vec<vk::PipelineLayout>,
    descriptor_set_layouts_per_pipeline: Vec<Vec<vk::DescriptorSetLayout>>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<Vec<Vec<vk::DescriptorSet>>>,
}

impl Descriptors {
    #[profiling::function]
    pub fn clean_up(&mut self, device: &Device) {
        unsafe { device.destroy_descriptor_pool(self.descriptor_pool, None) };

        for descriptor_set_layouts in &self.descriptor_set_layouts_per_pipeline {
            for &descriptor_set_layout in descriptor_set_layouts {
                unsafe { device.destroy_descriptor_set_layout(descriptor_set_layout, None) };
            }
        }

        for &pipeline_layout in &self.pipeline_layouts {
            unsafe { device.destroy_pipeline_layout(pipeline_layout, None) };
        }
    }

    #[profiling::function]
    pub fn new(device: &Device, descriptor_set_count: u32) -> Result<Descriptors, Error> {
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
                    .descriptor_count(descriptor_layout.descriptor_count * descriptor_set_count)
                    .build()
            })
            .collect::<Vec<vk::DescriptorPoolSize>>();
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(descriptor_set_count)
            .pool_sizes(&pool_sizes);
        let descriptor_pool = unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .map_err(Error::VulkanDescriptorPoolCreation)
        }?;

        let descriptor_sets = (0..descriptor_set_count)
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

        Ok(Descriptors {
            pipeline_layouts,
            descriptor_set_layouts_per_pipeline,
            descriptor_pool,
            descriptor_sets,
        })
    }

    #[profiling::function]
    pub(crate) fn set_uniform_buffer(
        &self,
        gpu: &Gpu,
        pipeline: Pipeline,
        frame_index: u32,
        set: u32,
        binding: u32,
        buffer: vk::Buffer,
    ) {
        let frame_idx = gpu.frame_mod(frame_index);
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
    pub(crate) fn descriptor_sets(
        &self,
        gpu: &Gpu,
        frame_index: u32,
        pipeline_idx: usize,
    ) -> &[vk::DescriptorSet] {
        &self.descriptor_sets[gpu.frame_mod(frame_index)][pipeline_idx]
    }
}
