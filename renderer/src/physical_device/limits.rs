//! `VkPhysicalDeviceLimits` checks that can be made against `PipelinesParameters`.

use ash::vk;

use crate::renderer::pipeline_parameters::{PipelineParameters, PIPELINE_PARAMETERS};

// TODO: Add a limit check for PhysicalDeviceLimits maxDrawIndirectCount, as some only support 0 or 1.
// Everything else supports >1 billion draws, so might as well set some ridiculous "minimum".

#[derive(Debug, thiserror::Error)]
pub enum PhysicalDeviceLimitBreak {
    #[error("physical device only supports uniform buffers of size {0}, but {1} are needed")]
    UniformBufferRange(u32, u32),
    #[error("physical device only supports storage buffers of size {0}, but {1} are needed")]
    StorageBufferRange(u32, u32),
    #[error("physical device only supports {0} descriptor sets, but {1} are needed")]
    DescriptorSetLimit(u32, u32),
    #[error("physical device only supports {1} {0:?} descriptors per stage, but {2} are needed")]
    DescriptorPerStageLimit(vk::DescriptorType, u32, u32),
    #[error("physical device only supports a total of {0:?} descriptors per stage, but {1} are needed")]
    DescriptorPerStageTotalLimit(u32, u32),
    #[error("physical device only supports {1} {0:?} descriptors per pipeline layout, but {2} are needed")]
    DescriptorPerSetLimit(vk::DescriptorType, u32, u32),
    #[error("physical device only supports {0} vertex input attributes, but {1} are needed")]
    VertexInputAttributes(u32, u32),
    #[error("physical device only supports {0} vertex input bindings, but {1} are needed")]
    VertexInputBindings(u32, u32),
    #[error("physical device only supports an vertex input attribute offset of {0}, but {1} is needed")]
    VertexInputAttributeOffset(u32, u32),
    #[error("physical device only supports an vertex input binding stride of {0}, but {1} is needed")]
    VertexInputBindingStride(u32, u32),
}

pub fn uniform_buffer_range(max_uniform_buffer_range: u32) -> Result<(), PhysicalDeviceLimitBreak> {
    let types = [vk::DescriptorType::UNIFORM_BUFFER, vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC];
    let max_buffer_range = |params: &PipelineParameters| get_max_buffer_range(params, &types);
    let req = PIPELINE_PARAMETERS.values().map(max_buffer_range).max().unwrap_or(0);
    if max_uniform_buffer_range >= req { Ok(()) } else { Err(PhysicalDeviceLimitBreak::UniformBufferRange(max_uniform_buffer_range, req)) }
}

pub fn storage_buffer_range(max_storage_buffer_range: u32) -> Result<(), PhysicalDeviceLimitBreak> {
    let types = [vk::DescriptorType::STORAGE_BUFFER, vk::DescriptorType::STORAGE_BUFFER_DYNAMIC];
    let max_buffer_range = |params: &PipelineParameters| get_max_buffer_range(params, &types);
    let req = PIPELINE_PARAMETERS.values().map(max_buffer_range).max().unwrap_or(0);
    if max_storage_buffer_range >= req { Ok(()) } else { Err(PhysicalDeviceLimitBreak::StorageBufferRange(max_storage_buffer_range, req)) }
}

pub fn bound_descriptor_sets(max_bound_descriptor_sets: u32) -> Result<(), PhysicalDeviceLimitBreak> {
    let descriptor_sets = |params: &PipelineParameters| params.descriptor_sets.len() as u32;
    let req = PIPELINE_PARAMETERS.values().map(descriptor_sets).max().unwrap_or(0);
    if max_bound_descriptor_sets >= req {
        Ok(())
    } else {
        Err(PhysicalDeviceLimitBreak::DescriptorSetLimit(max_bound_descriptor_sets, req))
    }
}

pub fn per_stage_descriptors(descriptor_type: vk::DescriptorType, max_per_stage_descriptors: u32) -> Result<(), PhysicalDeviceLimitBreak> {
    let descs = |params: &PipelineParameters| {
        let (frag, vert) = get_per_stage_descriptors_of_type(params, Some(descriptor_type));
        frag.max(vert)
    };
    let req = PIPELINE_PARAMETERS.values().map(descs).max().unwrap_or(0);
    let supported = max_per_stage_descriptors;
    if supported >= req { Ok(()) } else { Err(PhysicalDeviceLimitBreak::DescriptorPerStageLimit(descriptor_type, supported, req)) }
}

pub fn per_stage_resources(max_per_stage_resources: u32) -> Result<(), PhysicalDeviceLimitBreak> {
    let descs = |params: &PipelineParameters| {
        let (frag, vert) = get_per_stage_descriptors_of_type(params, None);
        frag.max(vert)
    };
    let req = PIPELINE_PARAMETERS.values().map(descs).max().unwrap_or(0);
    let supported = max_per_stage_resources;
    if supported >= req { Ok(()) } else { Err(PhysicalDeviceLimitBreak::DescriptorPerStageTotalLimit(supported, req)) }
}

pub fn per_set_descriptors(
    descriptor_type: vk::DescriptorType,
    max_descriptor_set_descriptors: u32,
) -> Result<(), PhysicalDeviceLimitBreak> {
    let descs = |params: &PipelineParameters| {
        let (frag, vert) = get_per_stage_descriptors_of_type(params, Some(descriptor_type));
        frag + vert
    };
    let req = PIPELINE_PARAMETERS.values().map(descs).max().unwrap_or(0);
    let supported = max_descriptor_set_descriptors;
    if supported >= req { Ok(()) } else { Err(PhysicalDeviceLimitBreak::DescriptorPerSetLimit(descriptor_type, supported, req)) }
}

pub fn vertex_input_attributes(max_vertex_input_attributes: u32) -> Result<(), PhysicalDeviceLimitBreak> {
    let vertex_input_attrs = |params: &PipelineParameters| params.attributes.len() as u32;
    let req = PIPELINE_PARAMETERS.values().map(vertex_input_attrs).max().unwrap_or(0);
    if max_vertex_input_attributes >= req {
        Ok(())
    } else {
        Err(PhysicalDeviceLimitBreak::VertexInputAttributes(max_vertex_input_attributes, req))
    }
}

pub fn vertex_input_bindings(max_vertex_input_bindings: u32) -> Result<(), PhysicalDeviceLimitBreak> {
    let vertex_input_bindings = |params: &PipelineParameters| params.bindings.len() as u32;
    let req = PIPELINE_PARAMETERS.values().map(vertex_input_bindings).max().unwrap_or(0);
    if max_vertex_input_bindings >= req {
        Ok(())
    } else {
        Err(PhysicalDeviceLimitBreak::VertexInputBindings(max_vertex_input_bindings, req))
    }
}

pub fn vertex_input_attribute_offset(max_vertex_input_attribute_offset: u32) -> Result<(), PhysicalDeviceLimitBreak> {
    let max_offset = |params: &PipelineParameters| params.attributes.iter().map(|attr| attr.offset).max().unwrap_or(0);
    let req = PIPELINE_PARAMETERS.values().map(max_offset).max().unwrap_or(0);
    let supported = max_vertex_input_attribute_offset;
    if supported >= req { Ok(()) } else { Err(PhysicalDeviceLimitBreak::VertexInputAttributeOffset(supported, req)) }
}

pub fn vertex_input_binding_stride(max_vertex_input_binding_stride: u32) -> Result<(), PhysicalDeviceLimitBreak> {
    let max_stride = |params: &PipelineParameters| params.bindings.iter().map(|binding| binding.stride).max().unwrap_or(0);
    let req = PIPELINE_PARAMETERS.values().map(max_stride).max().unwrap_or(0);
    let supported = max_vertex_input_binding_stride;
    if supported >= req { Ok(()) } else { Err(PhysicalDeviceLimitBreak::VertexInputBindingStride(supported, req)) }
}

fn get_per_stage_descriptors_of_type(params: &PipelineParameters, descriptor_type: Option<vk::DescriptorType>) -> (u32, u32) {
    let mut frag_stage_descriptors = 0;
    let mut vert_stage_descriptors = 0;
    for descriptor_sets in params.descriptor_sets {
        for params in *descriptor_sets {
            match descriptor_type {
                Some(descriptor_type) if descriptor_type != params.descriptor_type => continue,
                _ => {}
            }
            if params.stage_flags.contains(vk::ShaderStageFlags::FRAGMENT) {
                frag_stage_descriptors += params.descriptor_count;
            }
            if params.stage_flags.contains(vk::ShaderStageFlags::VERTEX) {
                vert_stage_descriptors += params.descriptor_count;
            }
        }
    }
    (frag_stage_descriptors, vert_stage_descriptors)
}

fn get_max_buffer_range(params: &PipelineParameters, descriptor_types: &[vk::DescriptorType]) -> u32 {
    let mut max_buffer_range = 0;
    for descriptor_sets in params.descriptor_sets {
        for params in *descriptor_sets {
            if descriptor_types.contains(&params.descriptor_type) {
                if let Some(buffer_range) = params.descriptor_size {
                    max_buffer_range = max_buffer_range.max(buffer_range);
                }
            }
        }
    }
    max_buffer_range as u32
}
