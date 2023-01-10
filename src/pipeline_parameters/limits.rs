//! VkPhysicalDeviceLimits checks that can be made against PipelinesParameters.

use crate::pipeline_parameters::{PipelineParameters, PIPELINE_PARAMETERS};
use ash::vk;

#[derive(Debug, thiserror::Error)]
pub enum PhysicalDeviceLimitBreak {
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
}

impl PipelineParameters {
    fn get_per_stage_descriptors_of_type(&self, descriptor_type: Option<vk::DescriptorType>) -> (u32, u32) {
        let mut frag_stage_descriptors = 0;
        let mut vert_stage_descriptors = 0;
        for descriptor_sets in self.descriptor_sets {
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
}

pub fn bound_descriptor_sets(max_bound_descriptor_sets: u32) -> Result<(), PhysicalDeviceLimitBreak> {
    let descriptor_sets = |params: &PipelineParameters| params.descriptor_sets.len() as u32;
    let req = PIPELINE_PARAMETERS.iter().map(descriptor_sets).max().unwrap_or(0);
    if max_bound_descriptor_sets > req {
        Ok(())
    } else {
        Err(PhysicalDeviceLimitBreak::DescriptorSetLimit(max_bound_descriptor_sets, req))
    }
}

pub fn per_stage_descriptors(descriptor_type: vk::DescriptorType, max_per_stage_descriptors: u32) -> Result<(), PhysicalDeviceLimitBreak> {
    let descs = |params: &PipelineParameters| {
        let (frag, vert) = params.get_per_stage_descriptors_of_type(Some(descriptor_type));
        frag.max(vert)
    };
    let req = PIPELINE_PARAMETERS.iter().map(descs).max().unwrap_or(0);
    let supported = max_per_stage_descriptors;
    if supported > req {
        Ok(())
    } else {
        Err(PhysicalDeviceLimitBreak::DescriptorPerStageLimit(descriptor_type, supported, req))
    }
}

pub fn per_stage_resources(max_per_stage_resources: u32) -> Result<(), PhysicalDeviceLimitBreak> {
    let descs = |params: &PipelineParameters| {
        let (frag, vert) = params.get_per_stage_descriptors_of_type(None);
        frag.max(vert)
    };
    let req = PIPELINE_PARAMETERS.iter().map(descs).max().unwrap_or(0);
    let supported = max_per_stage_resources;
    if supported > req {
        Ok(())
    } else {
        Err(PhysicalDeviceLimitBreak::DescriptorPerStageTotalLimit(supported, req))
    }
}

pub fn per_set_descriptors(
    descriptor_type: vk::DescriptorType,
    max_descriptor_set_descriptors: u32,
) -> Result<(), PhysicalDeviceLimitBreak> {
    let descs = |params: &PipelineParameters| {
        let (frag, vert) = params.get_per_stage_descriptors_of_type(Some(descriptor_type));
        frag + vert
    };
    let req = PIPELINE_PARAMETERS.iter().map(descs).max().unwrap_or(0);
    let supported = max_descriptor_set_descriptors;
    if supported > req {
        Ok(())
    } else {
        Err(PhysicalDeviceLimitBreak::DescriptorPerSetLimit(descriptor_type, supported, req))
    }
}
