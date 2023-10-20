//! This module exists for convenience, to make it ergonomic at all to have all meshes' vertex data
//! be in the same buffers. Even if there was a "buffer arena" which could sub-allocate from a
//! buffer, you'd need 4 or 6 of those to contain each of the buffers of the  different vertex
//! attributes.

use arrayvec::ArrayVec;
use ash::vk;

use crate::arena::{VulkanArena, VulkanArenaError};
use crate::arena::buffers::ForBuffers;
use crate::renderer::pipeline_parameters::{ALL_PIPELINES, PIPELINE_COUNT, PIPELINE_PARAMETERS, PipelineIndex, PipelineMap};

// TODO: figure out how to enforce this limit
const MAX_BINDINGS_PER_SET: usize = 8;

struct BindingSetOffset {
    offset: usize,
    description: vk::VertexInputBindingDescription,
}

/// Wrapper around a [`VulkanArena`] that holds vertex attribute data in a very
/// tightly packed, structure-of-arrays configuration.
pub struct VertexLibrary {
    arena: VulkanArena<ForBuffers>,
    distinct_binding_sets: DistinctBindingSets,
    offsets_per_distinct_binding_set: ArrayVec<ArrayVec<BindingSetOffset, MAX_BINDINGS_PER_SET>, PIPELINE_COUNT>,
}

impl VertexLibrary {
    pub fn new(mut arena: VulkanArena<ForBuffers>, measurements: VertexLibraryMeasurements) -> Result<VertexLibrary, VulkanArenaError> {
        let VertexLibraryMeasurements {
            vertex_counts_per_binding,
            distinct_binding_sets,
        } = measurements;
        let mut buffer_size = 0;
        let mut offsets_per_distinct_binding_set = ArrayVec::new();
        for (&vertex_count, &bindings) in vertex_counts_per_binding.iter().zip(&distinct_binding_sets.binding_sets) {
            let mut offsets = ArrayVec::<BindingSetOffset, MAX_BINDINGS_PER_SET>::new();
            for binding in bindings {
                offsets.push(BindingSetOffset {
                    offset: buffer_size,
                    description: *binding,
                });
                buffer_size += binding.stride as usize * vertex_count;
            }
            offsets_per_distinct_binding_set.push(offsets);
        }
        Ok(VertexLibrary {
            arena,
            distinct_binding_sets,
            offsets_per_distinct_binding_set,
        })
    }

    pub fn into_inner(self) -> VulkanArena<ForBuffers> {
        self.arena
    }
}

/// Records how many vertices need to be allocated for each distinct vertex layout.
pub struct VertexLibraryMeasurements {
    distinct_binding_sets: DistinctBindingSets,
    vertex_counts_per_binding: ArrayVec<usize, PIPELINE_COUNT>,
}

impl Default for VertexLibraryMeasurements {
    fn default() -> VertexLibraryMeasurements {
        let distinct_binding_sets = DistinctBindingSets::default();
        let vertex_counts_per_binding = distinct_binding_sets.binding_sets.iter().map(|_| 0).collect();
        VertexLibraryMeasurements {
            distinct_binding_sets,
            vertex_counts_per_binding,
        }
    }
}

impl VertexLibraryMeasurements {
    /// Record the amount of vertices represented by the data, and ensure that it's well-formed:
    /// same amount of vertices in each buffer, using the vertex binding descriptions from the given
    /// pipeline.
    #[track_caller]
    pub fn add_vertex_buffers(&mut self, pipeline: PipelineIndex, vertex_buffers: &[&[u8]]) {
        let binding_set_idx = self.distinct_binding_sets.binding_set_indices[pipeline];
        let descriptions = self.distinct_binding_sets.binding_sets[binding_set_idx];
        let mut vertex_count = None;
        for (i, desc) in descriptions
            .iter()
            .filter(|desc| desc.input_rate == vk::VertexInputRate::VERTEX)
            .enumerate()
        {
            assert!(
                vertex_buffers.len() > i,
                "provided only {} vertex buffers, but pipeline {pipeline:?} has a binding at index {i}",
                vertex_buffers.len(),
            );
            let new_vertex_count = vertex_buffers[i].len() / desc.stride as usize;
            vertex_count = Some(vertex_count.unwrap_or(new_vertex_count));
            assert_eq!(
                Some(new_vertex_count),
                vertex_count,
                "the {}. buffer contains {new_vertex_count} vertices while the previous buffers contained {}",
                i + 1,
                vertex_count.unwrap()
            );
        }
        if let Some(vertex_count) = vertex_count {
            self.vertex_counts_per_binding[binding_set_idx] += vertex_count;
        }
    }
}

struct DistinctBindingSets {
    binding_sets: ArrayVec<&'static [vk::VertexInputBindingDescription], PIPELINE_COUNT>,
    binding_set_indices: PipelineMap<usize>,
}

impl Default for DistinctBindingSets {
    fn default() -> DistinctBindingSets {
        let mut binding_indices = PipelineMap::from_infallible(|_| 0);
        let mut bindings = ArrayVec::new();
        for pipeline in ALL_PIPELINES {
            if let Some(i) = find_existing_binding(pipeline, &bindings) {
                binding_indices[pipeline] = i;
            } else {
                binding_indices[pipeline] = bindings.len();
                bindings.push(PIPELINE_PARAMETERS[pipeline].bindings);
            }
        }
        DistinctBindingSets { binding_sets: bindings, binding_set_indices: binding_indices }
    }
}

/// Returns an index to `distinct_bindings` where the descriptions fit the given pipeline.
fn find_existing_binding(pipeline: PipelineIndex, distinct_binding_sets: &[&'static [vk::VertexInputBindingDescription]]) -> Option<usize> {
    distinct_binding_sets.iter().position(|bindings| {
        let binding_descriptions = PIPELINE_PARAMETERS[pipeline].bindings;
        bindings.iter().zip(binding_descriptions).all(|(existing, new)| {
            existing.binding == new.binding && existing.input_rate == new.input_rate && existing.stride == new.stride
        })
    })
}
