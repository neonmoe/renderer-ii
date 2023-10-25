use alloc::rc::Rc;
use core::fmt::Arguments;
use core::mem;

use arrayvec::ArrayVec;
use ash::vk;

use crate::arena::buffers::{ForBuffers, MappedBuffer};
use crate::arena::{VulkanArena, VulkanArenaError};
use crate::memory_measurement::{VulkanArenaMeasurementError, VulkanArenaMeasurer};
use crate::renderer::pipeline_parameters::{
    PipelineIndex, PipelineMap, ALL_PIPELINES, PIPELINE_COUNT, PIPELINE_PARAMETERS, VERTEX_BINDING_COUNT,
};
use crate::renderer::scene::mesh::{IndexType, Mesh};
use crate::uploader::Uploader;
use crate::vulkan_raii::Buffer;

pub type VertexLibraryIndexType = u32;

const VERTEX_LIBRARY_INDEX_TYPE: vk::IndexType = vk::IndexType::UINT32;
const VERTEX_LIBRARY_INDEX_SIZE: usize = mem::size_of::<VertexLibraryIndexType>();

struct BindingOffset {
    offset: usize,
    size: usize,
    description: vk::VertexInputBindingDescription,
}

struct MeasurementResults {
    staging_vertex_buffer_info: vk::BufferCreateInfo<'static>,
    staging_index_buffer_info: vk::BufferCreateInfo<'static>,
    vertex_buffer_info: vk::BufferCreateInfo<'static>,
    index_buffer_info: vk::BufferCreateInfo<'static>,
    binding_offsets: ArrayVec<ArrayVec<BindingOffset, VERTEX_BINDING_COUNT>, PIPELINE_COUNT>,
}

pub struct VertexLibraryBuilder<'a> {
    debug_id: Arguments<'a>,
    vertex_staging: MappedBuffer,
    vertex_buffer: Rc<Buffer>,
    index_staging: MappedBuffer,
    index_buffer: Rc<Buffer>,
    distinct_binding_sets: DistinctBindingSets,
    binding_offsets: ArrayVec<ArrayVec<BindingOffset, VERTEX_BINDING_COUNT>, PIPELINE_COUNT>,
    vertices_allocated: ArrayVec<usize, PIPELINE_COUNT>,
    indices_allocated: ArrayVec<usize, PIPELINE_COUNT>,
}

impl<'a> VertexLibraryBuilder<'a> {
    pub fn new(
        staging_arena: &mut VulkanArena<ForBuffers>,
        measurer: VertexLibraryMeasurer,
        name: Arguments<'a>,
    ) -> Result<VertexLibraryBuilder<'a>, VulkanArenaError> {
        let MeasurementResults {
            staging_vertex_buffer_info,
            staging_index_buffer_info,
            vertex_buffer_info,
            index_buffer_info,
            binding_offsets,
        } = measurer.measure();
        let VertexLibraryMeasurer { distinct_binding_sets, .. } = measurer;

        let vertex_staging =
            staging_arena.create_staging_buffer(staging_vertex_buffer_info, format_args!("{name} (staging vertex buffer)"))?;
        let index_staging =
            staging_arena.create_staging_buffer(staging_index_buffer_info, format_args!("{name} (staging index buffer)"))?;
        let vertex_buffer = staging_arena.create_empty_buffer(vertex_buffer_info, format_args!("{name} (vertex buffer)"))?;
        let index_buffer = staging_arena.create_empty_buffer(index_buffer_info, format_args!("{name} (index buffer)"))?;

        let vertices_allocated = distinct_binding_sets.binding_sets.iter().map(|_| 0).collect();
        let indices_allocated = distinct_binding_sets.binding_sets.iter().map(|_| 0).collect();
        Ok(VertexLibraryBuilder {
            debug_id: name,
            vertex_staging,
            vertex_buffer: Rc::new(vertex_buffer),
            index_staging,
            index_buffer: Rc::new(index_buffer),
            distinct_binding_sets,
            binding_offsets,
            vertices_allocated,
            indices_allocated,
        })
    }

    /// Writes the mesh into the staging buffers and returns a [`Mesh`] that can be rendered after
    /// the uploader passed to [`VertexLibraryBuilder::upload`] has finished uploading it.
    #[track_caller]
    pub fn add_mesh<I: IndexType + Copy>(&mut self, pipeline: PipelineIndex, vertex_buffers: &[&[u8]], index_buffer: &[I]) -> Mesh {
        let (binding_set_idx, vertex_count) = self.distinct_binding_sets.find_set_and_vertex_count(pipeline, vertex_buffers);
        let vertex_offset = self.vertices_allocated[binding_set_idx] as i32;
        let first_index = self.indices_allocated[binding_set_idx] as u32;
        self.vertices_allocated[binding_set_idx] += vertex_count;
        self.indices_allocated[binding_set_idx] += index_buffer.len();

        let index_buffer_start = VERTEX_LIBRARY_INDEX_SIZE * first_index as usize;
        let index_buffer_end = index_buffer_start + VERTEX_LIBRARY_INDEX_SIZE * index_buffer.len();
        let indices_dst: &mut [u8] = &mut self.index_staging.data_mut()[index_buffer_start..index_buffer_end];
        let indices_dst: &mut [VertexLibraryIndexType] = bytemuck::cast_slice_mut(indices_dst);
        for (src_index, dst_index) in index_buffer.iter().zip(indices_dst) {
            let index = src_index.to_u32();
            debug_assert!(index < vertex_count as u32, "index is {index} but mesh has {vertex_count} vertices");
            *dst_index = index;
        }

        let mut vertex_buffer_offsets = ArrayVec::new();
        let vertex_buffer_offset_params = self.binding_offsets[binding_set_idx]
            .iter()
            .filter(|offs| offs.description.input_rate == vk::VertexInputRate::VERTEX);
        for (src, dst_offset) in vertex_buffers.iter().zip(vertex_buffer_offset_params) {
            let stride = dst_offset.description.stride as usize;
            let offset_into_buffer = dst_offset.offset + vertex_offset as usize * stride;
            let fits = offset_into_buffer + src.len() <= dst_offset.offset + dst_offset.size;
            assert!(fits, "given vertex buffer does not fit, check that the measurements are correct");
            let matches_stride = src.len() % stride == 0;
            assert!(matches_stride, "given vertices do not have the correct stride, check pipeline");
            let dst = &mut self.vertex_staging.data_mut()[offset_into_buffer..offset_into_buffer + src.len()];
            dst.copy_from_slice(src);
            vertex_buffer_offsets.push(dst_offset.offset as vk::DeviceSize);
        }

        Mesh {
            vertex_buffer: self.vertex_buffer.clone(),
            vertex_buffer_offsets,
            vertex_offset,
            index_buffer: self.index_buffer.clone(),
            index_buffer_offset: 0,
            first_index,
            index_count: index_buffer.len() as u32,
            index_type: VERTEX_LIBRARY_INDEX_TYPE,
        }
    }

    pub fn upload(self, arena: &mut VulkanArena<ForBuffers>, uploader: &mut Uploader) -> Result<(), VulkanArenaError> {
        arena.copy_buffer(
            self.vertex_staging.buffer,
            &self.vertex_buffer,
            uploader,
            format_args!("{} (vertex buffer)", self.debug_id),
        )?;
        arena.copy_buffer(
            self.index_staging.buffer,
            &self.index_buffer,
            uploader,
            format_args!("{} (index buffer)", self.debug_id),
        )?;
        Ok(())
    }
}

/// Records how many vertices need to be allocated for each distinct vertex layout.
pub struct VertexLibraryMeasurer {
    distinct_binding_sets: DistinctBindingSets,
    vertex_counts_per_binding: ArrayVec<usize, PIPELINE_COUNT>,
    index_counts_per_binding: ArrayVec<usize, PIPELINE_COUNT>,
}

impl VertexLibraryMeasurer {
    #[track_caller]
    pub fn add_mesh<I: IndexType + Copy>(&mut self, pipeline: PipelineIndex, vertex_buffers: &[&[u8]], index_buffers: &[I]) {
        let (binding_set_idx, vertex_count) = self.distinct_binding_sets.find_set_and_vertex_count(pipeline, vertex_buffers);
        self.vertex_counts_per_binding[binding_set_idx] += vertex_count;
        self.index_counts_per_binding[binding_set_idx] += index_buffers.len();
    }

    pub fn measure_required_arena(&self, arena_measurer: &mut VulkanArenaMeasurer<ForBuffers>) -> Result<(), VulkanArenaMeasurementError> {
        let MeasurementResults {
            vertex_buffer_info,
            index_buffer_info,
            ..
        } = self.measure();
        arena_measurer.add_buffer(vertex_buffer_info)?;
        arena_measurer.add_buffer(index_buffer_info)?;
        Ok(())
    }

    fn measure(&self) -> MeasurementResults {
        let index_buffer_size = self.index_counts_per_binding.iter().sum::<usize>() * VERTEX_LIBRARY_INDEX_SIZE;
        let mut vertex_buffer_size = 0;
        let mut binding_offsets = ArrayVec::new();
        for (&vertex_count, &bindings) in self.vertex_counts_per_binding.iter().zip(&self.distinct_binding_sets.binding_sets) {
            let mut offsets = ArrayVec::<BindingOffset, VERTEX_BINDING_COUNT>::new();
            for binding in bindings {
                if binding.input_rate != vk::VertexInputRate::VERTEX {
                    continue;
                }
                let offset = vertex_buffer_size;
                let size = binding.stride as usize * vertex_count;
                vertex_buffer_size += size;
                offsets.push(BindingOffset {
                    offset,
                    size,
                    description: *binding,
                });
            }
            binding_offsets.push(offsets);
        }
        let vertex_buffer_info_base = vk::BufferCreateInfo::default()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size(vertex_buffer_size as vk::DeviceSize);
        let index_buffer_info_base = vk::BufferCreateInfo::default()
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .size(index_buffer_size as vk::DeviceSize);
        MeasurementResults {
            staging_vertex_buffer_info: vertex_buffer_info_base.usage(vk::BufferUsageFlags::TRANSFER_SRC),
            staging_index_buffer_info: index_buffer_info_base.usage(vk::BufferUsageFlags::TRANSFER_SRC),
            vertex_buffer_info: vertex_buffer_info_base.usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST),
            index_buffer_info: index_buffer_info_base.usage(vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST),
            binding_offsets,
        }
    }
}

impl Default for VertexLibraryMeasurer {
    fn default() -> VertexLibraryMeasurer {
        let distinct_binding_sets = DistinctBindingSets::default();
        let vertex_counts_per_binding = distinct_binding_sets.binding_sets.iter().map(|_| 0).collect();
        let index_counts_per_binding = distinct_binding_sets.binding_sets.iter().map(|_| 0).collect();
        VertexLibraryMeasurer {
            distinct_binding_sets,
            vertex_counts_per_binding,
            index_counts_per_binding,
        }
    }
}

struct DistinctBindingSets {
    binding_sets: ArrayVec<&'static [vk::VertexInputBindingDescription], PIPELINE_COUNT>,
    binding_set_indices: PipelineMap<usize>,
}

impl DistinctBindingSets {
    fn find_set_and_vertex_count(&self, pipeline: PipelineIndex, vertex_buffers: &[&[u8]]) -> (usize, usize) {
        let binding_set_idx = self.binding_set_indices[pipeline];
        let descriptions = self.binding_sets[binding_set_idx];
        let mut vertex_count = None;
        for (i, desc) in descriptions
            .iter()
            .filter(|desc| desc.input_rate == vk::VertexInputRate::VERTEX)
            .enumerate()
        {
            assert!(
                i < vertex_buffers.len(),
                "provided only {i} vertex buffers, but pipeline {pipeline:?} has a binding at index {i}"
            );
            let vertex_buffer_length = vertex_buffers[i].len();
            let new_vertex_count = vertex_buffer_length / desc.stride as usize;
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
            (binding_set_idx, vertex_count)
        } else {
            assert!(vertex_buffers.is_empty(), "pipeline {pipeline:?} takes no input vertices");
            (binding_set_idx, 0)
        }
    }
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
        DistinctBindingSets {
            binding_sets: bindings,
            binding_set_indices: binding_indices,
        }
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
