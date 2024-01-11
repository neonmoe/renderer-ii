use alloc::rc::Rc;
use core::cmp::Ordering;
use core::fmt::Arguments;
use core::hash::{Hash, Hasher};
use core::mem;

use ash::vk;
use ash::vk::Handle;

use crate::arena::buffers::{ForBuffers, MappedBuffer};
use crate::arena::{VulkanArena, VulkanArenaError};
use crate::memory_measurement::VulkanArenaMeasurer;
use crate::renderer::pipeline_parameters::vertex_buffers::{VertexBindingVec, VertexLayout, VertexLayoutMap, VERTEX_BINDING_DESCRIPTIONS};
use crate::renderer::scene::mesh::{IndexType, Mesh};
use crate::uploader::Uploader;
use crate::vulkan_raii::Buffer;

// TODO: Packed vertex attributes (uvs don't need so much precision, etc.)

pub type VertexLibraryIndexType = u32;

pub const VERTEX_LIBRARY_INDEX_TYPE: vk::IndexType = vk::IndexType::UINT32;
const VERTEX_LIBRARY_INDEX_SIZE: usize = mem::size_of::<VertexLibraryIndexType>();

struct VertexBufferInfo {
    /// Offset into the vertex library's vertex buffer.
    offset: usize,
    /// Size of the buffer inside  the vertex library's vertex buffer.
    size: usize,
    /// Stride (in bytes) of the vertices in this buffer.
    stride: usize,
}

struct MeasurementResults {
    staging_vb_create_info: vk::BufferCreateInfo<'static>,
    staging_ib_create_info: vk::BufferCreateInfo<'static>,
    vb_create_info: vk::BufferCreateInfo<'static>,
    ib_create_info: vk::BufferCreateInfo<'static>,
    vertex_buffer_infos: VertexLayoutMap<VertexBindingVec<VertexBufferInfo>>,
    index_buffer_offsets: VertexLayoutMap<usize>,
}

pub struct VertexLibrary {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub vertex_buffer_offsets: VertexLayoutMap<VertexBindingVec<vk::DeviceSize>>,
    pub index_buffer_offsets: VertexLayoutMap<vk::DeviceSize>,
}

impl PartialEq for VertexLibrary {
    fn eq(&self, other: &Self) -> bool {
        self.vertex_buffer.inner.as_raw() == other.vertex_buffer.inner.as_raw()
            && self.index_buffer.inner.as_raw() == other.index_buffer.inner.as_raw()
    }
}

impl Eq for VertexLibrary {}

impl PartialOrd for VertexLibrary {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for VertexLibrary {
    fn cmp(&self, other: &Self) -> Ordering {
        let a = (self.vertex_buffer.inner.as_raw() as u128) | ((self.index_buffer.inner.as_raw() as u128) << 64);
        let b = (other.vertex_buffer.inner.as_raw() as u128) | ((other.index_buffer.inner.as_raw() as u128) << 64);
        a.cmp(&b)
    }
}

impl Hash for VertexLibrary {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vertex_buffer.hash(state);
        self.index_buffer.hash(state);
    }
}

pub struct VertexLibraryBuilder<'name> {
    debug_id: Arguments<'name>,
    library: Rc<VertexLibrary>,
    vertex_staging: MappedBuffer,
    index_staging: MappedBuffer,
    vertex_buffer_infos: VertexLayoutMap<VertexBindingVec<VertexBufferInfo>>,
    index_buffer_offsets: VertexLayoutMap<usize>,
    vertices_allocated: VertexLayoutMap<usize>,
    indices_allocated: VertexLayoutMap<usize>,
}

impl VertexLibraryBuilder<'_> {
    pub fn new<'name>(
        staging_arena: &mut VulkanArena<ForBuffers>,
        buffer_arena: &mut VulkanArena<ForBuffers>,
        measurer: &VertexLibraryMeasurer,
        name: Arguments<'name>,
    ) -> Result<VertexLibraryBuilder<'name>, VulkanArenaError> {
        let MeasurementResults {
            staging_vb_create_info,
            staging_ib_create_info,
            vb_create_info,
            ib_create_info,
            vertex_buffer_infos,
            index_buffer_offsets,
        } = measurer.measure();
        let vertex_staging = staging_arena.create_staging_buffer(staging_vb_create_info, format_args!("{name} (staging vertex buffer)"))?;
        let index_staging = staging_arena.create_staging_buffer(staging_ib_create_info, format_args!("{name} (staging index buffer)"))?;
        let vertex_buffer = buffer_arena.create_empty_buffer(vb_create_info, format_args!("{name} (vertex buffer)"))?;
        let index_buffer = buffer_arena.create_empty_buffer(ib_create_info, format_args!("{name} (index buffer)"))?;
        let vertex_buffer_offsets = vertex_buffer_infos
            .iter()
            .map(|(k, v)| (k, v.iter().map(|info| info.offset as vk::DeviceSize).collect::<VertexBindingVec<_>>()))
            .collect::<VertexLayoutMap<VertexBindingVec<vk::DeviceSize>>>();
        let library = Rc::new(VertexLibrary {
            vertex_buffer,
            index_buffer,
            vertex_buffer_offsets,
            index_buffer_offsets: index_buffer_offsets.map(|_, v| v as vk::DeviceSize),
        });
        let vertices_allocated = VertexLayoutMap::from_fn(|_| 0);
        let indices_allocated = VertexLayoutMap::from_fn(|_| 0);
        Ok(VertexLibraryBuilder {
            debug_id: name,
            library,
            vertex_staging,
            index_staging,
            vertex_buffer_infos,
            index_buffer_offsets,
            vertices_allocated,
            indices_allocated,
        })
    }

    /// Writes the mesh into the staging buffers and returns a [`Mesh`] that can be rendered after
    /// the uploader passed to [`VertexLibraryBuilder::upload`] has finished uploading it.
    #[track_caller]
    pub fn add_mesh<I: IndexType + Copy + core::fmt::Display>(
        &mut self,
        vertex_layout: VertexLayout,
        vertex_buffers: &[&[u8]],
        index_buffer: &[I],
    ) -> Mesh {
        let lengths: VertexBindingVec<usize> = vertex_buffers.iter().map(|buf| buf.len()).collect();
        let vertex_count = get_vertex_count(vertex_layout, &lengths);
        let vertex_offset = self.vertices_allocated[vertex_layout];
        let first_index = self.indices_allocated[vertex_layout];
        self.vertices_allocated[vertex_layout] += vertex_count;
        self.indices_allocated[vertex_layout] += index_buffer.len();

        let index_buffer_start = self.index_buffer_offsets[vertex_layout] + first_index * VERTEX_LIBRARY_INDEX_SIZE;
        let index_buffer_end = index_buffer_start + index_buffer.len() * VERTEX_LIBRARY_INDEX_SIZE;
        let index_bytes: &mut [u8] = &mut self.index_staging.data_mut()[index_buffer_start..index_buffer_end];
        let indices_dst: &mut [VertexLibraryIndexType] = bytemuck::cast_slice_mut(index_bytes);
        for (src_index, dst_index) in index_buffer.iter().zip(indices_dst) {
            let index = src_index.to_u32();
            debug_assert!(index < vertex_count as u32, "index is {index} but mesh has {vertex_count} vertices");
            *dst_index = index;
        }

        let vertex_buffer_offset_params = self.vertex_buffer_infos[vertex_layout].iter_mut();
        for (src, dst_offset) in vertex_buffers.iter().zip(vertex_buffer_offset_params) {
            let fits = src.len() <= dst_offset.size;
            assert!(fits, "given vertex buffer does not fit, check that the measurements are correct");
            let matches_stride = src.len() % dst_offset.stride == 0;
            assert!(matches_stride, "given vertices do not have the correct stride, check pipeline");
            let dst = &mut self.vertex_staging.data_mut()[dst_offset.offset..dst_offset.offset + src.len()];
            dst_offset.offset += src.len();
            dst_offset.size -= src.len();
            dst.copy_from_slice(src);
        }

        Mesh {
            library: self.library.clone(),
            vertex_layout,
            vertex_offset: vertex_offset as i32,
            first_index: first_index as u32,
            index_count: index_buffer.len() as u32,
            index_type: VERTEX_LIBRARY_INDEX_TYPE,
        }
    }

    pub fn upload(self, uploader: &mut Uploader, arena: &mut VulkanArena<ForBuffers>) {
        arena.copy_buffer(
            self.vertex_staging.buffer,
            &self.library.vertex_buffer,
            uploader,
            format_args!("{} (vertex buffer)", self.debug_id),
        );
        arena.copy_buffer(
            self.index_staging.buffer,
            &self.library.index_buffer,
            uploader,
            format_args!("{} (index buffer)", self.debug_id),
        );
    }
}

/// Records how many vertices need to be allocated for each distinct vertex layout.
pub struct VertexLibraryMeasurer {
    vertex_counts: VertexLayoutMap<usize>,
    index_counts: VertexLayoutMap<usize>,
}

impl Default for VertexLibraryMeasurer {
    fn default() -> VertexLibraryMeasurer {
        let vertex_counts = VertexLayoutMap::from_fn(|_| 0);
        let index_counts = VertexLayoutMap::from_fn(|_| 0);
        VertexLibraryMeasurer { vertex_counts, index_counts }
    }
}

impl VertexLibraryMeasurer {
    #[track_caller]
    pub fn add_mesh_by_len<I: IndexType + Copy>(&mut self, vertex_layout: VertexLayout, vertex_buffer_sizes: &[usize], index_count: usize) {
        let vertex_count = get_vertex_count(vertex_layout, vertex_buffer_sizes);
        self.vertex_counts[vertex_layout] += vertex_count;
        self.index_counts[vertex_layout] += index_count;
    }

    #[track_caller]
    pub fn add_mesh<I: IndexType + Copy>(&mut self, vertex_layout: VertexLayout, vertex_buffers: &[&[u8]], index_buffer: &[I]) {
        let lengths: VertexBindingVec<usize> = vertex_buffers.iter().map(|buf| buf.len()).collect();
        let vertex_count = get_vertex_count(vertex_layout, &lengths);
        self.vertex_counts[vertex_layout] += vertex_count;
        self.index_counts[vertex_layout] += index_buffer.len();
    }

    pub fn measure_required_arena(&self, arena_measurer: &mut VulkanArenaMeasurer<ForBuffers>) {
        let MeasurementResults { vb_create_info: vertex_buffer_info, ib_create_info: index_buffer_info, .. } = self.measure();
        arena_measurer.add_buffer(vertex_buffer_info);
        arena_measurer.add_buffer(index_buffer_info);
    }

    fn measure(&self) -> MeasurementResults {
        let mut vertex_buffer_size = 0;
        let mut vertex_buffer_infos = VertexLayoutMap::from_fn(|_| VertexBindingVec::<VertexBufferInfo>::new());
        for (vertex_layout, &vertex_count) in &self.vertex_counts {
            for binding in VERTEX_BINDING_DESCRIPTIONS[vertex_layout] {
                if binding.input_rate != vk::VertexInputRate::VERTEX {
                    continue;
                }
                let offset = vertex_buffer_size;
                let stride = binding.stride as usize;
                let size = stride * vertex_count;
                vertex_buffer_size += size;
                vertex_buffer_infos[vertex_layout].push(VertexBufferInfo { offset, size, stride });
            }
        }
        let mut index_buffer_size = 0;
        let mut index_buffer_offsets = VertexLayoutMap::from_fn(|_| 0);
        for (vertex_layout, index_count) in &self.index_counts {
            index_buffer_offsets[vertex_layout] = index_buffer_size;
            index_buffer_size += index_count * VERTEX_LIBRARY_INDEX_SIZE;
        }
        let vertex_buffer_info_base =
            vk::BufferCreateInfo::default().sharing_mode(vk::SharingMode::EXCLUSIVE).size(vertex_buffer_size as vk::DeviceSize);
        let index_buffer_info_base =
            vk::BufferCreateInfo::default().sharing_mode(vk::SharingMode::EXCLUSIVE).size(index_buffer_size as vk::DeviceSize);
        MeasurementResults {
            staging_vb_create_info: vertex_buffer_info_base.usage(vk::BufferUsageFlags::TRANSFER_SRC),
            staging_ib_create_info: index_buffer_info_base.usage(vk::BufferUsageFlags::TRANSFER_SRC),
            vb_create_info: vertex_buffer_info_base.usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST),
            ib_create_info: index_buffer_info_base.usage(vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST),
            vertex_buffer_infos,
            index_buffer_offsets,
        }
    }
}

fn get_vertex_count(vertex_layout: VertexLayout, vertex_buffer_lengths: &[usize]) -> usize {
    let descs = VERTEX_BINDING_DESCRIPTIONS[vertex_layout];
    let mut prev_vertex_count = None;
    for (desc, buf_len) in descs.iter().filter(|desc| desc.input_rate == vk::VertexInputRate::VERTEX).zip(vertex_buffer_lengths) {
        let vertex_count = buf_len / desc.stride as usize;
        if let Some(prev_vertex_count) = prev_vertex_count {
            assert_eq!(prev_vertex_count, vertex_count, "each buffer must describe the same amount of vertices");
        }
        prev_vertex_count = Some(vertex_count);
    }
    prev_vertex_count.unwrap_or(0)
}
