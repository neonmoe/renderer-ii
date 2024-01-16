use alloc::rc::Rc;
use core::cmp::Ordering;
use core::fmt::Arguments;
use core::hash::{Hash, Hasher};
use core::mem;

use arrayvec::ArrayVec;
use ash::vk;
use ash::vk::Handle;

use crate::arena::buffers::{BufferUsage, ForBuffers};
use crate::arena::{VulkanArena, VulkanArenaError};
use crate::memory_measurement::VulkanArenaMeasurer;
use crate::renderer::pipeline_parameters::vertex_buffers::{
    get_vertex_sizes, write_vertices, VertexBindingMap, VertexLayout, VertexLayoutMap, VertexSizes,
};
use crate::renderer::scene::mesh::{IndexType, Mesh};
use crate::uploader::Uploader;
use crate::vulkan_raii::Buffer;

pub type VertexLibraryIndexType = u16;

pub const VERTEX_LIBRARY_INDEX_TYPE: vk::IndexType = vk::IndexType::UINT16;
const VERTEX_LIBRARY_INDEX_SIZE: usize = mem::size_of::<VertexLibraryIndexType>();

struct BufferInfo {
    offset: usize,
    size: usize,
}

struct MeasurementResults {
    staging_vb_create_info: vk::BufferCreateInfo<'static>,
    staging_ib_create_info: vk::BufferCreateInfo<'static>,
    vb_create_info: vk::BufferCreateInfo<'static>,
    ib_create_info: vk::BufferCreateInfo<'static>,
    vertex_buffer_infos: VertexLayoutMap<VertexBindingMap<BufferInfo>>,
    // TODO: Use BufferInfo for index buffers as well, to avoid overflows
    index_buffer_offsets: VertexLayoutMap<usize>,
}

pub struct VertexLibrary {
    pub vertex_buffer: Buffer,
    pub index_buffer: Buffer,
    pub vertex_buffer_offsets: VertexLayoutMap<VertexBindingMap<vk::DeviceSize>>,
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
    vertex_staging: *mut [u8],
    index_staging: *mut [u8],
    backing_staging_buffers: ArrayVec<Buffer, 2>,
    vertex_buffer_infos: VertexLayoutMap<VertexBindingMap<BufferInfo>>,
    index_buffer_offsets: VertexLayoutMap<usize>,
    vertices_allocated: VertexLayoutMap<usize>,
    indices_allocated: VertexLayoutMap<usize>,
}

impl VertexLibraryBuilder<'_> {
    /// Create a new [`VertexLibrary`] builder, which can be used to create
    /// [`Mesh`]es ("vertex libraries" are the backing memory for meshes). If no
    /// `buffer_arena` is provided, the meshes will use the `staging_arena` as
    /// their backing memory, and [`VertexLibraryBuilder::upload`] does not need
    /// to be called.
    pub fn new<'name>(
        staging_arena: &mut VulkanArena<ForBuffers>,
        buffer_arena: Option<&mut VulkanArena<ForBuffers>>,
        measurer: &VertexLibraryMeasurer,
        name: Arguments<'name>,
    ) -> Result<VertexLibraryBuilder<'name>, VulkanArenaError> {
        let MeasurementResults {
            mut staging_vb_create_info,
            mut staging_ib_create_info,
            vb_create_info,
            ib_create_info,
            vertex_buffer_infos,
            index_buffer_offsets,
        } = measurer.measure();
        let vertex_buffer_offsets = VertexLayoutMap::from_fn(|vl| {
            VertexBindingMap::from_fn(|b| vertex_buffer_infos[vl][b].as_ref().map(|info| info.offset as vk::DeviceSize))
        });
        if buffer_arena.is_none() {
            staging_vb_create_info.usage |= vk::BufferUsageFlags::VERTEX_BUFFER;
            staging_ib_create_info.usage |= vk::BufferUsageFlags::INDEX_BUFFER;
        }
        let vertex_staging = staging_arena.create_staging_buffer(staging_vb_create_info, format_args!("{name} (staging vertex buffer)"))?;
        let index_staging = staging_arena.create_staging_buffer(staging_ib_create_info, format_args!("{name} (staging index buffer)"))?;
        let (vertex_staging_buffer, vertex_staging) = unsafe { vertex_staging.split() };
        let (index_staging_buffer, index_staging) = unsafe { index_staging.split() };
        let mut backing_staging_buffers = ArrayVec::new();
        let (vertex_buffer, index_buffer) = if let Some(buffer_arena) = buffer_arena {
            backing_staging_buffers.push(vertex_staging_buffer);
            backing_staging_buffers.push(index_staging_buffer);
            let vertex_buffer = buffer_arena.create_empty_buffer(vb_create_info, format_args!("{name} (vertex buffer)"))?;
            let index_buffer = buffer_arena.create_empty_buffer(ib_create_info, format_args!("{name} (index buffer)"))?;
            (vertex_buffer, index_buffer)
        } else {
            (vertex_staging_buffer, index_staging_buffer)
        };
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
            backing_staging_buffers,
            vertex_buffer_infos,
            index_buffer_offsets,
            vertices_allocated,
            indices_allocated,
        })
    }

    /// Writes the mesh into the staging buffers and returns a [`Mesh`] that can
    /// be rendered after the uploader passed to
    /// [`VertexLibraryBuilder::upload`] has finished uploading it. All buffers
    /// required by the vertex layout must be provided, others will be silently
    /// ignored.
    #[track_caller]
    pub fn add_mesh<I: IndexType + Copy>(
        &mut self,
        vertex_layout: VertexLayout,
        vertex_buffers: &VertexBindingMap<&[u8]>,
        index_buffer: &[I],
    ) -> Mesh {
        let lengths: VertexBindingMap<usize> = vertex_buffers.map(|_, buf| buf.map(<[u8]>::len));
        let vertex_count = get_vertex_count(vertex_layout, &lengths);
        let vertex_offset = self.vertices_allocated[vertex_layout];
        let first_index = self.indices_allocated[vertex_layout];
        self.vertices_allocated[vertex_layout] += vertex_count;
        self.indices_allocated[vertex_layout] += index_buffer.len();

        // Safety: these mapped buffers still point to valid memory, since the
        // buffers are either in self.backing_staging_buffers or
        // self.vertex_library, which in turn must still exist, since we have a
        // borrow to self for the entirety of this function.
        let vertex_staging: &mut [u8] = unsafe { &mut *self.vertex_staging };
        let index_staging: &mut [u8] = unsafe { &mut *self.index_staging };

        let index_buffer_start = self.index_buffer_offsets[vertex_layout] + first_index * VERTEX_LIBRARY_INDEX_SIZE;
        let index_buffer_end = index_buffer_start + index_buffer.len() * VERTEX_LIBRARY_INDEX_SIZE;
        let index_bytes: &mut [u8] = &mut index_staging[index_buffer_start..index_buffer_end];
        let indices_dst: &mut [VertexLibraryIndexType] = bytemuck::cast_slice_mut(index_bytes);
        for (src_index, dst_index) in index_buffer.iter().zip(indices_dst) {
            let index = src_index.to_index_type();
            debug_assert!(index < vertex_count as VertexLibraryIndexType, "index is {index} but mesh has {vertex_count} vertices");
            *dst_index = index;
        }

        for vertex_binding in vertex_layout.required_inputs() {
            let VertexSizes { in_vertex_size, out_vertex_size, out_vertex_alignment } = get_vertex_sizes(vertex_layout, vertex_binding);
            let dst_offset = self.vertex_buffer_infos[vertex_layout][vertex_binding].as_mut().unwrap();
            dst_offset.offset = dst_offset.offset.next_multiple_of(out_vertex_alignment);
            let src = vertex_buffers[vertex_binding].expect("all bindings required by the vertex layout must be provided");
            let matches_stride = src.len() % in_vertex_size == 0;
            assert!(matches_stride, "given vertices do not have the correct stride, check vertex layout and binding");
            let write_len = out_vertex_size * (src.len() / in_vertex_size);
            let fits = write_len <= dst_offset.size;
            assert!(fits, "given vertex buffer does not fit, check that the measurements are correct");
            let dst = &mut vertex_staging[dst_offset.offset..dst_offset.offset + write_len];
            dst_offset.offset += write_len;
            dst_offset.size -= write_len;
            write_vertices(vertex_layout, vertex_binding, src, dst);
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

    /// Copies the buffers from the staging buffers to the real ones. Must not
    /// be called if the [`VertexLibraryBuilder`] was created without the arena
    /// for the real buffers.
    pub fn upload(mut self, uploader: &mut Uploader) {
        assert_eq!(2, self.backing_staging_buffers.len());
        uploader.copy_buffer(
            BufferUsage::INDEX,
            self.backing_staging_buffers.pop().unwrap(),
            &self.library.index_buffer,
            format_args!("{} (index buffer)", self.debug_id),
        );
        uploader.copy_buffer(
            BufferUsage::VERTEX,
            self.backing_staging_buffers.pop().unwrap(),
            &self.library.vertex_buffer,
            format_args!("{} (vertex buffer)", self.debug_id),
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
    pub fn add_mesh_by_len<I: IndexType + Copy>(
        &mut self,
        vertex_layout: VertexLayout,
        vertex_buffer_sizes: &VertexBindingMap<usize>,
        index_count: usize,
    ) {
        let vertex_count = get_vertex_count(vertex_layout, vertex_buffer_sizes);
        self.vertex_counts[vertex_layout] += vertex_count;
        self.index_counts[vertex_layout] += index_count;
    }

    #[track_caller]
    pub fn add_mesh<I: IndexType + Copy>(
        &mut self,
        vertex_layout: VertexLayout,
        vertex_buffers: &VertexBindingMap<&[u8]>,
        index_buffer: &[I],
    ) {
        let lengths: VertexBindingMap<usize> = vertex_buffers.map(|_, buf| buf.map(<[u8]>::len));
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
        let mut vertex_buffer_size = 0usize;
        let mut vertex_buffer_infos = VertexLayoutMap::from_fn(|_| VertexBindingMap::<BufferInfo>::default());
        for (vertex_layout, &vertex_count) in &self.vertex_counts {
            for vertex_binding in vertex_layout.required_inputs() {
                let VertexSizes { out_vertex_size, out_vertex_alignment, .. } = get_vertex_sizes(vertex_layout, vertex_binding);
                let offset = vertex_buffer_size.next_multiple_of(out_vertex_alignment);
                let padding = offset - vertex_buffer_size;
                let size = out_vertex_size * vertex_count;
                vertex_buffer_size += padding + size;
                vertex_buffer_infos[vertex_layout][vertex_binding] = Some(BufferInfo { offset, size });
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

fn get_vertex_count(vertex_layout: VertexLayout, vertex_buffer_lengths: &VertexBindingMap<usize>) -> usize {
    let mut prev_vertex_count = None;
    for (vertex_binding, buf_len) in vertex_buffer_lengths.into_iter().filter_map(|(binding, len)| len.map(|len| (binding, len))) {
        let VertexSizes { in_vertex_size, .. } = get_vertex_sizes(vertex_layout, vertex_binding);
        assert_eq!(0, buf_len % in_vertex_size, "vertex buffer lengths must be divisible by their vertex sizes");
        let vertex_count = buf_len / in_vertex_size;
        if let Some(prev_vertex_count) = prev_vertex_count {
            assert_eq!(prev_vertex_count, vertex_count, "each buffer must describe the same amount of vertices");
        }
        prev_vertex_count = Some(vertex_count);
    }
    prev_vertex_count.unwrap_or(0)
}
