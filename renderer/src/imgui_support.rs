use alloc::rc::Rc;
use core::mem;

use arrayvec::{ArrayString, ArrayVec};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use enum_map::enum_map;
use glam::{Affine3A, Mat4};
use imgui::{BackendFlags, DrawData, DrawIdx, TextureId};

use crate::arena::buffers::ForBuffers;
use crate::arena::images::ForImages;
use crate::arena::{MemoryProps, VulkanArena};
use crate::image_loading::{self, ImageData, TextureKind};
use crate::instance::Instance;
use crate::memory_measurement::VulkanArenaMeasurer;
use crate::physical_device::PhysicalDevice;
use crate::renderer::descriptors::material::Material;
use crate::renderer::descriptors::Descriptors;
use crate::renderer::pipeline_parameters::vertex_buffers::VertexLayout;
use crate::renderer::scene::mesh::Mesh;
use crate::renderer::scene::Scene;
use crate::uploader::Uploader;
use crate::vertex_library::{VertexLibraryBuilder, VertexLibraryMeasurer};
use crate::vulkan_raii::Device;
use crate::VertexBinding;

const MIN_VTX_AND_IDX_BUF_SIZE: vk::DeviceSize = 10_000_000;

pub struct ImGuiRenderer {
    font_material: Rc<Material>,
    mesh_arena: VulkanArena<ForBuffers>,
    meshes: Vec<Mesh>,
}

impl ImGuiRenderer {
    pub fn new(
        imgui: &mut imgui::Context,
        instance: &Instance,
        device: &Device,
        physical_device: &PhysicalDevice,
        descriptors: &mut Descriptors,
    ) -> ImGuiRenderer {
        imgui.set_renderer_name(Some(format!(
            "{} {}.{}.{}",
            env!("CARGO_PKG_NAME"),
            env!("CARGO_PKG_VERSION_MAJOR"),
            env!("CARGO_PKG_VERSION_MINOR"),
            env!("CARGO_PKG_VERSION_PATCH")
        )));
        imgui.io_mut().backend_flags.insert(BackendFlags::RENDERER_HAS_VTX_OFFSET);

        let fonts = imgui.fonts();
        let font_atlas = fonts.build_rgba32_texture();
        let image_data: ImageData = ImageData {
            width: font_atlas.width,
            height: font_atlas.height,
            pixels: font_atlas.data,
            format: vk::Format::R8G8B8A8_UNORM,
            #[allow(clippy::single_range_in_vec_init)]
            mip_ranges: ArrayVec::from_iter([(0..font_atlas.data.len())]),
        };

        let mut image_measurer = VulkanArenaMeasurer::<ForImages>::new(device);
        image_measurer.add_image(image_data.get_create_info(TextureKind::LinearColor));

        // Used as a staging arena for the font texture as well
        let mut mesh_arena = VulkanArena::<ForBuffers>::new(
            &instance.inner,
            device,
            physical_device,
            (image_data.pixels.len() as vk::DeviceSize).max(MIN_VTX_AND_IDX_BUF_SIZE),
            MemoryProps::for_staging(),
            format_args!("imgui images (staging)"),
        )
        .unwrap();
        let mut image_arena = VulkanArena::<ForImages>::new(
            &instance.inner,
            device,
            physical_device,
            image_measurer.measured_size,
            MemoryProps::for_textures(),
            format_args!("imgui images"),
        )
        .unwrap();
        let mut uploader =
            Uploader::new(device, device.graphics_queue, device.transfer_queue, physical_device, "imgui font texture uploader");
        let fonts_texture = image_loading::load_image(
            device,
            &mut mesh_arena,
            &mut uploader,
            &mut image_arena,
            &image_data,
            TextureKind::LinearColor,
            "imgui font texture",
        )
        .unwrap();
        assert!(uploader.wait(None));
        drop(uploader);
        mesh_arena.reset().unwrap();
        let material =
            Material::for_imgui(descriptors, ArrayString::from("imgui font texture material").unwrap(), Rc::new(fonts_texture)).unwrap();
        fonts.tex_id = TextureId::new(material.material_id as usize);

        ImGuiRenderer { font_material: material, mesh_arena, meshes: Vec::new() }
    }

    pub fn render<'a>(&'a mut self, draw_data: &DrawData, scene: &mut Scene<'a>) {
        self.meshes.clear();
        self.mesh_arena.reset().unwrap();

        let mut vertex_library_measurer = VertexLibraryMeasurer::default();
        for draw_list in draw_data.draw_lists() {
            let vertex_buffer_len = mem::size_of_val(draw_list.vtx_buffer());
            let vertex_buffer_lens = enum_map! {
                VertexBinding::Position => Some(vertex_buffer_len),
                VertexBinding::Texcoord0 => Some(vertex_buffer_len),
                VertexBinding::NormalOrColor => Some(vertex_buffer_len),
                _ => None,
            };
            vertex_library_measurer.add_mesh_by_len::<DrawIdx>(VertexLayout::ImGui, &vertex_buffer_lens, draw_list.idx_buffer().len());
        }

        let mut vertex_library_builder =
            VertexLibraryBuilder::new(&mut self.mesh_arena, None, &vertex_library_measurer, format_args!("imgui vertices")).unwrap();
        for draw_list in draw_data.draw_lists() {
            #[derive(Clone, Copy, Zeroable, Pod)]
            #[repr(C)]
            struct MyPodDrawVert {
                pos: [f32; 2],
                uv: [f32; 2],
                col: [u8; 4],
            }
            let vertex_buffer = unsafe { draw_list.transmute_vtx_buffer::<MyPodDrawVert>() };
            let vertex_buffer = bytemuck::cast_slice::<MyPodDrawVert, u8>(vertex_buffer);
            let vertex_buffers = enum_map! {
                VertexBinding::Position => Some(vertex_buffer),
                VertexBinding::Texcoord0 => Some(vertex_buffer),
                VertexBinding::NormalOrColor => Some(vertex_buffer),
                _ => None,
            };
            let base_mesh = vertex_library_builder.add_mesh(VertexLayout::ImGui, &vertex_buffers, draw_list.idx_buffer());
            for draw_cmd in draw_list.commands() {
                if let imgui::DrawCmd::Elements { count, cmd_params } = draw_cmd {
                    self.meshes.push(Mesh {
                        library: base_mesh.library.clone(),
                        vertex_layout: base_mesh.vertex_layout,
                        vertex_offset: base_mesh.vertex_offset + cmd_params.vtx_offset as i32,
                        first_index: base_mesh.first_index + cmd_params.idx_offset as u32,
                        index_count: count as u32,
                        index_type: base_mesh.index_type,
                    });
                }
            }
        }

        let &DrawData { display_pos: [x, y], display_size: [w, h], .. } = draw_data;
        // Mat4 -> Affine3A misses out on the last row, but luckily, orthographic projections don't need the last one!
        let proj_matrix = Affine3A::from_mat4(Mat4::orthographic_rh(x, x + w, y, y + h, -1.0, 1.0));
        for mesh in &self.meshes {
            scene.queue_mesh(mesh, &self.font_material, None, proj_matrix);
        }
    }
}
