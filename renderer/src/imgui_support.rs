use core::mem;
use core::sync::atomic::{AtomicBool, Ordering};

use bytemuck::{Pod, Zeroable};
use enum_map::EnumMap;
use glam::Mat4;
use imgui::internal::RawWrapper;
use imgui::{BackendFlags, DrawData, TextureId};

use crate::renderer::scene::Scene;

pub fn init(imgui: &mut imgui::Context) {
    imgui.set_renderer_name(Some(format!(
        "{} {}.{}.{}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION_MAJOR"),
        env!("CARGO_PKG_VERSION_MINOR"),
        env!("CARGO_PKG_VERSION_PATCH")
    )));
    imgui.io_mut().backend_flags.insert(BackendFlags::RENDERER_HAS_VTX_OFFSET);

    let fonts = imgui.fonts();
    // TODO: Upload this to a cpu-mapped texture, probably will get updated later
    let font_atlas = fonts.build_alpha8_texture();
    fonts.tex_id = TextureId::new(0); // TODO: this should be the texture index to a texture array that the imgui shader uses
}

pub struct ImGuiRender {}

/// Prepares render data for the render thread to render later. Called from the
/// imgui thread.
pub fn prepare_render(draw_data: &DrawData) -> ImGuiRender {
    let render = ImGuiRender {};
    let &DrawData { display_pos: [x, y], display_size: [w, h], .. } = draw_data;
    let proj = Mat4::orthographic_rh(x, x + w, y + h, y, -1.0, 1.0);
    for draw_list in draw_data.draw_lists() {
        let vertex_buf = draw_list.vtx_buffer();
        let index_buf = draw_list.idx_buffer();
        for draw_cmd in draw_list.commands() {
            match draw_cmd {
                imgui::DrawCmd::Elements { count, cmd_params } => {}
                imgui::DrawCmd::ResetRenderState => {}
                imgui::DrawCmd::RawCallback { callback, raw_cmd } => unsafe { callback(draw_list.raw(), raw_cmd) },
            }
        }
    }
    render
}

/// Actually renders out the stuff prepared in [`prepare_render`]. These are two
/// separate functions to allow having imgui and the renderer on different
/// threads.
pub fn render(imgui_render: &ImGuiRender, scene: &mut Scene) {
    todo!();
}
