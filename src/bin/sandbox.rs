use log::LevelFilter;
use neonvk::vk;
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use std::time::{Duration, Instant};
use ultraviolet::Mat4;

use logger::Logger;
static LOGGER: Logger = Logger;

#[derive(thiserror::Error, Debug)]
enum SandboxError {
    #[error("sdl error: {0}")]
    Sdl(String),
}

#[profiling::function]
fn load_png(bytes: &[u8]) -> (u32, u32, Vec<u8>, vk::Format) {
    let decoder = png::Decoder::new(bytes);
    let (info, mut reader) = decoder.read_info().unwrap();
    let mut bytes = vec![0; info.buffer_size()];
    reader.next_frame(&mut bytes).unwrap();
    debug_assert_eq!((info.width * info.height * 4) as usize, bytes.len());
    (info.width, info.height, bytes, vk::Format::R8G8B8A8_SRGB)
}

#[profiling::function]
fn main() -> anyhow::Result<()> {
    log::set_logger(&LOGGER).map(|()| log::set_max_level(LevelFilter::Trace)).unwrap();

    let sdl_context = sdl2::init().map_err(SandboxError::Sdl)?;
    let video_subsystem = sdl_context.video().map_err(SandboxError::Sdl)?;

    // TODO: Another window for a loading splash screen?

    let mut window = video_subsystem
        .window("neonvk sandbox", 640, 480)
        .position_centered()
        .resizable()
        .allow_highdpi()
        .vulkan()
        .hidden()
        .build()?;
    let (width, height) = window.vulkan_drawable_size();

    let driver = neonvk::Driver::new(&window)?;
    let (gpu, _gpus) = neonvk::Gpu::new(&driver, None)?;
    let mut canvas = neonvk::Canvas::new(&gpu, None, width, height, false)?;

    let loading_frame_index = gpu.wait_frame(&canvas)?;
    let camera = neonvk::Camera::new();

    let cube_model = neonvk::Gltf::from_glb(&gpu, loading_frame_index, include_bytes!("testbox/testbox.glb"), None)?;

    let (tex_w, tex_h, tex_bytes, tex_format) = load_png(include_bytes!("testbox/testbox_albedo_texture.png"));
    let tree_texture = neonvk::Texture::new(&gpu, loading_frame_index, &tex_bytes, tex_w, tex_h, tex_format)?;

    let quad_vertices: &[&[u8]] = &[
        bytemuck::bytes_of(&[[-0.5f32, 0.5, 0.0], [-0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [0.5, -0.5, 0.0]]),
        bytemuck::bytes_of(&[[0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
    ];
    let quad_indices: &[u8] = bytemuck::bytes_of(&[0u16, 1, 2, 3, 2, 1]);
    let quad = neonvk::Mesh::new::<u16>(&gpu, loading_frame_index, quad_vertices, quad_indices, neonvk::Pipeline::Default)?;
    // Get the first frame out of the way, to upload the meshes.
    // TODO: Add a proper way to upload resources before the game loop
    gpu.set_pipeline_textures(loading_frame_index, neonvk::Pipeline::Default, &[&tree_texture]);
    gpu.render_frame(loading_frame_index, &canvas, &camera, &neonvk::Scene::new())?;

    let start_time = Instant::now();
    let mut frame_instants = Vec::with_capacity(10_000);
    frame_instants.push(Instant::now());

    window.show();
    let mut event_pump = sdl_context.event_pump().map_err(SandboxError::Sdl)?;
    let mut size_changed = false;
    let mut immediate_present = false;
    'running: loop {
        profiling::finish_frame!();
        let frame_start_seconds = (Instant::now() - start_time).as_secs_f32();

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,

                Event::KeyDown {
                    keycode: Some(Keycode::I), ..
                } => {
                    immediate_present = !immediate_present;
                    size_changed = true;
                }

                Event::Window {
                    win_event: WindowEvent::SizeChanged(_, _),
                    ..
                } => size_changed = true,

                _ => {}
            }
        }

        if size_changed {
            gpu.wait_idle()?;
            let (width, height) = window.vulkan_drawable_size();
            canvas = neonvk::Canvas::new(&gpu, Some(&canvas), width, height, immediate_present)?;
            size_changed = false;
        }

        let mut scene = neonvk::Scene::new();
        let rotation = Mat4::from_rotation_z(frame_start_seconds * 0.1)
            * Mat4::from_rotation_x(frame_start_seconds * 0.1)
            * Mat4::from_rotation_y(frame_start_seconds * 0.1);
        scene.queue(&quad, Mat4::from_translation(ultraviolet::Vec3::new(-0.5, 0.0, 0.0)) * rotation);
        for mesh in &cube_model.meshes {
            scene.queue(
                mesh,
                Mat4::from_translation(ultraviolet::Vec3::new(0.5, 0.0, 0.0)) * Mat4::from_scale(0.5) * rotation,
            );
        }

        let frame_index = gpu.wait_frame(&canvas)?;
        gpu.set_pipeline_textures(frame_index, neonvk::Pipeline::Default, &[&tree_texture]);
        match gpu.render_frame(frame_index, &canvas, &camera, &scene) {
            Ok(_) => {}
            Err(neonvk::Error::VulkanSwapchainOutOfDate(_)) => {}
            Err(err) => log::warn!("Error during regular frame rendering: {}", err),
        }

        frame_instants.push(Instant::now());
        frame_instants.retain(|time| (Instant::now() - *time) < Duration::from_secs(1));
        let interval_count = frame_instants.len() - 1;
        let interval_sum: Duration = frame_instants
            .windows(2)
            .map(|instants| {
                if let [before, after] = *instants {
                    after - before
                } else {
                    unreachable!()
                }
            })
            .sum();
        let vram_usage = gpu
            .calculate_vram_stats()
            .map(|stats| (stats.total.usedBytes, stats.total.usedBytes + stats.total.unusedBytes));
        if let (Some(avg_interval), Ok((used, alloced))) = (interval_sum.checked_div(interval_count as u32), vram_usage) {
            let _ = window.set_title(&format!(
                "{} ({:.2} ms frame interval, {} of VRAM in use, {} allocated)",
                env!("CARGO_PKG_NAME"),
                avg_interval.as_secs_f64() * 1000.0,
                display_bytes(used),
                display_bytes(alloced),
            ));
        }
    }

    gpu.wait_idle()?;

    Ok(())
}

fn display_bytes(bytes: u64) -> String {
    const KIBI: u64 = 1_024;
    const MEBI: u64 = KIBI * KIBI;
    const GIBI: u64 = KIBI * KIBI * KIBI;
    match bytes {
        x if x < KIBI => format!("{:.1} bytes", bytes as f32),
        x if x < MEBI => format!("{:.1} KiB", bytes as f32 / KIBI as f32),
        x if x < GIBI => format!("{:.1} MiB", bytes as f32 / MEBI as f32),
        _ => format!("{:.1} GiB", bytes as f32 / GIBI as f32),
    }
}

mod logger {
    use log::{Level, Log, Metadata, Record};
    use sdl2::messagebox::{show_simple_message_box, MessageBoxFlag};

    pub struct Logger;

    impl Log for Logger {
        fn enabled(&self, _metadata: &Metadata) -> bool {
            true
        }

        fn log(&self, record: &Record) {
            if self.enabled(record.metadata()) {
                let message = format!("{}", record.args());
                let (color_code, color_end) = if cfg!(target_family = "unix") {
                    let start = match record.level() {
                        Level::Trace => "\u{1B}[34m", /* blue */
                        Level::Debug => "\u{1B}[36m", /* cyan */
                        Level::Info => "\u{1B}[32m",  /* green */
                        Level::Warn => "\u{1B}[33m",  /* yellow */
                        Level::Error => "\u{1B}[31m", /* red */
                    };
                    (start, "\u{1B}[m")
                } else {
                    ("", "")
                };
                if record.level() < Level::Trace {
                    eprintln!(
                        "{}[{}:{}] {}{}",
                        color_code,
                        record.file().unwrap_or(""),
                        record.line().unwrap_or(0),
                        message,
                        color_end,
                    );
                }
                if record.level() == Level::Error {
                    if let Some(title_len) = message.char_indices().position(|(_, c)| c == ':') {
                        let title = &message[..title_len];
                        let message = &message[(title_len + 2).min(message.len())..];
                        let _ = show_simple_message_box(MessageBoxFlag::ERROR, title, message, None);
                    } else {
                        let _ = show_simple_message_box(MessageBoxFlag::ERROR, "Error", &message, None);
                    }
                }
            }
        }

        fn flush(&self) {}
    }
}
