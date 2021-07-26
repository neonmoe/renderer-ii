use log::LevelFilter;
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use std::time::{Duration, Instant};
use ultraviolet::{Bivec3, Rotor3, Vec3};

use logger::Logger;
static LOGGER: Logger = Logger;

#[derive(thiserror::Error, Debug)]
enum SandboxError {
    #[error("sdl error: {0}")]
    Sdl(String),
}

#[profiling::function]
fn main() -> anyhow::Result<()> {
    log::set_logger(&LOGGER)
        .map(|()| log::set_max_level(LevelFilter::Trace))
        .unwrap();

    let sdl_context = sdl2::init().map_err(SandboxError::Sdl)?;
    let video_subsystem = sdl_context.video().map_err(SandboxError::Sdl)?;
    let mut window = video_subsystem
        .window("neonvk sandbox", 640, 640)
        .position_centered()
        .resizable()
        .allow_highdpi()
        .vulkan()
        .build()?;
    let (width, height) = window.vulkan_drawable_size();

    let driver = neonvk::Driver::new(&window)?;
    let (gpu, _gpus) = neonvk::Gpu::new(&driver, None)?;
    let mut canvas = neonvk::Canvas::new(&gpu, None, width, height)?;
    let camera = neonvk::Camera::new(&gpu, &canvas)?;

    let red = Vec3::new(1.0, 0.1, 0.1);
    let yellow = Vec3::new(0.9, 0.9, 0.1);
    let pink = Vec3::new(0.9, 0.1, 0.9);
    let mut meshes = vec![
        neonvk::Mesh::new(
            &gpu,
            &[
                [Vec3::new(-0.5, -0.5, 0.0), red],
                [Vec3::new(0.5, -0.5, 0.0), pink],
                [Vec3::new(-0.5, 0.5, 0.0), yellow],
                [Vec3::new(0.5, 0.5, 0.0), red],
            ],
            &[0u16, 1, 2, 3, 2, 1],
            neonvk::Pipeline::PlainVertexColor,
            true,
        )?,
        neonvk::Mesh::new(
            &gpu,
            &[
                [Vec3::new(-1.0, -1.0, 0.0), red],
                [Vec3::new(-0.5, -1.0, 0.0), pink],
                [Vec3::new(-1.0, -0.5, 0.0), yellow],
            ],
            &[0u32, 1, 2],
            neonvk::Pipeline::PlainVertexColor,
            false,
        )?,
    ];
    gpu.wait_buffer_uploads()?;

    let start_time = Instant::now();
    let mut frame_instants = Vec::with_capacity(10_000);
    frame_instants.push(Instant::now());

    let mut event_pump = sdl_context.event_pump().map_err(SandboxError::Sdl)?;
    let mut size_changed = false;
    'running: loop {
        let frame_start_seconds = (Instant::now() - start_time).as_secs_f32();

        let rotor = Rotor3::from_angle_plane(frame_start_seconds * 1.0, Bivec3::new(1.0, 0.0, 0.0));
        let vertices = [
            [Vec3::new(-0.5, -0.5, 0.0).rotated_by(rotor), red],
            [Vec3::new(0.5, -0.5, 0.0).rotated_by(rotor), pink],
            [Vec3::new(-0.5, 0.5, 0.0).rotated_by(rotor), yellow],
            [Vec3::new(0.5, 0.5, 0.0).rotated_by(rotor), red],
        ];
        meshes[0].update_vertices(&vertices)?;

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,

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
            canvas = neonvk::Canvas::new(&gpu, Some(&canvas), width, height)?;
            size_changed = false;
        }

        match gpu.render_frame(&canvas, &camera, &meshes) {
            Ok(_) => {}
            Err(neonvk::Error::VulkanSwapchainOutOfDate(_)) => {}
            Err(err) => log::warn!("Error during regular frame rendering: {}", err),
        }
        gpu.wait_frame()?;

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
        if let Some(avg_interval) = interval_sum.checked_div(interval_count as u32) {
            let _ = window.set_title(&format!(
                "{} ({:.2} ms)",
                env!("CARGO_PKG_NAME"),
                avg_interval.as_secs_f64() * 1000.0
            ));
        }
    }

    gpu.wait_idle()?;

    Ok(())
}

mod logger {
    use log::{Level, Log, Metadata, Record};
    use sdl2::messagebox::{show_simple_message_box, MessageBoxFlag};
    use std::thread;

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
                    thread::spawn(move || {
                        if let Some(title_len) = message.char_indices().position(|(_, c)| c == ':')
                        {
                            let title = &message[..title_len];
                            let message = &message[(title_len + 2).min(message.len())..];
                            let _ = show_simple_message_box(
                                MessageBoxFlag::ERROR,
                                title,
                                message,
                                None,
                            );
                        } else {
                            let _ = show_simple_message_box(
                                MessageBoxFlag::ERROR,
                                "Error",
                                &message,
                                None,
                            );
                        }
                    });
                }
            }
        }

        fn flush(&self) {}
    }
}
