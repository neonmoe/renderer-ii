use log::LevelFilter;
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use std::time::{Duration, Instant};

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

    let sdl_context = sdl2::init().map_err(|err| SandboxError::Sdl(err))?;
    let video_subsystem = sdl_context.video().map_err(|err| SandboxError::Sdl(err))?;
    let mut window = video_subsystem
        .window("neonvk sandbox", 800, 600)
        .position_centered()
        .resizable()
        .allow_highdpi()
        .vulkan()
        .build()?;
    let (width, height) = window.vulkan_drawable_size();

    let foundation = neonvk::Foundation::new(&window)?;
    let renderer = neonvk::Renderer::new(&foundation, None)?;
    let mut swapchain = neonvk::Swapchain::new(&renderer, None, width, height)?;

    let mut frame_instants = Vec::with_capacity(10_000);
    frame_instants.push(Instant::now());

    let mut event_pump = sdl_context
        .event_pump()
        .map_err(|err| SandboxError::Sdl(err))?;
    let mut size_changed = false;
    'running: loop {
        renderer.wait_frame();

        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,

                Event::Window { win_event, .. } => match win_event {
                    WindowEvent::SizeChanged(_, _) => size_changed = true,
                    _ => {}
                },

                _ => {}
            }
        }

        if size_changed {
            let (width, height) = window.vulkan_drawable_size();
            swapchain = neonvk::Swapchain::new(&renderer, Some(swapchain), width, height)?;
            // Somehow, the old swapchain needs to not die. Perhaps with Rc's in Renderer?
            // Should render_frame and wait_frame be in renderer, borrowing swapchain? Yes.
            size_changed = false;
        }

        if let Err(err) = renderer.render_frame(&swapchain) {
            log::warn!("Error during regular frame rendering: {}", err);
        }

        frame_instants.push(Instant::now());
        frame_instants.retain(|time| (Instant::now() - *time) < Duration::from_secs(1));
        let interval_count = frame_instants.len() - 1;
        let interval_sum: Duration = frame_instants
            .windows(2)
            .map(|instants| {
                if let &[before, after] = instants {
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
