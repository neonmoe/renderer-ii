use log::LevelFilter;
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::messagebox::{show_simple_message_box, MessageBoxFlag};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::time::{Duration, Instant};

use logger::Logger;
static LOGGER: Logger = Logger;

#[derive(thiserror::Error, Debug)]
enum SandboxError {
    #[error("sdl error: {0}")]
    Sdl(String),
}

fn assert_last_drop<T>(t: Rc<T>) {
    assert!(Rc::strong_count(&t) == 1);
}

#[profiling::function]
fn main() -> anyhow::Result<()> {
    if let Err(err) = fallible_main() {
        let message = err.to_string();
        let _ = show_simple_message_box(MessageBoxFlag::ERROR, "Fatal Error", &message, None);
        Err(err)
    } else {
        Ok(())
    }
}

fn fallible_main() -> anyhow::Result<()> {
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
    let surface = Rc::new(neonvk::create_surface(&driver.entry, &driver.instance, &window)?);
    let physical_devices = neonvk::get_physical_devices(&driver.entry, &driver.instance, surface.inner)?;
    let physical_device = &physical_devices[0];
    let device = Rc::new(neonvk::create_device(&driver.instance, physical_device)?);

    let mut uploader = neonvk::Uploader::new(
        &driver.instance,
        &device,
        device.graphics_queue,
        device.transfer_queue,
        physical_device,
        300_000_000,
    )?;
    let mut assets_arena = neonvk::VulkanArena::new(
        &driver.instance,
        &device,
        physical_device.inner,
        500_000_000,
        neonvk::vk::MemoryPropertyFlags::DEVICE_LOCAL,
        neonvk::vk::MemoryPropertyFlags::DEVICE_LOCAL,
        "sandbox asset arena",
    )?;
    let pbr_defaults = neonvk::PbrDefaults::new(&device, &mut uploader, &mut assets_arena)?;
    // FIXME: Descriptors should get its frame_count based on the swapchain!
    let mut descriptors = neonvk::Descriptors::new(&device, &physical_device.properties, 3, pbr_defaults)?;

    let mut resources = neonvk::GltfResources::with_path(find_resources_path());
    let sponza_model = neonvk::Gltf::from_gltf(
        &device,
        &mut uploader,
        &mut descriptors,
        &mut assets_arena,
        include_str!("sponza/glTF/Sponza.gltf"),
        &mut resources,
    )?;
    profiling::scope!("resource uploading");
    let upload_wait_start = Instant::now();
    assert!(uploader.wait(Duration::from_secs(5))?);
    log::info!("Waited {:?} for the scene to upload.", Instant::now() - upload_wait_start);
    drop(uploader);
    drop(resources);

    let mut canvas = Rc::new(neonvk::Canvas::new(
        &driver.entry,
        &driver.instance,
        &surface,
        &device,
        physical_device,
        &descriptors,
        None,
        width,
        height,
        false,
    )?);
    let mut renderer = neonvk::Renderer::new(&driver.instance, &device, physical_device, &canvas)?;
    let camera = neonvk::Camera::default();

    let mut frame_instants = Vec::with_capacity(10_000);
    frame_instants.push(Instant::now());

    window.show();
    let mut event_pump = sdl_context.event_pump().map_err(SandboxError::Sdl)?;
    let mut size_changed = false;
    let mut immediate_present = false;
    let mut debug_value = 0;
    'running: loop {
        profiling::finish_frame!();

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

                Event::KeyDown { keycode, .. } => match keycode {
                    Some(Keycode::Num0) => debug_value = 0,
                    Some(Keycode::Num1) => debug_value = 1,
                    Some(Keycode::Num2) => debug_value = 2,
                    Some(Keycode::Num3) => debug_value = 3,
                    Some(Keycode::Num4) => debug_value = 4,
                    Some(Keycode::Num5) => debug_value = 5,
                    _ => {}
                },

                Event::Window {
                    win_event: WindowEvent::SizeChanged(_, _),
                    ..
                } => size_changed = true,

                _ => {}
            }
        }

        if size_changed {
            device.wait_idle()?;
            let (width, height) = window.vulkan_drawable_size();
            let old_canvas = canvas;
            canvas = Rc::new(neonvk::Canvas::new(
                &driver.entry,
                &driver.instance,
                &surface,
                &device,
                physical_device,
                &descriptors,
                Some(&old_canvas),
                width,
                height,
                immediate_present,
            )?);
            drop(renderer);
            assert_last_drop(old_canvas);
            renderer = neonvk::Renderer::new(&driver.instance, &device, physical_device, &canvas)?;
            size_changed = false;
        }

        let mut scene = neonvk::Scene::new();
        for (mesh, material, transform) in sponza_model.mesh_iter() {
            scene.queue(mesh, material, transform);
        }

        let frame_index = renderer.wait_frame(&canvas)?;
        match renderer.render_frame(frame_index, &mut descriptors, &canvas, &camera, &scene, debug_value) {
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
        if let Some(avg_interval) = interval_sum.checked_div(interval_count as u32) {
            // TODO: VRAM stats
            // NOTE: Remove the dead_code annotation from display_bytes when fixing this
            let _ = window.set_title(&format!(
                "{} ({:.2} ms frame interval ({:.0} fps), {} of VRAM in use, {} allocated)",
                env!("CARGO_PKG_NAME"),
                avg_interval.as_secs_f64() * 1000.0,
                1.0 / avg_interval.as_secs_f64(),
                -1,
                -1,
            ));
        }
    }

    device.wait_idle()?;

    // Per-resize objects.
    drop(renderer);
    assert_last_drop(canvas);

    // Per-device-objects.
    drop(sponza_model);
    drop(assets_arena);
    drop(descriptors);
    assert_last_drop(device);
    assert_last_drop(surface);

    Ok(())
}

#[allow(dead_code)]
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

/// Attempts to find the sponza/glTF directory to be used as a
/// resources-directory.
fn find_resources_path() -> PathBuf {
    let current_path = Path::new(".").canonicalize().unwrap();
    let path = if current_path.ends_with("src") {
        "bin/sponza/glTF"
    } else if current_path.ends_with("bin") {
        "sponza/glTF"
    } else if current_path.ends_with("sponza") {
        "glTF"
    } else if current_path.ends_with("glTF") {
        "."
    } else {
        "src/bin/sponza/glTF"
    };
    PathBuf::from(path)
}

mod logger {
    use log::{Level, Log, Metadata, Record};

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
            }
        }

        fn flush(&self) {}
    }
}
