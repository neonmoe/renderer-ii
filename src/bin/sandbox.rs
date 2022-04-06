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

    let sdl_context = {
        profiling::scope!("SDL init");
        sdl2::init().map_err(SandboxError::Sdl)?
    };
    let video_subsystem = {
        profiling::scope!("SDL video subsystem init");
        sdl_context.video().map_err(SandboxError::Sdl)?
    };

    let mut window = {
        profiling::scope!("SDL window creation");
        video_subsystem
            .window("neonvk sandbox", 640, 480)
            .position_centered()
            .resizable()
            .allow_highdpi()
            .vulkan()
            .hidden()
            .build()?
    };

    let (width, height) = window.vulkan_drawable_size();

    let instance = neonvk::Instance::new(&window)?;
    let surface = Rc::new(neonvk::create_surface(&instance.entry, &instance.inner, &window)?);
    let physical_devices = neonvk::get_physical_devices(&instance.entry, &instance.inner, surface.inner)?;
    let physical_device = &physical_devices[0];
    let device = Rc::new(neonvk::create_device(&instance.inner, physical_device)?);

    let mut uploader = neonvk::Uploader::new(
        &instance.inner,
        &device,
        device.graphics_queue,
        device.transfer_queue,
        physical_device,
        300_000_000,
        "sandbox assets",
    )?;
    let mut assets_buffers_arena = neonvk::VulkanArena::new(
        &instance.inner,
        &device,
        physical_device.inner,
        12_000_000,
        neonvk::vk::MemoryPropertyFlags::DEVICE_LOCAL
            | neonvk::vk::MemoryPropertyFlags::HOST_VISIBLE
            | neonvk::vk::MemoryPropertyFlags::HOST_COHERENT,
        neonvk::vk::MemoryPropertyFlags::DEVICE_LOCAL,
        format_args!("sandbox assets (buffers)"),
    )?;
    let mut assets_textures_arena = neonvk::VulkanArena::new(
        &instance.inner,
        &device,
        physical_device.inner,
        100_000_000,
        neonvk::vk::MemoryPropertyFlags::DEVICE_LOCAL,
        neonvk::vk::MemoryPropertyFlags::DEVICE_LOCAL,
        format_args!("sandbox assets (textures)"),
    )?;
    let pbr_defaults = neonvk::PbrDefaults::new(&device, &mut uploader, &mut assets_textures_arena)?;
    let mut descriptors = neonvk::Descriptors::new(&device, physical_device.properties, pbr_defaults, 1)?;

    let sponza_model;
    {
        profiling::scope!("loading Sponza.gltf from disk to vram");
        let resources_path = find_resources_path();
        let upload_start = Instant::now();
        sponza_model = neonvk::Gltf::from_gltf(
            &device,
            &mut uploader,
            &mut descriptors,
            (&mut assets_buffers_arena, &mut assets_textures_arena),
            &resources_path.join("Sponza.gltf"),
            &resources_path,
        )?;
        let upload_wait_start = Instant::now();
        assert!(uploader.wait(Duration::from_secs(5))?);
        let now = Instant::now();
        log::info!(
            "Spent {:?} loading resources, of which {:?} was waiting for upload.",
            now - upload_start,
            now - upload_wait_start
        );
        drop(uploader);
    }

    let pipelines = neonvk::Pipelines::new(&device, physical_device, &descriptors)?;
    let mut swapchain_settings = neonvk::SwapchainSettings {
        extent: neonvk::vk::Extent2D { width, height },
        immediate_present: false,
    };
    let mut swapchain = neonvk::Swapchain::new(
        &instance.entry,
        &instance.inner,
        &surface,
        &device,
        physical_device,
        None,
        &swapchain_settings,
    )?;
    let mut framebuffers = neonvk::Framebuffers::new(&instance.inner, &device, physical_device, &pipelines, &swapchain)?;
    let mut renderer = neonvk::Renderer::new(&instance.inner, &device, physical_device, swapchain.frame_count())?;
    descriptors = neonvk::Descriptors::from_existing(descriptors, swapchain.frame_count())?;

    let mut frame_processing_durations = Vec::with_capacity(10_000);

    window.show();
    let mut event_pump = sdl_context.event_pump().map_err(SandboxError::Sdl)?;
    let mut size_changed = false;
    let mut debug_value = 0;
    'running: loop {
        let update_start_time = Instant::now();
        for event in event_pump.poll_iter() {
            profiling::scope!("handle SDL event");
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => break 'running,

                Event::KeyDown {
                    keycode: Some(Keycode::I), ..
                } => {
                    swapchain_settings.immediate_present = !swapchain_settings.immediate_present;
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
            profiling::scope!("handle resize");
            device.wait_idle()?;
            let (width, height) = window.vulkan_drawable_size();
            drop(renderer);
            drop(framebuffers);
            let old_swapchain = swapchain;
            swapchain_settings.extent = neonvk::vk::Extent2D { width, height };
            swapchain = neonvk::Swapchain::new(
                &instance.entry,
                &instance.inner,
                &surface,
                &device,
                physical_device,
                Some(&old_swapchain),
                &swapchain_settings,
            )?;
            assert!(old_swapchain.no_external_refs());
            drop(old_swapchain);
            framebuffers = neonvk::Framebuffers::new(&instance.inner, &device, physical_device, &pipelines, &swapchain)?;
            renderer = neonvk::Renderer::new(&instance.inner, &device, physical_device, swapchain.frame_count())?;
            if swapchain.frame_count() != descriptors.frame_count {
                descriptors = neonvk::Descriptors::from_existing(descriptors, swapchain.frame_count())?;
            }
            size_changed = false;
        }

        let mut scene = neonvk::Scene::default();
        {
            profiling::scope!("queue meshes to render");
            for (mesh, material, transform) in sponza_model.mesh_iter() {
                scene.queue(mesh, material, transform);
            }
        }

        let update_duration = Instant::now() - update_start_time;
        let frame_index = {
            // Xwayland waits here for vsync, don't take into account for frame times
            renderer.wait_frame(&swapchain)?
        };
        let render_start_time = Instant::now();
        match renderer.render_frame(frame_index, &mut descriptors, &pipelines, &framebuffers, &scene, debug_value) {
            Ok(_) => {}
            Err(err) => log::warn!("Error during regular frame rendering: {}", err),
        }
        let render_duration = Instant::now() - render_start_time;
        match {
            // Wayland waits here for vsync, don't take into account for frame times
            renderer.present_frame(frame_index, &swapchain)
        } {
            Ok(_) => {}
            Err(neonvk::Error::VulkanSwapchainOutOfDate(_)) => size_changed = true,
            Err(err) => log::warn!("Error during regular frame present: {}", err),
        }

        {
            profiling::scope!("frame time tracking");
            let now = Instant::now();
            let frame_duration = update_duration + render_duration;
            frame_processing_durations.push((now, frame_duration));
            frame_processing_durations.retain(|(time, _)| (now - *time) < Duration::from_secs(1));
            let avg_frame_duration =
                frame_processing_durations.iter().map(|(_, d)| d.as_secs_f64()).sum::<f64>() / frame_processing_durations.len() as f64;
            let used_memory = neonvk::get_allocated_vram_in_use();
            let actual_used_memory = physical_device.get_memory_usage(&instance.inner).unwrap_or(used_memory);
            let external_used_memory = actual_used_memory - used_memory;
            let allocated_memory = neonvk::get_allocated_vram();
            let _ = window.set_title(&format!(
                "{} ({:.3} ms frametime, {:.0} fps, {} / {} of VRAM in use / allocated + {} not accounted for)",
                env!("CARGO_PKG_NAME"),
                avg_frame_duration * 1000.0,
                frame_processing_durations.len(),
                display_bytes(used_memory),
                display_bytes(allocated_memory),
                display_bytes(external_used_memory),
            ));
        }

        profiling::finish_frame!();
    }

    {
        profiling::scope!("clean up on exit");
        device.wait_idle()?;

        // Per-resize objects.
        drop(renderer);
        drop(framebuffers);
        drop(swapchain);
        drop(pipelines);

        // Per-device-objects.
        drop(sponza_model);
        drop(assets_textures_arena);
        drop(assets_buffers_arena);
        drop(descriptors);
        assert_last_drop(device);
        assert_last_drop(surface);
        drop(instance);
    }

    {
        profiling::scope!("clean up on exit (SDL)");
        drop(event_pump);
        drop(window);
        drop(video_subsystem);
        drop(sdl_context);
    }

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
