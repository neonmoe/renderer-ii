use glam::{Mat4, Quat, Vec3};
use log::LevelFilter;
use sdl2::controller::Axis;
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::messagebox::{show_simple_message_box, MessageBoxFlag};
use sdl2::mouse::MouseButton;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use logger::Logger;
static LOGGER: Logger = Logger;

#[derive(thiserror::Error, Debug)]
enum SandboxError {
    #[error("sdl error: {0}")]
    Sdl(String),
    #[error("MSAA sample count not supported: {0:?}")]
    MsaaSampleCountNotSupported(neonvk::vk::SampleCountFlags),
}

#[profiling::function]
fn main() -> anyhow::Result<()> {
    if let Err(err) = fallible_main() {
        let message = format!("{:?}", err);
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

    // Enable all controllers
    let controller_subsystem = sdl_context.game_controller().unwrap();
    let controller = controller_subsystem.open(0);

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
    let surface = neonvk::create_surface(&instance.entry, &instance.inner, &window)?;
    let mut physical_devices = neonvk::get_physical_devices(&instance.entry, &instance.inner, surface.inner)?;
    let physical_device = physical_devices.remove(0)?;
    let mut device = neonvk::create_device(&instance.inner, &physical_device)?;
    let mut descriptors = neonvk::Descriptors::new(&instance, &device, &physical_device)?;

    let msaa_samples = neonvk::vk::SampleCountFlags::TYPE_4;
    if !physical_device
        .properties
        .limits
        .framebuffer_color_sample_counts
        .contains(msaa_samples)
    {
        return Err(SandboxError::MsaaSampleCountNotSupported(msaa_samples).into());
    }

    let resources_path = find_resources_path();
    let mut assets_buffers_measurer = neonvk::VulkanArenaMeasurer::new(&device);
    let mut assets_textures_measurer = neonvk::VulkanArenaMeasurer::new(&device);
    neonvk::measure_gltf_memory_usage(
        (&mut assets_buffers_measurer, &mut assets_textures_measurer),
        &resources_path.join("sponza/glTF/Sponza.gltf"),
        &resources_path.join("sponza/glTF"),
    )?;
    neonvk::measure_gltf_memory_usage(
        (&mut assets_buffers_measurer, &mut assets_textures_measurer),
        &resources_path.join("smol-ame-by-seafoam/smol-ame.gltf"),
        &resources_path.join("smol-ame-by-seafoam"),
    )?;

    let mut uploader = neonvk::Uploader::new(
        &instance.inner,
        &device,
        device.graphics_queue,
        device.transfer_queue,
        &physical_device,
        assets_buffers_measurer.measured_size + assets_textures_measurer.measured_size,
        "sandbox assets",
    )?;
    let mut assets_buffers_arena = neonvk::VulkanArena::new(
        &instance.inner,
        &device,
        &physical_device,
        assets_buffers_measurer.measured_size,
        neonvk::vk::MemoryPropertyFlags::DEVICE_LOCAL
            | neonvk::vk::MemoryPropertyFlags::HOST_VISIBLE
            | neonvk::vk::MemoryPropertyFlags::HOST_COHERENT,
        neonvk::vk::MemoryPropertyFlags::DEVICE_LOCAL,
        format_args!("sandbox assets (buffers)"),
    )?;
    let mut assets_textures_arena = neonvk::VulkanArena::new(
        &instance.inner,
        &device,
        &physical_device,
        assets_textures_measurer.measured_size,
        neonvk::vk::MemoryPropertyFlags::DEVICE_LOCAL,
        neonvk::vk::MemoryPropertyFlags::DEVICE_LOCAL,
        format_args!("sandbox assets (textures)"),
    )?;

    let sponza_model;
    let smol_ame_model;
    {
        profiling::scope!("loading Sponza.gltf from disk to vram");
        let upload_start = Instant::now();
        sponza_model = neonvk::Gltf::from_gltf(
            &device,
            &mut uploader,
            &mut descriptors,
            (&mut assets_buffers_arena, &mut assets_textures_arena),
            &resources_path.join("sponza/glTF/Sponza.gltf"),
            &resources_path.join("sponza/glTF"),
        )?;
        smol_ame_model = neonvk::Gltf::from_gltf(
            &device,
            &mut uploader,
            &mut descriptors,
            (&mut assets_buffers_arena, &mut assets_textures_arena),
            &resources_path.join("smol-ame-by-seafoam/smol-ame.gltf"),
            &resources_path.join("smol-ame-by-seafoam"),
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

    assert_eq!(assets_buffers_arena.memory_in_use(), assets_buffers_measurer.measured_size);
    assert_eq!(assets_textures_arena.memory_in_use(), assets_textures_measurer.measured_size);

    let mut swapchain_settings = neonvk::SwapchainSettings {
        extent: neonvk::vk::Extent2D { width, height },
        immediate_present: false,
    };
    let mut swapchain = neonvk::Swapchain::new(
        &instance.entry,
        &instance.inner,
        &device,
        &physical_device,
        neonvk::SwapchainBase::Surface(surface),
        &swapchain_settings,
    )?;
    let mut pipelines = neonvk::Pipelines::new(&device, &physical_device, &descriptors, swapchain.extent, msaa_samples, None)?;
    let mut framebuffers = neonvk::Framebuffers::new(&instance.inner, &device, &physical_device, &pipelines, &swapchain)?;
    let mut renderer = neonvk::Renderer::new(&instance.inner, &device, &physical_device)?;

    let mut frame_processing_durations = Vec::with_capacity(10_000);

    window.show();
    let mut event_pump = sdl_context.event_pump().map_err(SandboxError::Sdl)?;
    let mut resize_timestamp = None;
    let mut debug_value = 0;
    let mut update_time = Instant::now();
    let mut game_time = 0.0;
    let (mut cam_x, mut cam_y, mut cam_z, mut cam_yaw, mut cam_pitch) = (3.0, 1.6, 0.0, 1.56, 0.0);
    let (mut dx, mut dy, mut dz) = (0.0, 0.0, 0.0);
    let mut analog_controls = false;
    let (mut mouse_look, mut sprinting) = (false, false);
    'running: loop {
        let update_start_time = Instant::now();
        let (mut cam_yaw_delta, mut cam_pitch_delta) = (0.0, 0.0);
        {
            profiling::scope!("handle SDL events");
            for event in event_pump.poll_iter() {
                match event {
                    Event::Quit { .. } => break 'running,

                    Event::KeyDown { keycode, .. } => {
                        analog_controls = false;
                        match keycode {
                            Some(Keycode::Num0) => debug_value = 0,
                            Some(Keycode::Num1) => debug_value = 1,
                            Some(Keycode::Num2) => debug_value = 2,
                            Some(Keycode::Num3) => debug_value = 3,
                            Some(Keycode::Num4) => debug_value = 4,
                            Some(Keycode::Num5) => debug_value = 5,
                            Some(Keycode::W) => dz = -1.0,
                            Some(Keycode::A) => dx = -1.0,
                            Some(Keycode::S) => dz = 1.0,
                            Some(Keycode::D) => dx = 1.0,
                            Some(Keycode::E) => dy = 1.0,
                            Some(Keycode::Q) => dy = -1.0,
                            Some(Keycode::LShift) => sprinting = true,
                            Some(Keycode::Escape) if mouse_look => {
                                mouse_look = false;
                                sdl_context.mouse().set_relative_mouse_mode(false);
                                sdl_context.mouse().show_cursor(true);
                            }
                            _ => {}
                        }
                    }

                    Event::KeyUp { keycode, .. } => match keycode {
                        Some(Keycode::I) => {
                            swapchain_settings.immediate_present = !swapchain_settings.immediate_present;
                            resize_timestamp = Some(Instant::now());
                        }
                        Some(Keycode::W) if dz == -1.0 => dz = 0.0,
                        Some(Keycode::A) if dx == -1.0 => dx = 0.0,
                        Some(Keycode::S) if dz == 1.0 => dz = 0.0,
                        Some(Keycode::D) if dx == 1.0 => dx = 0.0,
                        Some(Keycode::E) if dy == 1.0 => dy = 0.0,
                        Some(Keycode::Q) if dy == -1.0 => dy = 0.0,
                        Some(Keycode::LShift) => sprinting = false,
                        _ => {}
                    },

                    Event::ControllerAxisMotion { axis, value, .. } => {
                        analog_controls = true;
                        match axis {
                            Axis::LeftX => dx = get_axis_deadzoned(value),
                            Axis::LeftY => dz = get_axis_deadzoned(value),
                            Axis::TriggerRight if value != 0 => dy = value as f32 / i16::MAX as f32,
                            Axis::TriggerRight if dy > 0.0 => dy = 0.0,
                            Axis::TriggerLeft if value != 0 => dy = -(value as f32 / i16::MAX as f32),
                            Axis::TriggerLeft if dy < 0.0 => dy = 0.0,
                            _ => {}
                        }
                    }

                    Event::MouseButtonDown {
                        mouse_btn: MouseButton::Left,
                        ..
                    } => {
                        mouse_look = !mouse_look;
                        if mouse_look {
                            sdl_context.mouse().set_relative_mouse_mode(true);
                            sdl_context.mouse().show_cursor(false);
                        } else {
                            sdl_context.mouse().set_relative_mouse_mode(false);
                            sdl_context.mouse().show_cursor(true);
                        }
                    }

                    Event::MouseMotion { xrel, yrel, .. } => {
                        if mouse_look {
                            cam_yaw_delta += -xrel as f32 / 750.0;
                            cam_pitch_delta += -yrel as f32 / 750.0;
                        }
                    }

                    Event::Window {
                        win_event: WindowEvent::SizeChanged(_, _),
                        ..
                    } => {
                        mouse_look = false;
                        sdl_context.mouse().set_relative_mouse_mode(false);
                        sdl_context.mouse().show_cursor(true);
                        resize_timestamp = Some(Instant::now())
                    }

                    _ => {}
                }
            }
        }

        if analog_controls {
            if let Ok(controller) = &controller {
                cam_yaw_delta = -get_axis_deadzoned(controller.axis(Axis::RightX)) / 50.0;
                cam_pitch_delta = -get_axis_deadzoned(controller.axis(Axis::RightY)) / 50.0;
            }
        }
        cam_yaw += cam_yaw_delta;
        cam_pitch = (cam_pitch + cam_pitch_delta).clamp(-std::f32::consts::FRAC_PI_2, std::f32::consts::FRAC_PI_2);

        if let Some(duration_since_resize) = resize_timestamp.and_then(|t| Instant::now().checked_duration_since(t)) {
            if duration_since_resize > Duration::from_millis(100) {
                profiling::scope!("handle resize");
                device.wait_idle()?;
                drop(framebuffers);
                let (width, height) = window.vulkan_drawable_size();
                swapchain_settings.extent = neonvk::vk::Extent2D { width, height };
                swapchain = neonvk::Swapchain::new(
                    &instance.entry,
                    &instance.inner,
                    &device,
                    &physical_device,
                    neonvk::SwapchainBase::OldSwapchain(swapchain),
                    &swapchain_settings,
                )?;
                pipelines = neonvk::Pipelines::new(
                    &device,
                    &physical_device,
                    &descriptors,
                    swapchain.extent,
                    msaa_samples,
                    Some(pipelines),
                )?;
                framebuffers = neonvk::Framebuffers::new(&instance.inner, &device, &physical_device, &pipelines, &swapchain)?;
                resize_timestamp = None;
            }
        }

        let new_update_time = Instant::now();
        let dt = (new_update_time - update_time).as_secs_f32();
        game_time += dt;
        update_time = new_update_time;

        let mut scene = neonvk::Scene::default();
        scene.camera.orientation = Quat::from_rotation_y(cam_yaw) * Quat::from_rotation_x(cam_pitch);
        if dx != 0.0 || dz != 0.0 || dy != 0.0 {
            let speed = if sprinting { 10.0 } else { 5.0 };
            let mut control_vec = Vec3::new(dx, dy, dz);
            if !analog_controls {
                control_vec = control_vec.normalize();
            }
            let move_vec = scene.camera.orientation * control_vec * speed;
            cam_x += move_vec.x * dt;
            cam_y += move_vec.y * dt;
            cam_z += move_vec.z * dt;
        }
        scene.camera.position = Vec3::new(cam_x, cam_y, cam_z);

        {
            profiling::scope!("queue meshes to render");
            scene.queue(&sponza_model, Mat4::IDENTITY);

            let animations = smol_ame_model
                .animations
                .iter()
                .map(|animation| (game_time % animation.end_time, animation))
                .collect::<Vec<(f32, &neonvk::Animation)>>();
            let smol_ame_transform =
                Mat4::from_scale(Vec3::ONE * 0.7) * Mat4::from_quat(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2));
            scene.queue_animated(&smol_ame_model, smol_ame_transform, &animations)?;
        }

        let update_duration = Instant::now() - update_start_time;
        let frame_index = {
            // Xwayland waits here for vsync, don't take into account for frame times
            renderer.wait_frame(&swapchain)?
        };
        let render_start_time = Instant::now();
        match renderer.render_frame(&frame_index, &mut descriptors, &pipelines, &framebuffers, &scene, debug_value) {
            Ok(_) => {}
            Err(err) => log::warn!("Error during regular frame rendering: {}", err),
        }
        let render_duration = Instant::now() - render_start_time;
        match {
            // Wayland waits here for vsync, don't take into account for frame times
            renderer.present_frame(frame_index, &swapchain)
        } {
            Ok(_) => {}
            Err(err) => {
                log::error!("Error during regular frame present: {}", err);
                break 'running;
            }
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
                neonvk::display_utils::Bytes(used_memory),
                neonvk::display_utils::Bytes(allocated_memory),
                neonvk::display_utils::Bytes(external_used_memory),
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
        drop(smol_ame_model);
        drop(sponza_model);
        drop(assets_textures_arena);
        drop(assets_buffers_arena);
        drop(descriptors);
        device.destroy();
        drop(device);
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

/// Attempts to find the sponza/glTF directory to be used as a
/// resources-directory.
fn find_resources_path() -> PathBuf {
    let current_path = Path::new(".").canonicalize().unwrap();
    let path = if current_path.ends_with("src") {
        "bin"
    } else if current_path.ends_with("bin") {
        "."
    } else if current_path.ends_with("sponza") {
        ".."
    } else if current_path.ends_with("glTF") {
        "../.."
    } else {
        "src/bin"
    };
    PathBuf::from(path)
}

fn get_axis_deadzoned(raw: i16) -> f32 {
    if -9000 < raw && raw < 9000 {
        0.0
    } else {
        (raw as f32 / i16::MAX as f32).powf(3.0)
    }
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
