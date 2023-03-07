use glam::{Mat4, Quat, Vec3};
use log::LevelFilter;
use sdl2::controller::{Axis, GameController};
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::messagebox::{show_simple_message_box, MessageBoxFlag};
use sdl2::mouse::MouseButton;
use std::f32::consts::FRAC_PI_2;
use std::path::Path;
use std::sync::{Arc, Mutex, TryLockError};
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

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "profile-with-tracy")]
    let client = tracy_client::Client::start();

    if let Err(err) = main_() {
        let message = format!("{:?}", err);
        let _ = show_simple_message_box(MessageBoxFlag::ERROR, "Fatal Error", &message, None);
        Err(err)
    } else {
        #[cfg(feature = "profile-with-tracy")]
        drop(client);
        // Let the OS clean up the rest.
        std::process::exit(0);
    }
}

/// Shared between the three threads using a mutex, used as follows:
///
/// - SDL thread locks for very brief periods to update state based on events,
///   when they happen.
/// - Update thread locks for the duration of the update, bumps .frame each
///   time. Happens on a strict schedule based on the monitor refresh rate.
/// - Render thread waits in a lock-check-wait-repeat loop until .frame is
///   bumped, then renders a frame.
///
/// So in general: update thread drives the program, SDL thread pokes the state
/// on events, render thread renders as indicated by the update thread.
#[derive(Default)]
struct SharedState {
    // Game state
    debug_value: u32,
    game_time: f32,
    cam_x: f32,
    cam_y: f32,
    cam_z: f32,
    cam_yaw: f32,
    cam_pitch: f32,

    // Inputs
    cam_yaw_once_delta: f32,
    cam_pitch_once_delta: f32,
    cam_yaw_delta: f32,
    cam_pitch_delta: f32,
    dx: f32,
    dy: f32,
    dz: f32,
    mouse_look: bool,
    sprinting: bool,

    // Window stuff
    queued_resize: Option<Instant>,
    width: u32,
    height: u32,
    immediate_present: bool,
    refresh_rate: i32,
    frame: u64,
    running: bool,
}

fn main_() -> anyhow::Result<()> {
    log::set_logger(&LOGGER).map(|()| log::set_max_level(LevelFilter::Trace)).unwrap();

    #[cfg(feature = "profile-with-tracy")]
    tracy_client::set_thread_name!("SDL thread");

    let sdl_context = {
        profiling::scope!("SDL init");
        sdl2::init().map_err(SandboxError::Sdl)?
    };
    let video_subsystem = {
        profiling::scope!("SDL video subsystem init");
        sdl_context.video().map_err(SandboxError::Sdl)?
    };

    let window = {
        profiling::scope!("SDL window creation");
        video_subsystem
            .window("neonvk sandbox", 640, 480)
            .position_centered()
            .resizable()
            .allow_highdpi()
            .vulkan()
            .build()?
    };

    let (width, height) = window.vulkan_drawable_size();
    let state = SharedState {
        cam_x: 3.0,
        cam_y: 1.6,
        cam_yaw: 1.56,
        width,
        height,
        refresh_rate: window.display_mode().map(|dm| dm.refresh_rate).unwrap_or(60),
        running: true,
        ..Default::default()
    };
    let state_mutex = Arc::new(Mutex::new(state));

    let game_thread = std::thread::spawn({
        let state_mutex = state_mutex.clone();
        move || game_main(state_mutex)
    });

    let instance = neonvk::Instance::new(&window)?;
    let surface = neonvk::create_surface(&instance.entry, &instance.inner, &window)?;
    let rendering_thread = std::thread::spawn({
        let state_mutex = state_mutex.clone();
        move || rendering_main(instance, surface, state_mutex)
    });

    let controller_subsystem = {
        profiling::scope!("SDL controller subsystem init");
        sdl_context.game_controller().unwrap()
    };
    let mut controller: Option<GameController> = None;
    let mut analog_controls = false;
    fn get_axis_deadzoned(raw: i16) -> f32 {
        if -9000 < raw && raw < 9000 {
            0.0
        } else {
            (raw as f32 / i16::MAX as f32).powf(3.0)
        }
    }

    profiling::scope!("event loop");
    let mut event_pump = sdl_context.event_pump().map_err(SandboxError::Sdl)?;
    'main: for event in event_pump.wait_iter() {
        profiling::scope!("handle event");
        let mut state = state_mutex.lock().unwrap();
        match event {
            Event::Quit { .. } => {
                state.running = false;
                sdl_context.mouse().set_relative_mouse_mode(false);
                sdl_context.mouse().show_cursor(true);
                break 'main;
            }

            Event::KeyDown { keycode, .. } => {
                analog_controls = false;
                match keycode {
                    Some(Keycode::Num0) => state.debug_value = 0,
                    Some(Keycode::Num1) => state.debug_value = 1,
                    Some(Keycode::Num2) => state.debug_value = 2,
                    Some(Keycode::Num3) => state.debug_value = 3,
                    Some(Keycode::Num4) => state.debug_value = 4,
                    Some(Keycode::Num5) => state.debug_value = 5,
                    Some(Keycode::Num6) => state.debug_value = 6,
                    Some(Keycode::Num7) => state.debug_value = 7,
                    Some(Keycode::W) => state.dz = -1.0,
                    Some(Keycode::A) => state.dx = -1.0,
                    Some(Keycode::S) => state.dz = 1.0,
                    Some(Keycode::D) => state.dx = 1.0,
                    Some(Keycode::Q) => state.dy = 1.0,
                    Some(Keycode::X) => state.dy = -1.0,
                    Some(Keycode::LShift) => state.sprinting = true,
                    Some(Keycode::Escape) if state.mouse_look => {
                        state.mouse_look = false;
                        sdl_context.mouse().set_relative_mouse_mode(false);
                        sdl_context.mouse().show_cursor(true);
                    }
                    _ => {}
                }
            }

            Event::KeyUp { keycode, .. } => match keycode {
                Some(Keycode::I) => {
                    state.immediate_present = !state.immediate_present;
                    state.queued_resize = Some(Instant::now());
                }
                Some(Keycode::S) if state.dz > 0.0 => state.dz = 0.0,
                Some(Keycode::W) if state.dz < 0.0 => state.dz = 0.0,
                Some(Keycode::D) if state.dx > 0.0 => state.dx = 0.0,
                Some(Keycode::A) if state.dx < 0.0 => state.dx = 0.0,
                Some(Keycode::Q) if state.dy > 0.0 => state.dy = 0.0,
                Some(Keycode::X) if state.dy < 0.0 => state.dy = 0.0,
                Some(Keycode::LShift) => state.sprinting = false,
                _ => {}
            },

            Event::ControllerAxisMotion { axis, value, .. } => {
                analog_controls = true;
                match axis {
                    Axis::LeftX => state.dx = get_axis_deadzoned(value),
                    Axis::LeftY => state.dz = get_axis_deadzoned(value),
                    Axis::TriggerRight if value != 0 => state.dy = value as f32 / i16::MAX as f32,
                    Axis::TriggerRight if state.dy > 0.0 => state.dy = 0.0,
                    Axis::TriggerLeft if value != 0 => state.dy = -(value as f32 / i16::MAX as f32),
                    Axis::TriggerLeft if state.dy < 0.0 => state.dy = 0.0,
                    _ => {}
                }
            }

            Event::MouseButtonDown {
                mouse_btn: MouseButton::Left,
                ..
            } => {
                state.mouse_look = !state.mouse_look;
                if state.mouse_look {
                    sdl_context.mouse().set_relative_mouse_mode(true);
                    sdl_context.mouse().show_cursor(false);
                } else {
                    sdl_context.mouse().set_relative_mouse_mode(false);
                    sdl_context.mouse().show_cursor(true);
                }
            }

            Event::MouseMotion { xrel, yrel, .. } => {
                if state.mouse_look {
                    state.cam_yaw_once_delta += -xrel as f32 / 750.0;
                    state.cam_pitch_once_delta += -yrel as f32 / 750.0;
                }
            }

            Event::Window {
                win_event: WindowEvent::SizeChanged(width, height),
                ..
            } => {
                state.mouse_look = false;
                sdl_context.mouse().set_relative_mouse_mode(false);
                sdl_context.mouse().show_cursor(true);
                state.width = width as u32;
                state.height = height as u32;
                state.queued_resize = Some(Instant::now());
            }

            Event::ControllerDeviceAdded { which, .. } => {
                controller = Some(controller_subsystem.open(which).unwrap());
            }

            _ => {}
        }

        state.refresh_rate = window.display_mode().map(|dm| dm.refresh_rate).unwrap_or(60);

        if !analog_controls && (state.dx != 0.0 || state.dy != 0.0 || state.dz != 0.0) {
            let dl = (state.dx * state.dx + state.dy * state.dy + state.dz * state.dz).sqrt();
            state.dx /= dl;
            state.dy /= dl;
            state.dz /= dl;
        }

        if analog_controls {
            if let Some(controller) = &controller {
                let speed = 2.0 / state.refresh_rate as f32;
                state.cam_yaw_delta = -get_axis_deadzoned(controller.axis(Axis::RightX)) * speed;
                state.cam_pitch_delta = -get_axis_deadzoned(controller.axis(Axis::RightY)) * speed;
            }
        }
    }

    game_thread.join().unwrap();
    rendering_thread.join().unwrap()?;

    Ok(())
}

fn game_main(state_mutex: Arc<Mutex<SharedState>>) {
    #[cfg(feature = "profile-with-tracy")]
    tracy_client::set_thread_name!("Update thread");

    let mut last_wait_time = Instant::now();

    loop {
        profiling::scope!("frame");
        let mut state = state_mutex.lock().unwrap();
        if !state.running {
            break;
        }
        let dt_duration = Duration::from_micros(1_000_000 / state.refresh_rate as u64);

        {
            profiling::scope!("main loop update");
            let dt = dt_duration.as_secs_f32();
            {
                profiling::scope!("apply rotation and movement");
                state.cam_yaw += state.cam_yaw_delta + state.cam_yaw_once_delta;
                state.cam_pitch = (state.cam_pitch + state.cam_pitch_delta + state.cam_pitch_once_delta).clamp(-FRAC_PI_2, FRAC_PI_2);
                state.cam_yaw_once_delta = 0.0;
                state.cam_pitch_once_delta = 0.0;

                if state.dx != 0.0 || state.dz != 0.0 || state.dy != 0.0 {
                    let speed = if state.sprinting { 10.0 } else { 5.0 };
                    let control_vec = Vec3::new(state.dx, state.dy, state.dz);
                    let orientation = Quat::from_rotation_y(state.cam_yaw) * Quat::from_rotation_x(state.cam_pitch);
                    let move_vec = orientation * control_vec * speed * dt;
                    state.cam_x += move_vec.x;
                    state.cam_y += move_vec.y;
                    state.cam_z += move_vec.z;
                }
            }
            state.game_time += dt;
            state.frame += 1;
            drop(state);
        }

        {
            let deadline = last_wait_time + dt_duration;
            while let Some(wait_left) = deadline.checked_duration_since(Instant::now()) {
                if wait_left > Duration::from_millis(2) {
                    std::thread::sleep(wait_left - Duration::from_millis(1));
                } else {
                    std::thread::yield_now();
                }
            }
            last_wait_time = Instant::now();
        }

        profiling::finish_frame!();
    }
}

fn rendering_main(instance: neonvk::Instance, surface: neonvk::Surface, state_mutex: Arc<Mutex<SharedState>>) -> anyhow::Result<()> {
    #[cfg(feature = "profile-with-tracy")]
    tracy_client::set_thread_name!("Render thread");

    let mut physical_devices = neonvk::get_physical_devices(&instance.entry, &instance.inner, surface.inner);
    let physical_device = physical_devices.remove(0)?;
    let device = neonvk::create_device(&instance.inner, &physical_device)?;
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

    let resources_path = {
        let current_path = Path::new(".").canonicalize().unwrap();
        let path = if current_path.ends_with("examples") {
            "."
        } else if current_path.ends_with("sponza") {
            ".."
        } else if current_path.ends_with("glTF") {
            "../.."
        } else {
            "examples"
        };
        Path::new(path)
    };
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
        {
            profiling::scope!("wait for uploads to finish");
            assert!(uploader.wait(Duration::from_secs(5))?);
        }
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

    let state = state_mutex.lock().unwrap();
    let (width, height) = (state.width, state.height);
    let mut swapchain_settings = neonvk::SwapchainSettings {
        extent: neonvk::vk::Extent2D { width, height },
        immediate_present: state.immediate_present,
    };
    let mut prev_frame = state.frame;
    drop(state);
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

    'running: loop {
        // Rendering preparation, which needs the SharedState:
        let mut scene;
        let debug_value;
        let mut recreate_swapchain = false;
        {
            let mut state = {
                profiling::scope!("waiting for the next update");
                loop {
                    let state = state_mutex.lock().unwrap();
                    if !state.running {
                        break 'running;
                    } else if state.frame != prev_frame {
                        prev_frame = state.frame;
                        break state;
                    } else {
                        drop(state);
                        std::thread::sleep(Duration::from_micros(100));
                    }
                }
            };

            profiling::scope!("rendering (scene creation)");

            if let Some(resize_timestamp) = state.queued_resize {
                let duration_since_resize = Instant::now() - resize_timestamp;
                if duration_since_resize > Duration::from_millis(100) {
                    swapchain_settings.extent = neonvk::vk::Extent2D {
                        width: state.width,
                        height: state.height,
                    };
                    swapchain_settings.immediate_present = state.immediate_present;
                    recreate_swapchain = true;
                    state.queued_resize = None;
                }
            }

            debug_value = state.debug_value;

            scene = neonvk::Scene::new(&physical_device);
            scene.camera.orientation = Quat::from_rotation_y(state.cam_yaw) * Quat::from_rotation_x(state.cam_pitch);
            scene.camera.position = Vec3::new(state.cam_x, state.cam_y, state.cam_z);

            {
                profiling::scope!("queue meshes to render");
                scene.queue(&sponza_model, Mat4::IDENTITY);

                let animations = smol_ame_model
                    .animations
                    .iter()
                    .map(|animation| (state.game_time % animation.end_time, animation))
                    .collect::<Vec<(f32, &neonvk::Animation)>>();
                let smol_ame_transform =
                    Mat4::from_scale(Vec3::ONE * 0.7) * Mat4::from_quat(Quat::from_rotation_y(std::f32::consts::FRAC_PI_2));
                scene.queue_animated(&smol_ame_model, smol_ame_transform, &animations)?;
            }
        }

        if recreate_swapchain {
            profiling::scope!("handle resize");
            device.wait_idle()?;
            drop(framebuffers);
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
        }

        {
            profiling::scope!("rendering (vulkan calls)");
            let frame_index = { renderer.wait_frame(&swapchain)? };
            match renderer.render_frame(&frame_index, &mut descriptors, &pipelines, &framebuffers, scene, debug_value) {
                Ok(_) => {}
                Err(err) => log::warn!("Error during regular frame rendering: {}", err),
            }
            match { renderer.present_frame(frame_index, &swapchain) } {
                Ok(_) => {}
                Err(neonvk::RendererError::SwapchainOutOfDate(_)) => {}
                Err(err) => {
                    log::error!("Error during regular frame present: {}", err);
                    return Err(err.into());
                }
            }

            // Update prev_frame here. If it's a new frame (i.e. prev_frame
            // actually changes its value), we overran the frame deadline.
            // Instead of trying to play catchup, updating prev_frame here
            // ensures that we have the maximum amount of time to render the
            // *next* frame. That said, if locking fails, the game update is
            // happening, and it means we're right on time to render the next
            // frame.
            match state_mutex.try_lock() {
                Ok(state) => prev_frame = state.frame,
                Err(TryLockError::WouldBlock) => {}
                Err(TryLockError::Poisoned(err)) => panic!("main thread state mutex was poisoned: {err}"),
            }
        }
    }

    {
        profiling::scope!("wait for gpu to be idle before exit");
        device.wait_idle()?;
    }

    Ok(())
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
