use std::f32::consts::FRAC_PI_2;
use std::panic;
use std::path::Path;
use std::sync::{Arc, Mutex, TryLockError};
use std::time::{Duration, Instant};

use glam::{Mat4, Quat, Vec3};
use log::LevelFilter;
use logger::Logger;
use sdl2::controller::{Axis, GameController};
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::messagebox::{show_simple_message_box, MessageBoxFlag};
use sdl2::mouse::MouseButton;
static LOGGER: Logger = Logger;

fn main() {
    #[cfg(feature = "profile-with-tracy")]
    tracy_client::Client::start();

    panic::set_hook(Box::new(|panic_info| {
        let message = format!("Unexpected crash! Error details below.\n\n{panic_info}");
        log::error!("{message}");
        let _ = show_simple_message_box(MessageBoxFlag::ERROR, "Crash!", &message, None);
    }));

    main_();
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

    // Perf visualization stuff
    cumulative_render_time: Duration,
    cumulative_render_count: u32,
    cumulative_update_time: Duration,
    cumulative_update_count: u32,
}

fn main_() {
    log::set_logger(&LOGGER).map(|()| log::set_max_level(LevelFilter::Trace)).unwrap();

    let sdl_context = {
        profiling::scope!("SDL init");
        sdl2::init().unwrap()
    };
    let video_subsystem = {
        profiling::scope!("SDL video subsystem init");
        sdl_context.video().unwrap()
    };

    let mut window = {
        profiling::scope!("SDL window creation");
        video_subsystem
            .window("sandbox", 640, 480)
            .position_centered()
            .resizable()
            .allow_highdpi()
            .vulkan()
            .build()
            .unwrap()
    };

    let (width, height) = window.vulkan_drawable_size();
    let state = SharedState {
        cam_y: 1.6,
        cam_yaw: 1.56,
        width,
        height,
        refresh_rate: window.display_mode().map(|dm| dm.refresh_rate).unwrap_or(60),
        running: true,
        ..Default::default()
    };
    let state_mutex = Arc::new(Mutex::new(state));

    let game_thread = std::thread::Builder::new()
        .name(format!("{}-update", env!("CARGO_CRATE_NAME")))
        .spawn({
            let state_mutex = state_mutex.clone();
            move || game_main(state_mutex)
        })
        .unwrap();

    #[cfg(feature = "profile-with-tracy")]
    std::thread::Builder::new()
        .name(format!("{}-vram-monitor", env!("CARGO_CRATE_NAME")))
        .spawn(|| {
            let mut prev_allocated = -1.0;
            let mut prev_in_use = -1.0;
            loop {
                let allocated = renderer::get_allocated_vram() as f64;
                if allocated != prev_allocated {
                    tracy_client::plot!("allocated vram (bytes)", prev_allocated);
                    prev_allocated = allocated;
                }
                let in_use = renderer::get_allocated_vram_in_use() as f64;
                if in_use != prev_in_use {
                    tracy_client::plot!("used vram (bytes)", prev_in_use);
                    prev_in_use = in_use;
                }
                std::thread::sleep(Duration::from_micros(100));
            }
        })
        .unwrap();

    let instance = renderer::Instance::new(
        &window,
        unsafe { core::ffi::CStr::from_bytes_with_nul_unchecked(b"sandbox example application\0") },
        env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
        env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
        env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
    )
    .unwrap();
    let surface = renderer::create_surface(&instance.entry, &instance.inner, &window, &window).unwrap();
    let rendering_thread = std::thread::Builder::new()
        .name(format!("{}-render", env!("CARGO_CRATE_NAME")))
        .spawn({
            let state_mutex = state_mutex.clone();
            move || rendering_main(instance, surface, state_mutex.clone())
        })
        .unwrap();

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

    const FPS_COUNTER_UPDATES_PER_SECOND: usize = 20;
    let now = Instant::now();
    let mut fps_counter_accumulators = [(0, Duration::ZERO, Duration::ZERO); FPS_COUNTER_UPDATES_PER_SECOND];
    let mut fps_counter_update_deadlines = [now; FPS_COUNTER_UPDATES_PER_SECOND];
    let mut fps_counter_accumulator_index = 0;
    let mut fps_counter_ready = false;
    for (i, deadline) in fps_counter_update_deadlines.iter_mut().enumerate() {
        *deadline = now + Duration::from_secs(1) * i as u32 / FPS_COUNTER_UPDATES_PER_SECOND as u32;
    }

    let mut event_pump = sdl_context.event_pump().unwrap();
    let mut opened_controller = None;
    'main: loop {
        if let Some(which) = opened_controller {
            profiling::scope!("opening controller");
            controller = Some(controller_subsystem.open(which).unwrap());
            opened_controller = None;
        }

        let event = {
            profiling::scope!("waiting for an event");
            event_pump.wait_event_timeout(2)
        };
        profiling::scope!("handle event");
        let mut state = {
            profiling::scope!("waiting for lock");
            state_mutex.lock().unwrap()
        };
        if !state.running {
            break 'main;
        }

        if let Some(event) = event {
            profiling::scope!("event-specific processing", &format!("event: {event:?}"));
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
                        Some(Keycode::W) => state.dz = 1.0,
                        Some(Keycode::S) => state.dz = -1.0,
                        Some(Keycode::A) => state.dx = 1.0,
                        Some(Keycode::D) => state.dx = -1.0,
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
                        log::info!("immediate present set to: {}", state.immediate_present);
                    }
                    Some(Keycode::W) if state.dz > 0.0 => state.dz = 0.0,
                    Some(Keycode::S) if state.dz < 0.0 => state.dz = 0.0,
                    Some(Keycode::A) if state.dx > 0.0 => state.dx = 0.0,
                    Some(Keycode::D) if state.dx < 0.0 => state.dx = 0.0,
                    Some(Keycode::Q) if state.dy > 0.0 => state.dy = 0.0,
                    Some(Keycode::X) if state.dy < 0.0 => state.dy = 0.0,
                    Some(Keycode::LShift) => state.sprinting = false,
                    _ => {}
                },

                Event::ControllerAxisMotion { axis, value, .. } => {
                    analog_controls = true;
                    match axis {
                        Axis::LeftX => state.dx = -get_axis_deadzoned(value),
                        Axis::LeftY => state.dz = -get_axis_deadzoned(value),
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
                        state.cam_yaw_once_delta -= xrel as f32 / 750.0;
                        state.cam_pitch_once_delta += yrel as f32 / 750.0;
                    }
                }

                Event::Window {
                    win_event: WindowEvent::SizeChanged(_, _),
                    ..
                } => {
                    state.mouse_look = false;
                    sdl_context.mouse().set_relative_mouse_mode(false);
                    sdl_context.mouse().show_cursor(true);
                    let (width, height) = window.vulkan_drawable_size();
                    state.width = width;
                    state.height = height;
                    state.queued_resize = Some(Instant::now());
                }

                Event::ControllerDeviceAdded { which, .. } => {
                    opened_controller = Some(which);
                }

                _ => {}
            }
        }

        state.refresh_rate = if state.immediate_present {
            1_000_000_000
        } else {
            profiling::scope!("getting refresh rate");
            window.display_mode().map(|dm| dm.refresh_rate).unwrap_or(60)
        };

        {
            profiling::scope!("updating controls");
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
                    state.cam_pitch_delta = get_axis_deadzoned(controller.axis(Axis::RightY)) * speed;
                }
            }
        }

        let next_update_time = &mut fps_counter_update_deadlines[fps_counter_accumulator_index];
        if Instant::now() >= *next_update_time {
            profiling::scope!("updating performance counters");
            *next_update_time += Duration::from_secs(1);

            let latest_renders = state.cumulative_render_count;
            let latest_avg_render_time = state.cumulative_render_time / state.cumulative_render_count.max(1);
            let latest_avg_update_time = state.cumulative_update_time / state.cumulative_update_count.max(1);
            state.cumulative_render_count = 0;
            state.cumulative_update_count = 0;
            state.cumulative_render_time = Duration::ZERO;
            state.cumulative_update_time = Duration::ZERO;

            fps_counter_ready |= fps_counter_accumulator_index == fps_counter_accumulators.len() - 1;
            fps_counter_accumulator_index = (fps_counter_accumulator_index + 1) % fps_counter_accumulators.len();
            let (fps_store, render_time_store, update_time_store) = &mut fps_counter_accumulators[fps_counter_accumulator_index];
            *fps_store = latest_renders;
            *render_time_store = latest_avg_render_time;
            *update_time_store = latest_avg_update_time;

            if fps_counter_ready {
                let (fps, render_time, update_time) = fps_counter_accumulators.iter().fold(
                    (0, Duration::ZERO, Duration::ZERO),
                    |(acc_fps, acc_r, acc_u), (renders, render_time, update_time)| {
                        (acc_fps + *renders, acc_r + *render_time, acc_u + *update_time)
                    },
                );
                let render_time = render_time / fps_counter_accumulators.len() as u32;
                let update_time = update_time / fps_counter_accumulators.len() as u32;
                let title = format!("sandbox (fps: {fps:4}, render time: {render_time:.2?}, dt: {update_time:.2?})");

                drop(state);
                profiling::scope!("updating window title (dropped state lock)");
                let _ = window.set_title(&title);
            }
        }
    }

    game_thread.join().unwrap();
    rendering_thread.join().unwrap();
}

fn game_main(state_mutex: Arc<Mutex<SharedState>>) {
    let mut last_wait_time = Instant::now();
    let mut last_frame_start = Instant::now();

    loop {
        let too_slow;
        let dt_duration;
        {
            profiling::scope!("main loop update");

            let mut state = {
                profiling::scope!("waiting for lock");
                state_mutex.lock().unwrap()
            };
            if !state.running {
                break;
            }

            let frame_start = Instant::now();
            let real_dt = frame_start - last_frame_start;
            let fixed_dt = Duration::from_nanos(1_000_000_000 / state.refresh_rate as u64);
            too_slow = real_dt > (fixed_dt * 12 / 10); // Too slow if last frame was 20% longer than fixed timestep,
            dt_duration = if too_slow { real_dt } else { fixed_dt }; // fall back to variable timestep when too slow.
            last_frame_start = frame_start;
            state.cumulative_update_time += dt_duration;
            state.cumulative_update_count += 1;

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
        }

        if !too_slow {
            profiling::scope!("sleeping until the next update");
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
    }
}

fn rendering_main(instance: renderer::Instance, surface: renderer::Surface, state_mutex: Arc<Mutex<SharedState>>) {
    let mut physical_devices = renderer::get_physical_devices(&instance.entry, &instance.inner, surface.inner);
    let physical_device = physical_devices.remove(0).unwrap();
    let device = physical_device.create_device(&instance.entry, &instance.inner).unwrap();

    let attachment_formats = physical_device.attachment_formats();
    let msaa_samples = renderer::vk::SampleCountFlags::TYPE_4;
    if !physical_device
        .properties
        .limits
        .framebuffer_color_sample_counts
        .contains(msaa_samples)
    {
        panic!("msaa sample count not supported: {msaa_samples:?}");
    }

    fn print_memory_usage(when: &str) {
        use renderer::Bytes;
        let in_use = Bytes(renderer::get_allocated_vram_in_use());
        let allocated = Bytes(renderer::get_allocated_vram());
        let peak = Bytes(renderer::get_allocated_vram_peak());
        log::info!("VRAM usage {when:40} {in_use}/{allocated},\tpeaked at {peak}");
    }

    print_memory_usage("before measurements");

    let resources_path = {
        let current_path = Path::new(".").canonicalize().unwrap();
        let path = if current_path.ends_with("src") {
            "."
        } else if current_path.ends_with("sandbox") {
            "src"
        } else if current_path.ends_with("examples") {
            "sandbox/src"
        } else {
            "examples/sandbox/src"
        };
        Path::new(path)
    };
    let mut assets_buffers_measurer = renderer::VulkanArenaMeasurer::new(&device);
    let mut assets_textures_measurer = renderer::VulkanArenaMeasurer::new(&device);
    for image_create_info in renderer::image_loading::pbr_defaults::all_defaults_create_infos() {
        assets_textures_measurer.add_image(image_create_info);
    }
    let mut vertex_library_measurer = renderer::VertexLibraryMeasurer::default();
    let sponza_json = std::fs::read_to_string(resources_path.join("sponza/glTF/Sponza.gltf")).unwrap();
    let smol_ame_json = std::fs::read_to_string(resources_path.join("smol-ame-by-seafoam/smol-ame.gltf")).unwrap();
    let sponza_pending = gltf::Gltf::preload_gltf(
        &sponza_json,
        resources_path.join("sponza/glTF"),
        (&mut assets_textures_measurer, &mut vertex_library_measurer),
    )
    .unwrap();
    let smol_ame_pending = gltf::Gltf::preload_gltf(
        &smol_ame_json,
        resources_path.join("smol-ame-by-seafoam"),
        (&mut assets_textures_measurer, &mut vertex_library_measurer),
    )
    .unwrap();
    vertex_library_measurer.measure_required_arena(&mut assets_buffers_measurer);

    print_memory_usage("after measurements");

    // Allocate in order of importance: if budget runs out, the arenas allocated
    // later may be allocated from a slower heap.
    let mut texture_arena = renderer::VulkanArena::new(
        &instance.inner,
        &device,
        &physical_device,
        assets_textures_measurer.measured_size,
        renderer::MemoryProps::for_textures(),
        format_args!("sandbox assets (textures)"),
    )
    .unwrap();
    let mut buffer_arena = renderer::VulkanArena::new(
        &instance.inner,
        &device,
        &physical_device,
        assets_buffers_measurer.measured_size,
        renderer::MemoryProps::for_buffers(),
        format_args!("sandbox assets (buffers)"),
    )
    .unwrap();
    let mut staging_arena = renderer::VulkanArena::new(
        &instance.inner,
        &device,
        &physical_device,
        assets_buffers_measurer.measured_size + assets_textures_measurer.measured_size,
        renderer::MemoryProps::for_staging(),
        format_args!("sandbox assets (staging)"),
    )
    .unwrap();
    let mut uploader = renderer::Uploader::new(
        &device,
        device.graphics_queue,
        device.transfer_queue,
        &physical_device,
        "sandbox assets",
    );

    print_memory_usage("after arena creation");

    let pbr_defaults =
        renderer::image_loading::pbr_defaults::all_defaults(&device, &mut staging_arena, &mut uploader, &mut texture_arena).unwrap();
    let mut descriptors = renderer::Descriptors::new(&device, &physical_device, pbr_defaults);
    let mut vertex_library_builder = renderer::VertexLibraryBuilder::new(
        &mut staging_arena,
        &mut buffer_arena,
        vertex_library_measurer,
        format_args!("vertex library of babel"),
    )
    .unwrap();

    let upload_start = Instant::now();
    let sponza_model = sponza_pending
        .upload(
            &device,
            &mut staging_arena,
            &mut uploader,
            &mut descriptors,
            &mut texture_arena,
            &mut vertex_library_builder,
        )
        .unwrap();
    let smol_ame_model = smol_ame_pending
        .upload(
            &device,
            &mut staging_arena,
            &mut uploader,
            &mut descriptors,
            &mut texture_arena,
            &mut vertex_library_builder,
        )
        .unwrap();
    vertex_library_builder.upload(&mut uploader, &mut buffer_arena);
    let upload_wait_start = Instant::now();
    {
        profiling::scope!("wait for uploads to finish");
        assert!(uploader.wait(Some(Duration::from_secs(5))));
    }
    let now = Instant::now();
    log::info!(
        "Spent {:.2?} loading resources, of which {:.2?} was waiting for upload.",
        now - upload_start,
        now - upload_wait_start
    );
    drop(uploader);
    drop(staging_arena);

    print_memory_usage("after uploads");

    assert_eq!(buffer_arena.memory_in_use(), assets_buffers_measurer.measured_size);
    assert_eq!(texture_arena.memory_in_use(), assets_textures_measurer.measured_size);

    let mut swapchain_settings;
    let mut prev_frame;
    {
        let state = state_mutex.lock().unwrap();
        let (width, height) = (state.width, state.height);
        swapchain_settings = renderer::SwapchainSettings {
            extent: renderer::vk::Extent2D { width, height },
            immediate_present: state.immediate_present,
        };
        prev_frame = state.frame;
    }

    let mut swapchain = renderer::Swapchain::new(&device, &physical_device, surface, &swapchain_settings).unwrap();
    print_memory_usage("after swapchain creation");
    let mut pipelines = renderer::Pipelines::new(&device, &descriptors, swapchain.extent, msaa_samples, attachment_formats, None);
    print_memory_usage("after pipelines creation");
    let mut framebuffers = renderer::Framebuffers::new(&instance.inner, &device, &physical_device, &pipelines, &swapchain).unwrap();
    print_memory_usage("after framebuffers creation");
    let mut renderer = renderer::Renderer::new(&instance.inner, &device, &physical_device).unwrap();
    print_memory_usage("after renderer creation");

    let mut scene = renderer::Scene::new(&physical_device);
    let mut recreate_swapchain = false;
    let mut prev_duration = Duration::ZERO;
    'running: loop {
        // Rendering preparation, which needs the SharedState:
        let frame_start;
        let debug_value;
        {
            let mut state = {
                profiling::scope!("waiting for the next update");
                loop {
                    let state = {
                        profiling::scope!("waiting for lock");
                        state_mutex.lock().unwrap()
                    };
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

            profiling::scope!("rendering (scene creation, state locked)");

            frame_start = Instant::now();
            state.cumulative_render_time += prev_duration;
            state.cumulative_render_count += 1;

            debug_value = state.debug_value;

            if let Some(resize_timestamp) = state.queued_resize {
                let duration_since_resize = Instant::now() - resize_timestamp;
                if duration_since_resize > Duration::from_millis(100) {
                    swapchain_settings.extent = renderer::vk::Extent2D {
                        width: state.width,
                        height: state.height,
                    };
                    swapchain_settings.immediate_present = state.immediate_present;
                    recreate_swapchain = true;
                    state.queued_resize = None;
                }
            }

            scene.clear();
            scene.camera.orientation = Quat::from_rotation_y(state.cam_yaw) * Quat::from_rotation_x(state.cam_pitch);
            scene.camera.position = Vec3::new(state.cam_x, state.cam_y, state.cam_z);
            scene.world_space = renderer::CoordinateSystem::GLTF;

            {
                profiling::scope!("queue meshes to render");
                sponza_model.queue(&mut scene, Mat4::IDENTITY);

                let animations = smol_ame_model
                    .animations
                    .iter()
                    .map(|animation| (state.game_time % animation.end_time, animation))
                    .collect::<Vec<(f32, &gltf::Animation)>>();
                let smol_ame_transform = Mat4::from_scale_rotation_translation(
                    Vec3::ONE * 0.7,
                    Quat::from_rotation_y(-std::f32::consts::FRAC_PI_2),
                    Vec3::new(3.0, 0.0, -0.5),
                );
                smol_ame_model.queue_animated(&mut scene, smol_ame_transform, &animations).unwrap();
            }
        }

        if recreate_swapchain {
            profiling::scope!("handle resize");
            device.wait_idle();
            drop(framebuffers);
            swapchain.recreate(&device, &physical_device, &swapchain_settings).unwrap();
            pipelines = renderer::Pipelines::new(
                &device,
                &descriptors,
                swapchain.extent,
                msaa_samples,
                attachment_formats,
                Some(pipelines),
            );
            framebuffers = renderer::Framebuffers::new(&instance.inner, &device, &physical_device, &pipelines, &swapchain).unwrap();
            recreate_swapchain = false;
        }

        {
            profiling::scope!("rendering (vulkan calls)");
            let frame_index = match renderer.wait_frame(&swapchain) {
                Ok(i) => i,
                Err(renderer::RendererError::SwapchainOutOfDate) => {
                    recreate_swapchain = true;
                    continue;
                }
                Err(err) => {
                    log::error!("Error during regular frame wait: {}", err);
                    std::thread::sleep(Duration::from_millis(30));
                    continue;
                }
            };
            match renderer.render_frame(&frame_index, &mut descriptors, &pipelines, &framebuffers, &mut scene, debug_value) {
                Ok(_) => {}
                Err(err) => {
                    log::warn!("Error during regular frame rendering: {}", err);
                    std::thread::sleep(Duration::from_millis(30));
                }
            }
            match { renderer.present_frame(frame_index, &swapchain) } {
                Ok(_) => {}
                Err(renderer::RendererError::SwapchainOutOfDate) => recreate_swapchain = true,
                Err(err) => {
                    log::error!("Error during regular frame present: {}", err);
                    std::thread::sleep(Duration::from_millis(30));
                    continue;
                }
            }
            profiling::finish_frame!();

            prev_duration = Instant::now().duration_since(frame_start);

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
        device.wait_idle();
    }

    print_memory_usage("after ending the rendering loop");
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
                let file = record.file().unwrap_or("");
                let line = record.line().unwrap_or(0);
                let is_vk_debug_utils_print = file == "renderer/src/debug_utils.rs";
                let mut log_level = record.level();
                if is_vk_debug_utils_print && message.contains("[Loader Message]") {
                    log_level = Level::Trace;
                }
                let (color_code, color_end) = if cfg!(target_family = "unix") {
                    let start = match log_level {
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
                if log_level < Level::Trace {
                    if is_vk_debug_utils_print {
                        if let Some((tag, msg)) = message.split_once("] ") {
                            eprintln!("{color_code}{tag}]{color_end} {msg}");
                        } else {
                            eprintln!("{color_code}[VK_EXT_debug_utils]{color_end} {message}");
                        }
                    } else {
                        eprintln!("{color_code}[{file}:{line}]{color_end} {message}");
                    }
                }
            }
        }

        fn flush(&self) {
            use std::io::Write;
            let mut stderr = std::io::stderr().lock();
            let _ = stderr.flush();
        }
    }
}
