use std::f32::consts::FRAC_PI_2;
use std::panic;
use std::path::Path;
use std::time::{Duration, Instant};

use glam::{Affine3A, Quat, Vec3};
use log::LevelFilter;
use logger::Logger;
use sdl2::controller::{Axis, GameController};
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::messagebox::{MessageBoxFlag, show_simple_message_box};
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

fn main_() {
    #[cfg(feature = "profile-with-tracing")]
    let profiling_subscriber = Box::leak(Box::new(profiling_tracing_subscriber::ProfilingSubscriber::default()));
    #[cfg(feature = "profile-with-tracing")]
    let _ = profiling::tracing::subscriber::set_global_default(&*profiling_subscriber);
    #[cfg(feature = "profile-with-tracing")]
    let startup_span = profiling::tracing::info_span!("startup");
    #[cfg(feature = "profile-with-tracing")]
    let startup_span_guard = startup_span.enter();

    //
    // General state
    //

    // Game state
    let mut debug_value: u32 = 0;
    let mut game_time: f32 = 0.0;
    let mut cam_x: f32 = 0.0;
    let mut cam_y: f32 = 1.6;
    let mut cam_z: f32 = 0.0;
    let mut cam_yaw: f32 = 1.56;
    let mut cam_pitch: f32 = 0.0;

    // Inputs
    let mut cam_yaw_once_delta: f32 = 0.0;
    let mut cam_pitch_once_delta: f32 = 0.0;
    let mut cam_yaw_delta: f32 = 0.0;
    let mut cam_pitch_delta: f32 = 0.0;
    let mut dx: f32 = 0.0;
    let mut dy: f32 = 0.0;
    let mut dz: f32 = 0.0;
    let mut mouse_look: bool = false;
    let mut sprinting: bool = false;

    // Window stuff
    let mut queued_resize: Option<Instant> = None;
    let mut width: u32;
    let mut height: u32;
    let mut immediate_present: bool = false;
    let mut refresh_rate: i32;
    let mut recreate_swapchain = false;

    // Frame pacing
    let mut last_wait_time = Instant::now();
    let mut last_frame_start = Instant::now();

    //
    // Logging, SDL setup
    //

    log::set_logger(&LOGGER).map(|()| log::set_max_level(LevelFilter::Trace)).unwrap();

    let sdl_context = {
        profiling::scope!("SDL init");
        sdl2::init().unwrap()
    };
    let video_subsystem = {
        profiling::scope!("SDL video subsystem init");
        sdl_context.video().unwrap()
    };

    let window = {
        profiling::scope!("SDL window creation");
        video_subsystem.window("sandbox", 1280, 720).position_centered().resizable().allow_highdpi().vulkan().build().unwrap()
    };

    //
    // Renderer setup
    //

    let instance = renderer::Instance::new(
        &window,
        c"sandbox example application",
        env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap(),
        env!("CARGO_PKG_VERSION_MINOR").parse().unwrap(),
        env!("CARGO_PKG_VERSION_PATCH").parse().unwrap(),
    )
    .unwrap();
    let surface = renderer::create_surface(&instance.entry, &instance.inner, &window, &window).unwrap();

    let mut physical_devices = renderer::get_physical_devices(&instance.entry, &instance.inner, surface.inner);
    let physical_device = physical_devices.remove(0).unwrap();
    let device = physical_device.create_device(&instance.entry, &instance.inner).unwrap();

    let attachment_formats = physical_device.attachment_formats();
    let msaa_samples = renderer::vk::SampleCountFlags::TYPE_4;
    if !physical_device.properties.limits.framebuffer_color_sample_counts.contains(msaa_samples) {
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
    let sponza_json = std::fs::read_to_string(resources_path.join("sponza/glTF/Sponza.gltf")).unwrap();
    let smol_ame_json = std::fs::read_to_string(resources_path.join("smol-ame-by-seafoam/smol-ame.gltf")).unwrap();

    let (assets_buffers_measurer, assets_textures_measurer, vertex_library_measurer, sponza_pending, smol_ame_pending) = {
        profiling::scope!("gltf model memory measurement");
        let mut assets_buffers_measurer = renderer::VulkanArenaMeasurer::new(&device);
        let mut assets_textures_measurer = renderer::VulkanArenaMeasurer::new(&device);
        for image_create_info in renderer::image_loading::pbr_defaults::all_defaults_create_infos() {
            assets_textures_measurer.add_image(image_create_info);
        }
        let mut vertex_library_measurer = renderer::VertexLibraryMeasurer::default();
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
        (assets_buffers_measurer, assets_textures_measurer, vertex_library_measurer, sponza_pending, smol_ame_pending)
    };

    print_memory_usage("after measurements");

    // Allocate in order of importance: if budget runs out, the arenas allocated
    // later may be allocated from a slower heap.
    let mut texture_arena = {
        profiling::scope!("texture arena creation");
        renderer::VulkanArena::new(
            &instance.inner,
            &device,
            &physical_device,
            assets_textures_measurer.measured_size,
            renderer::MemoryProps::for_textures(),
            format_args!("sandbox assets (textures)"),
        )
        .unwrap()
    };
    let mut buffer_arena = {
        profiling::scope!("buffer arena creation");
        renderer::VulkanArena::new(
            &instance.inner,
            &device,
            &physical_device,
            assets_buffers_measurer.measured_size,
            renderer::MemoryProps::for_buffers(),
            format_args!("sandbox assets (buffers)"),
        )
        .unwrap()
    };
    let mut staging_arena = {
        profiling::scope!("staging arena creation");
        renderer::VulkanArena::new(
            &instance.inner,
            &device,
            &physical_device,
            assets_buffers_measurer.measured_size + assets_textures_measurer.measured_size,
            renderer::MemoryProps::for_staging(),
            format_args!("sandbox assets (staging)"),
        )
        .unwrap()
    };

    print_memory_usage("after arena creation");

    let mut uploader = renderer::Uploader::new(&device, device.graphics_queue, device.transfer_queue, &physical_device, "sandbox assets");

    let mut descriptors = {
        profiling::scope!("descriptor set creation");
        let pbr_defaults =
            renderer::image_loading::pbr_defaults::all_defaults(&device, &mut staging_arena, &mut uploader, &mut texture_arena).unwrap();
        renderer::Descriptors::new(&device, &physical_device, pbr_defaults)
    };

    let (sponza_model, smol_ame_model) = {
        profiling::scope!("upload gltf models to vram");

        let mut vertex_library_builder = renderer::VertexLibraryBuilder::new(
            &mut staging_arena,
            Some(&mut buffer_arena),
            &vertex_library_measurer,
            format_args!("vertex library of babel"),
        )
        .unwrap();

        let sponza_model = sponza_pending
            .upload(&device, &mut staging_arena, &mut uploader, &mut descriptors, &mut texture_arena, &mut vertex_library_builder)
            .unwrap();
        let smol_ame_model = smol_ame_pending
            .upload(&device, &mut staging_arena, &mut uploader, &mut descriptors, &mut texture_arena, &mut vertex_library_builder)
            .unwrap();
        vertex_library_builder.upload(&mut uploader);
        {
            profiling::scope!("wait for uploads to finish");
            assert!(uploader.wait(Some(Duration::from_secs(5))));
        }
        {
            profiling::scope!("drop the uploader and staging arena");
            drop(uploader);
            drop(staging_arena);
        }
        (sponza_model, smol_ame_model)
    };

    print_memory_usage("after uploads");

    assert_eq!(buffer_arena.memory_in_use(), assets_buffers_measurer.measured_size);
    assert_eq!(texture_arena.memory_in_use(), assets_textures_measurer.measured_size);

    (width, height) = window.vulkan_drawable_size();
    let mut swapchain_settings = renderer::SwapchainSettings { extent: renderer::vk::Extent2D { width, height }, immediate_present };

    let (mut swapchain, mut pipelines, mut framebuffers, mut renderer) = {
        profiling::scope!("rest of the vulkan renderer init");
        let swapchain = renderer::Swapchain::new(&device, &physical_device, surface, &swapchain_settings);
        print_memory_usage("after swapchain creation");
        let pipelines = renderer::Pipelines::new(&device, &descriptors, swapchain.extent, msaa_samples, attachment_formats, None);
        print_memory_usage("after pipelines creation");
        let framebuffers = renderer::Framebuffers::new(&instance.inner, &device, &physical_device, &pipelines, &swapchain);
        print_memory_usage("after framebuffers creation");
        let renderer = renderer::Renderer::new(&instance.inner, &device, &physical_device);
        print_memory_usage("after renderer creation");
        (swapchain, pipelines, framebuffers, renderer)
    };

    //
    // Imgui setup
    //

    let (mut imgui, mut imgui_platform, mut imgui_renderer) = {
        profiling::scope!("imgui setup");
        let mut imgui = imgui::Context::create();
        imgui.set_ini_filename(None);
        imgui.set_log_filename(None);
        imgui.fonts().add_font(&[
            imgui::FontSource::TtfData { data: include_bytes!("fonts/NotoSans-Regular.ttf"), size_pixels: 20.0, config: None },
            imgui::FontSource::DefaultFontData { config: None },
        ]);
        let imgui_platform = imgui_sdl2_support::SdlPlatform::init(&mut imgui);
        let imgui_renderer =
            renderer::imgui_support::ImGuiRenderer::new(&mut imgui, &instance, &device, &physical_device, &mut descriptors);
        imgui.style_mut();
        (imgui, imgui_platform, imgui_renderer)
    };

    print_memory_usage("after imgui setup");

    //
    // Setup HIDs and create the event pump
    //

    let controller_subsystem = {
        profiling::scope!("SDL controller subsystem init");
        sdl_context.game_controller().unwrap()
    };
    let mut controller: Option<GameController> = None;
    let mut opened_controller: Option<u32> = None;
    let mut analog_controls = false;
    fn get_axis_deadzoned(raw: i16) -> f32 {
        if -9000 < raw && raw < 9000 { 0.0 } else { (raw as f32 / i16::MAX as f32).powf(3.0) }
    }

    let mut event_pump = sdl_context.event_pump().unwrap();

    //
    // Save startup profile when profiling with tracing (for in-game timings)
    //

    #[cfg(feature = "profile-with-tracing")]
    let (startup_spans, mut frame_spans, mut capture_next_frame_spans) = {
        drop(startup_span_guard);
        (profiling_subscriber.drain_current_spans(), Vec::new(), true)
    };

    //
    // Main loop
    //

    'main: loop {
        #[cfg(feature = "profile-with-tracing")]
        if capture_next_frame_spans {
            frame_spans = profiling_subscriber.drain_current_spans();
        } else {
            let _ = profiling_subscriber.drain_current_spans();
        }

        profiling::scope!("frame");

        if let Some(which) = opened_controller {
            profiling::scope!("opening controller");
            controller = Some(controller_subsystem.open(which).unwrap());
            opened_controller = None;
        }

        {
            profiling::scope!("event handling");
            for event in event_pump.poll_iter() {
                {
                    profiling::scope!("imgui event processing", &format!("event: {event:?}"));
                    imgui_platform.handle_event(&mut imgui, &event);
                }
                let handle_mouse_events = !imgui.io().want_capture_mouse;
                let handle_keyboard_events = !imgui.io().want_capture_keyboard;

                profiling::scope!("event-specific processing", &format!("event: {event:?}"));
                match event {
                    Event::Quit { .. } => {
                        sdl_context.mouse().set_relative_mouse_mode(false);
                        sdl_context.mouse().show_cursor(true);
                        break 'main;
                    }

                    Event::KeyDown { keycode, .. } if handle_keyboard_events => {
                        analog_controls = false;
                        match keycode {
                            Some(Keycode::Num0) => debug_value = 0,
                            Some(Keycode::Num1) => debug_value = 1,
                            Some(Keycode::Num2) => debug_value = 2,
                            Some(Keycode::Num3) => debug_value = 3,
                            Some(Keycode::Num4) => debug_value = 4,
                            Some(Keycode::Num5) => debug_value = 5,
                            Some(Keycode::Num6) => debug_value = 6,
                            Some(Keycode::Num7) => debug_value = 7,
                            Some(Keycode::W) => dz = 1.0,
                            Some(Keycode::S) => dz = -1.0,
                            Some(Keycode::A) => dx = 1.0,
                            Some(Keycode::D) => dx = -1.0,
                            Some(Keycode::Q) => dy = 1.0,
                            Some(Keycode::X) => dy = -1.0,
                            Some(Keycode::LShift) => sprinting = true,
                            Some(Keycode::Escape) if mouse_look => {
                                mouse_look = false;
                                sdl_context.mouse().set_relative_mouse_mode(false);
                                sdl_context.mouse().show_cursor(true);
                                imgui.io_mut().config_flags.set(imgui::ConfigFlags::NO_MOUSE, mouse_look);
                            }
                            _ => {}
                        }
                    }

                    Event::KeyUp { keycode, .. } if handle_keyboard_events => match keycode {
                        Some(Keycode::I) => {
                            immediate_present = !immediate_present;
                            queued_resize = Some(Instant::now());
                            log::info!("immediate present set to: {}", immediate_present);
                        }
                        Some(Keycode::W) if dz > 0.0 => dz = 0.0,
                        Some(Keycode::S) if dz < 0.0 => dz = 0.0,
                        Some(Keycode::A) if dx > 0.0 => dx = 0.0,
                        Some(Keycode::D) if dx < 0.0 => dx = 0.0,
                        Some(Keycode::Q) if dy > 0.0 => dy = 0.0,
                        Some(Keycode::X) if dy < 0.0 => dy = 0.0,
                        Some(Keycode::LShift) => sprinting = false,
                        _ => {}
                    },

                    Event::ControllerAxisMotion { axis, value, .. } => {
                        analog_controls = true;
                        match axis {
                            Axis::LeftX => dx = -get_axis_deadzoned(value),
                            Axis::LeftY => dz = -get_axis_deadzoned(value),
                            Axis::TriggerRight if value != 0 => dy = value as f32 / i16::MAX as f32,
                            Axis::TriggerRight if dy > 0.0 => dy = 0.0,
                            Axis::TriggerLeft if value != 0 => dy = -(value as f32 / i16::MAX as f32),
                            Axis::TriggerLeft if dy < 0.0 => dy = 0.0,
                            _ => {}
                        }
                    }

                    Event::MouseButtonDown { mouse_btn: MouseButton::Left, .. } if handle_mouse_events => {
                        mouse_look = !mouse_look;
                        if mouse_look {
                            sdl_context.mouse().set_relative_mouse_mode(true);
                            sdl_context.mouse().show_cursor(false);
                        } else {
                            sdl_context.mouse().set_relative_mouse_mode(false);
                            sdl_context.mouse().show_cursor(true);
                        }
                        imgui.io_mut().config_flags.set(imgui::ConfigFlags::NO_MOUSE, mouse_look);
                    }

                    Event::MouseMotion { xrel, yrel, .. } => {
                        if mouse_look {
                            cam_yaw_once_delta -= xrel as f32 / 750.0;
                            cam_pitch_once_delta += yrel as f32 / 750.0;
                        }
                    }

                    Event::Window { win_event: WindowEvent::SizeChanged(_, _), .. } => {
                        mouse_look = false;
                        sdl_context.mouse().set_relative_mouse_mode(false);
                        sdl_context.mouse().show_cursor(true);
                        imgui.io_mut().config_flags.set(imgui::ConfigFlags::NO_MOUSE, mouse_look);
                        (width, height) = window.vulkan_drawable_size();

                        // Do an immediate_present = true resize already, the
                        // "queued resize" will reset it to the proper mode
                        swapchain_settings.extent = renderer::vk::Extent2D { width, height };
                        swapchain_settings.immediate_present = true;
                        recreate_swapchain = true;
                        queued_resize = Some(Instant::now());
                    }

                    Event::ControllerDeviceAdded { which, .. } => {
                        opened_controller = Some(which);
                    }

                    _ => {}
                }
            }
        }

        //
        // "Game" update
        //

        refresh_rate = {
            profiling::scope!("getting refresh rate");
            window.display_mode().map(|dm| dm.refresh_rate).unwrap_or(60)
        };

        {
            profiling::scope!("updating controls");
            if !analog_controls && (dx != 0.0 || dy != 0.0 || dz != 0.0) {
                let dl = (dx * dx + dy * dy + dz * dz).sqrt();
                dx /= dl;
                dy /= dl;
                dz /= dl;
            }

            if analog_controls {
                if let Some(controller) = &controller {
                    let speed = 2.0 / refresh_rate as f32;
                    cam_yaw_delta = -get_axis_deadzoned(controller.axis(Axis::RightX)) * speed;
                    cam_pitch_delta = get_axis_deadzoned(controller.axis(Axis::RightY)) * speed;
                }
            }
        }

        let too_slow;
        let dt_duration;
        {
            profiling::scope!("game update");

            let frame_start = Instant::now();
            let real_dt = frame_start - last_frame_start;
            let fixed_dt = Duration::from_nanos(1_000_000_000 / refresh_rate as u64);
            too_slow = if swapchain_settings.immediate_present {
                true
            } else {
                real_dt > (fixed_dt * 12 / 10) // Too slow if last frame was 20% longer than fixed timestep,
            };
            dt_duration = if too_slow { real_dt } else { fixed_dt }; // fall back to variable timestep when too slow.
            last_frame_start = frame_start;

            let dt = dt_duration.as_secs_f32();
            {
                profiling::scope!("apply rotation and movement");
                cam_yaw += cam_yaw_delta + cam_yaw_once_delta;
                cam_pitch = (cam_pitch + cam_pitch_delta + cam_pitch_once_delta).clamp(-FRAC_PI_2, FRAC_PI_2);
                cam_yaw_once_delta = 0.0;
                cam_pitch_once_delta = 0.0;

                if dx != 0.0 || dz != 0.0 || dy != 0.0 {
                    let speed = if sprinting { 10.0 } else { 5.0 };
                    let control_vec = Vec3::new(dx, dy, dz);
                    let orientation = Quat::from_rotation_y(cam_yaw) * Quat::from_rotation_x(cam_pitch);
                    let move_vec = orientation * control_vec * speed * dt;
                    cam_x += move_vec.x;
                    cam_y += move_vec.y;
                    cam_z += move_vec.z;
                }
            }
            game_time += dt;
        }

        //
        // User interface
        //

        {
            profiling::scope!("user interface (imgui)");
            imgui_platform.prepare_frame(&mut imgui, &window, &event_pump);
            #[allow(dead_code)]
            let ui = imgui.frame();
            #[cfg(feature = "profile-with-tracing")]
            ui.window("Performance stats")
                .size([640.0, 480.0], imgui::Condition::Appearing)
                .collapsed(true, imgui::Condition::Appearing)
                .build(|| {
                    ui.checkbox("Capture timings every frame", &mut capture_next_frame_spans);
                    ui.separator();
                    profiling_tracing_subscriber::span_tree(ui, &frame_spans);
                    ui.separator();
                    profiling_tracing_subscriber::span_tree(ui, &startup_spans);
                });
        }

        //
        // Rendering
        //

        let mut scene = renderer::Scene::default();
        {
            profiling::scope!("rendering (queueing draws)");

            if let Some(resize_timestamp) = queued_resize {
                let duration_since_resize = Instant::now() - resize_timestamp;
                if duration_since_resize > Duration::from_millis(500) {
                    swapchain_settings.extent = renderer::vk::Extent2D { width, height };
                    swapchain_settings.immediate_present = immediate_present;
                    recreate_swapchain = true;
                    queued_resize = None;
                }
            }

            scene.camera.orientation = Quat::from_rotation_y(cam_yaw) * Quat::from_rotation_x(cam_pitch);
            scene.camera.position = Vec3::new(cam_x, cam_y, cam_z);
            scene.world_space = renderer::CoordinateSystem::GLTF;

            {
                profiling::scope!("queue meshes to render");
                sponza_model.queue(&mut scene, Affine3A::IDENTITY);

                let animations = smol_ame_model
                    .animations
                    .iter()
                    .map(|animation| (game_time % animation.end_time, animation))
                    .collect::<Vec<(f32, &gltf::Animation)>>();
                let smol_ame_transform = Affine3A::from_scale_rotation_translation(
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
            swapchain.recreate(&device, &physical_device, &swapchain_settings);
            pipelines =
                renderer::Pipelines::new(&device, &descriptors, swapchain.extent, msaa_samples, attachment_formats, Some(pipelines));
            framebuffers = renderer::Framebuffers::new(&instance.inner, &device, &physical_device, &pipelines, &swapchain);
            recreate_swapchain = false;
        }

        {
            let frame_index = {
                profiling::scope!("wait for present", "or for the previous frame to finish");
                renderer.wait_frame(&swapchain)
            };
            match frame_index {
                Ok(frame_index) => {
                    {
                        profiling::scope!("rendering (imgui)");
                        imgui_renderer.render(imgui.render(), &mut scene, &mut descriptors);
                    }
                    {
                        profiling::scope!("rendering (the vulkan part)");
                        renderer.render_frame(&frame_index, &mut descriptors, &pipelines, &framebuffers, &mut scene, debug_value);
                    }
                    profiling::scope!("present");
                    match renderer.present_frame(frame_index, &swapchain) {
                        Ok(()) => {}
                        Err(renderer::SwapchainError::OutOfDate) => recreate_swapchain = true,
                    }
                    profiling::finish_frame!();
                }
                Err(renderer::SwapchainError::OutOfDate) => {
                    recreate_swapchain = true;
                }
            }
        }

        //
        // Frame pacing
        //

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

#[cfg(feature = "profile-with-tracing")]
mod profiling_tracing_subscriber {
    use std::collections::HashMap;
    use std::num::NonZeroU64;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use profiling::tracing::span::{Attributes, Id, Record};
    use profiling::tracing::{Event, Metadata, Subscriber};

    #[derive(Clone, Copy)]
    pub struct SpanData {
        pub accumulated: Duration,
        pub enter_time: Option<Instant>,
        pub name: &'static str,
        pub depth: u8,
    }

    #[derive(Default)]
    pub struct ProfilingSubscriber {
        spans: Arc<Mutex<Vec<(NonZeroU64, SpanData)>>>,
        span_counter: AtomicU64,
    }

    impl ProfilingSubscriber {
        pub fn drain_current_spans(&self) -> Vec<SpanData> {
            self.spans.lock().unwrap().drain(..).map(|(_, span)| span).collect::<Vec<_>>()
        }
    }

    impl Subscriber for &'static ProfilingSubscriber {
        fn enabled(&self, metadata: &Metadata<'_>) -> bool {
            metadata.is_span()
        }
        fn record(&self, _span: &Id, _values: &Record<'_>) {}
        fn record_follows_from(&self, _span: &Id, _follows: &Id) {}
        fn event(&self, _event: &Event<'_>) {}

        fn new_span(&self, span: &Attributes<'_>) -> Id {
            let mut spans = self.spans.lock().unwrap();
            let id = NonZeroU64::new(1 + self.span_counter.fetch_add(1, Ordering::Relaxed)).unwrap();
            let depth = spans.iter().rev().find(|(_, span)| span.enter_time.is_some()).map(|(_, span)| span.depth).unwrap_or(0) + 1;
            let data = SpanData { accumulated: Duration::ZERO, enter_time: None, name: span.metadata().name(), depth };
            spans.push((id, data));
            Id::from_non_zero_u64(id)
        }

        fn enter(&self, span: &Id) {
            if let Some((_, span_data)) = self.spans.lock().unwrap().iter_mut().find(|(span_u64, _)| span.into_non_zero_u64() == *span_u64)
            {
                span_data.enter_time = Some(Instant::now());
            }
        }

        fn exit(&self, span: &Id) {
            if let Some((_, span_data)) = self.spans.lock().unwrap().iter_mut().find(|(span_u64, _)| span.into_non_zero_u64() == *span_u64)
            {
                if let Some(enter_time) = span_data.enter_time.take() {
                    span_data.accumulated += Instant::now() - enter_time;
                }
            }
        }
    }

    pub fn span_tree(ui: &imgui::Ui, timed_spans: &[SpanData]) {
        fn fmt_ms(d: Duration) -> String {
            format!("{} µs", d.as_micros())
        }

        let mut tree_nodes: Vec<(&SpanData, imgui::TreeNodeToken)> = Vec::with_capacity(timed_spans.len());
        let mut span_iter = timed_spans.iter().peekable();
        let mut name_counts = Vec::new();
        name_counts.push(HashMap::new());
        while let Some(span) = span_iter.next() {
            // If this span is closer to the root than the latest
            // opened span, drop that span and its TreeNodeToken to
            // close it out.
            while let Some(would_be_parent) = tree_nodes.pop() {
                if would_be_parent.0.depth < span.depth {
                    tree_nodes.push(would_be_parent);
                    break;
                }
                name_counts.pop();
            }

            // Add the number of identical names seen thus far under this parent
            // to the id stack, to differentiate between spans with the same name.
            let _id_token = {
                let name_count = {
                    let name_counts = name_counts.last_mut().unwrap();
                    let new_count = name_counts.get(&span.name).copied().unwrap_or(0) + 1;
                    name_counts.insert(span.name, new_count);
                    new_count
                };
                ui.push_id_int(name_count)
            };

            if let Some(tree_node) = ui
                .tree_node_config(span.name)
                .allow_item_overlap(true)
                .default_open(tree_nodes.is_empty())
                .leaf(if let Some(next_span) = span_iter.peek() { next_span.depth <= span.depth } else { true })
                .push()
            {
                // Push this tree node into the vec, which results
                // everything being included within this node, until
                // it's dropped.
                tree_nodes.push((span, tree_node));
                name_counts.push(HashMap::new());
            } else {
                // This span's tree node is not open, skip over the child spans
                while let Some(next_span) = span_iter.peek() {
                    if next_span.depth > span.depth {
                        let _hidden_child = span_iter.next();
                    } else {
                        break;
                    }
                }
            }

            // Write out the % of the frame this span took
            let parent_duration = timed_spans.first().map(|span| span.accumulated).unwrap_or(span.accumulated);
            let pct = 100.0 * span.accumulated.as_secs_f32() / parent_duration.as_secs_f32();
            let pct_text = format!("{pct:.1} %");
            let [window_width, _] = ui.window_size();
            let [pct_text_width, _] = ui.calc_text_size(&pct_text);
            ui.same_line_with_pos(window_width - pct_text_width - 50.0);
            ui.text_colored(
                match pct {
                    50.0.. => [1.0, (100.0 - pct) / 50.0, 0.0, 1.0],
                    1.0.. => [1.0, 1.0, ((50.0 - pct) / 50.0).powf(1.0), 1.0],
                    _ => [0.2 + 0.8 * pct, 0.2 + 0.8 * pct, 0.2 + 0.8 * pct, 1.0],
                },
                pct_text,
            );

            // Write out the time this span spanned
            let timing_text = fmt_ms(span.accumulated);
            let [timing_text_width, _] = ui.calc_text_size(&timing_text);
            ui.same_line_with_pos(window_width - timing_text_width - 150.0);
            ui.text(timing_text);
        }
    }
}
