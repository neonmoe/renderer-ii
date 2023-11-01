use core::ffi::CStr;

use arrayvec::ArrayString;
use bytemuck::cast_slice;
use glam::{Mat4, Vec2, Vec3, Vec4};
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;

fn main() {
    use logger::Logger;
    static LOGGER: Logger = Logger;
    log::set_logger(&LOGGER)
        .map(|()| log::set_max_level(log::LevelFilter::Trace))
        .unwrap();

    let sdl = sdl2::init().unwrap();
    let time = sdl.timer().unwrap();
    let video = sdl.video().unwrap();
    let mut window = video
        .window("Hello, Triangles!", 640, 480)
        .vulkan()
        .allow_highdpi()
        .resizable()
        .build()
        .unwrap();
    let app_name = unsafe { CStr::from_bytes_with_nul_unchecked(b"triangle example application\0") };
    let instance = renderer::Instance::new(&window, app_name, 0, 1, 0).unwrap();
    let surface = renderer::create_surface(&instance.entry, &instance.inner, &window, &window).unwrap();

    let mut physical_devices = renderer::get_physical_devices(&instance.entry, &instance.inner, surface.inner);
    let physical_device = physical_devices.remove(0).unwrap();
    let device = physical_device.create_device(&instance.entry, &instance.inner).unwrap();

    let attachment_formats = physical_device.attachment_formats();
    let msaa_samples = renderer::vk::SampleCountFlags::TYPE_1;

    let mut texture_arena = renderer::VulkanArena::<renderer::ForImages>::new(
        &instance.inner,
        &device,
        &physical_device,
        262_144,
        renderer::MemoryProps::for_textures(),
        format_args!("triangle textures"),
    )
    .unwrap();
    let mut buffer_arena = renderer::VulkanArena::<renderer::ForBuffers>::new(
        &instance.inner,
        &device,
        &physical_device,
        262_144,
        renderer::MemoryProps::for_buffers(),
        format_args!("triangle buffers"),
    )
    .unwrap();
    let mut staging_arena = renderer::VulkanArena::<renderer::ForBuffers>::new(
        &instance.inner,
        &device,
        &physical_device,
        262_144,
        renderer::MemoryProps::for_staging(),
        format_args!("triangle staging"),
    )
    .unwrap();
    let mut uploader = renderer::Uploader::new(
        &device,
        device.graphics_queue,
        device.transfer_queue,
        &physical_device,
        "triangle assets",
    );

    let pbr_defaults =
        renderer::image_loading::pbr_defaults::all_defaults(&device, &mut staging_arena, &mut uploader, &mut texture_arena).unwrap();
    let mut descriptors = renderer::Descriptors::new(&device, &physical_device, pbr_defaults);

    let (width, height) = window.vulkan_drawable_size();
    let mut swapchain_settings = renderer::SwapchainSettings {
        extent: renderer::vk::Extent2D { width, height },
        immediate_present: false,
    };

    let mut swapchain = renderer::Swapchain::new(&device, &physical_device, surface, &swapchain_settings);
    let mut pipelines = renderer::Pipelines::new(&device, &descriptors, swapchain.extent, msaa_samples, attachment_formats, None);
    let mut framebuffers = renderer::Framebuffers::new(&instance.inner, &device, &physical_device, &pipelines, &swapchain);
    let mut renderer = renderer::Renderer::new(&instance.inner, &device, &physical_device);

    let (triangle_mesh1, triangle_mesh2) = {
        let positions = [Vec3::new(-0.5, 0.5, 0.8), Vec3::new(0.5, 0.5, 0.8), Vec3::new(-0.1, -0.5, 0.8)];
        let positions: &[u8] = cast_slice(&positions);
        let uvs = [Vec2::new(0.0, 1.0), Vec2::new(1.0, 1.0), Vec2::new(0.5, 0.0)];
        let uvs: &[u8] = cast_slice(&uvs);
        let norms: &[u8] = cast_slice(&[Vec3::X, Vec3::Y, Vec3::Z]);
        let tangents: &[u8] = cast_slice(&[Vec4::Y, Vec4::Z, Vec4::X]);
        let indices: &[u16] = &[0u16, 1, 2];
        let vertex_buffers: &[&[u8]; 4] = &[positions, uvs, norms, tangents];

        let mut measurer = renderer::VertexLibraryMeasurer::default();
        measurer.add_mesh(renderer::PipelineIndex::PbrOpaque, vertex_buffers, indices);
        measurer.add_mesh(renderer::PipelineIndex::PbrOpaque, vertex_buffers, indices);
        let mut builder =
            renderer::VertexLibraryBuilder::new(&mut staging_arena, &mut buffer_arena, measurer, format_args!("triangle mesh")).unwrap();
        let mesh1 = builder.add_mesh(renderer::PipelineIndex::PbrOpaque, vertex_buffers, indices);
        let mesh2 = builder.add_mesh(renderer::PipelineIndex::PbrOpaque, vertex_buffers, indices);
        builder.upload(&mut uploader, &mut buffer_arena);
        (mesh1, mesh2)
    };

    let triangle_material = renderer::Material::new(
        &mut descriptors,
        renderer::PipelineSpecificData::Pbr {
            base_color: None,
            metallic_roughness: None,
            normal: None,
            occlusion: None,
            emissive: None,
            factors: renderer::PbrFactors {
                base_color: Vec4::new(0.2, 0.8, 0.2, 1.0),
                emissive_and_occlusion: Vec4::ZERO,
                alpha_rgh_mtl_normal: Vec4::ONE * 0.5,
            },
            alpha_mode: renderer::AlphaMode::Opaque,
        },
        ArrayString::from("triangle material").unwrap(),
    )
    .unwrap();

    uploader.wait(None);
    drop(uploader);
    drop(staging_arena);

    let mut swapchain_recreation_requested = None;
    let mut event_pump = sdl.event_pump().unwrap();
    let mut scene = renderer::Scene::new(&physical_device);
    let mut fps_ticks = Vec::with_capacity(10_000);
    let mut prev_fps_update = time.ticks();
    'main: loop {
        for event in event_pump.poll_iter() {
            match event {
                Event::Quit { .. } => break 'main,
                Event::Window {
                    win_event: WindowEvent::SizeChanged(_, _),
                    ..
                } => {
                    let (w, h) = window.vulkan_drawable_size();
                    swapchain_settings.extent.width = w;
                    swapchain_settings.extent.height = h;
                    swapchain_recreation_requested = Some(time.ticks());
                }
                Event::KeyDown {
                    keycode: Some(Keycode::I),
                    repeat: false,
                    ..
                } => {
                    swapchain_settings.immediate_present = !swapchain_settings.immediate_present;
                    swapchain_recreation_requested = Some(time.ticks());
                }
                _ => {}
            }
        }

        if let Some(resize_time) = swapchain_recreation_requested {
            if time.ticks() - resize_time > 100 {
                swapchain_recreation_requested = None;
                device.wait_idle();
                drop(framebuffers);
                swapchain.recreate(&device, &physical_device, &swapchain_settings);
                pipelines = renderer::Pipelines::new(
                    &device,
                    &descriptors,
                    swapchain.extent,
                    msaa_samples,
                    attachment_formats,
                    Some(pipelines),
                );
                framebuffers = renderer::Framebuffers::new(&instance.inner, &device, &physical_device, &pipelines, &swapchain);
            }
        }

        scene.clear();
        scene.queue_mesh(&triangle_mesh1, &triangle_material, Mat4::from_scale(Vec3::new(1.0, 1.0, 1.0)));
        scene.queue_mesh(&triangle_mesh2, &triangle_material, Mat4::from_scale(Vec3::new(2.0, 0.5, 1.0)));

        let frame_index = renderer.wait_frame(&swapchain).unwrap();
        renderer.render_frame(&frame_index, &mut descriptors, &pipelines, &framebuffers, &mut scene, 3);
        match { renderer.present_frame(frame_index, &swapchain) } {
            Ok(_) => {}
            Err(renderer::SwapchainError::OutOfDate) => swapchain_recreation_requested = Some(time.ticks()),
        }

        let now = time.ticks();
        fps_ticks.push(now);
        fps_ticks.retain(|t| now - *t <= 1000);
        if now - prev_fps_update >= 1000 {
            prev_fps_update = now;
            let _ = window.set_title(&format!("Hello, Triangles! (FPS: {})", fps_ticks.len()));
        }
    }
    device.wait_idle();
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
                let is_vk_debug_utils_print = file == "src/debug_utils.rs";
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
