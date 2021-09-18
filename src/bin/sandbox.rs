use log::LevelFilter;
use neonvk::vk;
use sdl2::event::{Event, WindowEvent};
use sdl2::keyboard::Keycode;
use sdl2::messagebox::{show_simple_message_box, MessageBoxFlag};
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
    let (gpu, _gpus) = neonvk::Gpu::new(&driver, None)?;
    let mut canvas = neonvk::Canvas::new(&gpu, None, width, height, false)?;

    let loading_frame_index = gpu.wait_frame(&canvas)?;
    let camera = neonvk::Camera::new();

    let pink = [0xDD, 0x33, 0xDD, 0xFF];
    let fallback_texture = neonvk::Texture::new(&gpu, loading_frame_index, &pink, 1, 1, vk::Format::R8G8B8A8_SRGB)?;
    gpu.set_fallback_texture(&fallback_texture);

    let mut resources = neonvk::GltfResources::default();
    resources::load_resources(&mut resources);
    let sponza_model = neonvk::Gltf::from_gltf(&gpu, loading_frame_index, include_str!("sponza/glTF/Sponza.gltf"), &mut resources)?;
    let cube_model = neonvk::Gltf::from_glb(&gpu, loading_frame_index, include_bytes!("testbox/testbox.glb"), &mut resources)?;

    let (tex_w, tex_h, tex_format, tex_bytes) = neonvk::image_loading::load_png(include_bytes!("tree.png"), true)?;
    let tree_texture = neonvk::Texture::new(&gpu, loading_frame_index, &tex_bytes, tex_w, tex_h, tex_format)?;

    let quad_vertices: &[&[u8]] = &[
        bytemuck::bytes_of(&[[-0.5f32, 0.5, 0.0], [-0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [0.5, -0.5, 0.0]]),
        bytemuck::bytes_of(&[[0.0f32, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]),
    ];
    let quad_indices: &[u8] = bytemuck::bytes_of(&[0u16, 1, 2, 3, 2, 1]);
    let quad = neonvk::Mesh::new::<u16>(&gpu, loading_frame_index, quad_vertices, quad_indices, neonvk::Pipeline::Default)?;

    // Get the first frame out of the way, to upload the meshes.
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
        scene.queue(
            &quad,
            &tree_texture,
            Mat4::from_translation(ultraviolet::Vec3::new(-0.5, 1.5, 0.0)) * rotation,
        );
        for (mesh, texture, transform) in cube_model.mesh_iter() {
            scene.queue(
                mesh,
                texture,
                Mat4::from_translation(ultraviolet::Vec3::new(0.5, 1.5, 0.0)) * Mat4::from_scale(0.5) * rotation * transform,
            );
        }
        for (mesh, texture, transform) in sponza_model.mesh_iter() {
            scene.queue(mesh, texture, transform);
        }

        let frame_index = gpu.wait_frame(&canvas)?;
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

mod resources {
    macro_rules! insert_sponza_resource {
        ($resources:expr, $expression:expr) => {
            $resources.insert($expression, include_bytes!(concat!("sponza/glTF/", $expression)).as_ref())
        };
    }

    pub fn load_resources(resources: &mut neonvk::GltfResources) {
        insert_sponza_resource!(resources, "10381718147657362067.jpg");
        insert_sponza_resource!(resources, "10388182081421875623.jpg");
        insert_sponza_resource!(resources, "11474523244911310074.jpg");
        insert_sponza_resource!(resources, "11490520546946913238.jpg");
        insert_sponza_resource!(resources, "11872827283454512094.jpg");
        insert_sponza_resource!(resources, "11968150294050148237.jpg");
        insert_sponza_resource!(resources, "1219024358953944284.jpg");
        insert_sponza_resource!(resources, "12501374198249454378.jpg");
        insert_sponza_resource!(resources, "13196865903111448057.jpg");
        insert_sponza_resource!(resources, "13824894030729245199.jpg");
        insert_sponza_resource!(resources, "13982482287905699490.jpg");
        insert_sponza_resource!(resources, "14118779221266351425.jpg");
        insert_sponza_resource!(resources, "14170708867020035030.jpg");
        insert_sponza_resource!(resources, "14267839433702832875.jpg");
        insert_sponza_resource!(resources, "14650633544276105767.jpg");
        insert_sponza_resource!(resources, "15295713303328085182.jpg");
        insert_sponza_resource!(resources, "15722799267630235092.jpg");
        insert_sponza_resource!(resources, "16275776544635328252.png");
        insert_sponza_resource!(resources, "16299174074766089871.jpg");
        insert_sponza_resource!(resources, "16885566240357350108.jpg");
        insert_sponza_resource!(resources, "17556969131407844942.jpg");
        insert_sponza_resource!(resources, "17876391417123941155.jpg");
        insert_sponza_resource!(resources, "2051777328469649772.jpg");
        insert_sponza_resource!(resources, "2185409758123873465.jpg");
        insert_sponza_resource!(resources, "2299742237651021498.jpg");
        insert_sponza_resource!(resources, "2374361008830720677.jpg");
        insert_sponza_resource!(resources, "2411100444841994089.jpg");
        insert_sponza_resource!(resources, "2775690330959970771.jpg");
        insert_sponza_resource!(resources, "2969916736137545357.jpg");
        insert_sponza_resource!(resources, "332936164838540657.jpg");
        insert_sponza_resource!(resources, "3371964815757888145.jpg");
        insert_sponza_resource!(resources, "3455394979645218238.jpg");
        insert_sponza_resource!(resources, "3628158980083700836.jpg");
        insert_sponza_resource!(resources, "3827035219084910048.jpg");
        insert_sponza_resource!(resources, "4477655471536070370.jpg");
        insert_sponza_resource!(resources, "4601176305987539675.jpg");
        insert_sponza_resource!(resources, "466164707995436622.jpg");
        insert_sponza_resource!(resources, "4675343432951571524.jpg");
        insert_sponza_resource!(resources, "4871783166746854860.jpg");
        insert_sponza_resource!(resources, "4910669866631290573.jpg");
        insert_sponza_resource!(resources, "4975155472559461469.jpg");
        insert_sponza_resource!(resources, "5061699253647017043.png");
        insert_sponza_resource!(resources, "5792855332885324923.jpg");
        insert_sponza_resource!(resources, "5823059166183034438.jpg");
        insert_sponza_resource!(resources, "6047387724914829168.jpg");
        insert_sponza_resource!(resources, "6151467286084645207.jpg");
        insert_sponza_resource!(resources, "6593109234861095314.jpg");
        insert_sponza_resource!(resources, "6667038893015345571.jpg");
        insert_sponza_resource!(resources, "6772804448157695701.jpg");
        insert_sponza_resource!(resources, "7056944414013900257.jpg");
        insert_sponza_resource!(resources, "715093869573992647.jpg");
        insert_sponza_resource!(resources, "7268504077753552595.jpg");
        insert_sponza_resource!(resources, "7441062115984513793.jpg");
        insert_sponza_resource!(resources, "755318871556304029.jpg");
        insert_sponza_resource!(resources, "759203620573749278.jpg");
        insert_sponza_resource!(resources, "7645212358685992005.jpg");
        insert_sponza_resource!(resources, "7815564343179553343.jpg");
        insert_sponza_resource!(resources, "8006627369776289000.png");
        insert_sponza_resource!(resources, "8051790464816141987.jpg");
        insert_sponza_resource!(resources, "8114461559286000061.jpg");
        insert_sponza_resource!(resources, "8481240838833932244.jpg");
        insert_sponza_resource!(resources, "8503262930880235456.jpg");
        insert_sponza_resource!(resources, "8747919177698443163.jpg");
        insert_sponza_resource!(resources, "8750083169368950601.jpg");
        insert_sponza_resource!(resources, "8773302468495022225.jpg");
        insert_sponza_resource!(resources, "8783994986360286082.jpg");
        insert_sponza_resource!(resources, "9288698199695299068.jpg");
        insert_sponza_resource!(resources, "9916269861720640319.jpg");
        insert_sponza_resource!(resources, "Sponza.bin");
        insert_sponza_resource!(resources, "white.png");
    }
}
