use std::env;
use std::fmt::Write;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::SystemTime;

struct ShadersToCompile {
    src_modified: SystemTime,
    dst_modified: SystemTime,
    first_compile: bool,
    shader_srcs_and_dsts: Vec<(PathBuf, PathBuf)>,
}

fn main() {
    let mut shaders = ShadersToCompile {
        src_modified: SystemTime::UNIX_EPOCH,
        dst_modified: SystemTime::UNIX_EPOCH,
        first_compile: false,
        shader_srcs_and_dsts: Vec::new(),
    };
    compile_shaders(PathBuf::from("shaders/glsl"), PathBuf::from("shaders/spirv"), &mut shaders);

    if shaders.first_compile || shaders.src_modified > shaders.dst_modified {
        println!("cargo:warning={}: spirv out-of-date, recompiling shaders using glslc", env!("CARGO_PKG_NAME"),);
        for (src_path, dst_path) in shaders.shader_srcs_and_dsts {
            fs::create_dir_all(dst_path.parent().unwrap()).unwrap();
            let _ = fs::remove_file(&dst_path);
            glsl_to_spirv(&src_path, &dst_path);
        }
    }

    let shader_constants_glsl = include_str!("shaders/glsl/constants.glsl");
    let mut shader_constants_rust = String::with_capacity(shader_constants_glsl.len());
    for line in shader_constants_glsl.lines() {
        if let Some(name_and_value) = line.strip_prefix("#define ") {
            let (name, value) = name_and_value.split_once(' ').expect("malformed #define in constants.glsl");
            write!(&mut shader_constants_rust, "pub const {name}: u32 = {value};").unwrap();
            shader_constants_rust.push('\n');
        } else if line.starts_with("///") {
            shader_constants_rust.push_str(line);
            shader_constants_rust.push('\n');
        }
    }
    let mut shader_constants_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    shader_constants_path.push("shader_constants.rs");
    fs::write(shader_constants_path, shader_constants_rust).unwrap();
}

fn compile_shaders(src_path: PathBuf, mut dst_path: PathBuf, shaders: &mut ShadersToCompile) {
    if src_path.is_file() {
        let mut file_name = dst_path.file_name().unwrap().to_os_string();
        file_name.push(".spv");
        dst_path.set_file_name(file_name);
        if let Some(src_modified) = modified_time(&src_path) {
            shaders.src_modified = shaders.src_modified.max(src_modified);
        }
        if let Some(dst_modified) = modified_time(&dst_path) {
            shaders.dst_modified = shaders.dst_modified.max(dst_modified);
        }
        if is_glsl_module(&src_path) {
            shaders.first_compile |= !dst_path.exists();
            shaders.shader_srcs_and_dsts.push((src_path, dst_path));
        }
    } else if src_path.is_dir() {
        for dir_entry in fs::read_dir(src_path).unwrap() {
            let shader_path = dir_entry.unwrap().path();
            let output_path = dst_path.join(shader_path.components().next_back().unwrap());
            compile_shaders(shader_path, output_path, shaders);
        }
    }
}

fn is_glsl_module(glsl_path: &Path) -> bool {
    const EXPECTED_HEADER: &[u8] = b"#version ";
    let mut header = [0; EXPECTED_HEADER.len()];
    let mut src_file = File::open(glsl_path).unwrap();
    src_file.read_exact(&mut header).unwrap();
    header == EXPECTED_HEADER
}

fn modified_time(path: &Path) -> Option<SystemTime> {
    fs::metadata(path).ok()?.modified().ok()
}

fn glsl_to_spirv(glsl: &Path, spirv: &Path) {
    let command = Command::new("glslc")
        .args(["-O", "-g", "-std=450core", "--target-env=vulkan1.2", "-Werror"])
        .arg(glsl)
        .arg("-o")
        .arg(spirv)
        .output()
        .unwrap();
    if !command.status.success() {
        eprintln!(
            "GLSL to SPIR-V compilation failed!\n=== Stdout: ===\n{}\n=== Stderr: ===\n{}",
            String::from_utf8_lossy(&command.stdout),
            String::from_utf8_lossy(&command.stderr),
        );
        std::process::exit(1);
    }
}
