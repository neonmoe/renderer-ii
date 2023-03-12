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
        println!(
            "cargo:warning={}: spirv out-of-date, recompiling shaders using glslc",
            env!("CARGO_PKG_NAME"),
        );
        for (src_path, dst_path) in shaders.shader_srcs_and_dsts {
            fs::create_dir_all(dst_path.parent().unwrap()).unwrap();
            let _ = fs::remove_file(&dst_path);
            glsl_to_spirv(&src_path, &dst_path);
        }
    }
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
