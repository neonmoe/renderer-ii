use std::fs::{self};
use std::io::{BufRead, Cursor, Write};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use argh::FromArgs;
use image::imageops::FilterType;
use image::DynamicImage;
use rayon::prelude::*;

const VK_FORMAT_BC7_UNORM_BLOCK: u32 = 145;
//const VK_FORMAT_BC7_SRGB_BLOCK: u32 = 146;

const EXTENSION: &str = "ntex";
const HEADER: &[u8; 992] = b"The GPU decodable image container format this file follows:

the first 992 bytes: this null-terminated header including the null
u32: width
u32: height
u32: depth
u32: mip level count
u32: format from the vulkan 1.3 spec
u32: block width
u32: block height
u32: size of one block in bytes
the rest of the bytes: the raw images for each mip level with no padding

A u32 is a 32-bit little-endian unsigned integer.

The first mip level is this many bytes:

  ceil(width / block width) * ceil(height / block height) * (size of one block in bytes)

Each mip level's size after that is simply the previous mip level's size
divided by two, until it would go under the size of one block.

Files in this format should not be considered ground truth.
Handle your source images in a sane format such as PNG.
Convert them into this format for bundling with applications.

This header should be used to distinguish between versions of this format.

The header is 1024 bytes, hopefully it aligns well.\n\n\n\n\0";

#[derive(FromArgs)]
/// Compresses image files and writes them out next to the original file with
/// the ".ntex" file extension.
struct Opts {
    #[argh(positional)]
    images: Vec<String>,
    #[argh(switch)]
    /// overwrite files without asking
    overwrite: bool,
    #[argh(switch)]
    /// don't print anything (if --overwrite is not set, skips files that exist)
    silent: bool,
    #[argh(switch)]
    /// assume all input files are color files, use lanczos for all mip maps
    assume_color: bool,
    #[argh(switch)]
    /// assume all input files are not color files, use linear filter for all mips
    assume_linear: bool,
}

fn main() -> ExitCode {
    let opts: Opts = argh::from_env();
    if opts.images.is_empty() {
        print(&opts, "No images provided. Use --help to display usage info.");
        std::process::exit(2);
    }

    let count = opts.images.len();
    let counter = AtomicUsize::new(0);
    let failures: u32 = opts
        .images
        .par_iter()
        .map(|path| (path, convert(&opts, path, &counter, count)))
        .map(|(path, result)| {
            if let Err(err) = result {
                print(&opts, format_args!("Failed to convert {}, {}.", path, err));
                1
            } else {
                0
            }
        })
        .sum();
    if failures == 0 {
        ExitCode::SUCCESS
    } else {
        print(&opts, format_args!("Conversion failures: {failures}."));
        ExitCode::FAILURE
    }
}

fn print<T: std::fmt::Display>(opts: &Opts, message: T) {
    if !opts.silent {
        eprintln!("{message}");
    }
}

#[derive(Debug)]
enum Error {
    ReadFile(std::io::Error),
    WriteFile(std::io::Error),
    ImageDecode(image::ImageError),
    OverwriteCheckStdinRead(std::io::Error),
    ImageSmallerThanBlock,
}

impl std::fmt::Display for Error {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Error::ReadFile(err) => write!(fmt, "error reading file: {}", err),
            Error::WriteFile(err) => write!(fmt, "error writing file: {}", err),
            Error::ImageDecode(err) => write!(fmt, "error decoding image: {}", err),
            Error::OverwriteCheckStdinRead(err) => {
                write!(fmt, "error reading stdin for overwriting check: {}", err)
            }
            Error::ImageSmallerThanBlock => write!(fmt, "image is smaller than 4px on at least one axis"),
        }
    }
}

fn convert(opts: &Opts, path: &str, counter: &AtomicUsize, count: usize) -> Result<(), Error> {
    let mut dst_path = PathBuf::from(path);
    dst_path.set_extension(EXTENSION);
    if dst_path.exists() && !opts.overwrite {
        if opts.silent {
            return Ok(());
        }
        let mut line = String::new();
        let stdin = std::io::stdin();
        let mut stdin = stdin.lock();
        loop {
            eprint!("{} exists, replace? [y/n]: ", dst_path.display());
            let _ = std::io::stderr().flush();
            stdin.read_line(&mut line).map_err(Error::OverwriteCheckStdinRead)?;
            line.make_ascii_lowercase();
            let ans = line.trim();
            if ans == "y" || ans == "yes" {
                break;
            }
            if ans == "n" || ans == "no" {
                return Ok(());
            }
        }
    }
    let lowercase_path = path.to_lowercase();
    let sharpen = !opts.assume_linear && (opts.assume_color || lowercase_path.contains("color") || lowercase_path.contains("albedo"));
    if sharpen {
        print(opts, format_args!("Path {path} assumed color: using lanczos for mipmaps.",));
    } else {
        print(opts, format_args!("Path {path} assumed not color: making linear mipmaps."));
    }
    let start_time = Instant::now();

    let input_bytes = fs::read(path).map_err(Error::ReadFile)?;
    let image = image::io::Reader::new(Cursor::new(input_bytes))
        .with_guessed_format()
        .unwrap()
        .decode()
        .map_err(Error::ImageDecode)?;
    let mip_levels = (0..)
        .take_while(|i| {
            let d = 4 * (1 << i);
            image.width() % d == 0 && image.height() % d == 0
        })
        .count() as u32;
    if mip_levels == 0 {
        return Err(Error::ImageSmallerThanBlock);
    }
    let mut output_len = 1024;
    for mip in 0..mip_levels {
        let width = ((image.width() / (1 << mip)) as f32 / 4.0).ceil() as usize;
        let height = ((image.height() / (1 << mip)) as f32 / 4.0).ceil() as usize;
        output_len += (width * height * (128 / 8)).max(16);
    }
    let mut output_bytes = Vec::with_capacity(output_len);
    // the first 992 bytes: this null-terminated header including the null
    output_bytes.extend_from_slice(HEADER);
    // u32: width
    output_bytes.extend_from_slice(&image.width().to_le_bytes());
    // u32: height
    output_bytes.extend_from_slice(&image.height().to_le_bytes());
    // u32: depth
    output_bytes.extend_from_slice(&1u32.to_le_bytes());
    // u32: mip level count
    output_bytes.extend_from_slice(&mip_levels.to_le_bytes());
    // u32: format from the vulkan 1.3 spec
    output_bytes.extend_from_slice(&VK_FORMAT_BC7_UNORM_BLOCK.to_le_bytes());
    // u32: block width
    output_bytes.extend_from_slice(&4u32.to_le_bytes());
    // u32: block height
    output_bytes.extend_from_slice(&4u32.to_le_bytes());
    // u32: size of one block in bytes
    output_bytes.extend_from_slice(&(128u32 /* 128 bits */ / 8).to_le_bytes());
    assert_eq!(output_bytes.len(), 1024);
    // the rest of the bytes: the raw images for each mip level with no padding
    print(
        opts,
        format_args!(
            "Compressing {mip_levels} mips of {} ({}x{}).",
            dst_path.display(),
            image.width(),
            image.height(),
        ),
    );
    let compressed_images = (0..mip_levels)
        .collect::<Vec<u32>>()
        .into_par_iter()
        .map(|mip| {
            let mip_image = if mip == 0 {
                image.clone()
            } else {
                let filter = if sharpen { FilterType::Lanczos3 } else { FilterType::Triangle };
                image.resize_exact(image.width() / (1 << mip), image.height() / (1 << mip), filter)
            };
            compress_image(mip_image)
        })
        .collect::<Vec<Vec<u8>>>();
    for mut compressed_image in compressed_images {
        assert_eq!(output_bytes.len() % 16, 0);
        output_bytes.append(&mut compressed_image);
    }
    assert_eq!(output_bytes.len(), output_len);

    fs::write(&dst_path, output_bytes).map_err(Error::WriteFile)?;
    print(
        opts,
        format_args!(
            "Compressed {} in {:.2}s, {}/{} done.",
            dst_path.display(),
            (Instant::now() - start_time).as_secs_f32(),
            counter.fetch_add(1, Ordering::Relaxed) + 1,
            count,
        ),
    );
    Ok(())
}

fn compress_image(image: DynamicImage) -> Vec<u8> {
    let alpha = image.color().has_alpha();
    let image = image.into_rgba8();
    let width = image.width();
    let height = image.height();
    let stride = image.width() * 4;
    let pixels = image.into_raw();
    let surface = intel_tex::RgbaSurface {
        width,
        height,
        stride,
        data: &pixels,
    };
    let settings = if alpha {
        intel_tex::bc7::alpha_slow_settings()
    } else {
        intel_tex::bc7::opaque_slow_settings()
    };
    let compressed_bytes = intel_tex::bc7::compress_blocks(&settings, &surface);
    // The size of the image according to the format spec (from HEADER):
    // ceil(width / block width) * ceil(height / block height) * (size of one block in bytes)
    let expected_size = ((width as f32 / 4.0).ceil() as u32 * (height as f32 / 4.0).ceil() as u32 * (128 / 8)).max(16);
    assert_eq!(expected_size as usize, compressed_bytes.len());
    compressed_bytes
}
