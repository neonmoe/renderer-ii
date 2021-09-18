use crate::Error;
use ash::vk;

#[profiling::function]
pub fn load_png(bytes: &[u8], srgb: bool) -> Result<(u32, u32, vk::Format, Vec<u8>), Error> {
    use png::{BitDepth, ColorType, Decoder};
    let decoder = Decoder::new(bytes);
    let mut reader = decoder.read_info().map_err(Error::PngDecoding)?;
    let mut pixels = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut pixels).map_err(Error::PngDecoding)?;
    pixels.truncate(info.buffer_size());

    let (bits, colors) = (info.bit_depth, info.color_type);
    let (format, needs_padding) = {
        match (bits, colors) {
            (BitDepth::Eight, ColorType::Rgba) if srgb => (vk::Format::R8G8B8A8_SRGB, false),
            (BitDepth::Eight, ColorType::Rgba) => (vk::Format::R8G8B8A8_UNORM, false),
            (BitDepth::Eight, ColorType::Rgb) if srgb => (vk::Format::R8G8B8A8_SRGB, true),
            (BitDepth::Eight, ColorType::Rgb) => (vk::Format::R8G8B8A8_UNORM, true),
            (BitDepth::Eight, ColorType::Grayscale) => (vk::Format::R8_UNORM, false),
            (_, _) => return Err(Error::UnsupportedImageFormat(bits, colors)),
        }
    };
    if needs_padding {
        pixels = pad_rgb24_to_rgba32(&pixels);
    }
    Ok((info.width, info.height, format, pixels))
}

#[profiling::function]
pub fn load_jpeg(bytes: &[u8], srgb: bool) -> Result<(u32, u32, vk::Format, Vec<u8>), Error> {
    use jpeg_decoder::{Decoder, PixelFormat};
    let mut decoder = Decoder::new(bytes);
    let mut pixels = decoder.decode().map_err(Error::JpegDecoding)?;
    let info = decoder.info().unwrap(); // Should be Some if decode returned Ok
    let (width, height) = (info.width as u32, info.height as u32);
    let (format, needs_padding) = match info.pixel_format {
        PixelFormat::L8 => (vk::Format::R8_UNORM, false),
        PixelFormat::RGB24 if srgb => (vk::Format::R8G8B8A8_SRGB, true),
        PixelFormat::RGB24 => (vk::Format::R8G8B8A8_SRGB, true),
        PixelFormat::CMYK32 => return Err(Error::MiscImageDecoding("cmyk is not supported as pixel format")),
    };
    if needs_padding {
        pixels = pad_rgb24_to_rgba32(&pixels);
    }
    Ok((width, height, format, pixels))
}

/// Allocates a new, properly sized pixel array, and fills it out with
/// the rgb values + 0xFF for alpha.
#[profiling::function]
fn pad_rgb24_to_rgba32(bytes: &[u8]) -> Vec<u8> {
    let mut new_bytes = vec![0; bytes.len() * 4 / 3];
    for (src, dst) in bytes.chunks_exact(3).zip(new_bytes.chunks_exact_mut(4)) {
        if let ([src_r, src_g, src_b], [dst_r, dst_g, dst_b, dst_a]) = (src, dst) {
            *dst_r = *src_r;
            *dst_g = *src_g;
            *dst_b = *src_b;
            *dst_a = 0xFF;
        }
    }
    new_bytes
}
