use crate::Error;
use ash::vk;

/// If the `format` is a 'UNORM' format, and has an SRGB variant,
/// return that.
///
/// Does not work for the 3D ASTC ones.
pub fn to_srgb(format: vk::Format) -> vk::Format {
    match format {
        vk::Format::R8_UNORM => vk::Format::R8_SRGB,
        vk::Format::R8G8_UNORM => vk::Format::R8G8_SRGB,
        vk::Format::R8G8B8_UNORM => vk::Format::R8G8B8_SRGB,
        vk::Format::R8G8B8A8_UNORM => vk::Format::R8G8B8A8_SRGB,
        vk::Format::B8G8R8_UNORM => vk::Format::B8G8R8_SRGB,
        vk::Format::B8G8R8A8_UNORM => vk::Format::B8G8R8A8_SRGB,
        vk::Format::A8B8G8R8_UNORM_PACK32 => vk::Format::A8B8G8R8_SRGB_PACK32,
        vk::Format::BC2_UNORM_BLOCK => vk::Format::BC2_SRGB_BLOCK,
        vk::Format::BC3_UNORM_BLOCK => vk::Format::BC3_SRGB_BLOCK,
        vk::Format::BC7_UNORM_BLOCK => vk::Format::BC7_SRGB_BLOCK,
        vk::Format::BC1_RGB_UNORM_BLOCK => vk::Format::BC1_RGB_SRGB_BLOCK,
        vk::Format::BC1_RGBA_UNORM_BLOCK => vk::Format::BC1_RGBA_SRGB_BLOCK,
        vk::Format::ETC2_R8G8B8_UNORM_BLOCK => vk::Format::ETC2_R8G8B8_SRGB_BLOCK,
        vk::Format::ETC2_R8G8B8A1_UNORM_BLOCK => vk::Format::ETC2_R8G8B8A1_SRGB_BLOCK,
        vk::Format::ETC2_R8G8B8A8_UNORM_BLOCK => vk::Format::ETC2_R8G8B8A8_SRGB_BLOCK,
        vk::Format::PVRTC1_2BPP_UNORM_BLOCK_IMG => vk::Format::PVRTC1_2BPP_SRGB_BLOCK_IMG,
        vk::Format::PVRTC1_4BPP_UNORM_BLOCK_IMG => vk::Format::PVRTC1_4BPP_SRGB_BLOCK_IMG,
        vk::Format::PVRTC2_2BPP_UNORM_BLOCK_IMG => vk::Format::PVRTC2_2BPP_SRGB_BLOCK_IMG,
        vk::Format::PVRTC2_4BPP_UNORM_BLOCK_IMG => vk::Format::PVRTC2_4BPP_SRGB_BLOCK_IMG,
        vk::Format::ASTC_4X4_UNORM_BLOCK => vk::Format::ASTC_4X4_SRGB_BLOCK,
        vk::Format::ASTC_5X4_UNORM_BLOCK => vk::Format::ASTC_5X4_SRGB_BLOCK,
        vk::Format::ASTC_5X5_UNORM_BLOCK => vk::Format::ASTC_5X5_SRGB_BLOCK,
        vk::Format::ASTC_6X5_UNORM_BLOCK => vk::Format::ASTC_6X5_SRGB_BLOCK,
        vk::Format::ASTC_6X6_UNORM_BLOCK => vk::Format::ASTC_6X6_SRGB_BLOCK,
        vk::Format::ASTC_8X5_UNORM_BLOCK => vk::Format::ASTC_8X5_SRGB_BLOCK,
        vk::Format::ASTC_8X6_UNORM_BLOCK => vk::Format::ASTC_8X6_SRGB_BLOCK,
        vk::Format::ASTC_8X8_UNORM_BLOCK => vk::Format::ASTC_8X8_SRGB_BLOCK,
        vk::Format::ASTC_10X5_UNORM_BLOCK => vk::Format::ASTC_10X5_SRGB_BLOCK,
        vk::Format::ASTC_10X6_UNORM_BLOCK => vk::Format::ASTC_10X6_SRGB_BLOCK,
        vk::Format::ASTC_10X8_UNORM_BLOCK => vk::Format::ASTC_10X8_SRGB_BLOCK,
        vk::Format::ASTC_10X10_UNORM_BLOCK => vk::Format::ASTC_10X10_SRGB_BLOCK,
        vk::Format::ASTC_12X10_UNORM_BLOCK => vk::Format::ASTC_12X10_SRGB_BLOCK,
        vk::Format::ASTC_12X12_UNORM_BLOCK => vk::Format::ASTC_12X12_SRGB_BLOCK,
        f => f,
    }
}

/// If the `format` is a 'UNORM' format, return the 'SNORM' variant if
/// there is one.
pub fn to_snorm(format: vk::Format) -> vk::Format {
    match format {
        vk::Format::R8_UNORM => vk::Format::R8_SNORM,
        vk::Format::R8G8_UNORM => vk::Format::R8G8_SNORM,
        vk::Format::R8G8B8_UNORM => vk::Format::R8G8B8_SNORM,
        vk::Format::R8G8B8A8_UNORM => vk::Format::R8G8B8A8_SNORM,
        vk::Format::B8G8R8_UNORM => vk::Format::B8G8R8_SNORM,
        vk::Format::B8G8R8A8_UNORM => vk::Format::B8G8R8A8_SNORM,
        vk::Format::A8B8G8R8_UNORM_PACK32 => vk::Format::A8B8G8R8_SNORM_PACK32,
        vk::Format::A2R10G10B10_UNORM_PACK32 => vk::Format::A2R10G10B10_SNORM_PACK32,
        vk::Format::A2B10G10R10_UNORM_PACK32 => vk::Format::A2B10G10R10_SNORM_PACK32,
        vk::Format::R16_UNORM => vk::Format::R16_SNORM,
        vk::Format::R16G16_UNORM => vk::Format::R16G16_SNORM,
        vk::Format::R16G16B16_UNORM => vk::Format::R16G16B16_SNORM,
        vk::Format::R16G16B16A16_UNORM => vk::Format::R16G16B16A16_SNORM,
        vk::Format::BC4_UNORM_BLOCK => vk::Format::BC4_SNORM_BLOCK,
        vk::Format::BC5_UNORM_BLOCK => vk::Format::BC5_SNORM_BLOCK,
        vk::Format::EAC_R11_UNORM_BLOCK => vk::Format::EAC_R11_SNORM_BLOCK,
        vk::Format::EAC_R11G11_UNORM_BLOCK => vk::Format::EAC_R11G11_SNORM_BLOCK,
        f => f,
    }
}

/// Loads a png into (width, height, format, pixel bytes).
#[profiling::function]
pub fn load_png(bytes: &[u8]) -> Result<(u32, u32, vk::Format, Vec<u8>), Error> {
    use png::{BitDepth, ColorType, Decoder};
    let decoder = Decoder::new(bytes);
    let mut reader = decoder.read_info().map_err(Error::PngDecoding)?;
    let mut pixels = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut pixels).map_err(Error::PngDecoding)?;
    pixels.truncate(info.buffer_size());

    let (bits, colors) = (info.bit_depth, info.color_type);
    let (format, needs_padding) = {
        match (bits, colors) {
            (BitDepth::Eight, ColorType::Rgba) => (vk::Format::R8G8B8A8_UNORM, false),
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

/// Loads a jpeg into (width, height, format, pixel bytes).
#[profiling::function]
pub fn load_jpeg(bytes: &[u8]) -> Result<(u32, u32, vk::Format, Vec<u8>), Error> {
    use jpeg_decoder::{Decoder, PixelFormat};
    let mut decoder = Decoder::new(bytes);
    let mut pixels = decoder.decode().map_err(Error::JpegDecoding)?;
    let info = decoder.info().unwrap(); // Should be Some if decode returned Ok
    let (width, height) = (info.width as u32, info.height as u32);
    let (format, needs_padding) = match info.pixel_format {
        PixelFormat::L8 => (vk::Format::R8_UNORM, false),
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
