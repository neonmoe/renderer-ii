use crate::Error;
use ash::vk;
use std::convert::TryInto;
use std::ops::Range;

const KTX_IDENTIFIER: [u8; 12] = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x31, 0x31, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A];

pub struct KtxData {
    pub width: u32,
    pub height: u32,
    pub format: vk::Format,
    /// Note: not necessarily actual pixels, just the pixel data.
    pub pixels: Vec<u8>,
    /// The ranges from `pixels` that represent different mip levels.
    pub mip_ranges: Vec<Range<usize>>,
}

#[profiling::function]
pub fn decode(bytes: &[u8]) -> Result<KtxData, Error> {
    if KTX_IDENTIFIER != bytes[0..12] || bytes.len() < 68 {
        return Err(Error::BadKtx);
    }

    // let endianness = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    // let gl_type = u32::from_le_bytes(bytes[16..20].try_into().unwrap());
    // let gl_type_size = u32::from_le_bytes(bytes[20..24].try_into().unwrap());
    // let gl_format = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
    let gl_internal_format = u32::from_le_bytes(bytes[28..32].try_into().unwrap());
    // let gl_base_internal_format = u32::from_le_bytes(bytes[32..36].try_into().unwrap());
    let width = u32::from_le_bytes(bytes[36..40].try_into().unwrap());
    let height = u32::from_le_bytes(bytes[40..44].try_into().unwrap());
    let depth = u32::from_le_bytes(bytes[44..48].try_into().unwrap());
    let number_of_array_elements = u32::from_le_bytes(bytes[48..52].try_into().unwrap());
    let number_of_faces = u32::from_le_bytes(bytes[52..56].try_into().unwrap());
    let number_of_mipmap_levels = u32::from_le_bytes(bytes[56..60].try_into().unwrap()) as usize;
    let bytes_of_key_value_data = u32::from_le_bytes(bytes[60..64].try_into().unwrap()) as usize;

    if depth != 0 {
        return Err(Error::UnsupportedKtxFeature("3D textures are not supported"));
    } else if number_of_array_elements != 0 {
        return Err(Error::UnsupportedKtxFeature("array textures are not supported"));
    } else if number_of_faces != 1 {
        return Err(Error::UnsupportedKtxFeature("cube textures are not supported"));
    }

    // We have to allocate yet another buffer (as opposed to just reusing
    // `bytes`) because the KTX files' mip levels are aligned to 4 bytes, but
    // Vulkan requires the source of the vkCmdCopyBufferToImage to be aligned to
    // the texel size, which can be e.g. 16 for BC7. Tightly packed texels are
    // aligned to the texel size, so no manual alignment math needs to be done
    // for this secondary buffer.
    let mut pixels = Vec::with_capacity(bytes.len() + number_of_mipmap_levels * 12);
    let mut mip_ranges = Vec::with_capacity(number_of_mipmap_levels);
    let all_images_start = 64 + bytes_of_key_value_data;
    let mut image_start = all_images_start;
    while image_start + 4 < bytes.len() {
        profiling::scope!("processing mip level");
        let image_size = u32::from_le_bytes(bytes[image_start..image_start + 4].try_into().unwrap()) as usize;

        // TODO(low): Add configuration for not loading some mip levels
        // This would be an easy "texture quality" switch to add. It would be
        // enough to just not push the first N mip ranges and pixels, and set
        // width/height to the first mip's width/height.
        let mip_pixels = &bytes[image_start + 4..image_start + 4 + image_size];
        let mip_range_start = pixels.len();
        {
            profiling::scope!("copying pixels");
            pixels.extend_from_slice(mip_pixels);
        }
        let mip_range_end = pixels.len();
        mip_ranges.push(mip_range_start..mip_range_end);

        let offset = if image_size % 4 > 0 { 4 - (image_size % 4) } else { 0 };
        image_start += 4 + image_size + offset;
    }
    debug_assert_eq!(bytes.len(), image_start);

    let format = opengl_internal_format_to_vk_format(gl_internal_format)?;

    Ok(KtxData {
        width,
        height,
        format,
        pixels,
        mip_ranges,
    })
}

// From the `gl` crate's bindings.rs.
const COMPRESSED_RGB_BPTC_SIGNED_FLOAT: u32 = 0x8E8E;
const COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT: u32 = 0x8E8F;
const COMPRESSED_RGBA_BPTC_UNORM: u32 = 0x8E8C;
const COMPRESSED_SRGB_ALPHA_BPTC_UNORM: u32 = 0x8E8D;

fn opengl_internal_format_to_vk_format(internal_format: u32) -> Result<vk::Format, Error> {
    match internal_format {
        COMPRESSED_RGB_BPTC_SIGNED_FLOAT => Ok(vk::Format::BC6H_SFLOAT_BLOCK),
        COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT => Ok(vk::Format::BC6H_UFLOAT_BLOCK),
        COMPRESSED_RGBA_BPTC_UNORM => Ok(vk::Format::BC7_UNORM_BLOCK),
        COMPRESSED_SRGB_ALPHA_BPTC_UNORM => Ok(vk::Format::BC7_SRGB_BLOCK),
        _ => Err(Error::UnsupportedKtxFeature("only BC6H and BC7 textures are supported")),
    }
}
