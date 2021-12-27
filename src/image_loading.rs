use crate::arena::ImageAllocation;
use crate::{Error, FrameIndex, Gpu};
use ash::vk;

mod ktx;

#[derive(Clone, Copy)]
pub enum TextureKind {
    SrgbColor,
    LinearColor,
    NormalMap,
}

impl TextureKind {
    fn convert_format(self, format: vk::Format) -> vk::Format {
        match self {
            TextureKind::SrgbColor => to_srgb(format),
            TextureKind::NormalMap => to_snorm(format),
            TextureKind::LinearColor => format,
        }
    }
}

/// If the `format` is a 'UNORM' format, and has an SRGB variant,
/// return that.
///
/// Does not work for the 3D ASTC ones.
fn to_srgb(format: vk::Format) -> vk::Format {
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
fn to_snorm(format: vk::Format) -> vk::Format {
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

/// Loads a ktx into (width, height, format, pixel bytes). Note that
/// the pixel bytes do not map to a linear rgba array: the data is
/// compressed. The format reflects this.
#[profiling::function]
pub fn load_ktx(gpu: &Gpu, frame_index: FrameIndex, bytes: &[u8], kind: TextureKind) -> Result<(ImageAllocation, vk::ImageView), Error> {
    let ktx::KtxData {
        width,
        height,
        format,
        pixels,
        mip_ranges,
    } = ktx::decode(bytes)?;
    let format = kind.convert_format(format);

    let image = todo!();

    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(mip_ranges.len() as u32)
        .base_array_layer(0)
        .layer_count(1)
        .build();
    let image_view_create_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(subresource_range);

    todo!()
}
