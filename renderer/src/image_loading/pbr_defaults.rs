use arrayvec::ArrayVec;
use ash::vk;

use crate::arena::buffers::ForBuffers;
use crate::arena::images::ForImages;
use crate::arena::{VulkanArena, VulkanArenaError};
use crate::image_loading::{self, ImageData, TextureKind};
use crate::uploader::Uploader;
use crate::vulkan_raii::{Device, ImageView};

const WHITE: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF];
const BLACK: [u8; 4] = [0, 0, 0, 0xFF];
const NORMAL_Z: [u8; 4] = [0x7F, 0x7F, 0xFF, 0];
const M_AND_R: [u8; 4] = [0, 0xFF, 0xFF, 0];

pub fn all_defaults(
    device: &Device,
    staging_arena: &mut VulkanArena<ForBuffers>,
    uploader: &mut Uploader,
    arena: &mut VulkanArena<ForImages>,
) -> Result<crate::PbrDefaults, VulkanArenaError> {
    Ok(crate::PbrDefaults {
        base_color: base_color(device, staging_arena, uploader, arena)?,
        metallic_roughness: metallic_roughness(device, staging_arena, uploader, arena)?,
        normal: normal(device, staging_arena, uploader, arena)?,
        occlusion: occlusion(device, staging_arena, uploader, arena)?,
        emissive: emissive(device, staging_arena, uploader, arena)?,
    })
}

pub fn base_color(
    device: &Device,
    staging_arena: &mut VulkanArena<ForBuffers>,
    uploader: &mut Uploader,
    arena: &mut VulkanArena<ForImages>,
) -> Result<ImageView, VulkanArenaError> {
    image_loading::create_pixel(device, staging_arena, uploader, arena, WHITE, TextureKind::SrgbColor, "default pbr base color")
}

pub fn metallic_roughness(
    device: &Device,
    staging_arena: &mut VulkanArena<ForBuffers>,
    uploader: &mut Uploader,
    arena: &mut VulkanArena<ForImages>,
) -> Result<ImageView, VulkanArenaError> {
    image_loading::create_pixel(device, staging_arena, uploader, arena, M_AND_R, TextureKind::LinearColor, "default pbr metallic/roughness")
}

pub fn normal(
    device: &Device,
    staging_arena: &mut VulkanArena<ForBuffers>,
    uploader: &mut Uploader,
    arena: &mut VulkanArena<ForImages>,
) -> Result<ImageView, VulkanArenaError> {
    image_loading::create_pixel(device, staging_arena, uploader, arena, NORMAL_Z, TextureKind::NormalMap, "default pbr normals")
}

pub fn occlusion(
    device: &Device,
    staging_arena: &mut VulkanArena<ForBuffers>,
    uploader: &mut Uploader,
    arena: &mut VulkanArena<ForImages>,
) -> Result<ImageView, VulkanArenaError> {
    image_loading::create_pixel(device, staging_arena, uploader, arena, WHITE, TextureKind::LinearColor, "default pbr occlusion")
}

pub fn emissive(
    device: &Device,
    staging_arena: &mut VulkanArena<ForBuffers>,
    uploader: &mut Uploader,
    arena: &mut VulkanArena<ForImages>,
) -> Result<ImageView, VulkanArenaError> {
    image_loading::create_pixel(device, staging_arena, uploader, arena, BLACK, TextureKind::SrgbColor, "default pbr emissive")
}

/// Returns the image creation structs which can be used to
/// [measure](crate::VulkanArenaMeasurer::add_image) the amount of memory this
/// module allocates.
///
/// The image creation infos describe the images created by [`base_color`],
/// [`metallic_roughness`], [`normal`], [`occlusion`], and [`emissive`], in that
/// order.
pub fn all_defaults_create_infos() -> [vk::ImageCreateInfo<'static>; 5] {
    let image_kinds: [TextureKind; 5] = [
        TextureKind::SrgbColor,   // Base color
        TextureKind::LinearColor, // Metallic/roughness
        TextureKind::NormalMap,   // Normals
        TextureKind::LinearColor, // Ambient occlusion
        TextureKind::SrgbColor,   // Emissive
    ];
    let mut infos = [vk::ImageCreateInfo::default(); 5];
    assert_eq!(infos.len(), image_kinds.len());
    for (info, kind) in infos.iter_mut().zip(&image_kinds) {
        let mip_range = 0..4;
        let placeholder = ImageData {
            width: 1,
            height: 1,
            format: vk::Format::R8G8B8A8_UNORM,
            pixels: &[0, 0, 0, 0],
            mip_ranges: ArrayVec::from_iter([mip_range]),
        };
        *info = placeholder.get_create_info(*kind);
    }
    infos
}
