// Keep these in sync with shaders/glsl/constants.glsl.
// TODO: Generate this file with a proc macro, to avoid de-sync?

pub const MAX_TEXTURE_COUNT: u32 = 25;
pub const MAX_BONE_COUNT: u32 = 128;
pub const MAX_DRAW_CALLS: u32 = 1024;

pub const IN_TRANSFORM_LOCATION: u32 = 0;
pub const IN_POSITION_LOCATION: u32 = 4;
pub const IN_TEXCOORD_0_LOCATION: u32 = 5;
pub const IN_NORMAL_LOCATION: u32 = 6;
pub const IN_TANGENT_LOCATION: u32 = 7;
pub const IN_JOINTS_0_LOCATION: u32 = 8;
pub const IN_WEIGHTS_0_LOCATION: u32 = 9;

// Set 0
pub const UF_TRANSFORMS_BINDING: u32 = 0;
pub const UF_RENDER_SETTINGS_BINDING: u32 = 1;

// Set 1 (pbr pipelines)
pub const UF_SAMPLER_BINDING: u32 = 0;
pub const UF_TEX_BASE_COLOR_BINDING: u32 = 1;
pub const UF_TEX_METALLIC_ROUGHNESS_BINDING: u32 = 2;
pub const UF_TEX_NORMAL_BINDING: u32 = 3;
pub const UF_TEX_OCCLUSION_BINDING: u32 = 4;
pub const UF_TEX_EMISSIVE_BINDING: u32 = 5;
pub const UF_PBR_FACTORS_BINDING: u32 = 6;
pub const UF_DRAW_CALL_PARAMS_BINDING: u32 = 7;

// Set 1 (post-process pipelines)
pub const UF_HDR_FRAMEBUFFER_BINDING: u32 = 0;

// Set 2
pub const UF_SKELETON_BINDING: u32 = 0;
