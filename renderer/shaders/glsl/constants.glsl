#define MAX_TEXTURE_COUNT 64
#define MAX_JOINT_COUNT (65536 / (4 * 16))
/// 65536 (max uniform size) / 4 (max bytes of per-draw-call data available per draw call)
#define MAX_DRAW_CALLS (65536 / 4)
/// 65536 (max uniform size) / 4 (max bytes of per-draw-call data available per draw call)
#define MAX_IMGUI_DRAW_CALLS (65536 / (4 * 2 + 4 * 2))

#define IN_TRANSFORMS_LOCATION 0
#define IN_POSITION_LOCATION 7
#define IN_TEXCOORD_0_LOCATION 8
#define IN_NORMAL_LOCATION 9
#define IN_TANGENT_LOCATION 10
#define IN_JOINTS_0_LOCATION 11
#define IN_WEIGHTS_0_LOCATION 12
#define IN_COLOR_LOCATION 13

// Set 0
#define UF_TRANSFORMS_BINDING 0
#define UF_RENDER_SETTINGS_BINDING 1
#define UF_DRAW_CALL_VERT_PARAMS_BINDING 2
#define UF_DRAW_CALL_FRAG_PARAMS_BINDING 3

// Set 1 (pbr pipelines)
#define UF_SAMPLER_BINDING 0
#define UF_TEX_BASE_COLOR_BINDING 1
#define UF_TEX_METALLIC_ROUGHNESS_BINDING 2
#define UF_TEX_NORMAL_BINDING 3
#define UF_TEX_OCCLUSION_BINDING 4
#define UF_TEX_EMISSIVE_BINDING 5
#define UF_PBR_FACTORS_BINDING 6

// Set 1 (post-process pipelines)
#define UF_HDR_FRAMEBUFFER_BINDING 0

// Set 1 (imgui)
#define UF_IMGUI_DRAW_CALL_PARAMS_BINDING 0
#define UF_IMGUI_SAMPLER_BINDING 1
#define UF_IMGUI_TEXTURES_BINDING 2

// Set 2
#define UF_SKELETON_BINDING 0
