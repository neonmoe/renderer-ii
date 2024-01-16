#version 450 core
#extension GL_EXT_scalar_block_layout : require

#include "constants.glsl"

layout(location = 0) out vec4 out_color;

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec4 in_color;
layout(location = 2) in flat uint in_draw_id;
layout(location = 3) in vec2 out_screen_space_coords;

layout(set = 0, binding = UF_DRAW_CALL_FRAG_PARAMS_BINDING, std430) uniform DrawCallFragParams {
    uint material_index[MAX_DRAW_CALLS];
}
uf_draw_call;
layout(set = 0, binding = UF_SAMPLER_BINDING) uniform sampler uf_sampler;
layout(set = 0, binding = UF_TEXTURES_BINDING) uniform texture2D textures[MAX_TEXTURE_COUNT];

layout(set = 1, binding = UF_IMGUI_DRAW_CMD_PARAMS_BINDING, std430) uniform ImGuiDrawCmdParams {
    vec4 clip_rect[MAX_IMGUI_DRAW_CALLS];
    uint texture_index[MAX_IMGUI_DRAW_CALLS];
}
uf_imgui_params;

void main() {
    uint material_index = uf_draw_call.material_index[in_draw_id];
    vec4 clip_rect = uf_imgui_params.clip_rect[material_index];
    if (!(clip_rect.x < out_screen_space_coords.x && clip_rect.z >= out_screen_space_coords.x &&
        clip_rect.y < out_screen_space_coords.y && clip_rect.w >= out_screen_space_coords.y)) {
        discard;
    }
    uint texture_index = uf_imgui_params.texture_index[material_index];
    // TODO: Use uf_imgui_params.clip_rect
    out_color = in_color * texture(sampler2D(textures[texture_index], uf_sampler), in_uv);
}
