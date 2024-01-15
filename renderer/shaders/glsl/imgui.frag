#version 450 core
#extension GL_EXT_scalar_block_layout : require

#include "constants.glsl"

layout(location = 0) out vec4 out_color;

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec4 in_color;
layout(location = 2) in flat uint in_draw_id;

layout(set = 0, binding = UF_DRAW_CALL_FRAG_PARAMS_BINDING, std430) uniform DrawCallFragParams {
    uint texture_index[MAX_DRAW_CALLS];
}
uf_draw_call;

layout(set = 1, binding = UF_IMGUI_SAMPLER_BINDING) uniform sampler uf_sampler;
layout(set = 1, binding = UF_IMGUI_TEXTURES_BINDING) uniform texture2D textures[MAX_TEXTURE_COUNT];

void main() {
    uint texture_index = uf_draw_call.texture_index[in_draw_id];
    out_color = in_color * texture(sampler2D(textures[texture_index], uf_sampler), in_uv);
}
