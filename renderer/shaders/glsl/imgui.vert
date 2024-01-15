#version 450 core
#extension GL_ARB_shader_draw_parameters : require
#extension GL_EXT_scalar_block_layout : require

#include "constants.glsl"

layout(location = IN_POSITION_LOCATION) in vec2 in_position;
layout(location = IN_TEXCOORD_0_LOCATION) in vec2 in_uv;
layout(location = IN_COLOR_LOCATION) in vec4 in_color;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec4 out_color;
layout(location = 2) out uint out_draw_id;

layout(set = 1, binding = UF_IMGUI_DRAW_CALL_PARAMS_BINDING, std430) uniform ImGuiDrawCallParams {
    vec2 scale[MAX_IMGUI_DRAW_CALLS];
    vec2 translate[MAX_IMGUI_DRAW_CALLS];
}
uf_draw_call;

void main() {
    uint draw_id = gl_BaseInstanceARB;
    // TODO: Projection matrix for imgui draws?
    gl_Position =
        vec4(in_position * uf_draw_call.scale[draw_id] + uf_draw_call.translate[draw_id], 0, 1);
    out_color = in_color;
    out_uv = in_uv;
    out_draw_id = draw_id;
}
