#version 450 core
#extension GL_ARB_shader_draw_parameters : require
#extension GL_EXT_scalar_block_layout : require

#include "constants.glsl"

layout(location = IN_TRANSFORMS_LOCATION) in mat3 in_transform_rotationscale;
layout(location = IN_TRANSFORMS_LOCATION + 3) in vec3 in_transform_translation;
layout(location = IN_POSITION_LOCATION) in vec2 in_position;
layout(location = IN_TEXCOORD_0_LOCATION) in vec2 in_uv;
layout(location = IN_COLOR_LOCATION) in vec4 in_color;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec4 out_color;
layout(location = 2) out uint out_draw_id;

void main() {
    uint draw_id = gl_BaseInstanceARB;
    mat4 transform = mat4(in_transform_rotationscale);
    transform[3].xyz = in_transform_translation;
    gl_Position = transform * vec4(in_position, 0, 1);
    out_color = in_color;
    out_uv = in_uv;
    out_draw_id = draw_id;
}
