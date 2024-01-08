// This file is included in some variants in the "variants" directory.
#extension GL_ARB_shader_draw_parameters : require
#extension GL_EXT_scalar_block_layout : require

#include "constants.glsl"

layout(set = 0, binding = UF_TRANSFORMS_BINDING) uniform GlobalTransforms {
    mat4 proj;
    mat4 view;
}
uf_transforms;

layout(set = 0, binding = UF_DRAW_CALL_VERT_PARAMS_BINDING, std430) uniform DrawCallVertParams {
    uint joints_offset[MAX_DRAW_CALLS];
}
uf_draw_call;

#ifdef SKINNED
layout(set = 2, binding = UF_SKELETON_BINDING) uniform Joint {
    mat4 joints[MAX_JOINT_COUNT];
}
uf_skin;
#endif

layout(location = IN_TRANSFORM_LOCATION) in mat4 in_transform;
layout(location = IN_POSITION_LOCATION) in vec3 in_position;
layout(location = IN_TEXCOORD_0_LOCATION) in vec2 in_uv;
layout(location = IN_NORMAL_LOCATION) in vec3 in_normal;
layout(location = IN_TANGENT_LOCATION) in vec4 in_tangent;
#ifdef SKINNED
layout(location = IN_JOINTS_0_LOCATION) in uvec4 in_joints;
layout(location = IN_WEIGHTS_0_LOCATION) in vec4 in_weights;
#endif

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec4 out_tangent;
layout(location = 3) out vec3 out_debug_color;
layout(location = 4) out uint out_draw_id;

vec3 hsv(float hue, float saturation, float value) {
    float h = mod(hue * 6.0, 6.0);
    float c = value * saturation;
    float x = c * (1 - abs((int(h) % 2) - 1));
    if (0 <= h && h < 1) {
        return vec3(c, x, 0);
    } else if (1 <= h && h < 2) {
        return vec3(x, c, 0);
    } else if (2 <= h && h < 3) {
        return vec3(0, c, x);
    } else if (3 <= h && h < 4) {
        return vec3(0, x, c);
    } else if (4 <= h && h < 5) {
        return vec3(x, 0, c);
    } else if (5 <= h && h < 6) {
        return vec3(c, 0, x);
    } else {
        return vec3(1, 1, 1);
    }
}

// Adapted from: https://thebookofshaders.com/10/
float random(float x) {
    return fract(sin(x) * 43758.5453123);
}

void main() {
    uint draw_id = gl_BaseInstanceARB;
#ifdef SKINNED
    uint joints_offset = uf_draw_call.joints_offset[draw_id];
    mat4 transform = in_transform;
    transform *= uf_skin.joints[in_joints.x + joints_offset] * in_weights.x +
                 uf_skin.joints[in_joints.y + joints_offset] * in_weights.y +
                 uf_skin.joints[in_joints.z + joints_offset] * in_weights.z +
                 uf_skin.joints[in_joints.w + joints_offset] * in_weights.w;
    out_debug_color = hsv(in_joints.x / 256.0 * 37.0, 0.8, in_weights.x);
#else
    mat4 transform = in_transform;
    float d = 1.0 + draw_id;
    out_debug_color = vec3(random(d * 641.63), random(d * 1864.25), random(d * 182362.365));
    // out_debug_color = vec3(mod(d / 8.0, 8.0), mod(d / 64.0, 8.0), mod(d / 512.0, 8.0));
#endif
    gl_Position = uf_transforms.proj * uf_transforms.view * transform * vec4(in_position, 1.0);
    out_uv = in_uv;
    // Normals and tangents should not be translated, and the translation is
    // specified in the 4th column.
    mat3 rotation_scale = mat3(transform);
    out_tangent = vec4(normalize(rotation_scale * in_tangent.xyz), in_tangent.w);
    // For non-uniform scales, this scales the normals appropriately (otherwise
    // only rotation would be enough)
    mat3 normal_transform = transpose(inverse(rotation_scale));
    out_normal = normalize(normal_transform * in_normal);
    // Ensure 90 degree angle between normal and tangent.
    out_tangent.xyz = normalize(out_tangent.xyz - dot(out_tangent.xyz, out_normal) * out_normal);
    out_draw_id = draw_id;
}
