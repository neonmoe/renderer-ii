#version 450

#include "constants.glsl"

layout(location = 0) out vec4 out_color;

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_tangent;

layout(set = 1, binding = 0) uniform sampler tex_sampler;
layout(set = 1, binding = 1) uniform texture2D base_color[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 2) uniform texture2D metallic_roughness[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 3) uniform texture2D normal[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 4) uniform texture2D occlusion[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 5) uniform texture2D emissive[MAX_TEXTURE_COUNT];

layout(push_constant) uniform PushConstantStruct {
    int texture_index;
    int debug_texture;
}
push_constant;

void main() {
    vec4 base_color =
        texture(sampler2D(base_color[push_constant.texture_index], tex_sampler), in_uv);
    vec4 metallic_roughness =
        texture(sampler2D(metallic_roughness[push_constant.texture_index], tex_sampler), in_uv);
    vec4 normal_tex = texture(sampler2D(normal[push_constant.texture_index], tex_sampler), in_uv);
    vec4 occlusion = texture(sampler2D(occlusion[push_constant.texture_index], tex_sampler), in_uv);
    vec4 emissive = texture(sampler2D(emissive[push_constant.texture_index], tex_sampler), in_uv);

    vec3 bitangent = in_tangent.w * cross(in_normal, in_tangent.xyz);
    mat3 tangent_to_world = mat3(in_tangent.xyz, bitangent, in_normal);
    vec3 normal = tangent_to_world * normal_tex.xyz;

    switch (push_constant.debug_texture) {
    // The actual rendering case, enabled by default and by pressing 0 in
    // the sandbox:
    default:
        float brightness =
            max(0.0, dot(normal, normalize(vec3(-1.0, 1.0, 1.0)))) * 0.6 + 0.3 * occlusion.r;
        out_color = brightness * base_color;
        break;
    // Debugging cases, selectable with keys 1-5 in the sandbox:
    case 1:
        out_color = base_color;
        break;
    case 2:
        out_color = metallic_roughness;
        break;
    case 3:
        out_color = vec4(normal, 1.0);
        break;
    case 4:
        out_color = occlusion;
        break;
    case 5:
        out_color = emissive;
        break;
    }
}
