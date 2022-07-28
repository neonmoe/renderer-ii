#version 450

#include "constants.glsl"

layout(location = 0) out vec4 out_color;

layout(location = 0) in vec3 in_position;
layout(location = 1) in centroid vec2 in_uv;
layout(location = 2) in vec3 in_normal;
layout(location = 3) in vec4 in_tangent;

layout(set = 1, binding = 0) uniform sampler tex_sampler;
layout(set = 1, binding = 1) uniform texture2D base_color[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 2) uniform texture2D metallic_roughness[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 3) uniform texture2D normal[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 4) uniform texture2D occlusion[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 5) uniform texture2D emissive[MAX_TEXTURE_COUNT];

layout(push_constant) uniform PushConstantStruct { uint texture_index; }
push_constant;

layout(set = 0, binding = 1) uniform RenderSettings { vec4 lights[LIGHT_COUNT]; }
uf_render_settings;

// Takes 4 samples of the texture from different spots in the pixel. Very
// expensive, yes, but needed to avoid aliasing caused by small textures being
// magnified with the "nearest" algorithm.
vec4 sample_texture(texture2D tex) {
    float sample_sparseness = 0.5;
    vec2 sampling_offsets[] = {
        vec2(0.5, 0.25) * sample_sparseness,
        vec2(0.25, -0.5) * sample_sparseness,
        vec2(-0.5, -0.25) * sample_sparseness,
        vec2(-0.25, 0.5) * sample_sparseness,
    };
    vec2 d_uvx = dFdx(in_uv);
    vec2 d_uvy = dFdy(in_uv);
    vec4 sum = vec4(0.0, 0.0, 0.0, 0.0);
    for (int i = 0; i < 4; i++) {
        vec2 uv = in_uv + sampling_offsets[i].x * d_uvx + sampling_offsets[i].y * d_uvy;
        sum += texture(sampler2D(tex, tex_sampler), uv);
    }
    return sum / 4;
}

void main() {
    vec4 base_color = sample_texture(base_color[push_constant.texture_index]);
    vec2 metallic_roughness = sample_texture(metallic_roughness[push_constant.texture_index]).xy;
    vec4 normal_tex = sample_texture(normal[push_constant.texture_index]);
    vec4 occlusion = sample_texture(occlusion[push_constant.texture_index]);
    vec3 emissive = sample_texture(emissive[push_constant.texture_index]).xyz;

    vec3 bitangent = in_tangent.w * cross(in_normal, in_tangent.xyz);
    mat3 tangent_to_world = mat3(in_tangent.xyz, bitangent, in_normal);
    vec3 normal = tangent_to_world * normal_tex.xyz;

    float ambient = 0.3 * occlusion.r;
    float brightness = ambient;
    for (int i = 0; i < LIGHT_COUNT; i++) {
        vec3 light_position = uf_render_settings.lights[i].xyz;
        float light_brightness = uf_render_settings.lights[i].w;
        vec3 to_light = light_position - in_position;
        vec3 light_dir = normalize(to_light);
        float distance_to_light_squared = dot(to_light, to_light);
        float lambert = max(0, dot(light_dir, normal));
        brightness += lambert * light_brightness / distance_to_light_squared;
    }
    out_color = vec4(brightness * base_color.rgb + emissive, base_color.a);
}
