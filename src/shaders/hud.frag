#version 450

#include "constants.glsl"

layout(location = 0) out vec4 out_color;

layout(location = 0) in centroid vec2 in_uv;

layout(set = 1, binding = 0) uniform sampler tex_sampler;
layout(set = 1, binding = 1) uniform texture2D base_color[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 6) uniform GltfFactors {
    vec4 base_color;
    vec4 emissive;
    vec4 metallic_roughness_alpha_cutoff;
}
factors[MAX_TEXTURE_COUNT];

layout(push_constant) uniform PushConstantStruct { uint texture_index; }
push_constant;

layout(set = 0, binding = 1) uniform RenderSettings { uint debug_value; }
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
    base_color *= factors[push_constant.texture_index].base_color;
    out_color = base_color;
}
