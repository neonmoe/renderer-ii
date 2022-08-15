#version 450

#include "constants.glsl"

layout(location = 0) out vec4 out_color;

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_tangent;
layout(location = 3) in flat vec3 in_debug_color;

layout(set = 1, binding = 0) uniform sampler tex_sampler;
layout(set = 1, binding = 1) uniform texture2D base_color[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 2) uniform texture2D metallic_roughness[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 3) uniform texture2D normal[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 4) uniform texture2D occlusion[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 5) uniform texture2D emissive[MAX_TEXTURE_COUNT];
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

void main() {
    vec4 base_color =
        texture(sampler2D(base_color[push_constant.texture_index], tex_sampler), in_uv);
    vec2 metallic_roughness =
        texture(sampler2D(metallic_roughness[push_constant.texture_index], tex_sampler), in_uv).xy;
    vec4 normal_tex = texture(sampler2D(normal[push_constant.texture_index], tex_sampler), in_uv);
    vec4 occlusion = texture(sampler2D(occlusion[push_constant.texture_index], tex_sampler), in_uv);
    vec3 emissive =
        texture(sampler2D(emissive[push_constant.texture_index], tex_sampler), in_uv).xyz;

    vec4 base_color_factor = factors[push_constant.texture_index].base_color;
    vec3 emissive_factor = factors[push_constant.texture_index].emissive.xyz;
    vec2 metallic_roughness_factor =
        factors[push_constant.texture_index].metallic_roughness_alpha_cutoff.xy;
    float alpha_cutoff = factors[push_constant.texture_index].metallic_roughness_alpha_cutoff.z;

    base_color *= base_color_factor;
    if (base_color.a <= alpha_cutoff) {
        discard;
    }

    vec3 bitangent = in_tangent.w * cross(in_normal, in_tangent.xyz);
    mat3 tangent_to_world = mat3(in_tangent.xyz, bitangent, in_normal);
    vec3 normal = tangent_to_world * normal_tex.xyz;

    emissive *= emissive_factor;
    metallic_roughness *= metallic_roughness_factor;

    switch (uf_render_settings.debug_value) {
    // The actual rendering case, enabled by default and by pressing 0 in
    // the sandbox:
    default:
        if (length(emissive) > 0.0) {
            out_color = vec4(emissive, 1.0);
        } else {
            float ambient = 0.3 * occlusion.r;
            float sun_brightness = 2.0;
            float sun_dot = max(0.0, dot(normal, normalize(vec3(-1.0, 1.0, 1.0))));
            float brightness = ambient + sun_dot * sun_brightness;
            out_color = vec4(brightness * base_color.rgb, base_color.a);
        }
        break;
    // Debugging cases, selectable with keys 1-5 in the sandbox:
    case 1:
        out_color = base_color;
        break;
    case 2:
        out_color = vec4(in_debug_color, 1.0);
        break;
    case 3:
        out_color = vec4(normal, 1.0);
        break;
    case 4:
        out_color = occlusion;
        break;
    case 5:
        out_color = vec4(emissive, 1.0);
        break;
    }
}
