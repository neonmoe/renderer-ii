#version 450 core
#extension GL_EXT_scalar_block_layout : require

#include "constants.glsl"

layout(location = 0) out vec4 out_color;

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_tangent;
layout(location = 3) in flat vec3 in_debug_color;
layout(location = 4) in flat int in_draw_id;

layout(set = 1, binding = 0) uniform sampler tex_sampler;
layout(set = 1, binding = 1) uniform texture2D base_color[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 2) uniform texture2D metallic_roughness[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 3) uniform texture2D normal[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 4) uniform texture2D occlusion[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 5) uniform texture2D emissive[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 6, std430) uniform PbrFactorsSoa {
    vec4 base_color[MAX_TEXTURE_COUNT];
    vec4 emissive_and_occlusion[MAX_TEXTURE_COUNT];
    vec4 alpha_rgh_mtl_normal[MAX_TEXTURE_COUNT];
}
factors;
layout(set = 1, binding = 7, std430) uniform DrawCallParametersSoa {
    uint material_index[MAX_DRAW_CALLS];
}
draw_call_parameters;

layout(set = 0, binding = 1) uniform RenderSettings { uint debug_value; }
uf_render_settings;

void main() {
    uint texture_index = draw_call_parameters.material_index[in_draw_id];

    vec4 base_color = texture(sampler2D(base_color[texture_index], tex_sampler), in_uv);
    vec4 metallic_roughness_tex =
        texture(sampler2D(metallic_roughness[texture_index], tex_sampler), in_uv);
    vec3 normal_tex = texture(sampler2D(normal[texture_index], tex_sampler), in_uv).xyz * 2.0 - 1.0;
    vec4 occlusion_tex = texture(sampler2D(occlusion[texture_index], tex_sampler), in_uv);
    vec3 emissive = texture(sampler2D(emissive[texture_index], tex_sampler), in_uv).xyz;

    vec4 base_color_factor = factors.base_color[texture_index];
    vec3 emissive_factor = factors.emissive_and_occlusion[texture_index].rgb;
    float occlusion_strength = factors.emissive_and_occlusion[texture_index].a;
    vec4 alpha_rgh_mtl_normal = factors.alpha_rgh_mtl_normal[texture_index];
    float alpha_cutoff = alpha_rgh_mtl_normal.r;
    float roughness_factor = alpha_rgh_mtl_normal.g;
    float metallic_factor = alpha_rgh_mtl_normal.b;
    float normal_scale = alpha_rgh_mtl_normal.a;

    base_color *= base_color_factor;
    if (base_color.a <= alpha_cutoff) {
        discard;
    }

    vec3 bitangent = in_tangent.w * cross(in_normal, in_tangent.xyz);
    mat3 tangent_to_world = mat3(in_tangent.xyz, bitangent, in_normal);
    normal_tex.xy *= normal_scale;
    vec3 normal = tangent_to_world * normalize(normal_tex);

    emissive *= emissive_factor;
    float roughness = metallic_roughness_tex.g * roughness_factor;
    float metallic = metallic_roughness_tex.b * metallic_factor;
    float occlusion = 1.0 + occlusion_strength * (occlusion_tex.r - 1.0);

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
    // Debugging cases, selectable with keys 1-6 in the sandbox:
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
        out_color = vec4(0.0, roughness, metallic, 1.0);
        break;
    case 5:
        out_color = vec4(emissive, 1.0);
        break;
    case 6:
        out_color = vec4(vec3(occlusion), 1.0);
        break;
    }
}
