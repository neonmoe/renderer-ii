#version 450 core
#extension GL_EXT_scalar_block_layout : require

#include "constants.glsl"

layout(location = 0) out vec4 out_color;

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec4 in_tangent;
layout(location = 3) in flat vec3 in_debug_color;
layout(location = 4) in flat uint in_draw_id;

// TODO: Should render settings just be defines?
layout(set = 0, binding = UF_RENDER_SETTINGS_BINDING) uniform RenderSettings {
    uint debug_value;
}
uf_render_settings;

layout(set = 0, binding = UF_DRAW_CALL_FRAG_PARAMS_BINDING, std430) uniform DrawCallFragParams {
    uint material_index[MAX_DRAW_CALLS];
}
uf_draw_call;

layout(set = 1, binding = UF_SAMPLER_BINDING) uniform sampler uf_sampler;
layout(set = 1,
       binding = UF_TEX_BASE_COLOR_BINDING) uniform texture2D base_color[MAX_TEXTURE_COUNT];
layout(set = 1, binding = UF_TEX_METALLIC_ROUGHNESS_BINDING) uniform texture2D
    metallic_roughness[MAX_TEXTURE_COUNT];
layout(set = 1, binding = UF_TEX_NORMAL_BINDING) uniform texture2D normal[MAX_TEXTURE_COUNT];
layout(set = 1, binding = UF_TEX_OCCLUSION_BINDING) uniform texture2D occlusion[MAX_TEXTURE_COUNT];
layout(set = 1, binding = UF_TEX_EMISSIVE_BINDING) uniform texture2D emissive[MAX_TEXTURE_COUNT];

layout(set = 1, binding = UF_PBR_FACTORS_BINDING, std430) uniform PbrFactorsSoa {
    vec4 base_color[MAX_TEXTURE_COUNT];
    vec4 emissive_and_occlusion[MAX_TEXTURE_COUNT];
    vec4 alpha_rgh_mtl_normal[MAX_TEXTURE_COUNT];
}
uf_factors;

void main() {
    uint material_index = uf_draw_call.material_index[in_draw_id];

    vec4 base_color = texture(sampler2D(base_color[material_index], uf_sampler), in_uv);
    vec4 metallic_roughness_tex =
        texture(sampler2D(metallic_roughness[material_index], uf_sampler), in_uv);
    vec3 normal_tex = texture(sampler2D(normal[material_index], uf_sampler), in_uv).xyz * 2.0 - 1.0;
    vec4 occlusion_tex = texture(sampler2D(occlusion[material_index], uf_sampler), in_uv);
    vec3 emissive = texture(sampler2D(emissive[material_index], uf_sampler), in_uv).xyz;

    vec4 base_color_factor = uf_factors.base_color[material_index];
    vec3 emissive_factor = uf_factors.emissive_and_occlusion[material_index].rgb;
    float occlusion_strength = uf_factors.emissive_and_occlusion[material_index].a;
    vec4 alpha_rgh_mtl_normal = uf_factors.alpha_rgh_mtl_normal[material_index];
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
    default: {
        if (length(emissive) > 0.0) {
            out_color = vec4(emissive, 1.0);
        } else {
            float ambient = 0.3 * occlusion.r;
            float sun_brightness = 2.0;
            float sun_dot = max(0.0, dot(normal, normalize(vec3(-1.0, 1.0, 1.0))));
            float brightness = ambient + sun_dot * sun_brightness;
            out_color = vec4(brightness * base_color.rgb, base_color.a);
        }
    } break;
    // Debugging cases, selectable with keys 1-6 in the sandbox:
    case 1: {
        out_color = base_color;
    } break;
    case 2: {
        out_color = vec4(in_debug_color, 1.0);
    } break;
    case 3: {
        out_color = vec4(normal, 1.0);
    } break;
    case 4: {
        out_color = vec4(0.0, roughness, metallic, 1.0);
    } break;
    case 5: {
        out_color = vec4(emissive, 1.0);
    } break;
    case 6: {
        out_color = vec4(vec3(occlusion), 1.0);
    } break;
    }
}
