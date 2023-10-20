// This file is included in some variants in the "variants" directory.

#extension GL_EXT_samplerless_texture_functions : require

#include "constants.glsl"

layout(location = 0) out vec4 out_color;
layout(origin_upper_left) in vec4 gl_FragCoord;

#ifdef MULTISAMPLED
layout(set = 1, binding = UF_HDR_FRAMEBUFFER_BINDING)
uniform texture2DMS uf_hdr_framebuffer;
#else
layout(set = 1, binding = UF_HDR_FRAMEBUFFER_BINDING)
uniform texture2D uf_hdr_framebuffer;
#endif

layout(set = 0, binding = 1) uniform RenderSettings { uint debug_value; }
uf_render_settings;

vec3 aces(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1);
}

vec3 reinhard(vec3 x) {
    // These weights seem good, source: https://en.wikipedia.org/wiki/Relative_luminance
    float luminance = 0.2126 * x.r + 0.7152 * x.g + 0.0722 * x.b;
    // Reinhard tonemapping: https://en.wikipedia.org/wiki/Tone_mapping
    float tonemapped_luminance = luminance / (1.0 + luminance);
    return x / luminance * tonemapped_luminance;
}

void main() {
    // Handles post-processing effects that are done before MSAA resolve and down/upsampling passes:
    // - Tonemapping

    ivec2 texcoord = ivec2(gl_FragCoord.xy);
    #ifdef MULTISAMPLED
    vec4 linear = texelFetch(uf_hdr_framebuffer, texcoord, gl_SampleID);
    #else
    vec4 linear = texelFetch(uf_hdr_framebuffer, texcoord, 0);
    #endif
    float exposure = 0.8;
    out_color = vec4(aces(linear.rgb * exposure), 1.0);
}
