#version 450

layout(location = 0) out vec4 out_color;
layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInputMS in_color;

void main() {
    // Handles post-processing effects that are done before MSAA resolve and down/upsampling passes:
    // - Tonemapping

    vec3 linear = subpassLoad(in_color, gl_SampleID).rgb;
    // These weights seem good, source: https://en.wikipedia.org/wiki/Relative_luminance
    float luminance = 0.2126 * linear.r + 0.7152 * linear.g + 0.0722 * linear.b;
    // Reinhard tonemapping: https://en.wikipedia.org/wiki/Tone_mapping
    float tonemapped_luminance = luminance / (1.0 + luminance);
    out_color = vec4(linear / luminance * tonemapped_luminance, 1.0);
}
