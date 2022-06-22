#version 450

layout(location = 0) out vec4 out_color;
layout(input_attachment_index = 0, set = 1, binding = 0) uniform subpassInputMS in_color;

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

void main() {
    // Handles post-processing effects that are done before MSAA resolve and down/upsampling passes:
    // - Tonemapping

    vec3 linear = subpassLoad(in_color, gl_SampleID).rgb;
    out_color = vec4(aces(linear), 1.0);
}
