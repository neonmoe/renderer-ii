#version 450

layout(location = 0) out vec4 out_color;
layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput in_color;

void main() {
    // Handles post-processing effects that are done before MSAA resolve and down/upsampling passes:
    // - Tonemapping
    out_color = vec4(vec3(0.5, 0.0, 0.5) + subpassLoad(in_color).rgb, 1.0);
}
