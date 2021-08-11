#version 450

layout(set = 0, binding = 0) uniform GlobalTransforms {
    mat4 proj;
    mat4 view;
} uf_transforms;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_uv;

layout(location = 0) out vec2 out_uv;

void main() {
    gl_Position = uf_transforms.proj * uf_transforms.view * vec4(in_position, 1.0);
    out_uv = in_uv;
}
