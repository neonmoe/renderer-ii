#version 450

layout(set = 0, binding = 0) uniform GlobalTransforms {
    mat4 proj;
    mat4 view;
} uf_transforms;

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_color;
layout(location = 2) in mat4 in_transform;

layout(location = 0) out vec3 out_color;

void main() {
    gl_Position = uf_transforms.proj * uf_transforms.view * in_transform * vec4(in_position, 1.0);
    out_color = in_color;
}
