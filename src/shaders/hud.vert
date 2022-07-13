#version 450

layout(set = 0, binding = 2) uniform HudTransform { mat4 proj; }
uf_transform;

layout(location = 0) in mat4 in_transform;
layout(location = 4) in vec3 in_position;
layout(location = 5) in vec2 in_uv;

layout(location = 0) out vec2 out_uv;

void main() {
    gl_Position = uf_transform.proj * in_transform * vec4(in_position, 1.0);
    out_uv = in_uv;
}
