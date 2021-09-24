#version 450

layout(set = 0, binding = 0) uniform GlobalTransforms {
    mat4 proj;
    mat4 view;
} uf_transforms;

layout(location = 0) in mat4 in_transform;
layout(location = 4) in vec3 in_position;
layout(location = 5) in vec2 in_uv;
layout(location = 6) in vec3 in_normal;
layout(location = 7) in vec4 in_tangent;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec4 out_tangent;

void main() {
    gl_Position = uf_transforms.proj * uf_transforms.view * in_transform * vec4(in_position, 1.0);
    out_uv = in_uv;
    mat4 normalTransform = transpose(inverse(in_transform));
    out_normal = normalize((normalTransform * vec4(in_normal, 1.0)).xyz);
    // Not sure if tangent.w would be affected by the normalTransform. Would be neat if it wasn't.
    out_tangent = vec4(normalize((normalTransform * vec4(in_tangent.xyz, 1.0)).xyz), 1.0);
}
