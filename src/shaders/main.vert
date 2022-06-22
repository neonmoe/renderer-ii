#version 450

layout(set = 0, binding = 0) uniform GlobalTransforms {
    mat4 proj;
    mat4 view;
}
uf_transforms;

layout(location = 0) in mat4 in_transform;
layout(location = 4) in vec3 in_position;
layout(location = 5) in vec2 in_uv;
layout(location = 6) in vec3 in_normal;
layout(location = 7) in vec4 in_tangent;
#ifdef SKINNED
layout(location = 8) in uvec4 in_joints;
layout(location = 9) in vec4 in_weights;

layout(set = 2, binding = 0) uniform Bone { mat4 bones[256]; }
skeleton;
#endif

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec3 out_normal;
layout(location = 2) out vec4 out_tangent;

vec3 hsv(float hue, float saturation, float value) {
    float h = mod(hue * 6.0, 6.0);
    float c = value * saturation;
    float x = c * (1 - abs((int(h) % 2) - 1));
    if (0 <= h && h < 1) {
        return vec3(c, x, 0);
    } else if (1 <= h && h < 2) {
        return vec3(x, c, 0);
    } else if (2 <= h && h < 3) {
        return vec3(0, c, x);
    } else if (3 <= h && h < 4) {
        return vec3(0, x, c);
    } else if (4 <= h && h < 5) {
        return vec3(x, 0, c);
    } else if (5 <= h && h < 6) {
        return vec3(c, 0, x);
    } else {
        return vec3(1, 1, 1);
    }
}

void main() {
#ifdef SKINNED
    mat4 transform = in_transform;
    transform *=
        skeleton.bones[in_joints.x] * in_weights.x + skeleton.bones[in_joints.y] * in_weights.y +
        skeleton.bones[in_joints.z] * in_weights.z + skeleton.bones[in_joints.w] * in_weights.w;
#else
    mat4 transform = in_transform;
#endif
    gl_Position = uf_transforms.proj * uf_transforms.view * transform * vec4(in_position, 1.0);
    out_uv = in_uv;
    // Normals and tangents should not be translated, and the translation is
    // specified in the 4th column.
    mat3 rotation_scale = mat3(transform);
    out_tangent = vec4(normalize(rotation_scale * in_tangent.xyz), in_tangent.w);
    // For non-uniform scales, this scales the normals appropriately (otherwise
    // only rotation would be enough)
    mat3 normal_transform = transpose(inverse(rotation_scale));
    out_normal = normalize(normal_transform * in_normal);
    // Ensure 90 degree angle between normal and tangent.
    out_tangent.xyz = normalize(out_tangent.xyz - dot(out_tangent.xyz, out_normal) * out_normal);
}
