#version 450

#include "constants.glsl"

layout(location = 0) out vec4 out_color;
layout(location = 0) in vec2 in_uv;
layout(set = 1, binding = 0) uniform sampler tex_sampler;
layout(set = 1, binding = 1) uniform texture2D base_color[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 2) uniform texture2D metallic_roughness[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 3) uniform texture2D normal[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 4) uniform texture2D occlusion[MAX_TEXTURE_COUNT];
layout(set = 1, binding = 5) uniform texture2D emissive[MAX_TEXTURE_COUNT];

layout(push_constant) uniform PushConstantStruct {
    int texture_index;
} push_constant;

void main() {
    out_color = texture(sampler2D(base_color[push_constant.texture_index], tex_sampler), in_uv);
}
