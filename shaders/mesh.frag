#version 460
#include "scene_data.glsl"

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec2 inUv;
layout(location = 2) flat in uint inMaterialIndex;

layout(location = 0) out vec4 outColor;

const vec3 LIGHT_DIRECTION = vec3(0.4082, 0.8165, 0.4082);
const vec3 AMBIENT = vec3(0.12);

void main() {
    Material material = pushConstants.materials.items[inMaterialIndex];
    vec3 normal = normalize(inNormal);
    float diffuse = max(dot(normal, LIGHT_DIRECTION), 0.0);
    // 방사 항은 방사 텍스처가 들어오는 단계에서 더한다. 계수만 쓰면 전면 발광이 된다.
    vec3 color = material.baseColorFactor.rgb * (AMBIENT + vec3(diffuse));
    outColor = vec4(color, material.baseColorFactor.a);
}
