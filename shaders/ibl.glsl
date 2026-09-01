#ifndef IBL_GLSL
#define IBL_GLSL

#include "pbr.glsl"

// 큐브맵 면 번호와 면 안 좌표에서 월드 방향을 만든다. 면 순서는 Vulkan 의 층 순서
// (+X, -X, +Y, -Y, +Z, -Z)와 같아야 한다.
vec3 cubeDirection(uint face, vec2 uv) {
    vec2 c = uv * 2.0 - 1.0;
    switch (face) {
    case 0u:
        return normalize(vec3(1.0, -c.y, -c.x));
    case 1u:
        return normalize(vec3(-1.0, -c.y, c.x));
    case 2u:
        return normalize(vec3(c.x, 1.0, c.y));
    case 3u:
        return normalize(vec3(c.x, -1.0, -c.y));
    case 4u:
        return normalize(vec3(c.x, -c.y, 1.0));
    default:
        return normalize(vec3(-c.x, -c.y, -1.0));
    }
}

// 등정방형(equirectangular) 이미지의 좌표. u 는 방위각, v 는 천정각이다.
vec2 equirectUv(vec3 direction) {
    return vec2(atan(direction.z, direction.x) / (2.0 * PI) + 0.5, acos(clamp(direction.y, -1.0, 1.0)) / PI);
}

// Hammersley 저불일치 수열. 중요도 표본의 두 번째 축으로 쓴다.
vec2 hammersley(uint i, uint count) {
    uint bits = i;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return vec2(float(i) / float(count), float(bits) * 2.3283064365386963e-10);
}

// GGX 분포를 따르는 반벡터를 뽑는다. 법선 주위 접선 기저로 옮겨 돌려준다.
vec3 importanceSampleGgx(vec2 random, vec3 normal, float roughness) {
    float a = roughness * roughness;
    float phi = 2.0 * PI * random.x;
    float cosTheta = sqrt((1.0 - random.y) / (1.0 + (a * a - 1.0) * random.y));
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));

    vec3 halfway = vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
    vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);
    return normalize(tangent * halfway.x + bitangent * halfway.y + normal * halfway.z);
}

// IBL 용 Smith 기하항. 직접광과 k 값이 다르다 (Karis).
float geometrySmithIbl(float nDotV, float nDotL, float roughness) {
    float k = (roughness * roughness) / 2.0;
    float ggxV = nDotV / (nDotV * (1.0 - k) + k);
    float ggxL = nDotL / (nDotL * (1.0 - k) + k);
    return ggxV * ggxL;
}

// 거칠기를 반영한 프레넬. 거친 표면에서 가장자리 반사가 과해지는 것을 막는다.
vec3 fresnelSchlickRoughness(float cosTheta, vec3 f0, float roughness) {
    vec3 ceiling = max(vec3(1.0 - roughness), f0);
    return f0 + (ceiling - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

#endif
