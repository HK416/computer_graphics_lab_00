#ifndef SAMPLING_GLSL
#define SAMPLING_GLSL

#include "ibl.glsl"

// 난수와 중요도 표본. 경로 추적 광선 생성 셰이더와 반사 컴퓨트가 함께 쓴다. 푸시 상수에 기대지
// 않으므로 어느 스테이지에서든 include 할 수 있다.

// 접선 공간의 방향을 노멀 둘레의 월드 공간으로 옮긴다.
vec3 tangentToWorld(vec3 local, vec3 normal) {
    vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);
    return tangent * local.x + bitangent * local.y + normal * local.z;
}

// PCG 해시 기반 난수. 픽셀과 프레임마다 서로 다른 수열을 쓴다.
uint pcgHash(uint value) {
    uint state = value * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float randomFloat(inout uint seed) {
    seed = pcgHash(seed);
    return float(seed) * (1.0 / 4294967296.0);
}

// GGX 법선 분포로 반값 벡터를 뽑는다. 거친 표면일수록 넓게 퍼진다. IBL 프리필터와 같은 식이다.
vec3 sampleGgxHalfVector(vec3 normal, float roughness, inout uint seed) {
    float u1 = randomFloat(seed);
    float u2 = randomFloat(seed);
    return importanceSampleGgx(vec2(u1, u2), normal, roughness);
}

// 코사인 가중 반구 표본. 확산 반사의 중요도 표본이다.
vec3 sampleCosineHemisphere(vec3 normal, inout uint seed) {
    float u1 = randomFloat(seed);
    float u2 = randomFloat(seed);
    float radius = sqrt(u1);
    float angle = 2.0 * PI * u2;

    return normalize(tangentToWorld(vec3(radius * cos(angle), radius * sin(angle), sqrt(max(1.0 - u1, 0.0))), normal));
}

#endif
