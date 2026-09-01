#ifndef PATHTRACE_COMMON_GLSL
#define PATHTRACE_COMMON_GLSL

#extension GL_EXT_ray_tracing : require

#include "scene_types.glsl"

// 경로 추적 페이로드. 적중 셰이더가 표면 정보를 채우고 광선 생성 셰이더가 경로를 이어 간다.
struct PathPayload {
    vec3 albedo;
    vec3 emissive;
    vec3 normal;
    vec3 position;
    float hitDistance;
    bool missed;
};

#define PATH_FLAG_NEXT_EVENT 1u
#define PATH_FLAG_RUSSIAN_ROULETTE 2u

layout(push_constant) uniform PathTracePushConstants {
    VertexBuffer vertices;
    IndexBuffer indices;
    MeshBuffer meshes;
    InstanceBuffer instances;
    MaterialBuffer materials;
    MeshLodBuffer lods;
    CameraBuffer camera;
    LightBuffer lights;
    uint accumulationImage;
    uint outputImage;
    uint frameIndex;
    uint sampleCount;
    uint maxBounces;
    uint samplesPerFrame;
    uint flags;
    float radianceClamp;
    float skyIntensity;
} pathTrace;

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

// 코사인 가중 반구 표본. 확산 반사의 중요도 표본이다.
vec3 sampleCosineHemisphere(vec3 normal, inout uint seed) {
    float u1 = randomFloat(seed);
    float u2 = randomFloat(seed);
    float radius = sqrt(u1);
    float angle = 6.2831853 * u2;

    vec3 up = abs(normal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);
    return normalize(tangent * (radius * cos(angle)) + bitangent * (radius * sin(angle)) +
                     normal * sqrt(max(1.0 - u1, 0.0)));
}

#endif
