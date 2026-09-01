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
    float metallic;
    float roughness;
    float hitDistance;
    bool missed;
};

#define PATH_FLAG_NEXT_EVENT 1u
#define PATH_FLAG_RUSSIAN_ROULETTE 2u

layout(push_constant) uniform PathTracePushConstants {
    VertexBuffer vertices;
    // 스킨 컴퓨트가 뽑아 둔 변형 정점. 스킨 인스턴스는 여기서 정점을 읽는다.
    VertexBuffer skinnedVertices;
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

// GGX 법선 분포로 반값 벡터를 뽑는다. 거친 표면일수록 넓게 퍼진다.
vec3 sampleGgxHalfVector(vec3 normal, float roughness, inout uint seed) {
    float alpha = roughness * roughness;
    float u1 = randomFloat(seed);
    float u2 = randomFloat(seed);
    float phi = 6.2831853 * u1;
    float cosTheta = sqrt((1.0 - u2) / (1.0 + (alpha * alpha - 1.0) * u2));
    float sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    return normalize(tangentToWorld(vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta), normal));
}

// 코사인 가중 반구 표본. 확산 반사의 중요도 표본이다.
vec3 sampleCosineHemisphere(vec3 normal, inout uint seed) {
    float u1 = randomFloat(seed);
    float u2 = randomFloat(seed);
    float radius = sqrt(u1);
    float angle = 6.2831853 * u2;

    return normalize(tangentToWorld(vec3(radius * cos(angle), radius * sin(angle), sqrt(max(1.0 - u1, 0.0))), normal));
}

#endif
