#ifndef PATHTRACE_COMMON_GLSL
#define PATHTRACE_COMMON_GLSL

#extension GL_EXT_ray_tracing : require

#include "sampling.glsl"
#include "scene_types.glsl"

// 경로 추적 페이로드. 적중 셰이더가 표면 정보를 채우고 광선 생성 셰이더가 경로를 이어 간다.
struct PathPayload {
    vec3 albedo;
    vec3 emissive;
    vec3 normal;
    vec3 position;
    // 지난 프레임의 세계 위치. 모션 벡터에만 쓴다.
    vec3 previousPosition;
    // 디버그 뷰와 가이드 버퍼만 쓴다. 경로 자체는 필요로 하지 않는다.
    vec2 uv;
    float metallic;
    float roughness;
    float hitDistance;
    bool missed;
};

// 1차 히트에서 건져 두는 값들. 경로 루프가 페이로드를 덮어쓰기 전에 복사해 둔다.
struct PrimaryHit {
    vec3 normal;
    vec3 albedo;
    vec2 uv;
    float roughness;
    float metallic;
    // reverse-Z NDC 깊이. 안내 버퍼로 나간다.
    float ndcDepth;
    // 화면 UV 단위 모션 벡터. 래스터와 같은 규약(지난 프레임 - 이번 프레임)이다.
    vec2 velocity;
    // 카메라에서의 거리. 아무것도 못 맞히면 음수.
    float hitDistance;
    // 첫 그림자 조명의 가시성. 조명이 없으면 음수.
    float visibility;
};

#define PATH_FLAG_NEXT_EVENT 1u
#define PATH_FLAG_RUSSIAN_ROULETTE 2u
// DLSS Ray Reconstruction 용 안내 버퍼를 채운다. 이때는 누적하지 않고 1표본만 쓴다.
#define PATH_FLAG_WRITE_GUIDES 4u

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
    // 화면 UV 모션 벡터를 쓸 rg16f 스토리지 슬롯.
    uint velocityImage;
    uint frameIndex;
    uint sampleCount;
    uint maxBounces;
    uint samplesPerFrame;
    uint flags;
    float radianceClamp;
    float skyIntensity;
    // scene_types.glsl 의 DEBUG_MODE_*. 0 이 아니면 셰이딩 대신 중간 값을 그린다.
    uint debugMode;
    // 안내 버퍼 슬롯. 여기까지가 124 바이트로, 규격이 보장하는 128 바이트에 거의 닿는다.
    // 그래서 슬롯을 하나씩 두지 않고 둘씩 16비트로 묶는다. 더 늘려야 하면 버퍼 주소로 옮긴다.
    uint guideAlbedoSlots;          // 하위 16비트 확산, 상위 16비트 반사
    uint guideNormalRoughnessSlots; // 하위 노멀, 상위 거칠기
    uint guideDepthSlot;
} pathTrace;

uint lowSlot(uint packed) {
    return packed & 0xFFFFu;
}

uint highSlot(uint packed) {
    return packed >> 16;
}

#endif
