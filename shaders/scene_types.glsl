#ifndef SCENE_TYPES_GLSL
#define SCENE_TYPES_GLSL

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

#include "bindless.glsl"

// 아래 구조체는 src/gfx/geometry.h 및 src/asset/model.h 의 정의와 배치가 일치해야 한다.
struct Vertex {
    vec3 position;
    vec3 normal;
    vec4 tangent;
    vec2 uv;
    // 조인트 넷을 바이트 하나씩, 가중치 넷을 unorm8 로 담는다. 스킨이 없으면 둘 다 0 이다.
    uint joints;
    uint weights;
};

struct Mesh {
    vec4 boundingSphere;
    uint indexOffset;
    uint indexCount;
    int vertexOffset;
    uint materialIndex;
    uint meshletOffset;
    uint meshletCount;
    uint lodOffset;
    uint lodCount;
};

struct MeshLod {
    uint indexOffset;
    uint indexCount;
    uint meshletOffset;
    uint meshletCount;
};

struct Meshlet {
    vec4 boundingSphere;
    vec4 cone;
    vec4 errorSphere;
    vec4 parentSphere;
    float error;
    float parentError;
    uint indexOffset;
    uint vertexOffset;
    uint triangleOffset;
    uint vertexCount;
    uint triangleCount;
    uint level;
    uint padding;
};

#define INVALID_TEXTURE 0xFFFFFFFFu

#define ALPHA_MODE_SOLID 0u
#define ALPHA_MODE_CUTOFF 1u
#define ALPHA_MODE_TRANSLUCENT 2u

struct Material {
    vec4 baseColorFactor;
    vec4 emissiveAndCutoff;
    float metallicFactor;
    float roughnessFactor;
    float normalScale;
    float occlusionStrength;
    uint baseColorTexture;
    uint metallicRoughnessTexture;
    uint normalTexture;
    uint occlusionTexture;
    uint emissiveTexture;
    uint alphaMode;
    uint flags;
    uint padding;
};

// 스킨이 없는 인스턴스의 조인트 오프셋.
#define NO_JOINTS 0xFFFFFFFFu
// 스킨 결과를 따로 뽑아 두지 않은 인스턴스의 정점 오프셋.
#define NO_SKINNED_VERTICES 0xFFFFFFFFu

struct Instance {
    mat4 model;
    mat4 normalMatrix;
    uint meshIndex;
    uint bucket;
    uint bucketBase;
    uint jointOffset;
    // 스킨 컴퓨트가 이 인스턴스의 변형된 정점을 써 둔 위치. 래스터 경로는 정점 셰이더에서 직접
    // 스키닝하므로 쓰지 않고, 가속 구조를 세우는 광선 경로만 본다. 없으면 NO_SKINNED_VERTICES.
    uint skinnedVertexOffset;
};

// 태스크 셰이더 워크그룹 하나가 처리할 meshlet 구간.
struct MeshletGroup {
    uint instanceIndex;
    uint firstMeshlet;
    uint meshletCount;
    uint padding;
};

struct Camera {
    mat4 viewProjection;
    vec4 position;
    // x: 근평면, y: 화면 오차 환산 배율, z: LOD 오차 임계값, w: 고정 LOD 단계(-1 이면 자동)
    vec4 parameters;
    // xyz 환경광 색조와 세기, w 광선 그림자를 쓸 최대 거리(0 이면 안 쓴다).
    vec4 ambient;
    // xy 렌더 해상도, zw 그 역수.
    vec4 viewport;
    // 깊이에서 월드 위치를 되돌릴 때 쓴다.
    mat4 inverseViewProjection;
    // x: 조명 수, y: 그림자 배열 슬롯(없으면 INVALID_TEXTURE), z: SSAO 슬롯, w: 환경 큐브 슬롯.
    // bindless 슬롯은 상위 8비트에 샘플러 번호가 들어가 float 로는 정확히 담기지 않으므로 정수로 둔다.
    uvec4 shading;
    // x: 조도 큐브 슬롯, y: 프리필터 큐브 슬롯, z: BRDF 표 슬롯, w: 프리필터 밉 수(0 이면 IBL 꺼짐).
    uvec4 environment;
};

#define LIGHT_TYPE_DIRECTIONAL 0u
#define LIGHT_TYPE_POINT 1u
#define LIGHT_TYPE_SPOT 2u
#define LIGHT_TYPE_AREA 3u

// 조명 하나. 위치와 축은 모두 월드 공간이며 렌더러가 오브젝트 변환에서 뽑아 채운다.
struct Light {
    vec4 positionRange;      // xyz 위치, w 도달 거리
    vec4 directionIntensity; // xyz 앞 방향, w 세기
    vec4 colorType;          // xyz 색, w 종류
    vec4 coneSize;           // xy 원뿔 cos(안/바깥), zw 영역광 반크기
    vec4 rightShadow;        // xyz 가로축, w 그림자 첫 층(-1 이면 없음)
    vec4 up;                 // xyz 세로축, w 이 조명이 쓰는 그림자 시점 수
    vec4 cascadeSplits;      // 캐스케이드 i 의 끝 거리
    vec4 cascadeTexelSizes;  // 캐스케이드 i 의 월드 텍셀 크기
};

// 간접 그리기 명령. VkDrawIndexedIndirectCommand 와 배치가 같다.
struct DrawCommand {
    uint indexCount;
    uint instanceCount;
    uint firstIndex;
    int vertexOffset;
    uint firstInstance;
};

#define DEBUG_MODE_SHADED 0u
#define DEBUG_MODE_MESHLET 1u
#define DEBUG_MODE_NORMAL 2u
#define DEBUG_MODE_UV 3u
#define DEBUG_MODE_DEPTH 4u
#define DEBUG_MODE_LOD 5u
#define DEBUG_MODE_CASCADE 6u
#define DEBUG_MODE_SHADOW 7u

layout(buffer_reference, scalar) readonly buffer VertexBuffer {
    Vertex items[];
};
// 스킨 컴퓨트가 변형 결과를 쓰는 곳. 읽기는 VertexBuffer 로 같은 주소를 가리킨다.
layout(buffer_reference, scalar) writeonly buffer SkinnedVertexBuffer {
    Vertex items[];
};
layout(buffer_reference, scalar) readonly buffer IndexBuffer {
    uint items[];
};
layout(buffer_reference, scalar) readonly buffer MeshLodBuffer {
    MeshLod items[];
};
layout(buffer_reference, scalar) readonly buffer MeshBuffer {
    Mesh items[];
};
layout(buffer_reference, scalar) readonly buffer InstanceBuffer {
    Instance items[];
};
layout(buffer_reference, scalar) readonly buffer MaterialBuffer {
    Material items[];
};
layout(buffer_reference, scalar) readonly buffer CameraBuffer {
    Camera item;
};
layout(buffer_reference, scalar) readonly buffer MeshletBuffer {
    Meshlet items[];
};
layout(buffer_reference, scalar) readonly buffer MeshletTriangleBuffer {
    uint items[];
};
layout(buffer_reference, scalar) readonly buffer VertexMeshletBuffer {
    uint items[];
};
layout(buffer_reference, scalar) readonly buffer MeshletGroupBuffer {
    MeshletGroup items[];
};
layout(buffer_reference, scalar) readonly buffer JointBuffer {
    mat4 items[];
};
layout(buffer_reference, scalar) readonly buffer LightBuffer {
    Light items[];
};
layout(buffer_reference, scalar) readonly buffer ShadowMatrixBuffer {
    mat4 items[];
};
layout(buffer_reference, scalar) buffer DrawCommandBuffer {
    DrawCommand items[];
};
layout(buffer_reference, scalar) buffer CounterBuffer {
    uint items[];
};

// 조인트 넷을 가중치로 섞은 스킨 행렬. 양자화된 가중치는 합이 1 이 아닐 수 있어 다시 정규화한다.
mat4 skinMatrix(JointBuffer joints, uint offset, uint packedJoints, uint packedWeights) {
    vec4 weights = unpackUnorm4x8(packedWeights);
    float total = weights.x + weights.y + weights.z + weights.w;
    if (total <= 0.0) {
        return mat4(1.0);
    }
    weights /= total;

    uvec4 indices = (uvec4(packedJoints) >> uvec4(0, 8, 16, 24)) & 0xFFu;
    return weights.x * joints.items[offset + indices.x] + weights.y * joints.items[offset + indices.y] +
           weights.z * joints.items[offset + indices.z] + weights.w * joints.items[offset + indices.w];
}

// 값을 색상환에 흩어 meshlet 이나 LOD 를 구분한다. 채도를 유지해야 인접 값이 잘 구별된다.
vec3 debugPalette(uint value) {
    uint hashed = value * 2654435761u;
    float hue = float(hashed >> 8) / 16777215.0;
    return clamp(abs(mod(hue * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
}

#endif
