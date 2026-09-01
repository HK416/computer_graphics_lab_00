#ifndef SCENE_DATA_GLSL
#define SCENE_DATA_GLSL

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

#include "bindless.glsl"

// 아래 구조체는 src/gfx/geometry.h 및 src/asset/model.h 의 정의와 배치가 일치해야 한다.
struct Vertex {
    vec3 position;
    vec3 normal;
    vec4 tangent;
    vec2 uv;
};

struct Mesh {
    vec4 boundingSphere;
    uint indexOffset;
    uint indexCount;
    int vertexOffset;
    uint materialIndex;
    uint meshletOffset;
    uint meshletCount;
    uint padding0;
    uint padding1;
};

struct Meshlet {
    vec4 boundingSphere;
    vec4 cone;
    uint vertexOffset;
    uint triangleOffset;
    uint vertexCount;
    uint triangleCount;
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

struct Instance {
    mat4 model;
    mat4 normalMatrix;
    uint meshIndex;
    uint padding0;
    uint padding1;
    uint padding2;
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
    vec4 parameters; // x: 근평면
};

#define DEBUG_MODE_SHADED 0u
#define DEBUG_MODE_MESHLET 1u
#define DEBUG_MODE_NORMAL 2u
#define DEBUG_MODE_UV 3u
#define DEBUG_MODE_DEPTH 4u

layout(buffer_reference, scalar) readonly buffer VertexBuffer { Vertex items[]; };
layout(buffer_reference, scalar) readonly buffer MeshBuffer { Mesh items[]; };
layout(buffer_reference, scalar) readonly buffer InstanceBuffer { Instance items[]; };
layout(buffer_reference, scalar) readonly buffer MaterialBuffer { Material items[]; };
layout(buffer_reference, scalar) readonly buffer CameraBuffer { Camera item; };
layout(buffer_reference, scalar) readonly buffer MeshletBuffer { Meshlet items[]; };
layout(buffer_reference, scalar) readonly buffer MeshletTriangleBuffer { uint items[]; };
layout(buffer_reference, scalar) readonly buffer VertexMeshletBuffer { uint items[]; };
layout(buffer_reference, scalar) readonly buffer MeshletGroupBuffer { MeshletGroup items[]; };

layout(push_constant) uniform PushConstants {
    VertexBuffer vertices;
    MeshBuffer meshes;
    InstanceBuffer instances;
    MaterialBuffer materials;
    CameraBuffer camera;
    MeshletBuffer meshlets;
    MeshletTriangleBuffer meshletTriangles;
    VertexMeshletBuffer vertexMeshlets;
    MeshletGroupBuffer meshletGroups;
    // 재질 경로마다 meshlet 그룹 구간이 달라 디스패치 직전에 갱신한다.
    uint meshletGroupBase;
    uint debugMode;
} pushConstants;

// 값을 색상환에 흩어 meshlet 이나 LOD 를 구분한다. 채도를 유지해야 인접 값이 잘 구별된다.
vec3 debugPalette(uint value) {
    uint hashed = value * 2654435761u;
    float hue = float(hashed >> 8) / 16777215.0;
    return clamp(abs(mod(hue * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
}

#endif
