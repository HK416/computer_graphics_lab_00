#ifndef CLOTH_COMMON_GLSL
#define CLOTH_COMMON_GLSL

#include "fluid_types.glsl"
#include "scene_types.glsl"

// 아래는 src/gfx/cloth.h 의 GpuClothParams / ClothPushConstants, src/physics/cloth.h 의 ClothConstraint /
// ClothVertexInfo 와 배치가 같아야 한다(scalar). 알고리즘은 physics/cloth.cpp 의 CPU 솔버와 같은 순서다.

#define CLOTH_GROUP_SIZE 128u
#define CLOTH_FLAG_RESET 1u
#define CLOTH_FLAG_RAY_QUERY 2u
// src/gfx/cloth.h 의 CLOTH_SELF_RAY_MASK. 천 인스턴스는 이 비트가 내려가 있어 자기 자신을 맞히지 않는다.
#define CLOTH_SELF_RAY_MASK 0x02u

struct ClothConstraint {
    uint a;
    uint b;
    float restLength;
    float compliance;
};

struct ClothVertexInfo {
    uint gridX;
    uint gridZ;
    uint pad0;
    uint pad1;
};

struct ClothParams {
    vec4 accelerationDamping; // xyz 중력 + 바람, w 감쇠
    vec4 thicknessFriction;   // x 두께, y 마찰, z 서브스텝 간격, w 예약
    uint vertexCount;
    uint constraintCount;
    uint resolution;
    uint colliderCount;
    FluidCollider colliders[FLUID_MAX_COLLIDERS];
};

layout(buffer_reference, scalar) readonly buffer ClothParamsBuffer {
    ClothParams item;
};
layout(buffer_reference, scalar) buffer ClothVec4Buffer {
    vec4 items[];
};
layout(buffer_reference, scalar) buffer ClothFloatBuffer {
    float items[];
};
layout(buffer_reference, scalar) readonly buffer ClothConstraintBuffer {
    ClothConstraint items[];
};
layout(buffer_reference, scalar) readonly buffer ClothVertexInfoBuffer {
    ClothVertexInfo items[];
};
layout(buffer_reference, scalar) readonly buffer ClothGridBuffer {
    uint items[];
};

layout(push_constant, scalar) uniform ClothPushConstants {
    ClothParamsBuffer params;
    // xyz 위치, w 질량 역수(고정점 0).
    ClothVec4Buffer positions;
    ClothVec4Buffer predicted;
    ClothVec4Buffer velocities;
    ClothConstraintBuffer constraints;
    ClothFloatBuffer lambdas;
    // xyz 정지 위치(월드), w 질량 역수.
    ClothVec4Buffer rest;
    ClothVertexInfoBuffer info;
    ClothGridBuffer grid;
    SkinnedVertexBuffer vertices;
    uint destinationOffset;
    uint vertexCount;
    uint constraintFirst;
    uint constraintCount;
    float dt;
    uint flags;
}
push;

#endif
