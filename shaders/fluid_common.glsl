#ifndef FLUID_COMMON_GLSL
#define FLUID_COMMON_GLSL

#include "scene_types.glsl"

// 아래 구조체와 푸시 상수는 src/gfx/fluid.h 의 GpuFluidCollider / GpuFluidParams / FluidPushConstants 와 배치가
// 같아야 한다(scalar).

#define FLUID_MAX_COLLIDERS 8u
#define FLUID_CELL_CAPACITY 32u
#define FLUID_GROUP_SIZE 128u

#define FLUID_COLLIDER_SPHERE 0u
#define FLUID_COLLIDER_BOX 1u
#define FLUID_COLLIDER_PLANE 2u

// 입자 인스턴스가 지난 위치 대신 현재 위치를 이전 변환으로 쓴다(히스토리 없음).
#define FLUID_FLAG_RESET_HISTORY 1u
// 상위 가속 구조 인스턴스도 쓴다. 광선 기능이 켜져 있고 구의 하위 구조가 있을 때만 선다.
#define FLUID_FLAG_WRITE_TLAS 2u

// 강체 하나. 구: data0.xyz 중심, w 반지름. 상자: data0.xyz 반쪽 크기, inverseWorld 로 지역 공간에서 판정.
// 평면: data0.xyz 법선, w = dot(법선, 평면 위 점).
struct FluidCollider {
    vec4 data0;
    mat4 inverseWorld;
    mat4 world;
    uint type;
    uint pad0;
    uint pad1;
    uint pad2;
};

struct FluidParams {
    mat4 emitterWorld;
    vec4 emitterHalfExtents; // xyz 반쪽 크기, w 입자 간격
    vec4 containerMin;       // xyz, w 입자 반지름
    vec4 containerMax;       // xyz, w 벽 반발
    vec4 gravity;            // xyz, w 입자 질량
    float smoothingRadius;
    float restDensity;
    float stiffness;
    float viscosity;
    uvec4 lattice; // xyz 축별 방출 개수, w 셀 수(2 의 거듭제곱)
    uint colliderCount;
    uint pad0;
    uint pad1;
    uint pad2;
    FluidCollider colliders[FLUID_MAX_COLLIDERS];
};

// VkAccelerationStructureInstanceKHR 와 같은 64 바이트. 주소는 uint 둘로 나눠 담는다.
struct TlasInstance {
    vec4 row0;
    vec4 row1;
    vec4 row2;
    uint customIndexMask;
    uint sbtOffsetFlags;
    uint blasLow;
    uint blasHigh;
};

layout(buffer_reference, scalar) readonly buffer FluidParamsBuffer {
    FluidParams item;
};
layout(buffer_reference, scalar) buffer Vec4Buffer {
    vec4 items[];
};
layout(buffer_reference, scalar) writeonly buffer InstanceWriteBuffer {
    Instance items[];
};
layout(buffer_reference, scalar) writeonly buffer TlasInstanceBuffer {
    TlasInstance items[];
};

layout(push_constant) uniform FluidPushConstants {
    FluidParamsBuffer params;
    // xyz 위치, w 밀도. 서브스텝마다 In/Out 을 바꿔 쓴다.
    Vec4Buffer positionsIn;
    Vec4Buffer positionsOut;
    // xyz 속도, w 압력.
    Vec4Buffer velocitiesIn;
    Vec4Buffer velocitiesOut;
    CounterBuffer cellCounts;
    CounterBuffer cellParticles;
    // 지난 프레임에 그린 위치. 모션 벡터용.
    Vec4Buffer previousRendered;
    InstanceWriteBuffer instances;
    // 0 이면 쓰지 않는다.
    TlasInstanceBuffer tlasInstances;
    uint particleCount;
    uint instanceBase;
    uint tlasBase;
    uint flags;
    uint sphereMesh;
    float dt;
    // 구 하위 가속 구조의 주소. TLAS 인스턴스가 가리킨다.
    uint blasLow;
    uint blasHigh;
}
push;

#define PI 3.14159265358979

ivec3 fluidCell(vec3 position, float h) {
    return ivec3(floor(position / h));
}

uint fluidHash(ivec3 cell, uint cellCount) {
    uint hashed = uint(cell.x) * 73856093u ^ uint(cell.y) * 19349663u ^ uint(cell.z) * 83492791u;
    return hashed & (cellCount - 1u);
}

// Müller 2003 커널. poly6 는 밀도, spiky 기울기는 압력, 점성 라플라시안은 점성에 쓴다.
float poly6(float r2, float h) {
    float h2 = h * h;
    if (r2 >= h2) {
        return 0.0;
    }
    float d = h2 - r2;
    return 315.0 / (64.0 * PI * pow(h, 9.0)) * d * d * d;
}

vec3 spikyGradient(vec3 r, float length, float h) {
    if (length >= h || length <= 1e-6) {
        return vec3(0.0);
    }
    float d = h - length;
    return -45.0 / (PI * pow(h, 6.0)) * d * d * (r / length);
}

float viscosityLaplacian(float length, float h) {
    if (length >= h) {
        return 0.0;
    }
    return 45.0 / (PI * pow(h, 6.0)) * (h - length);
}

#endif
