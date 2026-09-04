#ifndef FLUID_TYPES_GLSL
#define FLUID_TYPES_GLSL

#include "pbr.glsl"
#include "scene_types.glsl"

// 아래 구조체는 src/gfx/fluid.h 의 GpuFluidCollider / GpuFluidParams 와 배치가 같아야 한다(scalar).
// 푸시 상수는 쓰는 쪽마다 다르다. 시뮬레이션 패스는 fluid_common.glsl, 표면 패스는
// fluid_surface_common.glsl 이 각각 제 블록을 둔다.

#define FLUID_MAX_COLLIDERS 8u
#define FLUID_CELL_CAPACITY 32u
#define FLUID_GROUP_SIZE 128u
// 물 표면 패스의 작업 그룹 한 변. src/gfx/fluid.h 의 FLUID_SURFACE_GROUP_SIZE 와 같아야 디스패치 수가 맞는다.
#define FLUID_SURFACE_GROUP_SIZE 4

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

// 물 표면 정점 하나. src/physics/marching_cubes.h 의 SurfaceVertex 와 배치가 같아야 한다(scalar).
struct FluidSurfaceVertex {
    vec3 position;
    // 8진법으로 접은 법선. scene_types.glsl 의 encodeUnitVector 규칙이다.
    uint normal;
};

layout(buffer_reference, scalar) buffer FluidFieldBuffer {
    float items[];
};
layout(buffer_reference, scalar) buffer FluidSurfaceVertexBuffer {
    FluidSurfaceVertex items[];
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

// PI 는 pbr.glsl 의 것을 쓴다. 여기서 매크로로 다시 정의하면 그쪽 상수 선언과 부딪힌다.

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
