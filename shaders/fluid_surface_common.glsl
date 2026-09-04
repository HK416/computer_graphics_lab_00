#ifndef FLUID_SURFACE_COMMON_GLSL
#define FLUID_SURFACE_COMMON_GLSL

#include "fluid_types.glsl"

// 물 표면 패스(장 만들기 · 마칭)의 푸시 상수. src/gfx/fluid.h 의 FluidSurfacePushConstants 와 배치가
// 같아야 한다(scalar). 시뮬레이션 패스와 나눈 것은 한 블록에 다 넣으면 128 바이트를 넘기 때문이다.

layout(push_constant, scalar) uniform FluidSurfacePushConstants {
    FluidParamsBuffer params;
    // xyz 위치, w 밀도. 이번 프레임의 최신 반쪽이다.
    Vec4Buffer positionsIn;
    CounterBuffer cellCounts;
    CounterBuffer cellParticles;
    // (resolution+1)³ 개의 스칼라 표본. x 가 가장 빠르다.
    FluidFieldBuffer surfaceField;
    FluidSurfaceVertexBuffer surfaceVertices;
    // [0] 이 정점 수. 마칭이 원자 덧셈으로 자리를 잡는다.
    CounterBuffer surfaceCounter;
    uint surfaceResolution;
    uint surfaceCapacity;
    float surfaceIso;
    uint pad0;
}
push;

#endif
