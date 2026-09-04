#ifndef FLUID_COMMON_GLSL
#define FLUID_COMMON_GLSL

#include "fluid_types.glsl"

// 시뮬레이션 패스(방출·격자·밀도·적분·인스턴스)의 푸시 상수. src/gfx/fluid.h 의
// FluidPushConstants 와 배치가 같아야 한다(scalar).
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

#endif
