#ifndef FLUID_DRAW_COMMON_GLSL
#define FLUID_DRAW_COMMON_GLSL

#include "fluid_types.glsl"
#include "lighting.glsl"
#include "shadow.glsl"

// 물 표면을 그리는 두 패스(두께 · 표면)의 푸시 상수. src/gfx/renderer_internal.h 의 FluidDrawPushConstants 와
// 배치가 같아야 한다(scalar). 그림자 행렬은 shadow.glsl 이 인자로 받으므로 장면 푸시 상수 블록 없이도 그림자
// 맵을 읽는다(광선 그림자 변종은 없다 — 집합 1 을 묶지 않는다).
layout(push_constant, scalar) uniform FluidDrawPushConstants {
    CameraBuffer camera;
    FluidSurfaceVertexBuffer vertices;
    LightBuffer lights;
    // rgb 물 색(투과광의 색), w 표면 거칠기.
    vec4 waterColor;
    // rgb 흡수 계수(1/m), w 두께 배율.
    vec4 absorption;
    // 두께 텍스처 슬롯. INVALID_TEXTURE 면 두께를 흡수 계수의 기준값으로 본다.
    uint thicknessTexture;
    uint pad0;
    ShadowMatrixBuffer shadowMatrices;
}
push;

// 장면 푸시 상수 블록이 없으므로 디버그 모드는 카메라에서 직접 읽는다.
uint sceneDebugMode() {
    return push.camera.item.flags.x;
}

#endif
