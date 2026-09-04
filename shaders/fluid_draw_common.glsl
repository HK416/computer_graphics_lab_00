#ifndef FLUID_DRAW_COMMON_GLSL
#define FLUID_DRAW_COMMON_GLSL

#include "fluid_types.glsl"
#include "lighting.glsl"

// 물 표면을 그리는 두 패스(두께 · 표면)의 푸시 상수. src/gfx/renderer.h 의 FluidDrawPushConstants 와
// 배치가 같아야 한다(scalar).
//
// 그림자는 넣지 않았다. shadow.glsl 이 장면 푸시 상수 블록(pushConstants)을 이름으로 참조하는데
// 여기서는 그 블록이 없다. 물은 거의 스페큘러라 그림자가 빠져도 눈에 잘 띄지 않는다.
// ponytail: 그림자를 넣으려면 shadow.glsl 이 블록 대신 인자를 받게 고쳐야 한다.
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
    uint pad1;
    uint pad2;
}
push;

// 장면 푸시 상수 블록이 없으므로 디버그 모드는 카메라에서 직접 읽는다.
uint sceneDebugMode() {
    return push.camera.item.flags.x;
}

#endif
