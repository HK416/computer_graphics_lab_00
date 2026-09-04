#version 460

#include "fluid_draw_common.glsl"

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inCurrentClip;
layout(location = 3) in vec4 inPreviousClip;

layout(location = 0) out float outThickness;

// 물을 통과하는 두께를 잰다. 뒷면은 더하고 앞면은 빼는 것을 가산 혼합으로 쌓으면 겹친 층까지 모두
// 더해진다. 앞뒷면을 모두 그리므로 컬링을 끈다.
void main() {
    // 역깊이 무한 투영이라 시야 거리는 근평면 / 깊이다.
    float viewDepth = push.camera.item.parameters.x / max(gl_FragCoord.z, 1e-6);
    outThickness = gl_FrontFacing ? -viewDepth : viewDepth;
}
