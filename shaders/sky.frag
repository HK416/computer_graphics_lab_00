#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

#include "scene_types.glsl"

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outVelocity;

layout(push_constant) uniform SkyPushConstants {
    CameraBuffer camera;
    uint environmentCube;
} pushConstants;

// 깊이 판정으로 아무것도 그려지지 않은 화소만 통과하므로 여기서는 다시 거르지 않는다. 하늘을
// 톤 매핑이 아니라 HDR 색상 대상에 직접 채우는 이유는 시간축 업스케일러가 하늘까지 함께 누적해야
// 실루엣 가장자리에 배경색 테두리가 남지 않기 때문이다.
void main() {
    Camera camera = pushConstants.camera.item;

    // 화면 좌표에서 카메라를 지나는 광선 방향. 지터가 들어간 시점 변환의 역이라 화소 중심과 맞는다.
    vec4 far = camera.inverseViewProjection * vec4(inUv * 2.0 - 1.0, 1.0, 1.0);
    vec3 direction = normalize(far.xyz / far.w - camera.position.xyz);
    outColor = vec4(sampleBindlessCube(pushConstants.environmentCube, direction).rgb, 1.0);

    // 하늘은 무한 원거리라 카메라 회전만 남는다. 방향 벡터를 지난 시점으로 되쏘면 그 회전분이 나온다.
    // 지터가 든 역변환으로 방향을 구했으므로 현재 NDC 는 정확히 (uv*2-1 - 지터)다.
    vec4 previousClip = camera.previousViewProjection * vec4(direction, 0.0);
    vec2 currentNdc = inUv * 2.0 - 1.0 - camera.jitter.xy;
    // 지난 프레임 카메라 뒤쪽이면 되짚을 자리가 없다. 변위를 0 으로 두어 히스토리를 버리게 한다.
    outVelocity = previousClip.w > 0.0 ? (previousClip.xy / previousClip.w - currentNdc) * 0.5 : vec2(0.0);
}
