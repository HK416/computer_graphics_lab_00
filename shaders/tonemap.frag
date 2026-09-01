#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

#include "pbr.glsl"
#include "scene_types.glsl"

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform TonemapPushConstants {
    CameraBuffer camera;
    uint colorTexture;
    float exposure;
    // 0 이 아니면 경로 추적 누적 버퍼이므로 표본 수로 나눈다.
    uint sampleCount;
    // 아무것도 그려지지 않은 화소를 환경 큐브맵으로 채운다. 슬롯이 INVALID_TEXTURE 면 건너뛴다.
    uint depthTexture;
    uint environmentCube;
} pushConstants;

// 화면 좌표에서 카메라를 지나는 광선 방향을 되돌린다.
vec3 viewRay(vec2 uv) {
    vec4 far = pushConstants.camera.item.inverseViewProjection * vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    return normalize(far.xyz / far.w - pushConstants.camera.item.position.xyz);
}

void main() {
    vec3 color = sampleBindless(pushConstants.colorTexture, inUv).rgb;
    if (pushConstants.sampleCount > 0u) {
        color /= float(pushConstants.sampleCount);
    } else if (pushConstants.environmentCube != INVALID_TEXTURE) {
        // reverse-Z 라 0 은 아무것도 그려지지 않은 곳이다. 그 화소만 하늘로 바꾼다.
        float depth = sampleBindless(pushConstants.depthTexture, inUv).r;
        if (depth <= 0.0) {
            color = sampleBindlessCube(pushConstants.environmentCube, viewRay(inUv)).rgb;
        }
    }
    color *= pushConstants.exposure;
    outColor = vec4(tonemapAces(color), 1.0);
}
