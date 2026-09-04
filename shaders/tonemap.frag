#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

#include "exposure.glsl"
#include "pbr.glsl"
#include "scene_types.glsl"

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform TonemapPushConstants {
    CameraBuffer camera;
    ExposureBuffer exposureBuffer;
    uint colorTexture;
    float exposure;
    // 0 이 아니면 경로 추적 누적 버퍼이므로 표본 수로 나눈다.
    uint sampleCount;
    // Bloom 0단계. 세기가 0 이면 읽지 않는다.
    uint bloomTexture;
    float bloomIntensity;
    uint autoExposure;
} pushConstants;

void main() {
    vec3 color = sampleBindless(pushConstants.colorTexture, inUv).rgb;
    if (pushConstants.sampleCount > 0u) {
        color /= float(pushConstants.sampleCount);
    }
    if (pushConstants.bloomIntensity > 0.0) {
        // Bloom 밉에는 임계값을 넘은 부분만 들어 있다. 섞으면 원본이 그만큼 어두워지므로 더한다.
        color += sampleBindless(pushConstants.bloomTexture, inUv).rgb * pushConstants.bloomIntensity;
    }
    float exposure = pushConstants.exposure;
    if (pushConstants.autoExposure != 0u) {
        exposure *= pushConstants.exposureBuffer.exposure;
    }
    color *= exposure;
    outColor = vec4(tonemapAces(color), 1.0);
}
