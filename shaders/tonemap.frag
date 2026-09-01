#version 460
#include "bindless.glsl"
#include "pbr.glsl"

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform TonemapPushConstants {
    uint colorTexture;
    float exposure;
    // 0 이 아니면 경로 추적 누적 버퍼이므로 표본 수로 나눈다.
    uint sampleCount;
} pushConstants;

void main() {
    vec3 color = sampleBindless(pushConstants.colorTexture, inUv).rgb;
    if (pushConstants.sampleCount > 0u) {
        color /= float(pushConstants.sampleCount);
    }
    color *= pushConstants.exposure;
    outColor = vec4(tonemapAces(color), 1.0);
}
