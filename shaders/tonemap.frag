#version 460
#include "bindless.glsl"
#include "pbr.glsl"

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform TonemapPushConstants {
    uint colorTexture;
    float exposure;
} pushConstants;

void main() {
    vec3 color = sampleBindless(pushConstants.colorTexture, inUv).rgb * pushConstants.exposure;
    outColor = vec4(tonemapAces(color), 1.0);
}
