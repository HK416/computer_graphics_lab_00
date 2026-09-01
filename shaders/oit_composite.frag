#version 460
#include "bindless.glsl"

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform CompositePushConstants {
    uint accumulationTexture;
    uint revealageTexture;
} pushConstants;

void main() {
    vec4 accumulation = sampleBindless(pushConstants.accumulationTexture, inUv);
    float revealage = sampleBindless(pushConstants.revealageTexture, inUv).r;
    vec3 color = accumulation.rgb / max(accumulation.a, 1e-5);
    // 알파에 잔여 투과율을 담아 블렌드 상태에서 src*(1-a) + dst*a 로 합성한다.
    outColor = vec4(color, revealage);
}
