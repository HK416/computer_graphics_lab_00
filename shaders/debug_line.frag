#version 460
#extension GL_EXT_nonuniform_qualifier : require

#include "bindless.glsl"
#include "debug_line_common.glsl"

layout(location = 0) in vec4 inColor;
layout(location = 0) out vec4 outColor;

void main() {
    if (push.occlude != 0u) {
        // 역깊이라 값이 클수록 가깝다. 장면보다 뒤에 있으면 가려진 것이다.
        //
        // ponytail: 가려진 구간을 통째로 지운다. Unity 처럼 흐리게 남기려면 색만 낮춰야 하는데,
        // 그러려면 가려진 선이 서로를 덮지 않도록 그리는 순서를 정해야 한다.
        float sceneDepth = sampleBindless(push.depthTexture, gl_FragCoord.xy / push.viewportSize).r;
        if (gl_FragCoord.z < sceneDepth) {
            discard;
        }
    }
    outColor = inColor;
}
