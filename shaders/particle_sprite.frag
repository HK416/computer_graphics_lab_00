#version 460

#include "fog.glsl"
#include "particle_common.glsl"

layout(location = 0) in vec2 inUv;
layout(location = 1) in vec3 inWorldPosition;
layout(location = 2) in vec4 inColor;
layout(location = 3) in float inSize;

// 미리 곱해진 알파로 낸다. 혼합은 (ONE, ONE_MINUS_SRC_ALPHA).
layout(location = 0) out vec4 outColor;

// 깊이 첨부물 없이 그린다. 장면 깊이를 텍스처로 읽어 손으로 판정하고, 표면에 가까운 곳은 지름만큼에 걸쳐
// 부드럽게 사라지게 한다(소프트 파티클). 거리 비교라 투영 종류에 기대지 않는다.
void main() {
    float radius2 = dot(inUv, inUv);
    if (radius2 > 1.0) {
        discard;
    }
    Camera camera = push.camera.item;
    vec2 screenUv = gl_FragCoord.xy * camera.viewport.zw;
    float sceneDepth = sampleBindless(push.depthTexture, screenUv).r;
    vec3 sceneWorld = worldFromDepth(camera, screenUv, sceneDepth);
    float sceneDistance = sceneDepth > 0.0 ? length(sceneWorld - camera.position.xyz) : 1.0e30;
    vec3 toFragment = inWorldPosition - camera.position.xyz;
    float fragmentDistance = length(toFragment);
    if (sceneDistance < fragmentDistance) {
        discard;
    }
    float soft = clamp((sceneDistance - fragmentDistance) / max(inSize, 1.0e-4), 0.0, 1.0);
    float falloff = 1.0 - radius2;
    float alpha = inColor.a * falloff * falloff * soft;

    // 안개는 불투명 표면과 같은 함수다. 미리 곱하는 알파라 인스캐터도 입자가 덮는 몫만큼만 얹는다.
    vec3 direction = toFragment / fragmentDistance;
    vec3 color = applyFog(camera, inColor.rgb, camera.position.xyz, direction, fragmentDistance);
    outColor = vec4(color * alpha, alpha);
}
