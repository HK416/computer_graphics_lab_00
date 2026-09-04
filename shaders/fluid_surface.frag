#version 460

#include "fluid_draw_common.glsl"
#include "fog.glsl"

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inCurrentClip;
layout(location = 3) in vec4 inPreviousClip;

// 미리 곱해진 알파로 낸다. 혼합은 (ONE, ONE_MINUS_SRC_ALPHA) 라 뒤에 있는 색을 따로 읽지 않아도
// 투과가 맞는다. 화면 색 사본을 뜰 필요가 없다.
layout(location = 0) out vec4 outColor;
layout(location = 1) out vec2 outVelocity;
// 광선 반사에 곱할 가중치. 물은 화면 공간 안내 버퍼에 법선을 쓰지 않으므로 0 으로 덮어 반사를 끈다.
layout(location = 2) out vec4 outReflectionWeight;

void main() {
    Camera camera = push.camera.item;
    vec3 view = normalize(camera.position.xyz - inWorldPosition);
    vec3 normal = normalize(inNormal);
    // 뒷면이 보이면(카메라가 물 안에 있으면) 법선을 뒤집어야 프레넬이 맞는다.
    if (dot(normal, view) < 0.0) {
        normal = -normal;
    }

    Surface surface;
    surface.position = inWorldPosition;
    surface.normal = normal;
    surface.view = view;
    // 물은 유전체다. 알베도를 0 으로 두면 확산이 사라지고 f0 는 기본 0.04 가 된다. 물의 색은
    // 확산이 아니라 «통과하면서 덜 먹힌 색» 이라 아래 흡수 항이 낸다.
    surface.albedo = vec3(0.0);
    surface.metallic = 0.0;
    surface.roughness = clamp(push.waterColor.w, 0.02, 1.0);

    // 반사. 환경광과 조명 모두 다른 표면과 같은 함수를 쓴다. 두 함수가 이미 프레넬을 먹여 돌려주므로
    // 여기서 다시 곱하면 안 된다(0.04² 이 되어 물이 새까매진다).
    vec3 reflection = environmentLight(camera, surface, 1.0, true);
    uint lightCount = camera.shading.x;
    for (uint i = 0u; i < lightCount; ++i) {
        vec3 lightDirection;
        // ponytail: 그림자를 곱하지 않는다. fluid_draw_common.glsl 의 주석 참조.
        reflection += lightContribution(push.lights.items[i], surface, lightDirection);
    }

    // 통과. 두께는 뒷면에서 앞면을 뺀 시야 거리 합이다. 텍스처가 없으면 한 뼘으로 친다.
    float thickness = 0.25;
    if (push.thicknessTexture != INVALID_TEXTURE) {
        thickness = max(sampleBindless(push.thicknessTexture, gl_FragCoord.xy * camera.viewport.zw).r, 0.0);
    }
    thickness *= push.absorption.w;
    // Beer-Lambert. 두꺼울수록 덜 통과한다. 혼합이 알파 «하나» 로만 뒤를 가리므로 파장별 투과율을
    // 그대로 쓸 수 없다. 평균을 «얼마나 가리는가» 로 쓰고, 물빛은 아래에서 더한다.
    //
    // ponytail: 파장마다 다르게 가리려면 이중 소스 혼합(dualSrcBlend)이나 곱하기 패스가 하나 더
    // 필요하다. 지금은 흡수량만 색에 반영한다.
    vec3 transmittance = exp(-push.absorption.rgb * thickness);
    float absorbed = clamp(1.0 - dot(transmittance, vec3(1.0 / 3.0)), 0.0, 1.0);

    // 반사를 내는 environmentLight 가 albedo=0, metallic=0 이라 f0 = 0.04 를 쓴다. 여기서 다른 값을
    // 쓰면 «반사로 막히는 몫» 과 «알파» 가 어긋나 정면에서 에너지가 는다.
    float nDotV = max(dot(normal, view), 1e-4);
    float fresnel = fresnelSchlick(nDotV, vec3(0.04)).x;

    // 미리 곱해진 알파. 알파는 «뒤가 얼마나 가려지는가» 다. 반사로 막히는 몫이 프레넬이고, 통과한
    // 빛 가운데 흡수로 사라지는 몫이 absorbed 다.
    float opacity = clamp(fresnel + (1.0 - fresnel) * absorbed, 0.0, 1.0);
    // 흡수로 사라진 자리를 물이 스스로 내보내는 색(산란)으로 채운다. 흡수량에 «물 색» 을 곱하는
    // 것이지, 흡수 계수를 색으로 쓰는 것이 아니다. 뒤집으면 파란 물이 주황으로 나온다.
    vec3 color = reflection + push.waterColor.rgb * (1.0 - fresnel) * absorbed;

    // 안개. 미리 곱해진 알파라 «들어오는 산란광» 은 물이 덮은 몫만큼만 얹어야 한다. applyFog 를 그대로
    // 쓰면 뒤에 있는 배경이 제 거리로 이미 먹은 안개가 한 번 더 더해진다. 색이 0 인 표면에 같은 함수를
    // 걸면 산란광만 따로 얻을 수 있어 감쇠와 산란을 나눠 쓸 수 있다.
    float distance = length(camera.position.xyz - inWorldPosition);
    vec3 inscatter = applyFog(camera, vec3(0.0), camera.position.xyz, -view, distance);
    vec3 attenuated = applyFog(camera, color, camera.position.xyz, -view, distance) - inscatter;
    outColor = vec4(attenuated + inscatter * opacity, opacity);
    // 다른 표면과 같이 디버그 뷰를 따른다. 노멀 뷰는 마칭 큐브가 만든 면을 눈으로 볼 수 있어 유용하다.
    if (sceneDebugMode() == DEBUG_MODE_NORMAL) {
        outColor = vec4(normal * 0.5 + 0.5, 1.0);
    }
    vec2 current = inCurrentClip.xy / inCurrentClip.w - camera.jitter.xy;
    vec2 previous = inPreviousClip.xy / inPreviousClip.w;
    outVelocity = (previous - current) * 0.5;
    outReflectionWeight = vec4(0.0);
}
