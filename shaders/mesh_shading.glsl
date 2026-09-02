#ifndef MESH_SHADING_GLSL
#define MESH_SHADING_GLSL

#include "fog.glsl"
#include "ibl.glsl"
#include "lighting.glsl"
#include "material.glsl"
#include "scene_data.glsl"
#include "shadow.glsl"

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inUv;
layout(location = 4) flat in uint inMaterialIndex;
layout(location = 5) flat in uint inMeshletIndex;
layout(location = 6) in vec4 inCurrentClip;
layout(location = 7) in vec4 inPreviousClip;

// 이 화소가 지난 프레임에 있던 자리로 가는 UV 변위. 업스케일러가 히스토리를 되짚는 데 쓴다.
// NDC 는 두 항이 같은 부호 규약이라 Y 뒤집기가 상쇄되고, 0.5 배는 NDC 폭 2 를 UV 폭 1 로 줄인다.
vec2 motionVector() {
    vec2 current = inCurrentClip.xy / inCurrentClip.w - pushConstants.camera.item.jitter.xy;
    vec2 previous = inPreviousClip.xy / inPreviousClip.w;
    return (previous - current) * 0.5;
}

// 재질 경로별로 파이프라인을 나누어, 불투명 경로에서는 discard 자체가 컴파일에서 사라지게 한다.
layout(constant_id = 0) const uint ALPHA_MODE_VARIANT = 0;

// SSAO 결과를 화면 좌표로 읽는다. 패스가 꺼져 있으면 슬롯이 INVALID_TEXTURE 다.
float screenSpaceOcclusion() {
    uint slot = pushConstants.camera.item.shading.z;
    if (slot == INVALID_TEXTURE) {
        return 1.0;
    }
    return sampleBindless(slot, gl_FragCoord.xy * pushConstants.camera.item.viewport.zw).r;
}

// 톤 매핑 이전의 선형 HDR 색과 알파를 돌려준다.
vec4 shadeSurface() {
    Material material = pushConstants.materials.items[inMaterialIndex];
    // 재질 읽기와 노멀 맵은 경로 추적 적중 셰이더와 같은 함수를 쓴다.
    MaterialSample sampled = sampleMaterial(material, inUv);
    if (ALPHA_MODE_VARIANT == ALPHA_MODE_CUTOFF && sampled.alpha < material.emissiveAndCutoff.w) {
        discard;
    }
    vec3 normal = perturbNormal(material, normalize(inNormal), inTangent.xyz, inTangent.w, inUv);

    Surface surface;
    surface.position = inWorldPosition;
    // 양면 재질의 뒷면은 노멀을 뒤집어야 조명이 뒤집히지 않는다.
    surface.normal = gl_FrontFacing ? normal : -normal;
    surface.view = normalize(pushConstants.camera.item.position.xyz - inWorldPosition);
    surface.albedo = sampled.albedo;
    surface.metallic = sampled.metallic;
    surface.roughness = sampled.roughness;

    // 조명이 많아도 그냥 훑는다.
    //
    // ponytail: 화면 전체에 대한 선형 순회라 조명이 수십 개를 넘어가면 타일/클러스터 컬링으로
    // 올려야 한다. 편집기 규모에서는 이 편이 훨씬 단순하다.
    vec3 color = vec3(0.0);
    uint lightCount = pushConstants.camera.item.shading.x;
    for (uint i = 0; i < lightCount; ++i) {
        Light light = pushConstants.lights.items[i];
        vec3 lightDirection;
        vec3 contribution = lightContribution(light, surface, lightDirection);
        if (contribution == vec3(0.0)) {
            continue;
        }
        color += contribution * shadowFactor(light, surface.position, surface.normal, lightDirection);
    }

    float ambientOcclusion = sampled.occlusion * screenSpaceOcclusion();
    color += environmentLight(pushConstants.camera.item, surface, ambientOcclusion, true);
    color += sampled.emissive;

    // 안개는 카메라에서 표면까지의 구간에 건다. 반투명도 같은 식으로 잠긴다.
    vec3 cameraPosition = pushConstants.camera.item.position.xyz;
    color = applyFog(pushConstants.camera.item,
                     color,
                     cameraPosition,
                     -surface.view,
                     length(inWorldPosition - cameraPosition));

    float alpha = ALPHA_MODE_VARIANT == ALPHA_MODE_TRANSLUCENT ? sampled.alpha : 1.0;
    return vec4(color, alpha);
}

#endif
