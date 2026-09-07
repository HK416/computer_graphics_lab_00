#version 460

#include "lighting.glsl"
#include "material.glsl"
#include "observation_common.glsl"

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inUv;
layout(location = 4) flat in uint inMaterialIndex;

layout(location = 0) out vec4 outColor;

void main() {
    Material material = observationPush.materials.items[inMaterialIndex];
    // 재질 읽기와 노멀 맵은 래스터·경로 추적과 **같은 함수**다.
    MaterialSample sampled = sampleMaterial(material, inUv);
    // 알파는 자르기만 한다. 관측에 반투명 정렬을 넣지 않는다 — 뒤섞인 순서가 프레임마다 달라지면
    // 같은 상태가 다른 그림이 된다.
    //
    // **재질이 실제로 CUTOFF 일 때만 자른다.** 컷오프 값은 alphaMode 와 무관하게 채워져 있어(glTF 기본
    // 0.5), 그냥 자르면 알파가 0 으로 남아 있는 불투명 재질이 관측에서 통째로 사라진다. 주 래스터 경로도
    // 이 판정을 CUTOFF 변종에서만 한다.
    if (material.alphaMode == ALPHA_MODE_CUTOFF && sampled.alpha < material.emissiveAndCutoff.w) {
        discard;
    }

    ObservationView view = observationPush.views.items[observationPush.view];
    Surface surface;
    surface.position = inWorldPosition;
    surface.normal = perturbNormal(material, normalize(inNormal), inTangent.xyz, inTangent.w, inUv);
    surface.view = normalize(view.cameraPosition.xyz - inWorldPosition);
    surface.albedo = sampled.albedo;
    surface.metallic = sampled.metallic;
    surface.roughness = sampled.roughness;
    clearSurfaceExtensions(surface);

    vec3 radiance = view.lightColor.rgb * view.lightDirection.w;
    vec3 color = evaluateBrdf(surface, view.lightDirection.xyz, radiance);
    // 상수 환경광. IBL 을 타지 않으므로 이것이 없으면 광원 반대쪽이 통째로 검다.
    color += sampled.albedo * sampled.occlusion * view.lightColor.w;
    color += sampled.emissive;
    // 톤 매핑도 같은 함수를 쓴다. **노출은 상수다** — 자동 노출을 태우면 장면이 조금만 밝아져도 관측이
    // 통째로 밀려, 같은 상태가 다른 입력이 된다.
    outColor = vec4(tonemapAces(color), 1.0);
}
