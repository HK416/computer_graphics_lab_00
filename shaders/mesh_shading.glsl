#ifndef MESH_SHADING_GLSL
#define MESH_SHADING_GLSL

#include "ibl.glsl"
#include "lighting.glsl"
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
    vec2 current = inCurrentClip.xy / inCurrentClip.w;
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

// 환경광. 프리필터 밉 수가 0 이면 IBL 이 꺼진 것이라 균일 환경광만 남긴다. ambient 는 IBL 에
// 곱하는 색조 겸 세기로 계속 쓰인다.
vec3 environmentLight(Surface surface, float occlusion) {
    vec3 tint = pushConstants.camera.item.ambient.rgb;
    uvec4 environment = pushConstants.camera.item.environment;
    if (environment.w == 0u) {
        return tint * surface.albedo * occlusion;
    }

    float nDotV = max(dot(surface.normal, surface.view), 1e-4);
    vec3 f0 = mix(vec3(0.04), surface.albedo, surface.metallic);
    vec3 fresnel = fresnelSchlickRoughness(nDotV, f0, surface.roughness);
    vec3 diffuseWeight = (vec3(1.0) - fresnel) * (1.0 - surface.metallic);

    vec3 irradiance = sampleBindlessCube(environment.x, surface.normal).rgb;
    vec3 reflection = reflect(-surface.view, surface.normal);
    vec3 prefiltered =
        sampleBindlessCubeLod(environment.y, reflection, surface.roughness * float(environment.w - 1u)).rgb;
    vec2 integrated = sampleBindlessArray(environment.z, vec2(nDotV, surface.roughness), 0.0).rg;

    vec3 diffuse = diffuseWeight * irradiance * surface.albedo;
    vec3 specular = prefiltered * (fresnel * integrated.x + integrated.y);
    return (diffuse + specular) * occlusion * tint;
}

vec3 shadingNormal(Material material) {
    vec3 normal = normalize(inNormal);
    if (material.normalTexture != INVALID_TEXTURE) {
        vec3 tangent = normalize(inTangent.xyz - normal * dot(normal, inTangent.xyz));
        vec3 bitangent = cross(normal, tangent) * inTangent.w;
        vec3 sampled = sampleBindless(material.normalTexture, inUv).xyz * 2.0 - 1.0;
        sampled.xy *= material.normalScale;
        normal = normalize(mat3(tangent, bitangent, normal) * sampled);
    }
    // 양면 재질의 뒷면은 노멀을 뒤집어야 조명이 뒤집히지 않는다.
    return gl_FrontFacing ? normal : -normal;
}

// 톤 매핑 이전의 선형 HDR 색과 알파를 돌려준다.
vec4 shadeSurface() {
    Material material = pushConstants.materials.items[inMaterialIndex];

    vec4 baseColor = material.baseColorFactor;
    if (material.baseColorTexture != INVALID_TEXTURE) {
        baseColor *= sampleBindless(material.baseColorTexture, inUv);
    }
    if (ALPHA_MODE_VARIANT == ALPHA_MODE_CUTOFF && baseColor.a < material.emissiveAndCutoff.w) {
        discard;
    }

    float metallic = material.metallicFactor;
    float roughness = material.roughnessFactor;
    if (material.metallicRoughnessTexture != INVALID_TEXTURE) {
        vec4 sampled = sampleBindless(material.metallicRoughnessTexture, inUv);
        roughness *= sampled.g;
        metallic *= sampled.b;
    }
    roughness = clamp(roughness, 0.03, 1.0);

    float occlusion = 1.0;
    if (material.occlusionTexture != INVALID_TEXTURE) {
        float sampled = sampleBindless(material.occlusionTexture, inUv).r;
        occlusion = mix(1.0, sampled, material.occlusionStrength);
    }

    vec3 emissive = material.emissiveAndCutoff.rgb;
    if (material.emissiveTexture != INVALID_TEXTURE) {
        emissive *= sampleBindless(material.emissiveTexture, inUv).rgb;
    }

    Surface surface;
    surface.position = inWorldPosition;
    surface.normal = shadingNormal(material);
    surface.view = normalize(pushConstants.camera.item.position.xyz - inWorldPosition);
    surface.albedo = baseColor.rgb;
    surface.metallic = metallic;
    surface.roughness = roughness;

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

    float ambientOcclusion = occlusion * screenSpaceOcclusion();
    color += environmentLight(surface, ambientOcclusion);
    color += emissive;

    float alpha = ALPHA_MODE_VARIANT == ALPHA_MODE_TRANSLUCENT ? baseColor.a : 1.0;
    return vec4(color, alpha);
}

#endif
