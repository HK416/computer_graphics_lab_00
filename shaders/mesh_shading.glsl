#ifndef MESH_SHADING_GLSL
#define MESH_SHADING_GLSL

#include "pbr.glsl"
#include "scene_data.glsl"

layout(location = 0) in vec3 inWorldPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec4 inTangent;
layout(location = 3) in vec2 inUv;
layout(location = 4) flat in uint inMaterialIndex;
layout(location = 5) flat in uint inMeshletIndex;

// 재질 경로별로 파이프라인을 나누어, 불투명 경로에서는 discard 자체가 컴파일에서 사라지게 한다.
layout(constant_id = 0) const uint ALPHA_MODE_VARIANT = 0;

const vec3 LIGHT_DIRECTION = vec3(0.4082, 0.8165, 0.4082);
const vec3 LIGHT_COLOR = vec3(3.0);
const vec3 AMBIENT_COLOR = vec3(0.25);

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

    vec3 normal = shadingNormal(material);
    vec3 view = normalize(pushConstants.camera.item.position.xyz - inWorldPosition);
    vec3 light = normalize(LIGHT_DIRECTION);
    vec3 halfway = normalize(view + light);

    float nDotL = max(dot(normal, light), 0.0);
    float nDotV = max(dot(normal, view), 1e-4);
    float nDotH = max(dot(normal, halfway), 0.0);
    float vDotH = max(dot(view, halfway), 0.0);

    vec3 f0 = mix(vec3(0.04), baseColor.rgb, metallic);
    vec3 fresnel = fresnelSchlick(vDotH, f0);
    float distribution = distributionGgx(nDotH, roughness);
    float geometry = geometrySmith(nDotV, nDotL, roughness);

    vec3 specular = (distribution * geometry * fresnel) / max(4.0 * nDotV * nDotL, 1e-4);
    vec3 diffuse = (vec3(1.0) - fresnel) * (1.0 - metallic) * baseColor.rgb / PI;

    vec3 color = (diffuse + specular) * LIGHT_COLOR * nDotL;
    color += AMBIENT_COLOR * baseColor.rgb * occlusion;
    color += emissive;

    float alpha = ALPHA_MODE_VARIANT == ALPHA_MODE_TRANSLUCENT ? baseColor.a : 1.0;
    return vec4(color, alpha);
}

#endif
