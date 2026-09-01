#ifndef SHADOW_GLSL
#define SHADOW_GLSL

#include "scene_data.glsl"

#ifdef RAY_QUERY_SHADOWS
#extension GL_EXT_ray_query : require

// 경로 추적기와 같은 상위 가속 구조. 집합 1 로 묶여 있다.
layout(set = 1, binding = 0) uniform accelerationStructureEXT shadowTopLevel;

// 방향광은 광원까지의 거리가 없다. 장면을 넉넉히 가로지르는 길이로 쏜다.
//
// ponytail: 장면 반지름을 넘겨 주면 더 정확하지만, 상수 하나로 UBO 를 늘리지 않는 편이 낫다.
const float RAY_SHADOW_MAX_DISTANCE = 1.0e4;

// 그림자 맵 대신 광선으로 가시성을 판정한다. 0 이면 완전히 가려진 것이다.
float rayQueryVisibility(vec3 position, vec3 normal, vec3 toLight, float maxDistance) {
    // 자기 자신을 맞히지 않도록 표면에서 살짝 띄운다. 접선 방향 오차가 텍셀 크기에 비례하지
    // 않으므로 그림자 맵보다 훨씬 작은 값이면 된다.
    vec3 origin = position + normal * 1.0e-3;

    rayQueryEXT query;
    rayQueryInitializeEXT(query,
                          shadowTopLevel,
                          gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
                          0xFFu,
                          origin,
                          1.0e-4,
                          toLight,
                          maxDistance);
    rayQueryProceedEXT(query);
    return rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionNoneEXT ? 1.0 : 0.0;
}
#endif

// src/gfx/renderer.h 의 SHADOW_MAP_SIZE 와 같아야 한다.
const float SHADOW_MAP_SIZE = 1024.0;
// 노멀 오프셋 배율. 실제 오프셋은 캐스케이드의 월드 텍셀 크기에 이 값을 곱한 것이다. 캐스케이드마다
// 텍셀 크기가 수십 배 다르므로 고정값으로는 가까운 쪽과 먼 쪽을 동시에 맞출 수 없다.
const float SHADOW_NORMAL_OFFSET = 1.5;

// 점광은 여섯 면을 층 여섯 장에 담는다. 면 번호는 방향의 최대 성분으로 고른다.
uint cubeFaceIndex(vec3 direction) {
    vec3 magnitude = abs(direction);
    if (magnitude.x >= magnitude.y && magnitude.x >= magnitude.z) {
        return direction.x > 0.0 ? 0u : 1u;
    }
    if (magnitude.y >= magnitude.z) {
        return direction.y > 0.0 ? 2u : 3u;
    }
    return direction.z > 0.0 ? 4u : 5u;
}

// 한 층을 3x3 PCF 로 읽는다. 절두체를 벗어나면 -1 을 돌려준다.
float sampleShadowLayer(uint layer, vec3 position, uint atlasSlot) {
    vec4 clip = pushConstants.shadowMatrices.items[layer] * vec4(position, 1.0);
    if (clip.w <= 0.0) {
        return -1.0;
    }
    vec3 ndc = clip.xyz / clip.w;
    if (any(greaterThan(abs(ndc.xy), vec2(1.0))) || ndc.z < 0.0 || ndc.z > 1.0) {
        return -1.0;
    }

    vec2 uv = ndc.xy * 0.5 + 0.5;
    float step = 1.0 / SHADOW_MAP_SIZE;
    float lit = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float depth = sampleBindlessArray(atlasSlot, uv + vec2(x, y) * step, float(layer)).r;
            lit += ndc.z <= depth ? 1.0 : 0.0;
        }
    }
    return lit / 9.0;
}

// 1 이면 완전히 밝고 0 이면 완전히 가려진 것이다.
float shadowFactor(Light light, vec3 position, vec3 normal, vec3 lightDirection) {
    int firstLayer = int(light.rightShadow.w);
    uint layerCount = uint(light.up.w);
    if (firstLayer < 0 || pushConstants.camera.item.shading.y == INVALID_TEXTURE || layerCount == 0u) {
        return 1.0;
    }

    uint type = uint(light.colorType.w);

#ifdef RAY_QUERY_SHADOWS
    // 하이브리드: 카메라 가까이는 광선으로 판정하고 먼 곳은 그림자 맵을 그대로 쓴다. ambient.w 가
    // 0 이면 광선 그림자가 꺼진 것이다.
    float rayDistance = pushConstants.camera.item.ambient.w;
    if (rayDistance > 0.0 && length(position - pushConstants.camera.item.position.xyz) < rayDistance) {
        float reach = type == LIGHT_TYPE_DIRECTIONAL ? RAY_SHADOW_MAX_DISTANCE
                                                     : length(light.positionRange.xyz - position);
        return rayQueryVisibility(position, normal, lightDirection, reach);
    }
#endif

    uint layer = uint(firstLayer);
    float texelSize = light.cascadeTexelSizes.x;
    uint cascade = 0u;

    if (type == LIGHT_TYPE_POINT) {
        layer += cubeFaceIndex(position - light.positionRange.xyz);
        // 점광은 원근 투영이라 텍셀 크기가 거리에 따라 달라진다. 거리에 비례해 잡는다.
        texelSize = 2.0 * length(position - light.positionRange.xyz) / SHADOW_MAP_SIZE;
    } else if (type == LIGHT_TYPE_DIRECTIONAL) {
        // 카메라까지의 반지름 거리로 후보를 고른다. 축 거리보다 크거나 같아 항상 더 넓은
        // 캐스케이드를 골라 보수적으로 안전하다.
        float distance = length(position - pushConstants.camera.item.position.xyz);
        while (cascade + 1u < layerCount && distance > light.cascadeSplits[cascade]) {
            ++cascade;
        }
    } else {
        texelSize = 2.0 * length(position - light.positionRange.xyz) / SHADOW_MAP_SIZE;
    }

    // 표면을 광원 쪽으로 텍셀 크기만큼 밀어 자기 그림자를 없앤다.
    float slope = clamp(1.0 - dot(normal, lightDirection), 0.0, 1.0);

    // 방향광은 고른 캐스케이드가 범위를 벗어나면 한 단계씩 물러난다. 스냅 경계에서 아슬아슬하게
    // 벗어나는 경우가 있다.
    for (uint attempt = cascade; attempt < layerCount; ++attempt) {
        float offset = SHADOW_NORMAL_OFFSET * (0.5 + slope) *
                       (type == LIGHT_TYPE_DIRECTIONAL ? light.cascadeTexelSizes[attempt] : texelSize);
        uint target = type == LIGHT_TYPE_DIRECTIONAL ? layer + attempt : layer;
        float lit = sampleShadowLayer(target, position + normal * offset, pushConstants.camera.item.shading.y);
        if (lit >= 0.0) {
            return lit;
        }
        if (type != LIGHT_TYPE_DIRECTIONAL) {
            break;
        }
    }
    return 1.0;
}

// 캐스케이드 디버그 뷰가 쓰는 색. 방향광이 아니면 0 을 돌려준다.
uint shadowCascadeIndex(Light light, vec3 position) {
    if (uint(light.colorType.w) != LIGHT_TYPE_DIRECTIONAL || light.rightShadow.w < 0.0) {
        return 0u;
    }
    uint layerCount = uint(light.up.w);
    float distance = length(position - pushConstants.camera.item.position.xyz);
    uint cascade = 0u;
    while (cascade + 1u < layerCount && distance > light.cascadeSplits[cascade]) {
        ++cascade;
    }
    return cascade;
}

#endif
