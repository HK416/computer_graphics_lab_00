#ifndef SHADOW_GLSL
#define SHADOW_GLSL

#include "scene_data.glsl"

// src/gfx/renderer.cpp 의 SHADOW_ATLAS_SIZE 와 같아야 한다.
const float SHADOW_ATLAS_SIZE = 4096.0;
// 표면을 광원 쪽으로 밀어 자기 그림자를 없앤다. 장면 크기에 비례하지 않는 고정 값이라
// 아주 큰 장면에서는 키워야 할 수 있다.
const float SHADOW_NORMAL_OFFSET = 0.05;

// 점광은 여섯 면을 타일 여섯 장에 담는다. 면 번호는 방향의 최대 성분으로 고른다.
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

// 아틀라스는 정사각형 타일 격자다. 타일 안의 UV 를 아틀라스 UV 로 옮긴다.
vec2 shadowAtlasUv(vec2 uv, uint tile, float tilesPerSide) {
    float column = mod(float(tile), tilesPerSide);
    float row = floor(float(tile) / tilesPerSide);
    return (vec2(column, row) + uv) / tilesPerSide;
}

// 1 이면 완전히 밝고 0 이면 완전히 가려진 것이다.
float shadowFactor(Light light, vec3 position, vec3 normal, vec3 lightDirection) {
    int firstTile = int(light.rightShadow.w);
    float tilesPerSide = float(pushConstants.camera.item.shading.w);
    if (firstTile < 0 || tilesPerSide < 1.0) {
        return 1.0;
    }

    uint tile = uint(firstTile);
    if (uint(light.colorType.w) == LIGHT_TYPE_POINT) {
        tile += cubeFaceIndex(position - light.positionRange.xyz);
    }

    float slope = clamp(1.0 - dot(normal, lightDirection), 0.0, 1.0);
    vec3 offsetPosition = position + normal * (SHADOW_NORMAL_OFFSET * (0.5 + slope));

    vec4 clip = pushConstants.shadowMatrices.items[tile] * vec4(offsetPosition, 1.0);
    if (clip.w <= 0.0) {
        return 1.0;
    }
    vec3 ndc = clip.xyz / clip.w;
    // 그림자 절두체를 벗어난 곳은 판정하지 않는다.
    if (any(greaterThan(abs(ndc.xy), vec2(1.0))) || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }

    uint atlasSlot = pushConstants.camera.item.shading.y;
    vec2 uv = ndc.xy * 0.5 + 0.5;
    // 아틀라스 한 텍셀은 타일 좌표계에서 이만큼이다.
    float step = tilesPerSide / SHADOW_ATLAS_SIZE;

    float lit = 0.0;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            // 이웃 타일을 넘겨다보지 않도록 타일 안으로 자른다.
            vec2 tap = clamp(uv + vec2(x, y) * step, vec2(0.0), vec2(1.0));
            float depth = sampleBindless(atlasSlot, shadowAtlasUv(tap, tile, tilesPerSide)).r;
            lit += ndc.z <= depth ? 1.0 : 0.0;
        }
    }
    return lit / 9.0;
}

#endif
