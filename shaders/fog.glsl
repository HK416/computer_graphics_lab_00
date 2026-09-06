#ifndef FOG_GLSL
#define FOG_GLSL

#extension GL_EXT_buffer_reference_uvec2 : require

#include "bindless.glsl"
#include "scene_types.glsl"

// 안개 인스캐터의 Henyey-Greenstein 비대칭 계수. 앞쪽으로 치우쳐 태양 쪽만 밝아진다.
const float FOG_SUN_ANISOTROPY = 0.6;

// 태양 인스캐터 항. 위상 함수는 정면 최댓값이 1 이 되게 정규화해 산란 세기 1 이면 태양을 정면으로 볼 때 태양 색 × 세기가
// 그대로 더해진다. 방향은 광선을 따라 일정하다고 본다(광학 두께와 달리 적분하지 않는다). 태양이 없으면 0.
vec3 fogSunTerm(Camera camera, vec3 direction) {
    float scatter = camera.fogSun.w;
    if (scatter <= 0.0 || camera.fogSunColor.w == 0.0) {
        return vec3(0.0);
    }
    float g = FOG_SUN_ANISOTROPY;
    // 태양 쪽을 볼수록 1. fogSun.xyz 는 빛이 «가는» 방향이다.
    float cosTheta = dot(direction, -camera.fogSun.xyz);
    float phase = (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * cosTheta, 1.5);
    float peak = (1.0 + g) / ((1.0 - g) * (1.0 - g));
    return camera.fogSunColor.rgb * (scatter * phase / peak);
}

// 안개 색에 태양 인스캐터를 더한 것. 그림자를 넣지 않는 해석식이 쓴다.
vec3 fogInScatter(Camera camera, vec3 direction) {
    return camera.fog.rgb + fogSunTerm(camera, direction);
}

#ifdef FOG_CUSTOM_VISIBILITY
// 부르는 셰이더가 정의한다(경로 추적기는 그림자 광선을 쏜다).
float fogSunVisibility(Camera camera, vec3 position);
#else
// 태양 그림자 맵에서 한 점의 가시성. 표면 셰이딩(shadow.glsl)과 같은 캐스케이드 선택이되 노멀 오프셋과 PCF 는 없다 —
// 안개는 표면이 없고 표본 수가 많아 한 탭이면 된다. 벗어나면 밝다고 본다.
float fogSunVisibility(Camera camera, vec3 position) {
    uint firstLayer = camera.fogShadow.z & 0xFFFFu;
    // 캐스케이드 경계는 vec4 에 넷까지다(방향광 캐스케이드 상한과 같다).
    uint layerCount = min(camera.fogShadow.z >> 16u, 4u);
    if (layerCount == 0u || camera.shading.y == INVALID_TEXTURE) {
        return 1.0;
    }
    float distanceToCamera = length(position - camera.position.xyz);
    uint cascade = 0u;
    while (cascade + 1u < layerCount && distanceToCamera > camera.fogCascadeSplits[cascade]) {
        ++cascade;
    }
    ShadowMatrixBuffer matrices = ShadowMatrixBuffer(camera.fogShadow.xy);
    for (uint attempt = cascade; attempt < layerCount; ++attempt) {
        uint layer = firstLayer + attempt;
        vec4 clip = matrices.items[layer] * vec4(position, 1.0);
        if (clip.w <= 0.0) {
            continue;
        }
        vec3 ndc = clip.xyz / clip.w;
        if (any(greaterThan(abs(ndc.xy), vec2(1.0))) || ndc.z < 0.0 || ndc.z > 1.0) {
            continue;
        }
        float depth = sampleBindlessArray(camera.shading.y, ndc.xy * 0.5 + 0.5, float(layer)).r;
        return ndc.z <= depth ? 1.0 : 0.0;
    }
    return 1.0;
}
#endif

// 광선을 따라 태양 인스캐터가 얼마나 가려지는지. 표본점마다 밀도 × 그때까지의 투과율로 가중해 «가려진 몫» 을
// 적분하고 해석적 총량(1 - 투과율)으로 나눈다. 전부 밝으면 정확히 0 이라 해석식과 같은 값이 나온다.
// 지터는 방향 해시 + 프레임 번호라 시간축 업스케일과 경로 추적 누적이 표본을 섞는다.
float fogSunShadowedFraction(Camera camera, vec3 origin, vec3 direction, float startDensity, float slope, float distance, float totalThickness) {
    uint samples = camera.fogShadow.w;
    // 그림자 맵은 마지막 캐스케이드 너머를 모른다. 그 밖은 밝다고 보고 그 안만 표본한다.
    uint layerCount = min(camera.fogShadow.z >> 16u, 4u);
    float reach = layerCount == 0u ? 1.0e4 : camera.fogCascadeSplits[layerCount - 1u];
    float end = min(distance, reach);
    if (end <= 0.0) {
        return 0.0;
    }
    float step = end / float(samples);
    float jitter = fract(sin(dot(direction, vec3(12.9898, 78.233, 37.719))) * 43758.5453 +
                         float(camera.flags.z) * 0.6180339887);
    float shadowed = 0.0;
    for (uint i = 0u; i < samples; ++i) {
        float s = (float(i) + jitter) * step;
        // 밀도와 광학 두께는 해석식과 같은 지수 높이 모형이다.
        float density = startDensity * exp(-slope * s);
        float thickness = abs(slope) < 1.0e-4 ? startDensity * s : startDensity * (1.0 - exp(min(-slope * s, 80.0))) / slope;
        float weight = density * exp(-max(thickness, 0.0)) * step;
        shadowed += weight * (1.0 - fogSunVisibility(camera, origin + direction * s));
    }
    float total = 1.0 - exp(-max(totalThickness, 0.0));
    return clamp(shadowed / max(total, 1.0e-6), 0.0, 1.0);
}

// 지수 높이 안개. 밀도는 기준 높이에서 camera.fog.w 이고 위로 갈수록 exp(-감쇠 * (y - 높이)) 로
// 옅어진다. 광선을 따라 광학 두께를 해석적으로 적분하므로 표본이 필요 없다. 래스터(불투명·반투명·
// 하늘)와 경로 추적 1차 구간이 같은 함수를 부른다. 안개 색은 상수에 태양 인스캐터(fogInScatter)를 더한 것이다.
//
// ponytail: 단일 산란·방향광 하나뿐이다. 다중 산란과 점광 인스캐터는 광선을 따라 표본해야 한다.
vec3 applyFog(Camera camera, vec3 color, vec3 origin, vec3 direction, float distance) {
    float density = camera.fog.w;
    if (density <= 0.0) {
        return color;
    }
    float height = camera.fogParameters.x;
    float falloff = camera.fogParameters.y;
    // 시작점의 밀도. 카메라가 안개 위에 있으면 작고, 아래에 있으면 크다.
    float startDensity = density * exp(-falloff * (origin.y - height));
    float slope = falloff * direction.y;
    bool infinite = distance > 1.0e6;

    float thickness;
    if (abs(slope) < 1.0e-4) {
        // 수평 광선이거나 감쇠가 0 이면 밀도가 일정하다. 무한 거리는 완전히 잠긴다.
        thickness = infinite ? 1.0e6 : startDensity * distance;
    } else if (infinite) {
        // 위로 향하면 적분이 수렴하고, 아래로 향하면 발산한다.
        thickness = slope > 0.0 ? startDensity / slope : 1.0e6;
    } else {
        // 아래로 향하는 긴 광선은 지수가 커져 넘친다. 어차피 완전히 잠기므로 지수를 자른다.
        thickness = startDensity * (1.0 - exp(min(-slope * distance, 80.0))) / slope;
    }
    float transmittance = exp(-max(thickness, 0.0));
    vec3 inScatter = fogInScatter(camera, direction);
    // 태양 그림자(빛줄기). 표본 수가 0 이면 해석식만 쓴다.
    if (camera.fogShadow.w > 0u && camera.fogSun.w > 0.0 && camera.fogSunColor.w != 0.0) {
        float shadowed = fogSunShadowedFraction(camera, origin, direction, startDensity, slope, distance, thickness);
        inScatter -= fogSunTerm(camera, direction) * shadowed;
    }
    return mix(inScatter, color, transmittance);
}

#endif
