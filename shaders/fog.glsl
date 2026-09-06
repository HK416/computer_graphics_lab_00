#ifndef FOG_GLSL
#define FOG_GLSL

#include "scene_types.glsl"

// 안개 인스캐터의 Henyey-Greenstein 비대칭 계수. 앞쪽으로 치우쳐 태양 쪽만 밝아진다.
const float FOG_SUN_ANISOTROPY = 0.6;

// 안개 색에 태양 인스캐터를 더한다. 위상 함수는 정면 최댓값이 1 이 되게 정규화해 산란 세기 1 이면 태양을 정면으로
// 볼 때 태양 색 × 세기가 그대로 더해진다. 방향은 광선을 따라 일정하다고 본다(광학 두께와 달리 적분하지 않는다).
vec3 fogInScatter(Camera camera, vec3 direction) {
    vec3 inScatter = camera.fog.rgb;
    float scatter = camera.fogSun.w;
    if (scatter <= 0.0 || camera.fogSunColor.w == 0.0) {
        return inScatter;
    }
    float g = FOG_SUN_ANISOTROPY;
    // 태양 쪽을 볼수록 1. fogSun.xyz 는 빛이 «가는» 방향이다.
    float cosTheta = dot(direction, -camera.fogSun.xyz);
    float phase = (1.0 - g * g) / pow(1.0 + g * g - 2.0 * g * cosTheta, 1.5);
    float peak = (1.0 + g) / ((1.0 - g) * (1.0 - g));
    return inScatter + camera.fogSunColor.rgb * (scatter * phase / peak);
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
    return mix(fogInScatter(camera, direction), color, transmittance);
}

#endif
