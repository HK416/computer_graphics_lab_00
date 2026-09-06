#ifndef RESTIR_GLSL
#define RESTIR_GLSL

#include "exposure.glsl"
#include "lighting.glsl"
#include "sampling.glsl"

// 저장소 표본 재추출(RIS). 광원 후보를 목표 함수(그림자 없는 기여의 휘도)로 가중해 하나만 남긴다.
// 래스터 ReSTIR 패스(restir_di.comp)와 경로 추적·반사의 NEE 가 같은 함수를 쓴다. 저장소 배치는
// src/gfx/renderer_internal.h 의 주석과 restir_di.comp 의 인코딩(광원 번호 비트, weightSum, m, w)이 따른다.

#define RESTIR_NO_LIGHT 0xFFFFFFFFu

struct Reservoir {
    uint light;
    // 지금까지 본 후보 가중치의 합과 후보 수, 그리고 고른 표본의 비편향 가중(1 / pdf 의 추정).
    float weightSum;
    float m;
    float w;
};

Reservoir emptyReservoir() {
    Reservoir r;
    r.light = RESTIR_NO_LIGHT;
    r.weightSum = 0.0;
    r.m = 0.0;
    r.w = 0.0;
    return r;
}

// 후보 하나(또는 다른 저장소 전체)를 합친다. 가중치에 비례해 고른 표본이 바뀐다.
bool reservoirUpdate(inout Reservoir r, uint light, float weight, float m, float random) {
    r.weightSum += weight;
    r.m += m;
    if (weight > 0.0 && random * r.weightSum < weight) {
        r.light = light;
        return true;
    }
    return false;
}

// 목표 함수: 그림자를 빼고 본 기여의 휘도. 래스터와 같은 lightContribution 이라 셰이딩과 일관된다.
float restirTarget(Light light, Surface surface) {
    vec3 lightDirection;
    return luminance(lightContribution(light, surface, lightDirection));
}

// 고른 표본의 가중을 확정한다. 목표 함수가 0 이면 그 표본은 기여가 없다.
void reservoirFinalize(inout Reservoir r, float target) {
    r.w = (target > 0.0 && r.m > 0.0) ? r.weightSum / (r.m * target) : 0.0;
}

// 초기 후보 뽑기. 광원을 균등하게(pdf 1/N) candidates 번 골라 목표 함수로 재추출한다. 광원이 후보 수보다
// 적어도 중복해 뽑는다(비편향).
Reservoir risPickLight(LightBuffer lights, uint lightCount, Surface surface, uint candidates, inout uint seed) {
    Reservoir r = emptyReservoir();
    if (lightCount == 0u) {
        return r;
    }
    for (uint c = 0u; c < candidates; ++c) {
        uint index = min(uint(randomFloat(seed) * float(lightCount)), lightCount - 1u);
        float target = restirTarget(lights.items[index], surface);
        reservoirUpdate(r, index, target * float(lightCount), 1.0, randomFloat(seed));
    }
    if (r.light != RESTIR_NO_LIGHT) {
        reservoirFinalize(r, restirTarget(lights.items[r.light], surface));
    }
    return r;
}

#endif
