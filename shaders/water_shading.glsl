#ifndef WATER_SHADING_GLSL
#define WATER_SHADING_GLSL

#include "pbr.glsl"

// 래스터 물 표면(fluid_surface.frag)과 광선 반사 컴퓨트(reflect.comp)가 함께 쓰는 물 셰이딩. 반사 항은 부르는
// 쪽이 환경광·조명으로 만들어 넘긴다(두 곳의 조명 표본 방식이 다르다). 여기서는 통과 항만 더한다.
//
// 두께는 Beer-Lambert 의 시야 거리다. 뒷면에서 앞면을 뺀 값이 없으면 한 뼘(0.25 m)으로 친다.
const float WATER_DEFAULT_THICKNESS = 0.25;

struct WaterShade {
    // 미리 곱해진 색(반사 + 물빛).
    vec3 color;
    // «뒤가 얼마나 가려지는가». 반사로 막히는 몫(프레넬) + 통과한 빛 가운데 흡수로 사라지는 몫.
    float opacity;
};

// nDotV 는 시선 쪽으로 세운 법선과 시선의 내적. waterColor.rgb 물 색, absorption 은 이미 두께 배율을 곱한 흡수
// 계수(1/m).
WaterShade shadeWater(vec3 reflection, vec3 waterColor, vec3 absorption, float thickness, float nDotV) {
    // Beer-Lambert. 두꺼울수록 덜 통과한다. 파장별 투과율의 평균을 «얼마나 가리는가» 로 쓰고 물빛은 아래에서
    // 더한다(래스터 혼합이 알파 하나로만 뒤를 가리기 때문. 경로 추적은 투과율을 그대로 쓴다).
    vec3 transmittance = exp(-absorption * thickness);
    float absorbed = clamp(1.0 - dot(transmittance, vec3(1.0 / 3.0)), 0.0, 1.0);
    // 반사를 내는 쪽이 albedo=0, metallic=0 이라 f0 = 0.04 를 쓴다. 여기서 다른 값을 쓰면 «반사로 막히는 몫» 과
    // «알파» 가 어긋나 정면에서 에너지가 는다.
    float fresnel = fresnelSchlick(max(nDotV, 1e-4), vec3(0.04)).x;
    WaterShade shade;
    shade.opacity = clamp(fresnel + (1.0 - fresnel) * absorbed, 0.0, 1.0);
    // 흡수로 사라진 자리를 물이 스스로 내보내는 색(산란)으로 채운다. 흡수량에 «물 색» 을 곱하는 것이지, 흡수
    // 계수를 색으로 쓰는 것이 아니다. 뒤집으면 파란 물이 주황으로 나온다.
    shade.color = reflection + waterColor * (1.0 - fresnel) * absorbed;
    return shade;
}

#endif
