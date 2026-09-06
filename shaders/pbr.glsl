#ifndef PBR_GLSL
#define PBR_GLSL

const float PI = 3.14159265359;

vec3 fresnelSchlick(float cosTheta, vec3 f0) {
    return f0 + (1.0 - f0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

float distributionGgx(float nDotH, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float denominator = nDotH * nDotH * (a2 - 1.0) + 1.0;
    return a2 / max(PI * denominator * denominator, 1e-7);
}

// Schlick-GGX 근사에 Smith 결합을 적용한 기하 감쇠항.
float geometrySmith(float nDotV, float nDotL, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;
    float ggxV = nDotV / (nDotV * (1.0 - k) + k);
    float ggxL = nDotL / (nDotL * (1.0 - k) + k);
    return ggxV * ggxL;
}

// 시인(sheen)의 Charlie 분포와 Ashikhmin 가시성. glTF KHR_materials_sheen 규격의 식이다.
float distributionCharlie(float nDotH, float roughness) {
    float alpha = max(roughness * roughness, 1e-3);
    float invAlpha = 1.0 / alpha;
    float sin2 = max(1.0 - nDotH * nDotH, 1e-4);
    return (2.0 + invAlpha) * pow(sin2, invAlpha * 0.5) / (2.0 * PI);
}

float visibilityAshikhmin(float nDotV, float nDotL) {
    return 1.0 / max(4.0 * (nDotL + nDotV - nDotL * nDotV), 1e-4);
}

// Narkowicz 의 ACES 필믹 톤 매핑 근사.
vec3 tonemapAces(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

#endif
