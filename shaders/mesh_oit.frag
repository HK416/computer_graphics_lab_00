#version 460
#include "mesh_shading.glsl"

// Weighted Blended OIT (McGuire & Bavoil). 누적 대상과 잔여 투과율 대상에 각각 기록한다.
layout(location = 0) out vec4 outAccumulation;
layout(location = 1) out float outRevealage;

// 깊이가 깊을수록 기여를 줄이는 가중치. reverse-Z 이므로 깊이를 뒤집어 넣는다.
float oitWeight(float depth, float alpha) {
    float z = 1.0 - depth;
    return clamp(pow(min(1.0, alpha * 10.0) + 0.01, 3.0) * 1e8 * pow(1.0 - z * 0.9, 3.0), 1e-2, 3e3);
}

void main() {
    // 반투명은 안내 버퍼를 쓰지 않는다. 파이프라인에 그 첨부물이 없어 값은 버려진다.
    vec4 normalRoughness;
    vec3 reflectionWeight;
    vec4 shaded = shadeSurface(normalRoughness, reflectionWeight);
    float weight = oitWeight(gl_FragCoord.z, shaded.a);
    outAccumulation = vec4(shaded.rgb * shaded.a, shaded.a) * weight;
    outRevealage = shaded.a;
}
