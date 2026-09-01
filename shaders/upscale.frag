#version 460
#include "bindless.glsl"

layout(location = 0) in vec2 inUv;
layout(location = 0) out vec4 outColor;

layout(push_constant) uniform UpscalePushConstants {
    uint sourceTexture;
    float sharpness;
    vec2 sourceSize;
    vec2 destinationSize;
} pushConstants;

// 업스케일 방식. 0 은 통과, 1 은 Lanczos-2 확대 + 대비 적응 선명화.
layout(constant_id = 0) const uint UPSCALE_MODE = 0;

float lanczosWeight(float x) {
    if (abs(x) < 1e-5) {
        return 1.0;
    }
    if (abs(x) >= 2.0) {
        return 0.0;
    }
    float pix = 3.14159265 * x;
    return (2.0 * sin(pix) * sin(pix * 0.5)) / (pix * pix);
}

void main() {
    if (UPSCALE_MODE == 0) {
        outColor = sampleBindless(pushConstants.sourceTexture, inUv);
        return;
    }

    vec2 texelSize = 1.0 / pushConstants.sourceSize;
    vec2 sourcePixel = inUv * pushConstants.sourceSize - 0.5;
    vec2 basePixel = floor(sourcePixel);
    vec2 fraction = sourcePixel - basePixel;

    // 4x4 이웃을 Lanczos-2 가중치로 모은다. 동시에 대비 적응 선명화에 쓸 국소 최대/최소도 구한다.
    vec3 accumulated = vec3(0.0);
    float weightSum = 0.0;
    vec3 minimumColor = vec3(1e9);
    vec3 maximumColor = vec3(-1e9);

    for (int y = -1; y <= 2; ++y) {
        float weightY = lanczosWeight(float(y) - fraction.y);
        for (int x = -1; x <= 2; ++x) {
            float weightX = lanczosWeight(float(x) - fraction.x);
            vec2 uv = (basePixel + vec2(float(x), float(y)) + 0.5) * texelSize;
            vec3 sampled = sampleBindless(pushConstants.sourceTexture, clamp(uv, vec2(0.0), vec2(1.0))).rgb;

            float weight = weightX * weightY;
            accumulated += sampled * weight;
            weightSum += weight;
            if (abs(x) <= 1 && abs(y) <= 1) {
                minimumColor = min(minimumColor, sampled);
                maximumColor = max(maximumColor, sampled);
            }
        }
    }

    vec3 color = accumulated / max(weightSum, 1e-5);
    // Lanczos 는 고리 무늬를 만들 수 있어 이웃 범위로 가둔다.
    color = clamp(color, minimumColor, maximumColor);

    vec3 average = (minimumColor + maximumColor) * 0.5;
    color = clamp(color + (color - average) * pushConstants.sharpness, minimumColor, maximumColor);

    outColor = vec4(color, 1.0);
}
