#ifndef LOD_NETWORK_GLSL
#define LOD_NETWORK_GLSL

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

#define LOD_NETWORK_INPUTS 4
#define LOD_NETWORK_HIDDEN 8

// src/gfx/lod_network.h 의 GpuLodNetwork 와 배치가 일치해야 한다.
struct LodNetwork {
    float hiddenWeights[LOD_NETWORK_HIDDEN][LOD_NETWORK_INPUTS];
    float hiddenBias[LOD_NETWORK_HIDDEN];
    float outputWeights[LOD_NETWORK_HIDDEN];
    float outputBias;
    float padding[3];
};

layout(buffer_reference, scalar) readonly buffer LodNetworkBuffer { LodNetwork item; };

// 입력은 (경계 구, 오차) 쌍에서만 만든다. meshlet 고유 정보를 넣으면 자식의 부모 판정과 부모의
// 자기 판정이 달라져 LOD 경계에 틈이 생긴다.
void lodNetworkFeatures(float viewDistance, float radius, float error, float projected, out float features[LOD_NETWORK_INPUTS]) {
    features[0] = log2(max(viewDistance, 1e-3)) / 8.0;
    features[1] = log2(max(radius, 1e-6)) / 8.0;
    features[2] = log2(max(error, 1e-6)) / 8.0;
    features[3] = log2(max(projected, 1e-6)) / 8.0;
}

float lodNetworkBias(LodNetwork network, float features[LOD_NETWORK_INPUTS]) {
    float result = network.outputBias;
    for (int hidden = 0; hidden < LOD_NETWORK_HIDDEN; ++hidden) {
        float sum = network.hiddenBias[hidden];
        for (int index = 0; index < LOD_NETWORK_INPUTS; ++index) {
            sum += network.hiddenWeights[hidden][index] * features[index];
        }
        result += network.outputWeights[hidden] * tanh(sum);
    }
    return clamp(result, -8.0, 8.0);
}

#endif
