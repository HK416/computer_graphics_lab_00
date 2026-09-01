#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace gfx {

inline constexpr uint32_t LOD_NETWORK_INPUTS = 4;
inline constexpr uint32_t LOD_NETWORK_HIDDEN = 8;

// shaders/lod_network.glsl 의 정의와 배치가 일치해야 한다.
struct GpuLodNetwork {
    float hiddenWeights[LOD_NETWORK_HIDDEN][LOD_NETWORK_INPUTS];
    float hiddenBias[LOD_NETWORK_HIDDEN];
    float outputWeights[LOD_NETWORK_HIDDEN];
    float outputBias;
    float padding[3];
};

struct LodSample {
    std::array<float, LOD_NETWORK_INPUTS> features{};
    float projectedError = 0.0F;
    float triangleCount = 0.0F;
};

// LOD 임계값 보정을 학습하는 4-8-1 다층 퍼셉트론.
//
// 신경망의 출력은 meshlet 이 아니라 (경계 구, 오차) 쌍의 함수다. DAG 선정은 자식의 부모 판정과
// 부모의 자기 판정이 같은 쌍을 쓰므로, 이렇게 두면 보정을 넣어도 두 판정이 정확히 일치해 경계에
// 틈이 생기지 않는다.
//
// 학습은 그린 삼각형 수를 예산에 맞추는 방향으로 진행한다. 선택 여부는 계단 함수라 미분할 수 없어
// 시그모이드로 부드럽게 만든 기대 삼각형 수를 손실로 쓴다.
class LodNetwork {
public:
    LodNetwork();

    void reset();
    // 표본으로 한 걸음 학습하고 손실을 돌려준다.
    float train(const std::vector<LodSample>& samples, float threshold, float triangleBudget);
    float predictBias(const std::array<float, LOD_NETWORK_INPUTS>& features) const;

    const GpuLodNetwork& weights() const { return parameters; }
    float lastLoss() const { return loss; }
    float lastSoftTriangleCount() const { return softTriangleCount; }

    float learningRate = 0.05F;
    // 로그 공간에서 선택 확률을 부드럽게 만드는 온도. 작을수록 실제 계단 함수에 가깝다.
    float temperature = 0.5F;

private:
    GpuLodNetwork parameters{};
    float loss = 0.0F;
    float softTriangleCount = 0.0F;
};

} // namespace gfx
