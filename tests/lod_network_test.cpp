#include <cassert>
#include <cmath>
#include <cstdio>
#include <vector>

#include "gfx/lod_network.h"

namespace {

// 오차와 거리가 제각각인 합성 표본. 임계값 그대로는 예산을 크게 넘기도록 잡는다.
std::vector<gfx::LodSample> makeSamples() {
    std::vector<gfx::LodSample> samples;
    for (int i = 0; i < 256; ++i) {
        float viewDistance = 1.0F + static_cast<float>(i % 16);
        float radius = 0.1F + static_cast<float>(i % 7) * 0.05F;
        float error = 0.01F + static_cast<float>(i % 11) * 0.01F;
        float projected = error * 800.0F / viewDistance;

        gfx::LodSample sample;
        sample.features[0] = std::log2(viewDistance) / 8.0F;
        sample.features[1] = std::log2(radius) / 8.0F;
        sample.features[2] = std::log2(error) / 8.0F;
        sample.features[3] = std::log2(projected) / 8.0F;
        sample.projectedError = projected;
        sample.triangleCount = 100.0F;
        samples.push_back(sample);
    }
    return samples;
}

} // namespace

int main() {
    std::vector<gfx::LodSample> samples = makeSamples();
    constexpr float THRESHOLD = 1.0F;
    constexpr float BUDGET = 6000.0F;

    gfx::LodNetwork network;
    network.learningRate = 0.2F;

    float firstLoss = network.train(samples, THRESHOLD, BUDGET);
    float lastLoss = firstLoss;
    for (int step = 0; step < 400; ++step) {
        lastLoss = network.train(samples, THRESHOLD, BUDGET);
    }

    std::printf("첫 손실 %.6f, 마지막 손실 %.6f, 기대 삼각형 %.1f (예산 %.0f)\n",
                static_cast<double>(firstLoss),
                static_cast<double>(lastLoss),
                static_cast<double>(network.lastSoftTriangleCount()),
                static_cast<double>(BUDGET));

    // 학습이 손실을 줄이고 삼각형 수를 예산 근처로 끌어와야 한다.
    assert(lastLoss < firstLoss * 0.1F);
    assert(std::fabs(network.lastSoftTriangleCount() - BUDGET) < BUDGET * 0.2F);

    // 가중치를 되돌리면 초기 상태로 돌아와야 한다.
    network.reset();
    assert(network.lastLoss() == 0.0F);

    std::puts("LOD 신경망 자체 점검 통과");
    return 0;
}
