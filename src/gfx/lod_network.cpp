#include "gfx/lod_network.h"

#include <algorithm>
#include <cmath>
#include <random>

namespace gfx {
namespace {

float sigmoid(float value) {
    return 1.0F / (1.0F + std::exp(-std::clamp(value, -30.0F, 30.0F)));
}

} // namespace

LodNetwork::LodNetwork() {
    reset();
}

void LodNetwork::reset() {
    // 은닉층이 tanh 이므로 입력 수에 맞춘 작은 균등 분포로 시작한다.
    std::mt19937 generator(12345);
    float limit = 1.0F / std::sqrt(static_cast<float>(LOD_NETWORK_INPUTS));
    std::uniform_real_distribution<float> distribution(-limit, limit);

    parameters = {};
    for (uint32_t hidden = 0; hidden < LOD_NETWORK_HIDDEN; ++hidden) {
        for (uint32_t input = 0; input < LOD_NETWORK_INPUTS; ++input) {
            parameters.hiddenWeights[hidden][input] = distribution(generator);
        }
        parameters.outputWeights[hidden] = distribution(generator);
    }
    loss = 0.0F;
    softTriangleCount = 0.0F;
}

float LodNetwork::predictBias(const std::array<float, LOD_NETWORK_INPUTS>& features) const {
    float output = parameters.outputBias;
    for (uint32_t hidden = 0; hidden < LOD_NETWORK_HIDDEN; ++hidden) {
        float sum = parameters.hiddenBias[hidden];
        for (uint32_t input = 0; input < LOD_NETWORK_INPUTS; ++input) {
            sum += parameters.hiddenWeights[hidden][input] * features[input];
        }
        output += parameters.outputWeights[hidden] * std::tanh(sum);
    }
    return output;
}

float LodNetwork::train(const std::vector<LodSample>& samples, float threshold, float triangleBudget) {
    if (samples.empty() || triangleBudget <= 0.0F) {
        return loss;
    }

    std::array<std::array<float, LOD_NETWORK_INPUTS>, LOD_NETWORK_HIDDEN> hiddenGradients{};
    std::array<float, LOD_NETWORK_HIDDEN> hiddenBiasGradients{};
    std::array<float, LOD_NETWORK_HIDDEN> outputGradients{};
    float outputBiasGradient = 0.0F;

    // 순전파를 두 번 하지 않도록 표본마다 은닉층 활성값과 선택 확률을 기억해 둔다.
    struct Cached {
        std::array<float, LOD_NETWORK_HIDDEN> hidden{};
        float bias = 0.0F;
        float selectionProbability = 0.0F;
        float selectedError = 0.0F;
    };
    std::vector<Cached> cache(samples.size());

    softTriangleCount = 0.0F;
    for (size_t i = 0; i < samples.size(); ++i) {
        const LodSample& sample = samples[i];
        Cached& cached = cache[i];

        float output = parameters.outputBias;
        for (uint32_t hidden = 0; hidden < LOD_NETWORK_HIDDEN; ++hidden) {
            float sum = parameters.hiddenBias[hidden];
            for (uint32_t input = 0; input < LOD_NETWORK_INPUTS; ++input) {
                sum += parameters.hiddenWeights[hidden][input] * sample.features[input];
            }
            cached.hidden[hidden] = std::tanh(sum);
            output += parameters.outputWeights[hidden] * cached.hidden[hidden];
        }
        cached.bias = output;
        // 화면 오차가 수십 배 범위에 걸치므로 선택 확률을 로그 공간에서 만든다. 선형 공간에서는
        // 임계값에서 멀어진 표본의 기울기가 곧바로 사라져 학습이 멈춘다.
        float logSelected = std::log(std::max(sample.projectedError, 1e-9F)) + std::clamp(output, -8.0F, 8.0F);
        cached.selectedError = std::exp(logSelected);
        cached.selectionProbability = sigmoid((std::log(std::max(threshold, 1e-9F)) - logSelected) / temperature);
        softTriangleCount += sample.triangleCount * cached.selectionProbability;
    }

    float relative = softTriangleCount / triangleBudget - 1.0F;
    loss = relative * relative;
    float lossGradient = 2.0F * relative / triangleBudget;

    for (size_t i = 0; i < samples.size(); ++i) {
        const LodSample& sample = samples[i];
        const Cached& cached = cache[i];

        float probabilityGradient = cached.selectionProbability * (1.0F - cached.selectionProbability);
        float outputDelta =
            lossGradient * sample.triangleCount * (-probabilityGradient / temperature) * cached.selectedError;

        outputBiasGradient += outputDelta;
        for (uint32_t hidden = 0; hidden < LOD_NETWORK_HIDDEN; ++hidden) {
            outputGradients[hidden] += outputDelta * cached.hidden[hidden];
            float hiddenDelta =
                outputDelta * parameters.outputWeights[hidden] * (1.0F - cached.hidden[hidden] * cached.hidden[hidden]);
            hiddenBiasGradients[hidden] += hiddenDelta;
            for (uint32_t input = 0; input < LOD_NETWORK_INPUTS; ++input) {
                hiddenGradients[hidden][input] += hiddenDelta * sample.features[input];
            }
        }
    }

    float scale = learningRate / static_cast<float>(samples.size());
    parameters.outputBias -= scale * outputBiasGradient;
    for (uint32_t hidden = 0; hidden < LOD_NETWORK_HIDDEN; ++hidden) {
        parameters.outputWeights[hidden] -= scale * outputGradients[hidden];
        parameters.hiddenBias[hidden] -= scale * hiddenBiasGradients[hidden];
        for (uint32_t input = 0; input < LOD_NETWORK_INPUTS; ++input) {
            parameters.hiddenWeights[hidden][input] -= scale * hiddenGradients[hidden][input];
        }
    }
    return loss;
}

} // namespace gfx
