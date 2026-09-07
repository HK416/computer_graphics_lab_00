#pragma once

// 유한차분 경사 검사. neural_test 와 rl_agent_test 가 함께 쓴다 — 두 벌로 두면 한쪽만 고쳐진다.
//
// 이 저장소가 «역전파가 맞다» 를 말하는 근거는 전부 여기를 지난다. 손으로 유도한 수식을 손으로
// 유도한 수식과 견주면 같은 실수를 두 번 하게 되므로, 기준은 늘 중앙 유한차분이다.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "gfx/neural_math.h"

namespace {

// 해석 경사와 유한차분 경사의 상대 오차. 눈금은 «이 그래프에서 경사가 대개 얼마나 큰가»로 잡는다.
// 자리마다 자기 크기로 나누면, 우연히 0 에 가까운 자리 하나가 반올림만으로 1e-2 를 찍어 구조적
// 버그(전치·첨자 밀림)와 구별할 수 없게 된다.
double relativeError(double a, double b, double typical) {
    double scale = std::max({std::abs(a), std::abs(b), 0.05 * typical, 1.0e-6});
    return std::abs(a - b) / scale;
}

// 그래프 하나를 세워 «해석 경사» 와 «유한차분 경사» 를 견준다. 흔드는 것은 파라미터와 입력 둘 다이며,
// 어느 자리를 흔들지는 씨앗으로 고른다(전부 흔들면 큰 그래프에서 너무 느리다).
struct GradientCheck {
    gfx::Graph graph;
    // 해석 경사는 **float** 로 낸다(실제로 쓰는 경로다). 유한차분은 **double** 로 잰다 — float 로 두
    // 손실의 차를 내면 반올림이 그 차이를 통째로 먹는다.
    std::vector<float> parameters;
    std::vector<float> activations;
    std::vector<float> parameterGradients;
    std::vector<float> activationGradients;
    std::vector<double> wideParameters;
    std::vector<double> wideActivations;
    // 흔들어 볼 입력 텐서들.
    std::vector<uint32_t> inputs;
    // 경사를 **일부러 끊은** 파라미터 텐서들(타깃망, 액터 손실에서 뗀 인코더 같은 것). 흔들면 손실이
    // 실제로 바뀌므로 유한차분과 어긋나는 것이 정상이다 — 훑지 않고, 대신 해석 경사가 **정확히 0** 인지
    // 본다. 그것이 stop-gradient 가 걸렸다는 증거다.
    std::vector<uint32_t> frozen;
    // 손실에 닿는 길이 여럿인데 **그중 일부만 끊긴** 파라미터 텐서들. 크리틱 표의 인코더가 그렇다 —
    // 온라인 갈래로는 경사를 받고 타깃 갈래로는 받지 않아, 해석 경사는 한쪽만 세고 유한차분은 양쪽을
    // 함께 잰다. 0 도 아니고 유한차분과 같지도 않으므로 여기서는 훑지 않는다. 그런 자리는 끊긴 길을
    // 없앤 판을 따로 세워 본다(rl_agent_test 의 checkCriticTerminal 이 그 예다).
    std::vector<uint32_t> partial;
    // randomize 뒤에 파라미터의 이 자리들을 이만큼 옮긴다. min 처럼 **갈래를 고르는** 연산이 있는
    // 그래프에서 어느 쪽이 이길지를 정해 놓는 데 쓴다 — 무작위 초기화에 맡기면 배치 전체가 한쪽으로
    // 쏠리고(진 쪽은 경사가 0 이라 위 «닿는가» 검사가 헛돈다) 반쪽만 밟게 된다.
    std::vector<std::pair<size_t, float>> nudges;

    void allocate() {
        parameters.assign(graph.parameterCount, 0.0F);
        activations.assign(graph.activationCount, 0.0F);
        parameterGradients.assign(graph.parameterCount, 0.0F);
        activationGradients.assign(graph.activationCount, 0.0F);
        wideParameters.assign(graph.parameterCount, 0.0);
        wideActivations.assign(graph.activationCount, 0.0);
    }

    // 파라미터와 입력을 흩뿌린다. 0 근처만 보면 ReLU 의 갈림과 min 의 갈림을 못 밟는다.
    void randomize(uint64_t seed) {
        for (size_t i = 0; i < parameters.size(); ++i) {
            parameters[i] = 0.7F * gfx::neuralGaussian(seed, i);
            wideParameters[i] = parameters[i];
        }
        for (const std::pair<size_t, float>& nudge : nudges) {
            parameters[nudge.first] += nudge.second;
            wideParameters[nudge.first] = parameters[nudge.first];
        }
        for (uint32_t tensor : inputs) {
            const gfx::Tensor& item = graph.tensors[tensor];
            for (uint32_t i = 0; i < item.count(); ++i) {
                activations[item.offset + i] = 0.9F * gfx::neuralGaussian(seed + 977, item.offset + i);
                wideActivations[item.offset + i] = activations[item.offset + i];
            }
        }
    }

    double loss() {
        gfx::forward(graph, wideParameters.data(), wideActivations.data());
        return wideActivations[graph.tensors[graph.ops.back().output].offset];
    }

    void analytic() {
        gfx::forward(graph, parameters.data(), activations.data());
        gfx::backward(
            graph, parameters.data(), activations.data(), parameterGradients.data(), activationGradients.data());
    }

    // value 를 흔들어 중앙 유한차분을 잰다. double 이라 h 를 작게 잡을 수 있고, 그만큼 절단 오차가
    // 사라진다(h 의 제곱에 비례한다).
    double numeric(double& value) {
        constexpr double H = 1.0e-5;
        double original = value;
        value = original + H;
        double plus = loss();
        value = original - H;
        double minus = loss();
        value = original;
        return (plus - minus) / (2.0 * H);
    }

    // 파라미터에서 자리를 골라 견준다. 돌려주는 값은 최대 상대 오차.
    double check(uint64_t seed, uint32_t samples) {
        randomize(seed);
        analytic();

        // 끊어 둔 파라미터는 훑지 않는다. 먼저 «정말 0 인가» 를 본다 — 이것이 이 검사에서 stop-gradient
        // 를 확인하는 자리다.
        std::vector<bool> skip(parameters.size(), false);
        for (uint32_t tensor : frozen) {
            const gfx::Tensor& item = graph.tensors[tensor];
            assert(item.arena == gfx::Arena::PARAMETER);
            for (uint32_t i = 0; i < item.count(); ++i) {
                assert(parameterGradients[item.offset + i] == 0.0F);
                skip[item.offset + i] = true;
            }
        }
        for (uint32_t tensor : partial) {
            const gfx::Tensor& item = graph.tensors[tensor];
            assert(item.arena == gfx::Arena::PARAMETER);
            for (uint32_t i = 0; i < item.count(); ++i) {
                skip[item.offset + i] = true;
            }
        }

        // **끊지 않은 파라미터 텐서는 하나도 빠짐없이 손실에 닿아야 한다.**
        //
        // 이것이 없으면 «해석 경사도 0, 유한차분도 0» 인 자리가 통과한다. 그래서 텐서 하나가 그래프에서
        // 통째로 떨어져 나가는 결함 — 쌍둥이 크리틱 둘이 같은 가중치를 보거나, min 이 한쪽만 보거나,
        // 액터가 타깃 크리틱을 최적화하거나 — 이 전부 조용히 지나간다. 자리마다 견주는 것만으로는
        // «없는 것» 을 볼 수 없다.
        for (uint32_t tensor : graph.parameterTensors()) {
            const gfx::Tensor& item = graph.tensors[tensor];
            if (skip[item.offset]) {
                continue;
            }
            bool reaches = false;
            for (uint32_t i = 0; i < item.count() && !reaches; ++i) {
                reaches = parameterGradients[item.offset + i] != 0.0F;
            }
            assert(reaches);
        }

        // 이 그래프에서 경사가 «대개 얼마나 큰가». 상대 오차의 눈금이 된다.
        double typical = 0.0;
        for (float value : parameterGradients) {
            typical = std::max(typical, static_cast<double>(std::abs(value)));
        }
        assert(typical > 0.0);

        double worst = 0.0;
        uint32_t checked = 0;
        auto compare = [&](size_t index) {
            if (skip[index]) {
                return;
            }
            double expected = parameterGradients[index];
            double measured = numeric(wideParameters[index]);
            worst = std::max(worst, relativeError(expected, measured, typical));
            ++checked;
        };
        // 파라미터 **텐서마다** 몇 자리씩 반드시 훑는다. 평탄한 첨자에 무작위로 때리기만 하면 작은
        // 텐서(편향은 두세 개다)가 체계적으로 밀려나 편향 경사가 통째로 검사되지 않는다.
        for (uint32_t tensor : graph.parameterTensors()) {
            const gfx::Tensor& item = graph.tensors[tensor];
            uint32_t sweep = std::min(item.count(), 4U);
            for (uint32_t i = 0; i < sweep; ++i) {
                compare(item.offset + static_cast<size_t>(i) * (item.count() / sweep));
            }
        }
        // 그 다음 나머지를 무작위로 흩어 본다.
        for (uint32_t s = 0; s < samples; ++s) {
            compare(static_cast<size_t>(std::abs(gfx::neuralGaussian(seed + 31, s)) * 1000.0F) % parameters.size());
        }
        // 입력을 흔들면 손실이 실제로 바뀌는지. 순전파가 그 값을 읽지 않으면 여기서 걸린다.
        for (uint32_t tensor : inputs) {
            const gfx::Tensor& item = graph.tensors[tensor];
            double moved = 0.0;
            for (uint32_t i = 0; i < item.count(); ++i) {
                moved = std::max(moved, std::abs(numeric(wideActivations[item.offset + i])));
            }
            assert(moved > 1.0e-9);
            ++checked;
        }
        assert(checked > 0);
        return worst;
    }
};

} // namespace
