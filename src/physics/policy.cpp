#include "physics/policy.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <random>

namespace physics {
namespace {

// splitmix64. 계수 하나를 섞어 균등한 64비트를 낸다. 씨앗이 1 씩 이어져도 결과가 이어지지 않는다.
uint64_t mix(uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

// [0, 1) 균등. 상위 24비트만 쓴다(float 가 그만큼만 담는다).
float uniform(uint64_t value) {
    return static_cast<float>(value >> 40) * (1.0F / 16777216.0F);
}

// 점수를 [-0.5, 0.5] 의 중심 순위로 바꾼다. 보상의 크기가 아니라 순서만 보므로 태스크마다 보상 눈금을
// 다시 맞출 필요가 없고, 한 표본이 크게 튀어도 걸음이 휘둘리지 않는다.
std::vector<float> centeredRanks(const std::vector<float>& scores) {
    std::vector<uint32_t> order(scores.size());
    std::iota(order.begin(), order.end(), 0U);
    // 점수가 같으면 번호 순. 세대 결과가 정렬 구현에 따라 갈리지 않게 한다. NaN 은 «가장 나쁨» 으로
    // 몰아 둔다 — 그냥 비교하면 NaN 이 모든 값과 «동등» 이 되어 엄격 약순서가 깨지고 std::sort 가
    // 미정의 동작이 된다. 롤아웃이 발산하면 점수에 NaN 이 실제로 나온다.
    std::sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        bool nanA = std::isnan(scores[a]);
        bool nanB = std::isnan(scores[b]);
        if (nanA != nanB) {
            return nanA;
        }
        return !nanA && scores[a] != scores[b] ? scores[a] < scores[b] : a < b;
    });
    std::vector<float> ranks(scores.size(), 0.0F);
    float last = static_cast<float>(scores.size() - 1);
    for (size_t i = 0; i < order.size(); ++i) {
        ranks[order[i]] = last > 0.0F ? static_cast<float>(i) / last - 0.5F : 0.0F;
    }
    return ranks;
}

} // namespace

float gaussianNoise(uint64_t seed, uint64_t index) {
    // Box-Muller. 균등 둘을 표준 정규 하나로 바꾼다. u1 이 0 이면 로그가 발산하므로 바닥을 둔다.
    uint64_t state = mix(seed * 0x9E3779B97F4A7C15ULL + index);
    float u1 = std::max(uniform(state), 1.0e-7F);
    float u2 = uniform(mix(state));
    return std::sqrt(-2.0F * std::log(u1)) * std::cos(6.283185307179586F * u2);
}

size_t Policy::parameterCount(uint32_t inputs, uint32_t hidden, uint32_t outputs) {
    hidden = std::min(hidden, POLICY_MAX_HIDDEN);
    return static_cast<size_t>(hidden) * inputs + hidden + static_cast<size_t>(hidden) * hidden + hidden +
           static_cast<size_t>(outputs) * hidden + outputs;
}

Policy::Policy(uint32_t inputs, uint32_t hidden, uint32_t outputs)
    : inputs(inputs), hidden(std::min(hidden, POLICY_MAX_HIDDEN)), outputs(outputs) {
    weights.resize(parameterCount(inputs, this->hidden, outputs));
    reset();
}

// ponytail: uniform_real_distribution 은 표준이 알고리즘을 정하지 않아 표준 라이브러리 구현마다 값이
// 다르다(mt19937 자체는 이식된다). 학습 결과를 플랫폼 사이에서 바이트로 견주려면 분포도 직접 써야 한다.
void Policy::reset() {
    std::mt19937 generator(20260907);
    auto fill = [&](size_t begin, size_t count, uint32_t fanIn) {
        float limit = 1.0F / std::sqrt(static_cast<float>(std::max(fanIn, 1U)));
        std::uniform_real_distribution<float> distribution(-limit, limit);
        for (size_t i = 0; i < count; ++i) {
            weights[begin + i] = distribution(generator);
        }
    };
    std::fill(weights.begin(), weights.end(), 0.0F);
    size_t offset = 0;
    fill(offset, static_cast<size_t>(hidden) * inputs, inputs);
    // 편향은 0 이다. 은닉층 두 개만 흔들어도 대칭이 깨진다.
    offset += static_cast<size_t>(hidden) * inputs + hidden;
    fill(offset, static_cast<size_t>(hidden) * hidden, hidden);
    // 출력층은 0 으로 둔다. 학습 전의 정책이 «아무 것도 하지 않음» 에서 출발한다.
}

void Policy::evaluate(const float* observation, float* action) const {
    // 은닉층 둘은 크기가 같으므로 자리 두 개를 번갈아 쓴다. 폭이 POLICY_MAX_HIDDEN 으로 묶여 있어
    // 스택에 둔다.
    std::array<float, POLICY_MAX_HIDDEN> first{};
    std::array<float, POLICY_MAX_HIDDEN> second{};
    const float* weight = weights.data();

    for (uint32_t h = 0; h < hidden; ++h) {
        float sum = 0.0F;
        for (uint32_t i = 0; i < inputs; ++i) {
            sum += weight[static_cast<size_t>(h) * inputs + i] * observation[i];
        }
        first[h] = sum;
    }
    weight += static_cast<size_t>(hidden) * inputs;
    for (uint32_t h = 0; h < hidden; ++h) {
        first[h] = std::tanh(first[h] + weight[h]);
    }
    weight += hidden;

    for (uint32_t h = 0; h < hidden; ++h) {
        float sum = 0.0F;
        for (uint32_t i = 0; i < hidden; ++i) {
            sum += weight[static_cast<size_t>(h) * hidden + i] * first[i];
        }
        second[h] = sum;
    }
    weight += static_cast<size_t>(hidden) * hidden;
    for (uint32_t h = 0; h < hidden; ++h) {
        second[h] = std::tanh(second[h] + weight[h]);
    }
    weight += hidden;

    for (uint32_t o = 0; o < outputs; ++o) {
        float sum = 0.0F;
        for (uint32_t h = 0; h < hidden; ++h) {
            sum += weight[static_cast<size_t>(o) * hidden + h] * second[h];
        }
        action[o] = std::tanh(sum + weight[static_cast<size_t>(outputs) * hidden + o]);
    }
}

EvolutionStrategy::EvolutionStrategy(size_t parameterCount, uint32_t population, uint64_t seed)
    : parameters(parameterCount), samples(population - (population % 2)), seed(seed) {
    // 반대 부호 짝을 이루므로 홀수는 뜻이 없다. 0 이면 걸음을 만들 수 없다.
    samples = std::max(samples, 2U);
}

uint64_t EvolutionStrategy::noiseStream(uint64_t pair) const {
    // 곱하는 상수는 홀수라 짝 번호가 다르면 결과도 다르고, 씨앗은 먼저 섞어 두 학습의 스트림이 서로
    // 가까이 놓이지 않게 한다.
    return mix(seed) + pair * 0x9E3779B97F4A7C15ULL;
}

void EvolutionStrategy::sample(uint32_t index, const std::vector<float>& mean, std::vector<float>& out) const {
    if (mean.size() != parameters) {
        out.clear();
        return;
    }
    out.resize(parameters);
    // 짝수는 +, 홀수는 - 로 같은 잡음을 쓴다.
    uint64_t stream = noiseStream(static_cast<uint64_t>(generationIndex) * samples + index / 2U);
    float sign = index % 2U == 0U ? 1.0F : -1.0F;
    for (size_t i = 0; i < parameters; ++i) {
        out[i] = mean[i] + sign * sigma * gaussianNoise(stream, i);
    }
}

void EvolutionStrategy::step(const std::vector<float>& scores, std::vector<float>& mean) {
    if (scores.size() != samples || mean.size() != parameters) {
        return;
    }
    std::vector<float> ranks = centeredRanks(scores);
    // theta += (learningRate / (표본 수 × sigma)) * sum(순위 × 잡음). sigma 로 나누므로 잡음의 크기를
    // 키워도 걸음의 크기는 learningRate 가 정한다.
    float scale = learningRate / (static_cast<float>(samples) * sigma);
    for (uint32_t index = 0; index < samples; ++index) {
        uint64_t stream = noiseStream(static_cast<uint64_t>(generationIndex) * samples + index / 2U);
        float weight = ranks[index] * (index % 2U == 0U ? 1.0F : -1.0F) * scale;
        for (size_t i = 0; i < parameters; ++i) {
            mean[i] += weight * gaussianNoise(stream, i);
        }
    }
    ++generationIndex;
}

} // namespace physics
