// 정책 망과 진화 전략. 시뮬레이션 없이 순수 계산만 본다.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "physics/policy.h"

namespace {

bool nearly(float a, float b, float tolerance = 1.0e-5F) {
    return std::abs(a - b) <= tolerance;
}

void testNoise() {
    // 같은 (씨앗, 번호)면 언제나 같은 값이다. 워커가 각자 다시 만들어 쓰는 것이 이 성질에 기댄다.
    assert(nearly(physics::gaussianNoise(7, 3), physics::gaussianNoise(7, 3)));
    assert(!nearly(physics::gaussianNoise(7, 3), physics::gaussianNoise(7, 4)));
    // 씨앗이 1 씩 이어져도 결과가 이어지지 않는다(계수 기반 해시).
    assert(!nearly(physics::gaussianNoise(7, 3), physics::gaussianNoise(8, 3)));

    // 표준 정규인지. 평균과 분산만 보면 균등분포도 통과하므로 꼬리 비율까지 본다.
    constexpr uint64_t COUNT = 200000;
    double sum = 0.0;
    double squared = 0.0;
    uint64_t beyondOne = 0;
    uint64_t beyondTwo = 0;
    for (uint64_t i = 0; i < COUNT; ++i) {
        double value = physics::gaussianNoise(1, i);
        sum += value;
        squared += value * value;
        beyondOne += std::abs(value) > 1.0 ? 1U : 0U;
        beyondTwo += std::abs(value) > 2.0 ? 1U : 0U;
    }
    double mean = sum / static_cast<double>(COUNT);
    double variance = squared / static_cast<double>(COUNT) - mean * mean;
    assert(std::abs(mean) < 0.02);
    assert(std::abs(variance - 1.0) < 0.03);
    // P(|z| > 1) = 0.3173, P(|z| > 2) = 0.0455. 평균 0·분산 1 인 균등분포는 각각 0.42 와 0 이라 걸린다.
    double outerOne = static_cast<double>(beyondOne) / static_cast<double>(COUNT);
    double outerTwo = static_cast<double>(beyondTwo) / static_cast<double>(COUNT);
    assert(std::abs(outerOne - 0.3173) < 0.01);
    assert(std::abs(outerTwo - 0.0455) < 0.01);
}

// 평평한 가중치 배열의 자리(W1, b1, W2, b2, W3, b3)를 못으로 박는다. 한 자리만 켜서 손으로 계산한
// 값과 견주므로, 층이 밀리거나 행렬이 전치되거나 편향 첨자가 어긋나면 바로 걸린다. 모양을 일부러
// 비대칭으로 잡아야 전치가 드러난다.
void testWeightLayout() {
    constexpr uint32_t INPUTS = 3;
    constexpr uint32_t HIDDEN = 4;
    constexpr uint32_t OUTPUTS = 2;
    constexpr size_t W1 = 0;
    constexpr size_t B1 = W1 + static_cast<size_t>(HIDDEN) * INPUTS;
    constexpr size_t W2 = B1 + HIDDEN;
    constexpr size_t B2 = W2 + static_cast<size_t>(HIDDEN) * HIDDEN;
    constexpr size_t W3 = B2 + HIDDEN;
    constexpr size_t B3 = W3 + static_cast<size_t>(OUTPUTS) * HIDDEN;
    assert(B3 + OUTPUTS == physics::Policy::parameterCount(INPUTS, HIDDEN, OUTPUTS));

    physics::Policy policy(INPUTS, HIDDEN, OUTPUTS);
    std::vector<float>& weights = policy.parameters();
    std::vector<float> observation{0.5F, -0.25F, 2.0F};
    std::vector<float> action(OUTPUTS, 0.0F);

    // 0) reset() 이 채운 구간. 은닉층 «가중치» 둘만 흔들고 편향과 출력층은 0 으로 남아야 한다.
    //    구간이 하나라도 밀리면 0 이어야 할 자리가 0 이 아니게 된다.
    auto anyNonZero = [&](size_t begin, size_t count) {
        for (size_t i = 0; i < count; ++i) {
            if (!nearly(weights[begin + i], 0.0F)) {
                return true;
            }
        }
        return false;
    };
    assert(anyNonZero(W1, static_cast<size_t>(HIDDEN) * INPUTS));
    assert(!anyNonZero(B1, HIDDEN));
    assert(anyNonZero(W2, static_cast<size_t>(HIDDEN) * HIDDEN));
    assert(!anyNonZero(B2, HIDDEN));
    assert(!anyNonZero(W3, static_cast<size_t>(OUTPUTS) * HIDDEN));
    assert(!anyNonZero(B3, OUTPUTS));

    // 1) 출력 편향만. 다른 층이 전부 0 이라 은닉층은 tanh(0) = 0 을 내고 출력은 tanh(b3) 다.
    std::fill(weights.begin(), weights.end(), 0.0F);
    weights[B3 + 0] = 0.5F;
    weights[B3 + 1] = -0.75F;
    policy.evaluate(observation.data(), action.data());
    assert(nearly(action[0], std::tanh(0.5F)));
    assert(nearly(action[1], std::tanh(-0.75F)));

    // 2) 입력 하나에서 출력 하나까지 한 줄만 켠다. 층마다 tanh 를 한 번씩 지난다.
    //    W1[h=2][i=1] · W2[h=3][h=2] · W3[o=0][h=3] 이라 행이 바뀌면(전치) 값이 0 이 된다. 첨자는
    //    전치해도 같은 자리를 가리키지 않는 것으로 고른다 - (o=1, h=3) 은 o·hidden+h 와 h·outputs+o
    //    가 둘 다 7 이라 전치를 못 잡는다.
    std::fill(weights.begin(), weights.end(), 0.0F);
    weights[W1 + 2 * INPUTS + 1] = 3.0F;
    weights[W2 + 3 * HIDDEN + 2] = 2.0F;
    weights[W3 + 0 * HIDDEN + 3] = 1.5F;
    policy.evaluate(observation.data(), action.data());
    float first = std::tanh(3.0F * observation[1]);
    float second = std::tanh(2.0F * first);
    assert(nearly(action[0], std::tanh(1.5F * second)));
    assert(nearly(action[1], 0.0F));

    // 3) 은닉 편향 둘. 관측이 0 이어도 편향만으로 값이 흐른다.
    std::fill(weights.begin(), weights.end(), 0.0F);
    std::vector<float> zero(INPUTS, 0.0F);
    weights[B1 + 1] = 0.8F;
    weights[W2 + 2 * HIDDEN + 1] = 1.0F;
    weights[B2 + 2] = -0.3F;
    weights[W3 + 1 * HIDDEN + 2] = 1.0F;
    policy.evaluate(zero.data(), action.data());
    assert(nearly(action[0], 0.0F));
    assert(nearly(action[1], std::tanh(std::tanh(std::tanh(0.8F) - 0.3F))));
}

void testPolicyShape() {
    constexpr uint32_t INPUTS = 5;
    constexpr uint32_t HIDDEN = 8;
    constexpr uint32_t OUTPUTS = 3;
    physics::Policy policy(INPUTS, HIDDEN, OUTPUTS);
    assert(policy.parameters().size() == physics::Policy::parameterCount(INPUTS, HIDDEN, OUTPUTS));
    assert(policy.inputCount() == INPUTS);
    assert(policy.outputCount() == OUTPUTS);

    std::vector<float> observation{0.3F, -1.2F, 5.0F, 0.0F, -0.7F};
    std::vector<float> action(OUTPUTS, 9.0F);
    policy.evaluate(observation.data(), action.data());
    // 출력층이 0 으로 시작하므로 학습 전에는 «아무 것도 하지 않음» 이다.
    for (float value : action) {
        assert(nearly(value, 0.0F));
    }

    // 가중치를 흔들면 움직이고, 흔들어도 출력은 tanh 라 [-1, 1] 안이다.
    for (size_t i = 0; i < policy.parameters().size(); ++i) {
        policy.parameters()[i] += 3.0F * physics::gaussianNoise(11, i);
    }
    policy.evaluate(observation.data(), action.data());
    bool moved = false;
    for (float value : action) {
        assert(value >= -1.0F && value <= 1.0F);
        moved = moved || !nearly(value, 0.0F);
    }
    assert(moved);

    // 같은 관측이면 언제나 같은 행동이다.
    std::vector<float> again(OUTPUTS, 0.0F);
    policy.evaluate(observation.data(), again.data());
    for (uint32_t i = 0; i < OUTPUTS; ++i) {
        assert(nearly(action[i], again[i]));
    }

    // 은닉층 폭은 상한으로 잘린다. 자리를 스택에 잡기 때문이다.
    physics::Policy wide(2, physics::POLICY_MAX_HIDDEN * 4, 1);
    assert(wide.hiddenCount() == physics::POLICY_MAX_HIDDEN);
    assert(wide.parameters().size() == physics::Policy::parameterCount(2, physics::POLICY_MAX_HIDDEN, 1));
    // 정적 함수도 생성자와 같은 규칙으로 잘라야 한다. 자르지 않으면 이 값으로 잡은 진화 전략의 가중치
    // 배열이 실제 망보다 커져, 남는 자리가 조용히 아무 일도 하지 않는다.
    assert(physics::Policy::parameterCount(2, physics::POLICY_MAX_HIDDEN * 4, 1) == wide.parameters().size());
}

void testAntitheticSampling() {
    constexpr size_t PARAMETERS = 16;
    physics::EvolutionStrategy strategy(PARAMETERS, 8);
    strategy.sigma = 0.1F;
    std::vector<float> mean(PARAMETERS, 0.5F);
    std::vector<float> plus;
    std::vector<float> minus;
    strategy.sample(0, mean, plus);
    strategy.sample(1, mean, minus);
    // 짝을 이룬 두 표본은 평균을 사이에 두고 정확히 반대쪽이다.
    for (size_t i = 0; i < PARAMETERS; ++i) {
        assert(nearly(plus[i] + minus[i], 2.0F * mean[i]));
        assert(!nearly(plus[i], mean[i]));
    }
    // 다른 짝은 다른 잡음을 쓴다.
    std::vector<float> other;
    strategy.sample(2, mean, other);
    assert(!nearly(other[0], plus[0]));

    // 흔드는 폭이 sigma 에 그대로 비례한다. step 의 1/sigma 와 상쇄되므로 여기서 따로 봐야 한다.
    strategy.sigma = 0.2F;
    std::vector<float> wider;
    strategy.sample(0, mean, wider);
    for (size_t i = 0; i < PARAMETERS; ++i) {
        assert(nearly(wider[i] - mean[i], 2.0F * (plus[i] - mean[i])));
    }
    strategy.sigma = 0.1F;

    // 홀수 개체군은 짝수로 내려 맞춘다.
    physics::EvolutionStrategy odd(PARAMETERS, 7);
    assert(odd.population() == 6);
}

// 흔들어 보고 좋은 쪽으로 옮기는 것이 실제로 목표에 다가가는지. 미분하지 않는다.
void testConvergence() {
    constexpr size_t PARAMETERS = 4;
    const std::vector<float> target{1.0F, -2.0F, 0.5F, 3.0F};
    auto score = [&](const std::vector<float>& value) {
        float sum = 0.0F;
        for (size_t i = 0; i < PARAMETERS; ++i) {
            float error = value[i] - target[i];
            sum -= error * error;
        }
        return sum;
    };
    auto train = [&](physics::EvolutionStrategy& strategy, std::vector<float>& mean) {
        std::vector<float> candidate;
        std::vector<float> scores(strategy.population(), 0.0F);
        for (uint32_t generation = 0; generation < 300; ++generation) {
            for (uint32_t i = 0; i < strategy.population(); ++i) {
                strategy.sample(i, mean, candidate);
                scores[i] = score(candidate);
            }
            strategy.step(scores, mean);
        }
    };

    physics::EvolutionStrategy strategy(PARAMETERS, 32);
    strategy.sigma = 0.2F;
    strategy.learningRate = 0.3F;
    std::vector<float> mean(PARAMETERS, 0.0F);
    float first = score(mean);
    train(strategy, mean);
    assert(strategy.generation() == 300);
    float last = score(mean);
    assert(last > first);
    // 최적값(0)에 충분히 가까워야 한다. 잡음 크기 sigma 가 남기는 바닥이 있다.
    assert(last > -0.05F);
    for (size_t i = 0; i < PARAMETERS; ++i) {
        assert(std::abs(mean[i] - target[i]) < 0.15F);
    }

    // 같은 씨앗이면 결과가 같다. 학습을 다시 돌려 견줄 수 있어야 한다.
    physics::EvolutionStrategy again(PARAMETERS, 32);
    again.sigma = 0.2F;
    again.learningRate = 0.3F;
    std::vector<float> repeated(PARAMETERS, 0.0F);
    train(again, repeated);
    for (size_t i = 0; i < PARAMETERS; ++i) {
        assert(nearly(mean[i], repeated[i]));
    }

    // 씨앗이 다르면 경로가 다르다. 그래도 같은 곳으로 간다.
    physics::EvolutionStrategy seeded(PARAMETERS, 32, 99);
    seeded.sigma = 0.2F;
    seeded.learningRate = 0.3F;
    std::vector<float> other(PARAMETERS, 0.0F);
    train(seeded, other);
    assert(!nearly(other[0], mean[0]));
    assert(std::abs(other[0] - target[0]) < 0.15F);
}

// 걸음의 크기는 sigma 에 반비례한다(theta += lr/(N·sigma) · sum(순위 × 잡음)). 선형 목적에서는 순위가
// sigma 와 무관하므로 걸음만 정확히 배로 갈린다. sample 이 뿌린 sigma 를 step 이 되돌리는지 본다.
void testSigmaScale() {
    constexpr size_t PARAMETERS = 6;
    auto run = [](float sigma) {
        physics::EvolutionStrategy strategy(PARAMETERS, 16);
        strategy.sigma = sigma;
        std::vector<float> mean(PARAMETERS, 0.0F);
        std::vector<float> candidate;
        std::vector<float> scores(strategy.population(), 0.0F);
        for (uint32_t i = 0; i < strategy.population(); ++i) {
            strategy.sample(i, mean, candidate);
            scores[i] = candidate[0] + 2.0F * candidate[1];
        }
        strategy.step(scores, mean);
        return mean;
    };
    std::vector<float> narrow = run(0.1F);
    std::vector<float> wide = run(0.2F);
    for (size_t i = 0; i < PARAMETERS; ++i) {
        assert(nearly(narrow[i], 2.0F * wide[i], 1.0e-4F));
    }
    assert(!nearly(narrow[0], 0.0F));
}

// 세대가 넘어가면 다른 잡음을 쓴다. 보폭이 표본 수가 아니면 다음 세대의 앞 짝이 이번 세대의 뒤 짝과
// 겹쳐 같은 방향만 되풀이해 흔들게 된다.
void testGenerationStride() {
    constexpr size_t PARAMETERS = 8;
    physics::EvolutionStrategy strategy(PARAMETERS, 8);
    strategy.sigma = 0.1F;
    std::vector<float> mean(PARAMETERS, 0.0F);
    std::vector<float> before;
    strategy.sample(2, mean, before);
    strategy.step(std::vector<float>(strategy.population(), 1.0F), mean);
    assert(strategy.generation() == 1);

    std::vector<float> zero(PARAMETERS, 0.0F);
    std::vector<float> after;
    strategy.sample(0, zero, after);
    std::vector<float> beforeNoise;
    physics::EvolutionStrategy fresh(PARAMETERS, 8);
    fresh.sigma = 0.1F;
    fresh.sample(2, zero, beforeNoise);
    assert(!nearly(after[0], beforeNoise[0]));

    // 씨앗이 다르면 잡음 자체가 다르다. 씨앗을 그냥 더하면 «세대만 어긋난 같은 잡음» 이 되어 씨앗을
    // 바꿔 여러 번 돌려 견주는 것이 뜻을 잃는다. 씨앗 1 의 세대 1 은 표본이 8 개이므로, 그냥 더하는
    // 방식이라면 씨앗 9 의 세대 0 과 정확히 같은 잡음이 된다.
    physics::EvolutionStrategy other(PARAMETERS, 8, 2);
    other.sigma = 0.1F;
    std::vector<float> seeded;
    other.sample(2, zero, seeded);
    assert(!nearly(seeded[0], beforeNoise[0]));

    physics::EvolutionStrategy shifted(PARAMETERS, 8, 1 + 8);
    shifted.sigma = 0.1F;
    std::vector<float> collided;
    shifted.sample(0, zero, collided);
    assert(!nearly(collided[0], after[0]));

    // mean 크기가 맞지 않으면 out 을 비우고 아무 것도 하지 않는다(step 과 같은 규칙).
    std::vector<float> shortMean(PARAMETERS - 1, 0.0F);
    std::vector<float> out(PARAMETERS, 1.0F);
    fresh.sample(0, shortMean, out);
    assert(out.empty());
}

// 점수의 눈금이 아니라 순서만 본다. 보상에 상수를 더하거나 곱해도 걸음이 같아야 한다.
void testRankInvariance() {
    constexpr size_t PARAMETERS = 6;
    auto run = [](float offset, float gain) {
        physics::EvolutionStrategy strategy(PARAMETERS, 16);
        std::vector<float> mean(PARAMETERS, 0.0F);
        std::vector<float> candidate;
        std::vector<float> scores(strategy.population(), 0.0F);
        for (uint32_t i = 0; i < strategy.population(); ++i) {
            strategy.sample(i, mean, candidate);
            scores[i] = gain * (candidate[0] + 2.0F * candidate[1]) + offset;
        }
        strategy.step(scores, mean);
        return mean;
    };
    std::vector<float> plain = run(0.0F, 1.0F);
    std::vector<float> shifted = run(1000.0F, 7.0F);
    for (size_t i = 0; i < PARAMETERS; ++i) {
        assert(nearly(plain[i], shifted[i]));
    }
    // 실제로 움직이기는 해야 한다.
    assert(!nearly(plain[0], 0.0F));

    // 점수 개수가 맞지 않으면 아무 것도 하지 않는다.
    physics::EvolutionStrategy strategy(PARAMETERS, 16);
    std::vector<float> mean(PARAMETERS, 0.25F);
    strategy.step(std::vector<float>(3, 1.0F), mean);
    assert(strategy.generation() == 0);
    assert(nearly(mean[0], 0.25F));
}

} // namespace

int main() {
    testNoise();
    testPolicyShape();
    testWeightLayout();
    testAntitheticSampling();
    testSigmaScale();
    testGenerationStride();
    testConvergence();
    testRankInvariance();
    std::printf("정책 망과 진화 전략 테스트 통과\n");
    return 0;
}
