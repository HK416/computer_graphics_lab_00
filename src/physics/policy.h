#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace physics {

// 은닉층의 최대 폭. evaluate 가 자리를 잡지 않고 도는 데 쓴다 — 롤아웃 한 번이 수천 스텝이고 표본이
// 워커에 나뉘므로, 스텝마다 배열을 잡으면 그것이 곧 병목이자 공유 상태가 된다.
inline constexpr uint32_t POLICY_MAX_HIDDEN = 64;

// 관측을 행동으로 옮기는 작은 다층 퍼셉트론. 은닉층 둘 다 tanh 이고 출력도 tanh 라 행동이 [-1, 1] 에
// 든다 — 부르는 쪽이 그것을 관절의 목표 범위로 늘린다.
//
// gfx::LodNetwork 과 달리 **경사를 쓰지 않는다.** 강화 학습의 보상은 시뮬레이션을 통째로 지나야
// 나오고 접촉은 미분할 수 없어서, 가중치를 흔들어 보고 좋은 쪽으로 옮기는 진화 전략(아래
// EvolutionStrategy)으로 학습한다.
class Policy {
public:
    // hidden 은 POLICY_MAX_HIDDEN 을 넘을 수 없다.
    Policy(uint32_t inputs, uint32_t hidden, uint32_t outputs);

    // 가중치를 시작값으로 되돌린다. 은닉층은 팬인에 맞춘 작은 균등 분포이고 출력층은 0 이라, 학습 전의
    // 정책은 «아무 것도 하지 않음»(행동 0)에서 시작한다.
    void reset();

    // observation 은 inputs 개, action 은 outputs 개다. 부르는 쪽이 자리를 잡아 넘긴다. 자리를 잡지
    // 않고 상태도 남기지 않으므로 여러 워커가 같은 정책을 함께 불러도 된다.
    void evaluate(const float* observation, float* action) const;

    uint32_t inputCount() const { return inputs; }
    // 생성자가 POLICY_MAX_HIDDEN 으로 자른 뒤의 값이다.
    uint32_t hiddenCount() const { return hidden; }
    uint32_t outputCount() const { return outputs; }
    // 학습이 흔드는 평평한 가중치 배열. 진화 전략은 이 배열만 보고 망의 모양을 모른다.
    std::vector<float>& parameters() { return weights; }
    const std::vector<float>& parameters() const { return weights; }
    // 생성자와 같은 규칙으로 hidden 을 POLICY_MAX_HIDDEN 까지 자르고 센다. 자르지 않으면 이 값으로
    // 잡은 진화 전략의 가중치 배열이 실제 망보다 커진다.
    static size_t parameterCount(uint32_t inputs, uint32_t hidden, uint32_t outputs);

private:
    uint32_t inputs;
    uint32_t hidden;
    uint32_t outputs;
    std::vector<float> weights;
};

// 진화 전략(OpenAI ES). 세대마다 평균 가중치에 가우시안 잡음을 더한 표본을 여럿 만들어 각각 점수를
// 재고, 점수의 «순위»로 가중한 잡음 평균만큼 평균을 옮긴다.
//
// 이 저장소에 맞는 이유 둘:
//
// 1. 표본끼리 완전히 독립이라 롤아웃을 core::JobSystem 에 그냥 나눌 수 있다. 공유 상태가 없다.
// 2. 잡음을 (세대, 표본, 가중치) 번호에서 다시 만들 수 있어 저장하지 않는다. 워커가 자기 표본의
//    잡음을 직접 만들어 쓰므로 잠금도, 세대마다 잡는 큰 배열도 없다.
//
// 표본은 반대 부호끼리 짝을 이룬다(antithetic). 같은 잡음을 더한 것과 뺀 것을 함께 재면 점수의
// 치우침이 상쇄되어 적은 표본으로도 방향이 잡힌다. 그래서 population 은 짝수여야 한다.
class EvolutionStrategy {
public:
    EvolutionStrategy(size_t parameterCount, uint32_t population, uint64_t seed = 1);

    // 표본 index 의 가중치를 out 에 채운다(mean + 부호 × sigma × 잡음). out 은 알아서 크기를 맞춘다.
    // mean 이 parameterCount() 개가 아니면 out 을 비우고 아무 것도 하지 않는다(step 과 같은 규칙).
    void sample(uint32_t index, const std::vector<float>& mean, std::vector<float>& out) const;
    // 표본마다 잰 점수(클수록 좋다)를 받아 mean 을 한 걸음 옮기고 세대를 하나 올린다. scores 는
    // population 개여야 한다.
    void step(const std::vector<float>& scores, std::vector<float>& mean);

    uint32_t population() const { return samples; }
    uint32_t generation() const { return generationIndex; }
    size_t parameterCount() const { return parameters; }

    // 잡음의 크기와 한 걸음의 크기. sample 이 뿌린 잡음을 step 이 sigma 로 나눠 되돌리므로 **한 세대
    // 안에서 sigma 를 바꾸면 안 된다** — 걸음의 크기가 조용히 어긋난다.
    float sigma = 0.05F;
    float learningRate = 0.03F;

private:
    // (씨앗, 짝) 을 잡음 스트림 번호 하나로 섞는다. 그냥 더하면 씨앗이 다른 두 학습이 «세대만 어긋난
    // 같은 잡음» 을 쓰게 되어 씨앗을 바꿔 여러 번 돌려 견주는 것이 뜻을 잃는다.
    uint64_t noiseStream(uint64_t pair) const;

    size_t parameters;
    uint32_t samples;
    uint64_t seed;
    uint32_t generationIndex = 0;
};

// 표본 번호와 가중치 번호만으로 정해지는 표준 가우시안 잡음. 저장하지 않고 필요할 때마다 다시 만들 수
// 있어야 표본을 워커에 그냥 나눌 수 있다. 계수 기반 해시(splitmix64)라 씨앗이 이어져도 상관이 없다.
float gaussianNoise(uint64_t seed, uint64_t index);

} // namespace physics
