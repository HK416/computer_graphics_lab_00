#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "gfx/neural_math.h"

namespace gfx {

// DrQ-v2 식 DDPG/TD3 에이전트의 그래프 명세. neural_math 의 연산 표만 쓰고 Vulkan 을 끌어오지 않는다 —
// 유한차분으로 검사할 수 있는 자리에 두려는 것이다.
//
// 왜 «그래프 셋» 인가: 역전파는 마지막 연산의 출력에 1 을 심어 시작하므로 표 하나에 손실은 하나다.
// 크리틱과 액터는 손실이 둘이라 표도 둘이고, 행동을 고르는 경로는 배치가 1 이라 셋째 표가 된다.
// **셋이 파라미터를 같은 순서로 잡아** 배열 하나를 함께 본다(parameterLayout 이 셋 다 같다).
//
// 갱신 한 번의 얼개:
//
//     forward(critic) -> backward(critic) -> adamStep(크리틱 구간)
//     forward(actor)  -> backward(actor)  -> adamStep(액터 구간)
//     polyakStep(온라인 -> 타깃)
//
// 두 adamStep 이 **파라미터 배열의 다른 구간**을 밟는다. PyTorch 로 치면 최적화기 둘을 따로 스텝하는
// 것과 같고, 그래서 파라미터의 순서가 계약이다(AgentParameterMap).
//
// **모멘트 배열은 구간마다 따로 잡는다.** adamStep 은 받은 배열의 앞 절반을 1차, 뒤 절반을 2차로 쓰므로
// 크기가 adamMomentCount(구간 길이) 여야 한다. 전체 길이로 하나 잡아 놓고 구간의 begin 만큼 밀어 쓰면
// 크리틱의 2차 모멘트와 액터의 1차 모멘트가 겹친다(84x84 판에서 28,640 개가 겹친다).
//
//     std::vector<float> criticMoments(adamMomentCount(map.criticUpdate.count()));
//     std::vector<float> actorMoments(adamMomentCount(map.actorUpdate.count()));
//
// 논문과 다른 자리:
//
// ponytail: DrQ-v2 는 액터와 크리틱에 **trunk 를 하나씩** 준다. 여기서는 크리틱의 trunk 하나를 액터가
// 끊은 채로 함께 본다. trunk 가 (39,200 -> 50) 이라 전체 파라미터의 90% 를 차지해, 두 벌로 두면 온라인·
// 타깃까지 넷이 되어 학습 상태가 배로 는다. 액터는 자기 은닉 둘을 그대로 학습하므로 표현만 크리틱이
// 정한다 — 그것이 인코더에 대해서는 원래 DrQ-v2 의 규약이기도 하다.
// ponytail: 타깃 정책 평활화 잡음을 tanh **앞** 에 더한다. TD3 는 tanh 뒤에 더하고 [-1, 1] 로 자르는데,
// 연산 표에 자르기가 없어서다. 잡음이 클 때 실효 세기가 tanh 의 기울기만큼 줄어든다.

// 픽셀 관측의 한 변. DrQ-v2 와 같다. 84 -(3x3 s2)- 41 - 39 - 37 - 35.
inline constexpr uint32_t AGENT_IMAGE_SIZE = 84;
// 인코더의 합성곱 층 수. 첫 층만 보폭 2 고 나머지는 1 이며, 여백은 전부 0 이다.
inline constexpr uint32_t AGENT_CONV_LAYERS = 4;
// 크리틱·액터 머리의 선형 층 수(은닉 둘 + 출력 하나).
inline constexpr uint32_t AGENT_HEAD_LAYERS = 3;

struct AgentConfig {
    // 픽셀 관측이면 한 변의 길이. **0 이면 저차원 관측**이라 인코더를 건너뛰고 observationCount 개
    // 실수를 그대로 trunk 에 넣는다(진자 각·각속도가 그 경우다).
    uint32_t imageSize = AGENT_IMAGE_SIZE;
    // 프레임 스택 = 관측 이미지의 채널 수. 회색 세 장이 기본이다(속도를 보려면 한 장으로는 모자란다).
    uint32_t frameStack = 3;
    // imageSize 가 0 일 때의 관측 수.
    uint32_t observationCount = 0;

    uint32_t actionCount = 1;
    // 학습 한 걸음이 보는 전이 수.
    uint32_t batch = 32;
    // trunk 가 내는 특징 수. 액터와 크리틱이 함께 본다.
    uint32_t featureCount = 50;
    // 머리의 은닉 폭. 논문은 1024 지만 우리 태스크에는 256 이면 넉넉하다.
    uint32_t hidden = 256;
    uint32_t convChannels = 32;

    bool pixels() const { return imageSize > 0; }

    // 관측 하나가 차지하는 float 개수. 픽셀이면 프레임 스택 x 변 x 변이다.
    uint32_t observationSize() const { return pixels() ? frameStack * imageSize * imageSize : observationCount; }
};

// 파라미터 배열 안의 구간(float 첨자). [begin, end) 다.
struct ParameterRange {
    size_t begin = 0;
    size_t end = 0;

    size_t count() const { return end - begin; }
    bool operator==(const ParameterRange&) const = default;
};

// 파라미터를 어떤 순서로 잡았는가. **통짜 구간으로 잘라 쓸 수 있게** 잡은 것이 요점이다 — 최적화도
// polyak 도 «어디부터 몇 개» 만 알면 되므로 GPU 에서 디스패치 하나로 끝난다.
//
//     [ 인코더 | 온라인 trunk | 온라인 크리틱 둘 | 타깃 trunk | 타깃 크리틱 둘 | 액터 ]
//       \______________ criticUpdate _________/   \_______ target ________/   \ actorUpdate
//                       \______ online ______/
//
// 액터가 맨 뒤인 이유: 크리틱 손실이 갱신하는 것(인코더 + 온라인 trunk + 크리틱 둘)이 앞에서부터
// 이어져야 한 구간이 된다. 액터를 가운데 두면 크리틱 구간이 둘로 갈린다.
struct AgentParameterMap {
    // 크리틱 손실이 갱신하는 구간. 배열 맨 앞이라 begin 이 0 이다.
    ParameterRange criticUpdate;
    // 인코더. 저차원 관측이면 빈 구간이다.
    ParameterRange encoder;
    // 온라인 trunk. 액터 손실은 encoder 와 이 구간에 경사를 내지 않는다.
    ParameterRange trunk;
    // polyak 의 원본과 목적지. 길이가 같고 안쪽 배치도 같은 순서다.
    ParameterRange online;
    ParameterRange target;
    // 액터 손실이 갱신하는 구간. 배열 맨 뒤다.
    ParameterRange actorUpdate;
    size_t total = 0;

    bool operator==(const AgentParameterMap&) const = default;
};

// 관측 하나에서 행동 하나. 배치가 1 이고 손실이 없어 validateForward 만 지난다.
struct AgentActGraph {
    Graph graph;
    uint32_t observation = NO_TENSOR;
    // tanh 를 지난 [-1, 1] 이다. physics::act 가 그대로 받는다.
    uint32_t action = NO_TENSOR;
};

// 크리틱 손실. **시간차 목표를 같은 표 안에서** 만들고 그 구간을 통째로 detach 한다 — PyTorch 의
// with no_grad() 와 같은 자리이고, 역전파가 그 구간의 연산을 아예 건너뛴다.
struct AgentCriticGraph {
    Graph graph;
    uint32_t observation = NO_TENSOR;
    uint32_t nextObservation = NO_TENSOR;
    // 리플레이에 담긴 «그때 실제로 한 행동». 액터가 지금 낼 행동이 아니다.
    uint32_t action = NO_TENSOR;
    uint32_t reward = NO_TENSOR;
    // 감가 마스크 γ(1 - 끝). 표본마다 다르므로 상수 배율이 아니라 입력이다.
    uint32_t discount = NO_TENSOR;
    // 타깃 정책 평활화 잡음. 액터의 **tanh 앞** 에 더한다 — tanh 가 [-1, 1] 을 지켜 주므로 따로 자를
    // 필요가 없다. TD3 는 tanh 뒤에 더하고 잘라 내지만, 자르기 연산이 없어 앞에 두었다.
    uint32_t noise = NO_TENSOR;

    uint32_t q1 = NO_TENSOR;
    uint32_t q2 = NO_TENSOR;
    // 타깃 크리틱 둘의 출력. min 을 취하기 전 값이라 테스트가 «정말 둘 다 보는가» 를 볼 수 있다 —
    // 드러내지 않으면 min 이 한쪽만 보도록 망가져도 밖에서 알 길이 없다.
    uint32_t targetTwin1 = NO_TENSOR;
    uint32_t targetTwin2 = NO_TENSOR;
    // y = r + γ(1 - 끝)·min(Q1', Q2'). 경사를 받지 않는다.
    uint32_t targetValue = NO_TENSOR;
    // mse(Q1, y) + mse(Q2, y).
    uint32_t loss = NO_TENSOR;
};

// 액터 손실 -mean(min(Q1, Q2)). 특징을 detach 해 **인코더와 trunk 를 갱신하지 않는다**(DrQ-v2 규약).
struct AgentActorGraph {
    Graph graph;
    uint32_t observation = NO_TENSOR;
    uint32_t action = NO_TENSOR;
    // 온라인 크리틱 둘이 지금 행동에 매긴 값. value 는 그 최소다.
    uint32_t q1 = NO_TENSOR;
    uint32_t q2 = NO_TENSOR;
    uint32_t value = NO_TENSOR;
    uint32_t loss = NO_TENSOR;
};

struct Agent {
    AgentConfig config;
    AgentActGraph act;
    AgentCriticGraph critic;
    AgentActorGraph actor;
    AgentParameterMap parameters;
};

// 설정에서 그래프 셋을 짓는다. 설정이 말이 안 되면(합성곱이 이미지를 다 먹거나, 저차원인데 관측이 0
// 이거나) 거짓을 돌려주고 out 을 건드리지 않는다.
bool buildAgent(const AgentConfig& config, Agent& out);

// 가중치를 초기화하고 타깃을 온라인과 똑같이 맞춘다. parameters 는 agent.parameters.total 개다.
//
// 크리틱 표가 파라미터를 **전부** 밟으므로 초기화의 기준은 그 표다. 액터 표는 타깃을 안 밟고 행동 표는
// 크리틱도 안 밟아, 그 둘로 초기화하면 밟지 않은 자리가 0 으로 남는다.
void initializeAgent(const Agent& agent, uint64_t seed, float* parameters);

// 타깃을 온라인 쪽으로 끌어당긴다. tau 가 1 이면 복사라 initializeAgent 가 이것을 쓴다.
void updateAgentTarget(const Agent& agent, float tau, float* parameters);

// 가중치를 저장하고 읽는다. **크리틱 표** 를 기준으로 삼는다(전부를 밟는 유일한 표라 그 해시가 곧
// 에이전트의 신원이다).
//
// ponytail: 그 해시에는 배치 크기도 든다. 파라미터는 배치와 무관한데도 배치 32 로 학습한 체크포인트가
// 배치 16 으로 지은 에이전트에 읽히지 않는다. 조용히 엉뚱한 파일을 읽는 것보다는 낫지만, 학습과 재생의
// 배치를 다르게 두고 싶어지면 «파라미터에 닿는 부분만» 접는 해시가 따로 필요하다.
bool saveAgent(const Agent& agent, const float* parameters, const std::string& path);
bool loadAgent(const Agent& agent, float* parameters, const std::string& path);

// ---- 갱신 루프.

struct AgentUpdateSettings {
    AdamSettings critic;
    AdamSettings actor;
    // 타깃을 온라인 쪽으로 끌어당기는 세기.
    float tau = 0.01F;
    // 타깃 정책 평활화 잡음의 세기와 자르는 폭. 잡음은 액터의 tanh **앞** 에 실린다.
    float targetNoise = 0.2F;
    float noiseClip = 0.5F;
};

// 갱신 한 번이 보는 전이 묶음. 배열은 부르는 쪽이 들고 있고 update 가 읽기만 한다.
struct AgentBatch {
    // batch x observationSize().
    const float* observations = nullptr;
    // batch x actionCount. **그때 실제로 한 행동** 이다.
    const float* actions = nullptr;
    // batch 개.
    const float* rewards = nullptr;
    // batch 개. 감가 마스크 γ(1 - 끝)이라 끝난 전이면 0 이다.
    const float* discounts = nullptr;
    const float* nextObservations = nullptr;
};

struct AgentUpdateStats {
    float criticLoss = 0.0F;
    // 액터가 본 min(Q1, Q2) 의 평균. 액터 손실은 이것의 음수다.
    float value = 0.0F;
};

// 파라미터·모멘트·활성을 들고 갱신을 돌린다. **build 뒤에는 힙 할당이 없다** — 배열을 한 번 잡고
// 되쓰므로 프레임 안에서 불러도 된다(편집기 안 학습이 그 경우다).
//
// 최적화기가 둘인 것이 요점이다. 크리틱 손실의 역전파는 액터 가중치에도 경사를 내지만 크리틱 구간만
// 밟고, 액터 손실은 크리틱 가중치에 경사를 내지만 액터 구간만 밟는다. PyTorch 로 치면 최적화기 둘을
// 따로 스텝하는 것과 같다.
class AgentTrainer {
public:
    // 그래프를 짓고 가중치를 초기화한다. 설정이 말이 안 되면 거짓이다. 두 번 불러도 되며, 그때는
    // 걸음 수와 모멘트가 함께 되돌아간다.
    //
    // **아래 것들은 build 가 참을 돌려준 뒤에만 부른다.** 그러지 않으면 빈 표를 훑는다 —
    // gfx::forward 가 «validate 를 통과한 표» 를 전제하는 것과 같은 규약이다.
    bool build(const AgentConfig& config, uint64_t seed);

    const Agent& graphs() const { return agent; }
    const AgentConfig& config() const { return agent.config; }
    // 지금까지 밟은 갱신 걸음 수. Adam 의 편향 보정이 이것을 쓴다.
    uint32_t steps() const { return step; }

    // 관측 하나로 행동 하나. 탐험 잡음은 부르는 쪽이 얹는다(학습이냐 평가냐를 여기서 정하지 않는다).
    void act(const float* observation, float* action);

    // 크리틱 한 걸음, 액터 한 걸음, 그리고 polyak. 순서가 계약이다 — 액터는 **이번 걸음에 갱신된**
    // 크리틱을 보고 오른다(DrQ-v2 와 같다).
    AgentUpdateStats update(const AgentBatch& batch, const AgentUpdateSettings& settings);

    std::vector<float>& weights() { return parameters; }
    const std::vector<float>& weights() const { return parameters; }

    bool save(const std::string& path) const;
    // 가중치를 읽어 들이고 **최적화기 상태를 되돌린다**(모멘트 0, 걸음 수 0). 파일에 담기는 것이
    // 가중치뿐이라 그대로 두면 읽어 들인 가중치와 상관 없는 모멘트로 첫 걸음을 밟는다.
    //
    // ponytail: 그래서 학습을 «이어서» 하지는 못한다. 체크포인트에서 다시 시작하면 Adam 이 처음부터
    // 감을 잡아야 한다. 이어 하려면 모멘트와 걸음 수도 파일에 담아야 한다.
    bool load(const std::string& path);

private:
    Agent agent;
    std::vector<float> parameters;
    // 역전파가 파라미터 배열 전체 크기로 쓴다. 두 손실이 번갈아 쓰므로 한 벌이면 된다.
    std::vector<float> gradients;
    std::vector<float> criticMoments;
    std::vector<float> actorMoments;
    std::vector<float> actActivations;
    std::vector<float> criticActivations;
    std::vector<float> criticActivationGradients;
    std::vector<float> actorActivations;
    std::vector<float> actorActivationGradients;
    uint32_t step = 0;
    uint64_t noiseStream = 0;
};

} // namespace gfx
