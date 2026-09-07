// DrQ-v2 식 DDPG/TD3 에이전트의 그래프 명세.
//
// neural_test 가 «연산 하나하나의 역전파» 를 보는 자리라면, 여기는 그 연산들을 엮어 만든 **손실 전체**
// 를 본다. 같은 유한차분 기준(gradient_check.h)을 쓰되 확인하는 것이 다르다:
//
//   - 크리틱·액터 손실의 경사가 유한차분과 맞는가
//   - **끊어 둔 자리의 경사가 정확히 0 인가** — 타깃망과, 액터 손실이 본 인코더·trunk
//   - 세 표(행동·크리틱·액터)가 파라미터를 같은 순서로 잡는가
//   - 파라미터 구간이 통짜라 Adam 과 polyak 이 «어디부터 몇 개» 만으로 돌아가는가
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

#include "gfx/neural_math.h"
#include "gfx/rl_agent.h"
#include "gradient_check.h"

namespace {

// 유한차분에 쓸 작은 판. 실제 크기(2.2M 파라미터)는 흔들어 볼 수 없으므로 **구조만 같고 작은** 설정을
// 쓴다. 16 -(3x3 s2)- 7 - 5 - 3 - 1 이라 합성곱 넷이 그대로 산다.
gfx::AgentConfig smallPixelConfig() {
    gfx::AgentConfig config;
    config.imageSize = 16;
    config.frameStack = 2;
    config.convChannels = 3;
    // 행동이 둘이어야 액터 머리의 전치가 드러난다.
    config.actionCount = 2;
    // 배치와 은닉을 너무 좁게 잡으면 크리틱 머리의 ReLU 층이 **통째로 죽는다**(은닉 5·배치 3 에서
    // 실제로 그랬다). 그러면 그 머리가 상수를 내뱉어 손실의 절반이 검사되지 않는데, 값만 봐서는
    // 멀쩡해 보인다. 8x4 면 여덟 유닛이 네 표본 모두에서 죽을 일이 사실상 없다.
    config.batch = 4;
    config.featureCount = 4;
    config.hidden = 8;
    return config;
}

gfx::AgentConfig smallVectorConfig() {
    gfx::AgentConfig config = smallPixelConfig();
    config.imageSize = 0;
    // 진자의 관측(sin, cos, 정규화한 각속도)과 같은 꼴이다.
    config.observationCount = 3;
    return config;
}

// 파라미터 구간 안에 드는 텐서들. 첨자를 손으로 적지 않고 배치에서 끌어낸다.
std::vector<uint32_t> tensorsIn(const gfx::Graph& graph, const gfx::ParameterRange& range) {
    std::vector<uint32_t> result;
    for (uint32_t tensor : graph.parameterTensors()) {
        uint32_t begin = graph.tensors[tensor].offset;
        if (begin >= range.begin && begin < range.end) {
            result.push_back(tensor);
        }
    }
    return result;
}

void append(std::vector<uint32_t>& into, const std::vector<uint32_t>& more) {
    into.insert(into.end(), more.begin(), more.end());
}

// 구간 안에서 크리틱 머리의 **마지막 편향** 둘을 찾는다. Q 를 통째로 올리고 내릴 수 있는 손잡이다.
//
// 원소가 하나인 파라미터 텐서는 그것뿐이라 배치 순서를 손으로 적지 않아도 된다(trunk 의 편향·이득은
// 특징 수, 은닉 편향은 은닉 폭이므로 모두 둘 이상이다).
std::vector<uint32_t> outputBiases(const gfx::Graph& graph, const gfx::ParameterRange& range) {
    std::vector<uint32_t> result;
    for (uint32_t tensor : tensorsIn(graph, range)) {
        if (graph.tensors[tensor].count() == 1) {
            result.push_back(tensor);
        }
    }
    return result;
}

// ---- 모양과 배치

// 논문 그대로의 설정에서 합성곱 사슬과 파라미터 총수를 못 박는다. 여기가 흔들리면 저장한 가중치가
// 통째로 못 쓰게 되므로, 숫자를 눈에 보이게 적어 둔다.
void testFullShapes() {
    gfx::AgentConfig config;
    config.actionCount = 6;
    gfx::Agent agent;
    assert(gfx::buildAgent(config, agent));

    // 84 -(3x3 보폭 2)- 41 -(보폭 1)- 39 - 37 - 35. 특징은 32 * 35 * 35 = 39,200.
    assert(gfx::convOutputSize(84, 3, 2, 0) == 41);
    assert(gfx::convOutputSize(41, 3, 1, 0) == 39);
    assert(gfx::convOutputSize(39, 3, 1, 0) == 37);
    assert(gfx::convOutputSize(37, 3, 1, 0) == 35);

    const gfx::AgentParameterMap& map = agent.parameters;
    // 인코더: (32,3,3,3)+32 그리고 (32,32,3,3)+32 셋.
    size_t encoder = (32 * 3 * 3 * 3 + 32) + 3 * (32 * 32 * 3 * 3 + 32);
    assert(map.encoder.count() == encoder);
    // trunk: (50,39200)+50 에 layernorm 의 이득·편향 50 씩.
    size_t trunk = 50 * 39200 + 50 + 50 + 50;
    assert(map.trunk.count() == trunk);
    // 크리틱 하나: (256,56)+256, (256,256)+256, (1,256)+1.
    size_t critic = (256 * 56 + 256) + (256 * 256 + 256) + (256 + 1);
    assert(map.online.count() == trunk + 2 * critic);
    assert(map.target.count() == map.online.count());
    // 액터: (256,50)+256, (256,256)+256, (6,256)+6.
    size_t actor = (256 * 50 + 256) + (256 * 256 + 256) + (6 * 256 + 6);
    assert(map.actorUpdate.count() == actor);
    assert(map.total == encoder + 2 * (trunk + 2 * critic) + actor);
    std::printf("  84x84 6행동 에이전트 파라미터 %zu 개 (%.1f MB)\n",
                map.total,
                static_cast<double>(map.total) * 4.0 / (1024.0 * 1024.0));
}

// 구간이 통짜로 이어지고 겹치지 않는지. Adam 두 벌과 polyak 이 이 배치를 그대로 믿는다.
void testParameterMap() {
    for (const gfx::AgentConfig& config : {smallPixelConfig(), smallVectorConfig()}) {
        gfx::Agent agent;
        assert(gfx::buildAgent(config, agent));
        const gfx::AgentParameterMap& map = agent.parameters;

        // 앞에서부터 빈틈없이 이어진다.
        assert(map.encoder.begin == 0);
        assert(map.encoder.end == map.online.begin);
        assert(map.trunk.begin == map.online.begin);
        assert(map.trunk.end <= map.online.end);
        assert(map.criticUpdate.begin == 0);
        assert(map.criticUpdate.end == map.online.end);
        assert(map.target.begin == map.online.end);
        assert(map.actorUpdate.begin == map.target.end);
        assert(map.actorUpdate.end == map.total);
        assert(map.total == agent.critic.graph.parameterCount);
        // polyak 은 두 구간을 나란히 훑는다. 길이가 같아야 한다.
        assert(map.online.count() == map.target.count());
        assert(map.online.count() > 0 && map.actorUpdate.count() > 0);
        // 저차원이면 인코더가 없다.
        assert(config.pixels() == (map.encoder.count() > 0));

        // 타깃 구간의 텐서 모양이 온라인 구간과 **같은 순서로 같아야** 한다. 어긋나면 polyak 이 엉뚱한
        // 가중치를 덮어쓰는데, 길이만 보면 알 수 없다.
        std::vector<uint32_t> online = tensorsIn(agent.critic.graph, map.online);
        std::vector<uint32_t> target = tensorsIn(agent.critic.graph, map.target);
        assert(online.size() == target.size());
        for (size_t i = 0; i < online.size(); ++i) {
            const gfx::Tensor& a = agent.critic.graph.tensors[online[i]];
            const gfx::Tensor& b = agent.critic.graph.tensors[target[i]];
            for (uint32_t axis = 0; axis < 4; ++axis) {
                assert(a.dims[axis] == b.dims[axis]);
            }
            assert(b.offset - a.offset == map.target.begin - map.online.begin);
            // 타깃은 경사를 아예 받지 않는다.
            assert(!b.receivesGradient());
            assert(a.receivesGradient());
        }
    }
}

// 세 표가 파라미터 배열 하나를 함께 본다. 배치가 어긋나면 buildAgent 가 거짓을 내야 한다.
void testGraphsShareParameters() {
    gfx::Agent agent;
    assert(gfx::buildAgent(smallPixelConfig(), agent));
    gfx::ParameterLayout layout = gfx::parameterLayout(agent.critic.graph);
    assert(gfx::parameterLayout(agent.act.graph) == layout);
    assert(gfx::parameterLayout(agent.actor.graph) == layout);

    // 그러나 **그래프는 서로 다르다.** 해시가 같으면 체크포인트가 엉뚱한 표에도 읽힌다.
    uint64_t criticHash = gfx::graphHash(agent.critic.graph);
    assert(gfx::graphHash(agent.act.graph) != criticHash);
    assert(gfx::graphHash(agent.actor.graph) != criticHash);

    assert(gfx::validateForward(agent.act.graph));
    assert(gfx::validate(agent.critic.graph));
    assert(gfx::validate(agent.actor.graph));
    // 행동 표는 손실이 없어 역전파의 씨앗을 심을 수 없다.
    assert(!gfx::validate(agent.act.graph));
}

void testRejectsBadConfig() {
    gfx::Agent agent;
    gfx::AgentConfig config = smallPixelConfig();
    // 합성곱 넷이 다 먹어 버리는 크기.
    config.imageSize = 8;
    assert(!gfx::buildAgent(config, agent));
    config = smallVectorConfig();
    config.observationCount = 0;
    assert(!gfx::buildAgent(config, agent));
    config = smallPixelConfig();
    config.actionCount = 0;
    assert(!gfx::buildAgent(config, agent));
    config = smallPixelConfig();
    config.batch = 0;
    assert(!gfx::buildAgent(config, agent));
    config = smallPixelConfig();
    config.frameStack = 0;
    assert(!gfx::buildAgent(config, agent));
    config = smallPixelConfig();
    config.featureCount = 0;
    assert(!gfx::buildAgent(config, agent));
    config = smallPixelConfig();
    config.hidden = 0;
    assert(!gfx::buildAgent(config, agent));
    config = smallPixelConfig();
    config.convChannels = 0;
    assert(!gfx::buildAgent(config, agent));
    // 멀쩡한 설정은 통과한다. 위가 «늘 거절» 이 아니라는 확인이다.
    assert(gfx::buildAgent(smallPixelConfig(), agent));
}

// ---- 초기화와 타깃

void testInitializeAndTarget() {
    gfx::Agent agent;
    assert(gfx::buildAgent(smallPixelConfig(), agent));
    std::vector<float> parameters(agent.parameters.total, 0.0F);
    gfx::initializeAgent(agent, 7, parameters.data());

    const gfx::AgentParameterMap& map = agent.parameters;
    // 타깃이 온라인과 비트까지 같아야 한다. 처음부터 어긋나면 크리틱이 자기 그림자를 쫓는다.
    for (size_t i = 0; i < map.online.count(); ++i) {
        assert(parameters[map.online.begin + i] == parameters[map.target.begin + i]);
    }
    // 밟지 않은 자리가 없어야 한다. 액터·타깃까지 채워야 학습이 0 에서 시작하지 않는다.
    size_t nonZero = 0;
    for (float value : parameters) {
        nonZero += value != 0.0F ? 1 : 0;
    }
    // 편향은 0 이 정상이므로 «전부» 는 아니지만, 구간마다 뭔가는 들어 있어야 한다.
    assert(nonZero > parameters.size() / 2);
    auto rangeMoved = [&](const gfx::ParameterRange& range) {
        for (size_t i = range.begin; i < range.end; ++i) {
            if (parameters[i] != 0.0F) {
                return true;
            }
        }
        return false;
    };
    assert(rangeMoved(map.encoder));
    assert(rangeMoved(map.trunk));
    assert(rangeMoved(map.online));
    assert(rangeMoved(map.target));
    assert(rangeMoved(map.actorUpdate));

    // 온라인을 흔들고 조금 끌어당기면 타깃이 그 사이로 간다.
    std::vector<float> before = parameters;
    for (size_t i = 0; i < map.online.count(); ++i) {
        parameters[map.online.begin + i] += 1.0F;
    }
    gfx::updateAgentTarget(agent, 0.25F, parameters.data());
    for (size_t i = 0; i < map.online.count(); ++i) {
        float expected = before[map.target.begin + i] + 0.25F * 1.0F;
        assert(std::abs(parameters[map.target.begin + i] - expected) < 1.0e-6F);
    }
    // 타깃을 끌어당겨도 온라인·액터는 그대로다.
    for (size_t i = map.actorUpdate.begin; i < map.actorUpdate.end; ++i) {
        assert(parameters[i] == before[i]);
    }
}

// ---- 순전파의 뜻

// 행동 표(배치 1)와 액터 표(배치 B)가 **같은 관측에 같은 행동** 을 내야 한다. 둘이 갈리면 학습한
// 정책이 재생에서 다르게 움직이는데, 손실만 봐서는 알 수 없다.
void testActMatchesActor() {
    gfx::AgentConfig config = smallPixelConfig();
    gfx::Agent agent;
    assert(gfx::buildAgent(config, agent));
    std::vector<float> parameters(agent.parameters.total, 0.0F);
    gfx::initializeAgent(agent, 11, parameters.data());
    // **타깃을 온라인에서 떼어 놓는다.** 초기화 직후에는 둘이 비트까지 같아, 행동 표가 온라인 대신
    // 타깃 trunk 를 보도록 망가져도 두 표가 같은 값을 낸다. 학습이 한 걸음이라도 지나면 갈리는 값이라
    // 여기서 미리 갈라 둔다.
    for (size_t i = agent.parameters.target.begin; i < agent.parameters.target.end; ++i) {
        parameters[i] += 0.5F + gfx::neuralGaussian(97, i);
    }

    std::vector<float> actActivations(agent.act.graph.activationCount, 0.0F);
    std::vector<float> actorActivations(agent.actor.graph.activationCount, 0.0F);
    uint32_t pixels = config.frameStack * config.imageSize * config.imageSize;
    float* actObservation = gfx::tensorValues(agent.act.graph, agent.act.observation, actActivations.data());
    float* actorObservation = gfx::tensorValues(agent.actor.graph, agent.actor.observation, actorActivations.data());
    for (uint32_t sample = 0; sample < config.batch; ++sample) {
        for (uint32_t i = 0; i < pixels; ++i) {
            // 표본마다 다른 그림을 넣는다. 그래야 «첫 줄만 보고 나머지를 무시하는» 배치 첨자 버그가
            // 드러난다.
            actorObservation[sample * pixels + i] = gfx::neuralGaussian(100 + sample, i);
        }
    }
    // 행동 표에는 **가운데 표본** 을 준다. 0 번을 주면 배치 첨자를 통째로 빼먹어도 맞는다.
    uint32_t chosen = config.batch / 2;
    for (uint32_t i = 0; i < pixels; ++i) {
        actObservation[i] = actorObservation[chosen * pixels + i];
    }

    gfx::forward(agent.act.graph, parameters.data(), actActivations.data());
    gfx::forward(agent.actor.graph, parameters.data(), actorActivations.data());
    const float* one = gfx::tensorValues(agent.act.graph, agent.act.action, actActivations.data());
    const float* many = gfx::tensorValues(agent.actor.graph, agent.actor.action, actorActivations.data());
    for (uint32_t i = 0; i < config.actionCount; ++i) {
        assert(std::abs(one[i] - many[chosen * config.actionCount + i]) < 1.0e-5F);
        // tanh 를 지났으므로 행동은 [-1, 1] 이다. physics::act 가 그 전제로 편다.
        assert(one[i] > -1.0F && one[i] < 1.0F);
    }
    // 다른 표본과는 달라야 한다(관측이 다르므로).
    bool differs = false;
    for (uint32_t sample = 0; sample < config.batch; ++sample) {
        for (uint32_t i = 0; sample != chosen && i < config.actionCount; ++i) {
            differs = differs || std::abs(many[sample * config.actionCount + i] - one[i]) > 1.0e-6F;
        }
    }
    assert(differs);
}

// 시간차 목표 y = r + γ(1-끝)·min(Q1', Q2') 의 배선. 값 자체는 망에 달렸지만 **마스크의 뜻** 은 못 박을
// 수 있다: 감가가 0 이면 y 는 보상 그대로여야 한다.
void testTargetValue() {
    gfx::AgentConfig config = smallPixelConfig();
    gfx::Agent agent;
    assert(gfx::buildAgent(config, agent));
    std::vector<float> parameters(agent.parameters.total, 0.0F);
    gfx::initializeAgent(agent, 13, parameters.data());
    std::vector<float> activations(agent.critic.graph.activationCount, 0.0F);

    const gfx::Graph& graph = agent.critic.graph;
    uint32_t pixels = config.frameStack * config.imageSize * config.imageSize;
    float* nextObservation = gfx::tensorValues(graph, agent.critic.nextObservation, activations.data());
    for (uint32_t i = 0; i < config.batch * pixels; ++i) {
        nextObservation[i] = gfx::neuralGaussian(211, i);
    }
    float* reward = gfx::tensorValues(graph, agent.critic.reward, activations.data());
    float* discount = gfx::tensorValues(graph, agent.critic.discount, activations.data());
    float* noise = gfx::tensorValues(graph, agent.critic.noise, activations.data());
    for (uint32_t sample = 0; sample < config.batch; ++sample) {
        reward[sample] = 0.5F * static_cast<float>(sample) - 1.0F;
        discount[sample] = 0.0F;
    }
    for (uint32_t i = 0; i < config.batch * config.actionCount; ++i) {
        noise[i] = 0.0F;
    }
    gfx::forward(graph, parameters.data(), activations.data());
    const float* value = gfx::tensorValues(graph, agent.critic.targetValue, activations.data());
    for (uint32_t sample = 0; sample < config.batch; ++sample) {
        // 끝난 전이(감가 0)는 «앞으로 받을 것이 없다» 는 뜻이라 목표가 보상 그대로다.
        assert(std::abs(value[sample] - reward[sample]) < 1.0e-6F);
    }

    // 감가를 켜면 목표가 보상에서 떨어진다.
    for (uint32_t sample = 0; sample < config.batch; ++sample) {
        discount[sample] = 0.99F;
    }
    gfx::forward(graph, parameters.data(), activations.data());
    std::vector<float> discounted(value, value + config.batch);
    bool moved = false;
    for (uint32_t sample = 0; sample < config.batch; ++sample) {
        moved = moved || std::abs(discounted[sample] - reward[sample]) > 1.0e-6F;
    }
    assert(moved);

    // 목표가 **쌍둥이 둘의 최소** 인지 자리마다 견준다.
    //
    // 무작위 초기화가 양쪽을 갈라 주기를 바라면 안 된다 — 머리의 마지막 층이 ReLU 뒤의 음이 아닌 값을
    // 받으므로 Q 의 크기를 표본 차이가 아니라 **출력 가중치 벡터**가 정하고, 그래서 한 쌍둥이가 배치
    // 전체에서 이기는 것이 보통이다(실측 1.24 차이). 그러면 min 을 «늘 첫째» 로 바꿔도 통과한다.
    // 그래서 진 쪽의 출력 편향을 직접 끌어내려 **승자를 바꿔 가며** 양쪽 갈래를 다 밟는다.
    std::vector<uint32_t> biases = outputBiases(graph, agent.parameters.target);
    assert(biases.size() == 2);
    float* firstBias = parameters.data() + graph.tensors[biases[0]].offset;
    float authored = *firstBias;
    // 첫째 쌍둥이의 출력 편향을 크게 내렸다 올려 **승자를 양쪽으로 정해 놓고** 본다. 무작위 초기화가
    // 갈라 주기를 바라면 안 된다 — 머리의 마지막 층이 ReLU 뒤의 음이 아닌 값을 받으므로 Q 의 크기를
    // 표본 차이가 아니라 출력 가중치 벡터가 정하고, 그래서 한 쌍둥이가 배치 전체에서 이기는 것이
    // 보통이다(실측 1.24 차이). 그러면 min 을 «늘 첫째» 로 바꿔도 통과한다.
    auto expectWinner = [&](bool wantFirst) {
        *firstBias = authored + (wantFirst ? -20.0F : 20.0F);
        gfx::forward(graph, parameters.data(), activations.data());
        const float* twin1 = gfx::tensorValues(graph, agent.critic.targetTwin1, activations.data());
        const float* twin2 = gfx::tensorValues(graph, agent.critic.targetTwin2, activations.data());
        const float* result = gfx::tensorValues(graph, agent.critic.targetValue, activations.data());
        for (uint32_t sample = 0; sample < config.batch; ++sample) {
            assert(wantFirst == (twin1[sample] < twin2[sample]));
            float expected = reward[sample] + discount[sample] * std::min(twin1[sample], twin2[sample]);
            assert(std::abs(result[sample] - expected) < 1.0e-5F);
        }
    };
    expectWinner(true);
    expectWinner(false);
    *firstBias = authored;

    // 잡음이 실제로 타깃 행동에 실리는지. 실리지 않으면 평활화가 죽은 코드가 된다.
    for (uint32_t i = 0; i < config.batch * config.actionCount; ++i) {
        noise[i] = 1.5F * gfx::neuralGaussian(307, i);
    }
    gfx::forward(graph, parameters.data(), activations.data());
    moved = false;
    for (uint32_t sample = 0; sample < config.batch; ++sample) {
        moved = moved || std::abs(discounted[sample] - value[sample]) > 1.0e-6F;
    }
    assert(moved);

    // **어느 trunk 가 어디에 쓰이는지** 를 가른다. 초기화 직후에는 온라인과 타깃이 비트까지 같아 둘을
    // 맞바꿔도 값이 같으므로, 한쪽씩 흔들어 y 가 따라 움직이는지 본다.
    //
    //   - 타깃 trunk 는 **크리틱** 이 본다: 흔들면 Q' 가 바뀌어 y 가 움직인다.
    //   - 온라인 trunk 는 **액터** 가 본다: 흔들면 a' 가 바뀌어 y 가 움직인다(DrQ-v2 규약).
    //
    // 둘 중 하나만 확인하면 둘을 맞바꾼 배선이 그대로 지나간다.
    std::vector<float> baseline(value, value + config.batch);
    auto shakeMovesTarget = [&](const gfx::ParameterRange& range) {
        std::vector<float> saved(parameters.begin() + static_cast<ptrdiff_t>(range.begin),
                                 parameters.begin() + static_cast<ptrdiff_t>(range.end));
        for (size_t i = range.begin; i < range.end; ++i) {
            parameters[i] += 0.25F;
        }
        gfx::forward(graph, parameters.data(), activations.data());
        bool changed = false;
        for (uint32_t sample = 0; sample < config.batch; ++sample) {
            changed = changed || std::abs(value[sample] - baseline[sample]) > 1.0e-5F;
        }
        std::copy(saved.begin(), saved.end(), parameters.begin() + static_cast<ptrdiff_t>(range.begin));
        return changed;
    };
    gfx::ParameterRange targetTrunk{agent.parameters.target.begin,
                                    agent.parameters.target.begin + agent.parameters.trunk.count()};
    assert(shakeMovesTarget(targetTrunk));
    assert(shakeMovesTarget(agent.parameters.trunk));
}

// ---- 경사

// 손실 그래프 하나를 유한차분으로 본다. frozen 에 적은 구간은 «해석 경사가 0» 인지만 확인한다.
double checkLoss(const gfx::Graph& graph,
                 const std::vector<uint32_t>& inputs,
                 const std::vector<uint32_t>& frozen,
                 const std::vector<uint32_t>& partial,
                 const std::vector<std::pair<size_t, float>>& nudges,
                 uint64_t seed) {
    GradientCheck test;
    test.graph = graph;
    test.inputs = inputs;
    test.frozen = frozen;
    test.partial = partial;
    test.nudges = nudges;
    test.allocate();
    return test.check(seed, 48);
}

// 크리틱 손실. 끊은 자리는 타깃 구간 전체와 **액터** 다 — 액터는 타깃 행동을 낼 때만 쓰이는데 그
// 구간이 통째로 끊겨 있으므로 경사가 나오면 안 된다.
//
// **인코더와 온라인 trunk 는 여기서 흔들 수 없다.** 둘 다 온라인 갈래와 타깃 갈래를 함께 지나는데
// 타깃 쪽만 끊겨 있어, 해석 경사는 한쪽만 세고 유한차분은 양쪽을 함께 잰다. 그것이 stop-gradient 의 정의라 «어긋나는
// 것이 정상» 이다. 그 둘은 아래 checkCriticTerminal 이 따로 본다.
double checkCritic(const gfx::Agent& agent, uint64_t seed) {
    const gfx::AgentCriticGraph& critic = agent.critic;
    std::vector<uint32_t> frozen = tensorsIn(critic.graph, agent.parameters.target);
    append(frozen, tensorsIn(critic.graph, agent.parameters.actorUpdate));
    // 인코더와 온라인 trunk 는 온라인 갈래로 경사를 받고 타깃 갈래로는 받지 않는다.
    std::vector<uint32_t> partial = tensorsIn(critic.graph, agent.parameters.encoder);
    append(partial, tensorsIn(critic.graph, agent.parameters.trunk));
    // discount 를 빼먹으면 0 으로 남아 y = r 이 되고, 타깃 갈래가 통째로 손실에 닿지 않는다. 그러면
    // «다음 관측을 흔들어도 손실이 안 움직인다» 로 걸린다 — 입력을 모두 적는 것이 이 검사의 전제다.
    return checkLoss(
        critic.graph,
        {critic.observation, critic.nextObservation, critic.action, critic.reward, critic.discount, critic.noise},
        frozen,
        partial,
        {},
        seed);
}

// **끝난 전이만 담긴 배치.** 감가를 입력 목록에서 빼면 0 으로 남아 y = r 이 되고, 타깃 갈래가 손실에
// 아예 닿지 않는다. 그러면 인코더와 온라인 trunk 가 손실에 미치는 길이 온라인 갈래 하나뿐이 되어
// 유한차분이 다시 진리가 된다. 이 판이 없으면 «합성곱 -> ReLU -> 펴기 -> trunk -> 크리틱» 배선이
// 통째로 검사되지 않는다.
double checkCriticTerminal(const gfx::Agent& agent, uint64_t seed) {
    const gfx::AgentCriticGraph& critic = agent.critic;
    std::vector<uint32_t> frozen = tensorsIn(critic.graph, agent.parameters.target);
    append(frozen, tensorsIn(critic.graph, agent.parameters.actorUpdate));
    return checkLoss(critic.graph, {critic.observation, critic.action, critic.reward}, frozen, {}, {}, seed);
}

// 액터 손실. 끊은 자리는 인코더·trunk(액터가 표현을 끌고 가지 않는다)와 타깃 구간이다. 온라인 크리틱은
// 경사를 **받는다** — 버리는 것은 부르는 쪽이지 표가 아니다.
// 손실이 min(Q1, Q2) 를 지나므로 **진 쪽 크리틱은 경사가 0 이다.** 그것이 MIN2 의 정의라, 어느 쪽이
// 이길지를 정해 놓고 두 번 본다: 이긴 쪽은 유한차분과 맞아야 하고 진 쪽은 정확히 0 이어야 한다.
// 무작위 초기화에 맡기면 배치 전체가 한쪽으로 쏠려(머리의 마지막 층이 ReLU 뒤의 음이 아닌 값을 받아
// Q 의 크기를 표본 차이가 아니라 출력 가중치가 정한다) 어느 쪽이 검사되는지가 씨앗에 달리게 된다.
double checkActor(const gfx::Agent& agent, uint64_t seed, bool firstWins) {
    const gfx::AgentActorGraph& actor = agent.actor;
    std::vector<uint32_t> frozen = tensorsIn(actor.graph, agent.parameters.encoder);
    append(frozen, tensorsIn(actor.graph, agent.parameters.trunk));
    append(frozen, tensorsIn(actor.graph, agent.parameters.target));

    // 온라인 구간에서 trunk 를 뺀 나머지가 크리틱 머리 둘이다. 진 쪽 머리 전체를 frozen 으로 둔다.
    gfx::ParameterRange heads{agent.parameters.trunk.end, agent.parameters.online.end};
    std::vector<uint32_t> headTensors = tensorsIn(actor.graph, heads);
    assert(headTensors.size() == 2 * 2 * gfx::AGENT_HEAD_LAYERS);
    size_t half = headTensors.size() / 2;
    for (size_t i = 0; i < half; ++i) {
        frozen.push_back(firstWins ? headTensors[half + i] : headTensors[i]);
    }
    // 이길 쪽의 출력 편향을 크게 내려 min 이 늘 그쪽을 고르게 한다.
    std::vector<uint32_t> biases = outputBiases(actor.graph, agent.parameters.online);
    assert(biases.size() == 2);
    std::vector<std::pair<size_t, float>> nudges{{actor.graph.tensors[biases[firstWins ? 0 : 1]].offset, -20.0F}};
    return checkLoss(actor.graph, {actor.observation}, frozen, {}, nudges, seed);
}

void testGradients() {
    struct Case {
        const char* name;
        double error;
    };
    gfx::Agent pixel;
    gfx::Agent vector;
    assert(gfx::buildAgent(smallPixelConfig(), pixel));
    assert(gfx::buildAgent(smallVectorConfig(), vector));
    Case cases[] = {
        {"크리틱(픽셀)", checkCritic(pixel, 17)},
        {"끝난 전이(픽셀)", checkCriticTerminal(pixel, 19)},
        {"액터(픽셀) Q1", checkActor(pixel, 23, true)},
        {"액터(픽셀) Q2", checkActor(pixel, 23, false)},
        {"크리틱(저차원)", checkCritic(vector, 29)},
        {"끝난 전이(저차원)", checkCriticTerminal(vector, 31)},
        {"액터(저차원) Q1", checkActor(vector, 37, true)},
        {"액터(저차원) Q2", checkActor(vector, 37, false)},
    };
    std::printf("  손실별 최대 상대 오차 (해석 경사 대 중앙 유한차분)\n");
    for (const Case& item : cases) {
        std::printf("    %-16s %.3e\n", item.name, item.error);
    }
    std::fflush(stdout);
    for (const Case& item : cases) {
        assert(item.error < 1.0e-4);
    }
}

// stop-gradient 가 «영향이 없어서 0» 이 아니라 «영향이 있는데도 0» 인지. 타깃 크리틱의 가중치는 손실을
// 실제로 움직이지만 경사는 정확히 0 이어야 한다. 이것을 보지 않으면 타깃 갈래가 통째로 죽은 코드여도
// 위의 frozen 검사가 통과한다.
void testStopGradientIsReal() {
    gfx::Agent agent;
    assert(gfx::buildAgent(smallPixelConfig(), agent));
    const gfx::Graph& graph = agent.critic.graph;
    std::vector<float> parameters(agent.parameters.total, 0.0F);
    gfx::initializeAgent(agent, 53, parameters.data());
    std::vector<float> activations(graph.activationCount, 0.0F);
    std::vector<float> parameterGradients(agent.parameters.total, 0.0F);
    std::vector<float> activationGradients(graph.activationCount, 0.0F);

    auto fillInput = [&](uint32_t tensor, uint64_t stream) {
        float* values = gfx::tensorValues(graph, tensor, activations.data());
        for (uint32_t i = 0; i < graph.tensors[tensor].count(); ++i) {
            values[i] = gfx::neuralGaussian(stream, i);
        }
    };
    fillInput(agent.critic.observation, 601);
    fillInput(agent.critic.nextObservation, 607);
    fillInput(agent.critic.action, 613);
    fillInput(agent.critic.reward, 617);
    fillInput(agent.critic.noise, 619);
    float* discount = gfx::tensorValues(graph, agent.critic.discount, activations.data());
    for (uint32_t i = 0; i < agent.config.batch; ++i) {
        discount[i] = 0.99F;
    }

    gfx::forward(graph, parameters.data(), activations.data());
    assert(gfx::backward(
        graph, parameters.data(), activations.data(), parameterGradients.data(), activationGradients.data()));
    float before = activations[graph.tensors[agent.critic.loss].offset];
    assert(before > 0.0F);

    // 타깃 구간과 액터 구간의 경사는 한 자리도 남김없이 0 이다.
    for (size_t i = agent.parameters.target.begin; i < agent.parameters.total; ++i) {
        assert(parameterGradients[i] == 0.0F);
    }
    // 그런데 그 자리를 흔들면 손실이 움직인다. 두 문장이 함께여야 «끊었다» 는 말이 뜻을 갖는다.
    //
    // 구간을 **통째로** 흔든다. 자리 하나만 집으면 하필 죽은 ReLU 뒤라 아무 것도 안 움직일 수 있고
    // (작은 판에서는 흔한 일이다) 그러면 검사가 조용히 헛돈다.
    auto shakeMovesLoss = [&](const gfx::ParameterRange& range) {
        std::vector<float> saved(parameters.begin() + static_cast<ptrdiff_t>(range.begin),
                                 parameters.begin() + static_cast<ptrdiff_t>(range.end));
        for (size_t i = range.begin; i < range.end; ++i) {
            parameters[i] += 0.25F;
        }
        gfx::forward(graph, parameters.data(), activations.data());
        float moved = std::abs(activations[graph.tensors[agent.critic.loss].offset] - before);
        std::copy(saved.begin(), saved.end(), parameters.begin() + static_cast<ptrdiff_t>(range.begin));
        return moved > 1.0e-5F;
    };
    assert(shakeMovesLoss(agent.parameters.target));
    assert(shakeMovesLoss(agent.parameters.actorUpdate));
}

// 액터 손실이 온라인 크리틱에는 **경사를 낸다**. 위 frozen 목록이 «전부 0» 이라 통과하는 것이 아님을
// 여기서 못 박는다.
void testActorTouchesCritic() {
    gfx::Agent agent;
    assert(gfx::buildAgent(smallPixelConfig(), agent));
    std::vector<float> parameters(agent.parameters.total, 0.0F);
    gfx::initializeAgent(agent, 41, parameters.data());
    std::vector<float> activations(agent.actor.graph.activationCount, 0.0F);
    std::vector<float> parameterGradients(agent.parameters.total, 0.0F);
    std::vector<float> activationGradients(agent.actor.graph.activationCount, 0.0F);

    const gfx::Graph& graph = agent.actor.graph;
    float* observation = gfx::tensorValues(graph, agent.actor.observation, activations.data());
    uint32_t pixels = agent.config.frameStack * agent.config.imageSize * agent.config.imageSize;
    for (uint32_t i = 0; i < agent.config.batch * pixels; ++i) {
        observation[i] = gfx::neuralGaussian(419, i);
    }
    gfx::forward(graph, parameters.data(), activations.data());
    // 액터가 보는 가치도 쌍둥이 둘의 최소다. 위와 같은 이유로 승자를 직접 뒤집어 양쪽을 밟는다.
    std::vector<uint32_t> biases = outputBiases(graph, agent.parameters.online);
    assert(biases.size() == 2);
    float* firstBias = parameters.data() + graph.tensors[biases[0]].offset;
    float authored = *firstBias;
    auto expectWinner = [&](bool wantFirst) {
        *firstBias = authored + (wantFirst ? -20.0F : 20.0F);
        gfx::forward(graph, parameters.data(), activations.data());
        const float* q1 = gfx::tensorValues(graph, agent.actor.q1, activations.data());
        const float* q2 = gfx::tensorValues(graph, agent.actor.q2, activations.data());
        const float* value = gfx::tensorValues(graph, agent.actor.value, activations.data());
        for (uint32_t sample = 0; sample < agent.config.batch; ++sample) {
            assert(wantFirst == (q1[sample] < q2[sample]));
            assert(value[sample] == std::min(q1[sample], q2[sample]));
        }
    };
    expectWinner(true);
    expectWinner(false);
    // 아래 경사 검사는 흔들지 않은 원래 가중치로 본다.
    *firstBias = authored;
    gfx::forward(graph, parameters.data(), activations.data());
    assert(gfx::backward(
        graph, parameters.data(), activations.data(), parameterGradients.data(), activationGradients.data()));

    auto moved = [&](const gfx::ParameterRange& range) {
        for (size_t i = range.begin; i < range.end; ++i) {
            if (parameterGradients[i] != 0.0F) {
                return true;
            }
        }
        return false;
    };
    assert(moved(agent.parameters.actorUpdate));
    // 크리틱 머리(온라인 구간에서 trunk 를 뺀 나머지)는 경사를 받는다.
    gfx::ParameterRange heads{agent.parameters.trunk.end, agent.parameters.online.end};
    assert(moved(heads));
    // 인코더·trunk·타깃은 받지 않는다.
    assert(!moved(agent.parameters.encoder));
    assert(!moved(agent.parameters.trunk));
    assert(!moved(agent.parameters.target));
}

// 최적화가 구간으로 갈리는지. 크리틱 걸음이 액터를 건드리면 «둘이 서로를 쫓는» 학습이 된다.
void testUpdateSlices() {
    gfx::Agent agent;
    assert(gfx::buildAgent(smallPixelConfig(), agent));
    const gfx::AgentParameterMap& map = agent.parameters;
    std::vector<float> parameters(map.total, 0.0F);
    gfx::initializeAgent(agent, 43, parameters.data());
    std::vector<float> gradients(map.total, 0.0F);
    for (size_t i = 0; i < gradients.size(); ++i) {
        gradients[i] = 0.5F + 0.001F * static_cast<float>(i % 7);
    }
    std::vector<float> before = parameters;

    gfx::AdamSettings settings;
    std::vector<float> criticMoments(gfx::adamMomentCount(map.criticUpdate.count()), 0.0F);
    gfx::adamStep(settings,
                  1,
                  map.criticUpdate.count(),
                  gradients.data() + map.criticUpdate.begin,
                  criticMoments.data(),
                  parameters.data() + map.criticUpdate.begin);
    for (size_t i = map.criticUpdate.begin; i < map.criticUpdate.end; ++i) {
        assert(parameters[i] != before[i]);
    }
    // 타깃과 액터는 한 자리도 움직이지 않는다.
    for (size_t i = map.target.begin; i < map.total; ++i) {
        assert(parameters[i] == before[i]);
    }

    before = parameters;
    std::vector<float> actorMoments(gfx::adamMomentCount(map.actorUpdate.count()), 0.0F);
    gfx::adamStep(settings,
                  1,
                  map.actorUpdate.count(),
                  gradients.data() + map.actorUpdate.begin,
                  actorMoments.data(),
                  parameters.data() + map.actorUpdate.begin);
    for (size_t i = 0; i < map.actorUpdate.begin; ++i) {
        assert(parameters[i] == before[i]);
    }
    for (size_t i = map.actorUpdate.begin; i < map.total; ++i) {
        assert(parameters[i] != before[i]);
    }
}

// ---- 손실이 «어느 쪽» 인가
//
// 유한차분은 «적어 둔 손실의 경사가 맞는가» 만 본다. 손실 자체의 부호가 뒤집혀도(액터가 가치를 **낮추는**
// 방향으로 배우도록) 경사는 그 손실에 대해 여전히 정확하므로 통과한다. 그래서 걸음을 실제로 밟아 값이
// 어느 쪽으로 가는지를 따로 본다.
//
// 저차원 판을 쓴다. 인코더가 없어 크리틱을 밟는 동안 목표 y 가 **한 자리도 움직이지 않으므로**(타깃
// trunk·타깃 크리틱·액터가 모두 크리틱 구간 밖이다) 손실이 줄어야 한다는 말이 정확해진다.
void fillInputs(const gfx::Graph& graph, const std::vector<uint32_t>& tensors, float* activations, uint64_t seed) {
    uint64_t stream = seed;
    for (uint32_t tensor : tensors) {
        float* values = gfx::tensorValues(graph, tensor, activations);
        for (uint32_t i = 0; i < graph.tensors[tensor].count(); ++i) {
            values[i] = gfx::neuralGaussian(stream, i);
        }
        ++stream;
    }
}

void testCriticLossFalls() {
    gfx::Agent agent;
    assert(gfx::buildAgent(smallVectorConfig(), agent));
    const gfx::Graph& graph = agent.critic.graph;
    const gfx::ParameterRange& range = agent.parameters.criticUpdate;
    std::vector<float> parameters(agent.parameters.total, 0.0F);
    gfx::initializeAgent(agent, 61, parameters.data());
    std::vector<float> activations(graph.activationCount, 0.0F);
    std::vector<float> parameterGradients(agent.parameters.total, 0.0F);
    std::vector<float> activationGradients(graph.activationCount, 0.0F);
    fillInputs(graph,
               {agent.critic.observation,
                agent.critic.nextObservation,
                agent.critic.action,
                agent.critic.reward,
                agent.critic.noise},
               activations.data(),
               701);
    float* discount = gfx::tensorValues(graph, agent.critic.discount, activations.data());
    for (uint32_t i = 0; i < agent.config.batch; ++i) {
        discount[i] = 0.99F;
    }

    gfx::AdamSettings settings;
    settings.learningRate = 1.0e-2F;
    std::vector<float> moments(gfx::adamMomentCount(range.count()), 0.0F);
    float first = 0.0F;
    float last = 0.0F;
    float targetBefore = 0.0F;
    for (uint32_t step = 1; step <= 40; ++step) {
        gfx::forward(graph, parameters.data(), activations.data());
        float loss = activations[graph.tensors[agent.critic.loss].offset];
        if (step == 1) {
            first = loss;
            targetBefore = activations[graph.tensors[agent.critic.targetValue].offset];
        }
        last = loss;
        assert(gfx::backward(
            graph, parameters.data(), activations.data(), parameterGradients.data(), activationGradients.data()));
        gfx::adamStep(settings,
                      step,
                      range.count(),
                      parameterGradients.data() + range.begin,
                      moments.data(),
                      parameters.data() + range.begin);
    }
    float drift = std::abs(activations[graph.tensors[agent.critic.targetValue].offset] - targetBefore);
    std::printf("  고정 배치 40 걸음: 크리틱 손실 %.4f -> %.4f (목표 이동 %.5f)\n", first, last, drift);
    assert(last < first * 0.5F);
}

void testActorRaisesValue() {
    gfx::Agent agent;
    assert(gfx::buildAgent(smallVectorConfig(), agent));
    const gfx::Graph& graph = agent.actor.graph;
    const gfx::ParameterRange& range = agent.parameters.actorUpdate;
    std::vector<float> parameters(agent.parameters.total, 0.0F);
    gfx::initializeAgent(agent, 67, parameters.data());
    std::vector<float> activations(graph.activationCount, 0.0F);
    std::vector<float> parameterGradients(agent.parameters.total, 0.0F);
    std::vector<float> activationGradients(graph.activationCount, 0.0F);
    fillInputs(graph, {agent.actor.observation}, activations.data(), 809);

    auto meanValue = [&]() {
        const float* value = gfx::tensorValues(graph, agent.actor.value, activations.data());
        float sum = 0.0F;
        for (uint32_t i = 0; i < agent.config.batch; ++i) {
            sum += value[i];
        }
        return sum / static_cast<float>(agent.config.batch);
    };

    gfx::AdamSettings settings;
    settings.learningRate = 1.0e-2F;
    std::vector<float> moments(gfx::adamMomentCount(range.count()), 0.0F);
    float first = 0.0F;
    float last = 0.0F;
    for (uint32_t step = 1; step <= 40; ++step) {
        gfx::forward(graph, parameters.data(), activations.data());
        if (step == 1) {
            first = meanValue();
            // 손실은 그 평균의 음수여야 한다. 부호가 뒤집히면 아래 걸음이 가치를 깎는다.
            assert(std::abs(activations[graph.tensors[agent.actor.loss].offset] + first) < 1.0e-5F);
        }
        last = meanValue();
        assert(gfx::backward(
            graph, parameters.data(), activations.data(), parameterGradients.data(), activationGradients.data()));
        gfx::adamStep(settings,
                      step,
                      range.count(),
                      parameterGradients.data() + range.begin,
                      moments.data(),
                      parameters.data() + range.begin);
    }
    std::printf("  고정 배치 40 걸음: 액터가 보는 가치 %.4f -> %.4f\n", first, last);
    // 액터는 크리틱이 높게 치는 행동을 찾아가야 한다.
    assert(last > first + 0.05F);
}

// ---- 저장

void testCheckpoint() {
    gfx::Agent agent;
    assert(gfx::buildAgent(smallPixelConfig(), agent));
    std::vector<float> parameters(agent.parameters.total, 0.0F);
    gfx::initializeAgent(agent, 47, parameters.data());
    const std::string path = "rl_agent_parameters.json";
    assert(gfx::saveAgent(agent, parameters.data(), path));

    std::vector<float> restored(agent.parameters.total, 1.0F);
    assert(gfx::loadAgent(agent, restored.data(), path));
    assert(restored == parameters);

    // 모양이 다른 에이전트에는 읽히지 않는다.
    gfx::AgentConfig other = smallPixelConfig();
    other.hidden = 6;
    gfx::Agent otherAgent;
    assert(gfx::buildAgent(other, otherAgent));
    std::vector<float> otherParameters(otherAgent.parameters.total, 2.0F);
    std::vector<float> untouched = otherParameters;
    assert(!gfx::loadAgent(otherAgent, otherParameters.data(), path));
    assert(otherParameters == untouched);

    // 배치만 다른 에이전트도 다른 그래프다. 파라미터 배치는 똑같아 모양만 견주면 통과한다.
    gfx::AgentConfig wider = smallPixelConfig();
    wider.batch = smallPixelConfig().batch + 1;
    gfx::Agent widerAgent;
    assert(gfx::buildAgent(wider, widerAgent));
    assert(gfx::parameterLayout(widerAgent.critic.graph) == gfx::parameterLayout(agent.critic.graph));
    std::vector<float> widerParameters(widerAgent.parameters.total, 3.0F);
    assert(!gfx::loadAgent(widerAgent, widerParameters.data(), path));

    std::remove(path.c_str());
}

} // namespace

int main() {
    testFullShapes();
    testParameterMap();
    testGraphsShareParameters();
    testRejectsBadConfig();
    testInitializeAndTarget();
    testActMatchesActor();
    testTargetValue();
    testStopGradientIsReal();
    testActorTouchesCritic();
    testUpdateSlices();
    testCriticLossFalls();
    testActorRaisesValue();
    testCheckpoint();
    testGradients();
    std::printf("에이전트 그래프 테스트 통과\n");
    return 0;
}
