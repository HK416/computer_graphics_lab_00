#include "gfx/rl_agent.h"

#include <cstddef>
#include <cstdint>
#include <utility>

#include "gfx/neural_math.h"

namespace gfx {

namespace {

// 그래프 셋이 공유하는 파라미터 텐서 번호. **세 표가 이것을 같은 순서로 잡으므로** 번호도 같다
// (addParameter 가 표에 이어 붙이기만 하기 때문이다). 그래서 parameterLayout 이 셋 다 같다.
struct AgentParameterIds {
    uint32_t convWeight[AGENT_CONV_LAYERS] = {};
    uint32_t convBias[AGENT_CONV_LAYERS] = {};

    struct Trunk {
        uint32_t weight = NO_TENSOR;
        uint32_t bias = NO_TENSOR;
        uint32_t gain = NO_TENSOR;
        uint32_t shift = NO_TENSOR;
    };
    struct Head {
        uint32_t weight[AGENT_HEAD_LAYERS] = {};
        uint32_t bias[AGENT_HEAD_LAYERS] = {};
    };

    Trunk onlineTrunk;
    Trunk targetTrunk;
    Head onlineCritic[2];
    Head targetCritic[2];
    Head actor;
};

// 인코더가 내는 특징 수. 픽셀이면 마지막 합성곱의 C·H·W, 저차원이면 관측 수 그대로다. 설정이 말이
// 안 되면(합성곱이 이미지를 다 먹으면) 0 이다.
uint32_t representationCount(const AgentConfig& config) {
    if (!config.pixels()) {
        return config.observationCount;
    }
    if (config.frameStack == 0 || config.convChannels == 0) {
        return 0;
    }
    uint32_t size = config.imageSize;
    for (uint32_t layer = 0; layer < AGENT_CONV_LAYERS; ++layer) {
        size = convOutputSize(size, 3, layer == 0 ? 2 : 1, 0);
        if (size == 0) {
            return 0;
        }
    }
    return config.convChannels * size * size;
}

AgentParameterIds::Trunk allocateTrunk(GraphBuilder& builder, const AgentConfig& config, uint32_t inputs) {
    AgentParameterIds::Trunk trunk;
    trunk.weight = builder.addParameter(config.featureCount, inputs, 1, 1);
    trunk.bias = builder.addParameter(config.featureCount, 1, 1, 1);
    trunk.gain = builder.addParameter(config.featureCount, 1, 1, 1);
    trunk.shift = builder.addParameter(config.featureCount, 1, 1, 1);
    return trunk;
}

// 은닉 둘 + 출력 하나. 크리틱은 (특징 + 행동) 을 받아 1 을 내고, 액터는 특징을 받아 행동 수를 낸다.
AgentParameterIds::Head
allocateHead(GraphBuilder& builder, const AgentConfig& config, uint32_t inputs, uint32_t outputs) {
    static_assert(AGENT_HEAD_LAYERS == 3, "아래 폭 표가 층 수와 함께 바뀌어야 한다");
    AgentParameterIds::Head head;
    uint32_t widths[AGENT_HEAD_LAYERS] = {config.hidden, config.hidden, outputs};
    uint32_t previous = inputs;
    for (uint32_t layer = 0; layer < AGENT_HEAD_LAYERS; ++layer) {
        head.weight[layer] = builder.addParameter(widths[layer], previous, 1, 1);
        head.bias[layer] = builder.addParameter(widths[layer], 1, 1, 1);
        previous = widths[layer];
    }
    return head;
}

// 파라미터를 **표마다 똑같은 순서로** 잡는다. 이 함수가 세 표의 계약이다 — 순서를 바꾸면 배열 하나를
// 함께 보는 전제가 무너지고, 저장한 가중치도 엉뚱한 자리에 들어간다.
AgentParameterIds
allocateParameters(GraphBuilder& builder, const AgentConfig& config, uint32_t representation, AgentParameterMap& map) {
    AgentParameterIds ids;
    map.criticUpdate.begin = 0;
    map.encoder.begin = 0;

    for (uint32_t layer = 0; layer < AGENT_CONV_LAYERS; ++layer) {
        ids.convWeight[layer] = NO_TENSOR;
        ids.convBias[layer] = NO_TENSOR;
    }
    if (config.pixels()) {
        uint32_t inputChannels = config.frameStack;
        for (uint32_t layer = 0; layer < AGENT_CONV_LAYERS; ++layer) {
            ids.convWeight[layer] = builder.addParameter(config.convChannels, inputChannels, 3, 3);
            ids.convBias[layer] = builder.addParameter(config.convChannels, 1, 1, 1);
            inputChannels = config.convChannels;
        }
    }

    map.encoder.end = builder.parameterCount();

    map.online.begin = builder.parameterCount();
    map.trunk.begin = builder.parameterCount();
    ids.onlineTrunk = allocateTrunk(builder, config, representation);
    map.trunk.end = builder.parameterCount();
    uint32_t criticInputs = config.featureCount + config.actionCount;
    for (uint32_t twin = 0; twin < 2; ++twin) {
        ids.onlineCritic[twin] = allocateHead(builder, config, criticInputs, 1);
    }
    map.online.end = builder.parameterCount();
    map.criticUpdate.end = builder.parameterCount();

    // 타깃은 온라인과 **완전히 같은 순서** 로 잡는다. 그래야 polyak 이 두 구간을 나란히 훑는 것으로
    // 끝난다. 지어 두고 detach 해 경사를 아예 받지 않게 한다.
    map.target.begin = builder.parameterCount();
    ids.targetTrunk = allocateTrunk(builder, config, representation);
    for (uint32_t twin = 0; twin < 2; ++twin) {
        ids.targetCritic[twin] = allocateHead(builder, config, criticInputs, 1);
    }
    map.target.end = builder.parameterCount();

    map.actorUpdate.begin = builder.parameterCount();
    ids.actor = allocateHead(builder, config, config.featureCount, config.actionCount);
    map.actorUpdate.end = builder.parameterCount();
    map.total = builder.parameterCount();

    builder.detach(ids.targetTrunk.weight);
    builder.detach(ids.targetTrunk.bias);
    builder.detach(ids.targetTrunk.gain);
    builder.detach(ids.targetTrunk.shift);
    for (uint32_t twin = 0; twin < 2; ++twin) {
        for (uint32_t layer = 0; layer < AGENT_HEAD_LAYERS; ++layer) {
            builder.detach(ids.targetCritic[twin].weight[layer]);
            builder.detach(ids.targetCritic[twin].bias[layer]);
        }
    }
    return ids;
}

// 관측 텐서를 잡는다. 픽셀이면 (배치, 프레임 스택, 변, 변), 저차원이면 (배치, 관측 수) 다.
uint32_t addObservationInput(GraphBuilder& builder, const AgentConfig& config, uint32_t batch) {
    if (config.pixels()) {
        return builder.addInput(batch, config.frameStack, config.imageSize, config.imageSize);
    }
    return builder.addInput(batch, config.observationCount, 1, 1);
}

// 관측을 평탄한 특징으로. 픽셀이면 3x3 합성곱 넷(첫 층만 보폭 2)에 ReLU 를 끼우고 마지막을 편다.
// 저차원이면 관측이 이미 평탄해 **그대로 돌려준다** — 인코더가 항등인 것이 저차원 경로다.
uint32_t encode(GraphBuilder& builder,
                const AgentConfig& config,
                const AgentParameterIds& ids,
                uint32_t observation,
                uint32_t batch) {
    if (!config.pixels()) {
        return observation;
    }
    uint32_t value = observation;
    for (uint32_t layer = 0; layer < AGENT_CONV_LAYERS; ++layer) {
        value = builder.addConv2d(value, ids.convWeight[layer], ids.convBias[layer], layer == 0 ? 2 : 1, 0);
        if (value == NO_TENSOR) {
            return NO_TENSOR;
        }
        value = builder.addRelu(value);
    }
    // 선형은 (배치, 특징) 만 받으므로 (배치, C, H, W) 를 편다. 자리를 새로 잡지 않는 뷰다.
    return builder.addReshape(value, batch, builder.featureCount(value), 1, 1);
}

uint32_t applyTrunk(GraphBuilder& builder, const AgentParameterIds::Trunk& trunk, uint32_t input) {
    uint32_t linear = builder.addLinear(input, trunk.weight, trunk.bias);
    uint32_t normalized = builder.addLayerNorm(linear, trunk.gain, trunk.shift);
    // tanh 로 특징을 [-1, 1] 에 묶는다. layernorm 과 함께 크리틱이 발산하지 않게 하는 짝이다.
    return builder.addTanh(normalized);
}

// 은닉 둘에 ReLU 를 끼우고 마지막 선형은 활성 없이 낸다. 액터는 부르는 쪽이 tanh 를 얹는다.
uint32_t applyHead(GraphBuilder& builder, const AgentParameterIds::Head& head, uint32_t input) {
    uint32_t value = input;
    for (uint32_t layer = 0; layer < AGENT_HEAD_LAYERS; ++layer) {
        value = builder.addLinear(value, head.weight[layer], head.bias[layer]);
        if (value == NO_TENSOR) {
            return NO_TENSOR;
        }
        if (layer + 1 < AGENT_HEAD_LAYERS) {
            value = builder.addRelu(value);
        }
    }
    return value;
}

uint32_t applyCritic(GraphBuilder& builder, const AgentParameterIds::Head& head, uint32_t feature, uint32_t action) {
    return applyHead(builder, head, builder.addConcat(feature, action));
}

// [begin, 지금까지) 의 텐서를 통째로 경사에서 끊는다. 마지막 하나만 끊어도 값은 같지만, 이렇게 두면
// 역전파가 그 연산들을 **아예 건너뛴다**(출력 경사가 없는 연산은 훑지 않는다).
void detachSince(GraphBuilder& builder, const Graph& graph, size_t begin) {
    for (size_t tensor = begin; tensor < graph.tensors.size(); ++tensor) {
        builder.detach(static_cast<uint32_t>(tensor));
    }
}

bool sane(const AgentConfig& config, uint32_t representation) {
    if (representation == 0 || config.batch == 0 || config.actionCount == 0) {
        return false;
    }
    return config.featureCount != 0 && config.hidden != 0;
}

bool buildActGraph(const AgentConfig& config, AgentActGraph& out, AgentParameterMap& map) {
    GraphBuilder builder(out.graph);
    uint32_t representation = representationCount(config);
    AgentParameterIds ids = allocateParameters(builder, config, representation, map);

    out.observation = addObservationInput(builder, config, 1);
    uint32_t feature = applyTrunk(builder, ids.onlineTrunk, encode(builder, config, ids, out.observation, 1));
    out.action = builder.addTanh(applyHead(builder, ids.actor, feature));
    // 손실이 없는 표라 validateForward 로 본다.
    return out.action != NO_TENSOR && validateForward(out.graph);
}

bool buildCriticGraph(const AgentConfig& config, AgentCriticGraph& out, AgentParameterMap& map) {
    GraphBuilder builder(out.graph);
    uint32_t representation = representationCount(config);
    AgentParameterIds ids = allocateParameters(builder, config, representation, map);
    uint32_t batch = config.batch;

    out.observation = addObservationInput(builder, config, batch);
    out.action = builder.addInput(batch, config.actionCount, 1, 1);
    out.reward = builder.addInput(batch, 1, 1, 1);
    out.discount = builder.addInput(batch, 1, 1, 1);
    out.nextObservation = addObservationInput(builder, config, batch);
    out.noise = builder.addInput(batch, config.actionCount, 1, 1);

    uint32_t feature = applyTrunk(builder, ids.onlineTrunk, encode(builder, config, ids, out.observation, batch));
    out.q1 = applyCritic(builder, ids.onlineCritic[0], feature, out.action);
    out.q2 = applyCritic(builder, ids.onlineCritic[1], feature, out.action);

    // ---- 여기부터가 시간차 목표다. PyTorch 의 with no_grad() 와 같은 구간이다.
    size_t frozenBegin = out.graph.tensors.size();
    uint32_t nextEncoded = encode(builder, config, ids, out.nextObservation, batch);
    // 액터는 **온라인 trunk** 의 특징을 본다. a' 는 «정책이 s' 에서 고를 행동» 이어야 하는데 배포 때
    // 정책이 보는 것이 온라인 특징이라서다(DrQ-v2 는 액터에게 자기 trunk 를 주고 그것을 갱신한다).
    // 타깃 trunk 를 먹이면 온라인도 타깃도 아닌 잡종이 되고, tau 가 작을수록 그 어긋남이 오래 남는다.
    uint32_t nextActorFeature = applyTrunk(builder, ids.onlineTrunk, nextEncoded);
    uint32_t nextRaw = applyHead(builder, ids.actor, nextActorFeature);
    uint32_t nextAction = builder.addTanh(builder.addAdd(nextRaw, out.noise));
    // 크리틱만 타깃 trunk 다. 여기가 뒤처져야 목표가 한 걸음 안에서 고정된다.
    uint32_t nextFeature = applyTrunk(builder, ids.targetTrunk, nextEncoded);
    out.targetTwin1 = applyCritic(builder, ids.targetCritic[0], nextFeature, nextAction);
    out.targetTwin2 = applyCritic(builder, ids.targetCritic[1], nextFeature, nextAction);
    uint32_t smallest = builder.addMin2(out.targetTwin1, out.targetTwin2);
    uint32_t discounted = builder.addMul(out.discount, smallest);
    out.targetValue = builder.addAdd(out.reward, discounted);
    if (out.targetValue == NO_TENSOR) {
        return false;
    }
    detachSince(builder, out.graph, frozenBegin);

    // 두 손실을 **먼저 지어 둔다.** 인자 자리에서 부르면 평가 순서가 정해져 있지 않아 컴파일러마다
    // 연산 표의 순서가 갈리고, 그러면 그래프 해시도 갈린다.
    uint32_t first = builder.addMse(out.q1, out.targetValue);
    uint32_t second = builder.addMse(out.q2, out.targetValue);
    out.loss = builder.addAdd(first, second);
    return out.loss != NO_TENSOR && validate(out.graph);
}

bool buildActorGraph(const AgentConfig& config, AgentActorGraph& out, AgentParameterMap& map) {
    GraphBuilder builder(out.graph);
    uint32_t representation = representationCount(config);
    AgentParameterIds ids = allocateParameters(builder, config, representation, map);
    uint32_t batch = config.batch;

    out.observation = addObservationInput(builder, config, batch);
    // **인코더와 trunk 를 통째로 끊는다.** 액터 손실이 표현을 끌고 가면 «크리틱을 크게 만드는 방향» 으로
    // 특징이 무너진다(DrQ-v2 가 인코더를 액터 손실에서 떼는 이유). 끊으면 역전파 비용도 함께 준다.
    size_t frozenBegin = out.graph.tensors.size();
    uint32_t feature = applyTrunk(builder, ids.onlineTrunk, encode(builder, config, ids, out.observation, batch));
    if (feature == NO_TENSOR) {
        return false;
    }
    detachSince(builder, out.graph, frozenBegin);

    out.action = builder.addTanh(applyHead(builder, ids.actor, feature));
    out.q1 = applyCritic(builder, ids.onlineCritic[0], feature, out.action);
    out.q2 = applyCritic(builder, ids.onlineCritic[1], feature, out.action);
    out.value = builder.addMin2(out.q1, out.q2);
    // 손실은 가치의 **음수** 평균이다. 크리틱 가중치에도 경사가 나지만 액터 구간만 갱신하므로 버려진다
    // (PyTorch 로 치면 액터 최적화기만 스텝하는 것과 같다).
    out.loss = builder.addScale(builder.addMean(out.value), -1.0F);
    return out.loss != NO_TENSOR && validate(out.graph);
}

} // namespace

bool buildAgent(const AgentConfig& config, Agent& out) {
    uint32_t representation = representationCount(config);
    if (!sane(config, representation)) {
        return false;
    }
    Agent built;
    built.config = config;
    AgentParameterMap actMap;
    AgentParameterMap actorMap;
    if (!buildActGraph(config, built.act, actMap) || !buildCriticGraph(config, built.critic, built.parameters) ||
        !buildActorGraph(config, built.actor, actorMap)) {
        return false;
    }
    // 세 표가 파라미터를 같은 순서로 잡았는지. 여기서 어긋나면 배열 하나를 함께 보는 전제가 깨진다.
    if (!(actMap == built.parameters) || !(actorMap == built.parameters)) {
        return false;
    }
    if (parameterLayout(built.act.graph) != parameterLayout(built.critic.graph) ||
        parameterLayout(built.actor.graph) != parameterLayout(built.critic.graph)) {
        return false;
    }
    // 구간이 통짜로 이어지고 온라인과 타깃의 길이가 같은지. updateAgentTarget 은 online.count() 개를
    // target.begin 에 쓰므로, 온라인 쪽에만 파라미터가 하나 더 붙으면 **액터 가중치를 덮어쓴다.**
    // 테스트만 보는 불변식으로 두면 나중에 층을 더할 때 조용히 무너진다.
    const AgentParameterMap& map = built.parameters;
    if (map.online.count() != map.target.count() || map.criticUpdate.begin != 0 ||
        map.criticUpdate.end != map.online.end || map.target.begin != map.online.end ||
        map.actorUpdate.begin != map.target.end || map.actorUpdate.end != map.total ||
        map.total != built.critic.graph.parameterCount) {
        return false;
    }
    out = std::move(built);
    return true;
}

void initializeAgent(const Agent& agent, uint64_t seed, float* parameters) {
    initializeParameters(agent.critic.graph, seed, parameters);
    updateAgentTarget(agent, 1.0F, parameters);
}

void updateAgentTarget(const Agent& agent, float tau, float* parameters) {
    const AgentParameterMap& map = agent.parameters;
    polyakStep(tau, map.online.count(), parameters + map.online.begin, parameters + map.target.begin);
}

bool saveAgent(const Agent& agent, const float* parameters, const std::string& path) {
    return saveParameters(agent.critic.graph, parameters, path);
}

bool loadAgent(const Agent& agent, float* parameters, const std::string& path) {
    return loadParameters(agent.critic.graph, parameters, path);
}

} // namespace gfx
