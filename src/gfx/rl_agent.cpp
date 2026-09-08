#include "gfx/rl_agent.h"

#include <algorithm>
#include <cmath>
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
// 뷰마다 입력 텐서를 하나씩, **연달아** 잡는다. 빌더가 순서대로 자리를 잡으므로 결과가 이어져 있고,
// 그래서 표집이 주소 하나로 전부 채우면서 뷰별 인코더가 자기 조각만 본다. 이음은 buildAgent 가 본다.
std::vector<uint32_t> addObservationInputs(GraphBuilder& builder, const AgentConfig& config, uint32_t batch) {
    std::vector<uint32_t> inputs;
    inputs.reserve(config.views);
    for (uint32_t view = 0; view < config.views; ++view) {
        inputs.push_back(config.pixels()
                             ? builder.addInput(batch, config.frameStack, config.imageSize, config.imageSize)
                             : builder.addInput(batch, config.observationCount, 1, 1));
    }
    return inputs;
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

// 뷰마다 **같은 인코더**를 태우고 특징을 더해 합친다. M = sum(V_i) 다.
//
// 병합이 덧셈인 것이 이 방법의 요점이다. 덧셈은 역전파가 «경사를 모든 입력에 복사» 라 병합에 학습할
// 것이 없고(파라미터가 늘지 않는다), 뷰 수가 인코더의 모양을 바꾸지 않아 **배포 때 카메라를 빼도 같은
// 인코더가 돈다.** 채널로 이어 붙이면 둘 다 무너진다.
//
// views 를 함께 돌려주는 것은 SADA 가 단일 뷰 특징을 따로 쓰기 때문이다.
uint32_t encodeViews(GraphBuilder& builder,
                     const AgentConfig& config,
                     const AgentParameterIds& ids,
                     const std::vector<uint32_t>& observations,
                     uint32_t batch,
                     std::vector<uint32_t>* views) {
    uint32_t merged = NO_TENSOR;
    for (uint32_t view = 0; view < observations.size(); ++view) {
        uint32_t encoded = encode(builder, config, ids, observations[view], batch);
        if (encoded == NO_TENSOR) {
            return NO_TENSOR;
        }
        if (views != nullptr) {
            views->push_back(encoded);
        }
        merged = view == 0 ? encoded : builder.addAdd(merged, encoded);
        if (merged == NO_TENSOR) {
            return NO_TENSOR;
        }
    }
    return merged;
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
    if (config.views == 0 || config.deployViews == 0 || config.deployViews > config.views) {
        return false;
    }
    if (!(config.sadaAlpha >= 0.0F) || !(config.sadaAlpha <= 1.0F)) {
        return false;
    }
    return config.featureCount != 0 && config.hidden != 0;
}

// usedViews 가 config.views 보다 작으면 앞의 몇 뷰만 더한다. 배포 때 카메라를 뺀 상태가 그 경우다 —
// 입력 자리는 그대로 두고(표집·관측이 채우는 자리가 같아야 한다) 더하는 것만 줄인다.
bool buildActGraph(const AgentConfig& config, AgentActGraph& out, AgentParameterMap& map, uint32_t usedViews) {
    GraphBuilder builder(out.graph);
    uint32_t representation = representationCount(config);
    AgentParameterIds ids = allocateParameters(builder, config, representation, map);

    out.observation = addObservationInputs(builder, config, 1);
    std::vector<uint32_t> used(out.observation.begin(),
                               out.observation.begin() + std::min<size_t>(usedViews, out.observation.size()));
    out.merged = encodeViews(builder, config, ids, used, 1, &out.viewFeature);
    if (out.merged == NO_TENSOR) {
        return false;
    }
    uint32_t feature = applyTrunk(builder, ids.onlineTrunk, out.merged);
    out.action = builder.addTanh(applyHead(builder, ids.actor, feature));
    // 손실이 없는 표라 validateForward 로 본다.
    return out.action != NO_TENSOR && validateForward(out.graph);
}

bool buildCriticGraph(const AgentConfig& config, AgentCriticGraph& out, AgentParameterMap& map) {
    GraphBuilder builder(out.graph);
    uint32_t representation = representationCount(config);
    AgentParameterIds ids = allocateParameters(builder, config, representation, map);
    uint32_t batch = config.batch;

    out.observation = addObservationInputs(builder, config, batch);
    out.action = builder.addInput(batch, config.actionCount, 1, 1);
    out.reward = builder.addInput(batch, 1, 1, 1);
    out.discount = builder.addInput(batch, 1, 1, 1);
    out.nextObservation = addObservationInputs(builder, config, batch);
    out.noise = builder.addInput(batch, config.actionCount, 1, 1);
    // 뷰가 여럿일 때만 SADA 가중치를 둔다. 뷰가 하나면 표가 지금까지와 글자 그대로 같아야 한다.
    if (config.multiView()) {
        for (uint32_t view = 0; view < config.views; ++view) {
            out.sadaWeight.push_back(builder.addInput(1, 1, 1, 1));
        }
    }

    std::vector<uint32_t> viewFeatures;
    uint32_t merged = encodeViews(builder, config, ids, out.observation, batch, &viewFeatures);
    if (merged == NO_TENSOR) {
        return false;
    }
    uint32_t feature = applyTrunk(builder, ids.onlineTrunk, merged);
    out.q1 = applyCritic(builder, ids.onlineCritic[0], feature, out.action);
    out.q2 = applyCritic(builder, ids.onlineCritic[1], feature, out.action);

    // ---- 여기부터가 시간차 목표다. PyTorch 의 with no_grad() 와 같은 구간이다.
    size_t frozenBegin = out.graph.tensors.size();
    // **타깃값은 병합 특징 M 으로 만든다.** 단일 뷰로 만들면 목표 자체가 뷰마다 흔들려, 학습이 «뷰가
    // 달라도 같은 값» 을 배우는 대신 «뷰마다 다른 값» 을 배운다.
    uint32_t nextEncoded = encodeViews(builder, config, ids, out.nextObservation, batch, nullptr);
    if (nextEncoded == NO_TENSOR) {
        return false;
    }
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
    out.mergedLoss = builder.addAdd(first, second);
    if (out.mergedLoss == NO_TENSOR) {
        return false;
    }
    out.loss = out.mergedLoss;

    // ---- SADA. 같은 머리를 **단일 뷰 특징**에도 태워 손실을 한 번 더 재고, 갱신마다 뷰 하나만 살린다.
    //
    // 표가 고정이라 «무작위 뷰» 를 가중치 입력으로 푼다. 뷰마다 항을 지어 두고 호스트가 고른 뷰에
    // alpha 를, 나머지에 0 을 넣는다 — 표는 그대로이고 값만 갈린다. 목표값 y 는 병합 특징으로 만든 것을
    // 그대로 쓴다(위에서 이미 끊어 두었다).
    //
    // ponytail: 가중치가 0 인 뷰의 갈래도 **순전파는 그대로 돈다.** 뷰 하나를 더할 때마다 trunk 한 벌
    // (39,200 -> 50, 이 망에서 가장 비싼 층)과 크리틱 머리 둘이 헛돌아, 2뷰에서 갱신 비용이 1.5 배쯤
    // 된다. 없애려면 손실이 아니라 **특징을 고르면** 된다 — sum(MUL(V_i, 1_i)) 로 뽑아 갈래 하나만
    // 태우는 것이고, 경사도 고른 뷰에만 간다. 그러려면 스칼라를 텐서에 퍼뜨려 곱하는 연산(브로드캐스트
    // MUL)이 있어야 하는데 지금 연산 표에는 같은 모양끼리의 곱뿐이다.
    if (config.multiView()) {
        uint32_t total = builder.addScale(out.mergedLoss, 1.0F - config.sadaAlpha);
        for (uint32_t view = 0; view < viewFeatures.size(); ++view) {
            uint32_t single = applyTrunk(builder, ids.onlineTrunk, viewFeatures[view]);
            uint32_t singleQ1 = applyCritic(builder, ids.onlineCritic[0], single, out.action);
            uint32_t singleQ2 = applyCritic(builder, ids.onlineCritic[1], single, out.action);
            uint32_t singleFirst = builder.addMse(singleQ1, out.targetValue);
            uint32_t singleSecond = builder.addMse(singleQ2, out.targetValue);
            uint32_t singleLoss = builder.addAdd(singleFirst, singleSecond);
            total = builder.addAdd(total, builder.addMul(singleLoss, out.sadaWeight[view]));
            if (total == NO_TENSOR) {
                return false;
            }
        }
        out.loss = total;
    }
    return out.loss != NO_TENSOR && validate(out.graph);
}

bool buildActorGraph(const AgentConfig& config, AgentActorGraph& out, AgentParameterMap& map) {
    GraphBuilder builder(out.graph);
    uint32_t representation = representationCount(config);
    AgentParameterIds ids = allocateParameters(builder, config, representation, map);
    uint32_t batch = config.batch;

    out.observation = addObservationInputs(builder, config, batch);
    if (config.multiView()) {
        for (uint32_t view = 0; view < config.views; ++view) {
            out.sadaWeight.push_back(builder.addInput(1, 1, 1, 1));
        }
    }
    // **인코더와 trunk 를 통째로 끊는다.** 액터 손실이 표현을 끌고 가면 «크리틱을 크게 만드는 방향» 으로
    // 특징이 무너진다(DrQ-v2 가 인코더를 액터 손실에서 떼는 이유). 끊으면 역전파 비용도 함께 준다.
    size_t frozenBegin = out.graph.tensors.size();
    std::vector<uint32_t> viewFeatures;
    uint32_t merged = encodeViews(builder, config, ids, out.observation, batch, &viewFeatures);
    if (merged == NO_TENSOR) {
        return false;
    }
    uint32_t feature = applyTrunk(builder, ids.onlineTrunk, merged);
    std::vector<uint32_t> singleFeatures;
    if (config.multiView()) {
        for (uint32_t view = 0; view < viewFeatures.size(); ++view) {
            singleFeatures.push_back(applyTrunk(builder, ids.onlineTrunk, viewFeatures[view]));
        }
    }
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
    out.mergedLoss = builder.addScale(builder.addMean(out.value), -1.0F);
    if (out.mergedLoss == NO_TENSOR) {
        return false;
    }
    out.loss = out.mergedLoss;

    // SADA. 액터도 같은 꼴인데, **행동까지 그 뷰의 특징에서 다시 고른다.**
    //
    // 여기가 이 방법의 급소다. 행동을 병합 특징 M 으로만 고르면 액터 머리가 학습 내내 trunk(M) 만
    // 보게 되고, 그러면 카메라를 뺀 배포에서 trunk(V_0) 를 먹여 **한 번도 본 적 없는 분포**로 돌게 된다.
    // 크리틱과 trunk 만 분리되고 정작 정책은 분리되지 않는 것이라, 그 상태로 잰 «1뷰 평가» 점수는
    // MAD 의 주장에 대해 아무 말도 하지 못한다. DrQ-v2 의 액터 손실이 -Q(f, pi(f)) 로 **같은 특징**을
    // 두 자리에 쓰는 것과 같은 이유다.
    if (config.multiView()) {
        uint32_t total = builder.addScale(out.mergedLoss, 1.0F - config.sadaAlpha);
        for (uint32_t view = 0; view < singleFeatures.size(); ++view) {
            uint32_t singleAction = builder.addTanh(applyHead(builder, ids.actor, singleFeatures[view]));
            uint32_t singleQ1 = applyCritic(builder, ids.onlineCritic[0], singleFeatures[view], singleAction);
            uint32_t singleQ2 = applyCritic(builder, ids.onlineCritic[1], singleFeatures[view], singleAction);
            uint32_t singleValue = builder.addMin2(singleQ1, singleQ2);
            uint32_t singleLoss = builder.addScale(builder.addMean(singleValue), -1.0F);
            total = builder.addAdd(total, builder.addMul(singleLoss, out.sadaWeight[view]));
            if (total == NO_TENSOR) {
                return false;
            }
        }
        out.loss = total;
    }
    return out.loss != NO_TENSOR && validate(out.graph);
}

} // namespace

bool tensorsContiguous(const Graph& graph, const std::vector<uint32_t>& tensors) {
    if (tensors.empty()) {
        return false;
    }
    for (size_t i = 1; i < tensors.size(); ++i) {
        const Tensor& previous = graph.tensors[tensors[i - 1]];
        if (previous.offset + previous.count() != graph.tensors[tensors[i]].offset) {
            return false;
        }
    }
    return true;
}

bool buildAgent(const AgentConfig& config, Agent& out) {
    uint32_t representation = representationCount(config);
    if (!sane(config, representation)) {
        return false;
    }
    Agent built;
    built.config = config;
    AgentParameterMap actMap;
    AgentParameterMap deployMap;
    AgentParameterMap actorMap;
    if (!buildActGraph(config, built.act, actMap, config.views) ||
        !buildActGraph(config, built.actDeploy, deployMap, config.deployViews) ||
        !buildCriticGraph(config, built.critic, built.parameters) || !buildActorGraph(config, built.actor, actorMap)) {
        return false;
    }
    // 네 표가 파라미터를 같은 순서로 잡았는지. 여기서 어긋나면 배열 하나를 함께 보는 전제가 깨진다.
    // **actDeploy 가 여기에 함께 걸리는 것이 요점이다** — 뷰를 뺀 표가 같은 가중치를 같은 자리에서
    // 읽어야 «학습한 그대로 카메라 하나로 돈다» 가 성립한다.
    if (!(actMap == built.parameters) || !(deployMap == built.parameters) || !(actorMap == built.parameters)) {
        return false;
    }
    if (parameterLayout(built.act.graph) != parameterLayout(built.critic.graph) ||
        parameterLayout(built.actDeploy.graph) != parameterLayout(built.critic.graph) ||
        parameterLayout(built.actor.graph) != parameterLayout(built.critic.graph)) {
        return false;
    }
    // 뷰 입력이 이어져 있는가. 표마다 본다.
    if (!tensorsContiguous(built.act.graph, built.act.observation) ||
        !tensorsContiguous(built.actDeploy.graph, built.actDeploy.observation) ||
        !tensorsContiguous(built.critic.graph, built.critic.observation) ||
        !tensorsContiguous(built.critic.graph, built.critic.nextObservation) ||
        !tensorsContiguous(built.actor.graph, built.actor.observation)) {
        return false;
    }
    // SADA 가중치도 이어져 있어야 한다. 호스트가 뷰마다 복사를 따로 걸지 않고 한 번에 올린다.
    if (config.multiView() && (!tensorsContiguous(built.critic.graph, built.critic.sadaWeight) ||
                               !tensorsContiguous(built.actor.graph, built.actor.sadaWeight))) {
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

bool AgentTrainer::build(const AgentConfig& config, uint64_t seed) {
    if (!buildAgent(config, agent)) {
        return false;
    }
    parameters.assign(agent.parameters.total, 0.0F);
    gradients.assign(agent.parameters.total, 0.0F);
    criticMoments.assign(adamMomentCount(agent.parameters.criticUpdate.count()), 0.0F);
    actorMoments.assign(adamMomentCount(agent.parameters.actorUpdate.count()), 0.0F);
    actActivations.assign(agent.act.graph.activationCount, 0.0F);
    actDeployActivations.assign(agent.actDeploy.graph.activationCount, 0.0F);
    criticActivations.assign(agent.critic.graph.activationCount, 0.0F);
    criticActivationGradients.assign(agent.critic.graph.activationCount, 0.0F);
    actorActivations.assign(agent.actor.graph.activationCount, 0.0F);
    actorActivationGradients.assign(agent.actor.graph.activationCount, 0.0F);
    initializeAgent(agent, seed, parameters.data());
    // 평활화 잡음의 흐름도 씨앗에서 갈라 둔다. 걸음 번호만 쓰면 씨앗이 다른 두 학습이 **같은 잡음**을
    // 먹어, 씨앗을 여럿 돌려 본다는 말이 반쯤 거짓이 된다.
    noiseStream = seed * 0x9E3779B97F4A7C15ULL + 0x2545F4914F6CDD1DULL;
    step = 0;
    return true;
}

void AgentTrainer::act(const float* observation, float* action) {
    const Graph& graph = agent.act.graph;
    // 뷰 입력이 이어져 있으므로(buildAgent 가 확인한다) 첫 뷰 자리에 통째로 부어 넣는다.
    std::copy(observation,
              observation + agent.config.observationSize(),
              tensorValues(graph, agent.act.observation[0], actActivations.data()));
    forward(graph, parameters.data(), actActivations.data());
    const float* result = tensorValues(graph, agent.act.action, actActivations.data());
    std::copy(result, result + agent.config.actionCount, action);
}

void AgentTrainer::actDeploy(const float* observation, float* action) {
    const Graph& graph = agent.actDeploy.graph;
    float* values = actDeployActivations.data();
    // 쓰지 않는 뷰 자리는 그대로 둔다 — 표가 그 자리를 읽지 않는다(더하는 항에서 빠져 있다).
    std::copy(observation,
              observation + agent.config.deployViews * agent.config.viewSize(),
              tensorValues(graph, agent.actDeploy.observation[0], values));
    forward(graph, parameters.data(), values);
    const float* result = tensorValues(graph, agent.actDeploy.action, values);
    std::copy(result, result + agent.config.actionCount, action);
}

namespace {

// 갱신마다 뷰 하나를 뽑아 그 자리에만 alpha 를 세운다. 표가 고정이라 «무작위 뷰» 가 이렇게 풀린다.
void setSadaWeights(
    const Graph& graph, const std::vector<uint32_t>& weights, float* values, uint32_t chosen, float alpha) {
    for (uint32_t view = 0; view < weights.size(); ++view) {
        *tensorValues(graph, weights[view], values) = view == chosen ? alpha : 0.0F;
    }
}

} // namespace

AgentUpdateStats AgentTrainer::update(const AgentBatch& batch, const AgentUpdateSettings& settings) {
    AgentUpdateStats stats;
    const AgentConfig& config = agent.config;
    uint32_t observationSize = config.observationSize();
    size_t observationTotal = static_cast<size_t>(config.batch) * observationSize;
    size_t actionTotal = static_cast<size_t>(config.batch) * config.actionCount;

    // ---- 크리틱 한 걸음.
    const Graph& critic = agent.critic.graph;
    float* criticValues = criticActivations.data();
    std::copy(batch.observations,
              batch.observations + observationTotal,
              tensorValues(critic, agent.critic.observation[0], criticValues));
    std::copy(batch.nextObservations,
              batch.nextObservations + observationTotal,
              tensorValues(critic, agent.critic.nextObservation[0], criticValues));
    // **크리틱과 액터가 같은 뷰를 본다.** 갱신 한 번 안에서 갈리면 액터가 «다른 뷰로 잰 크리틱» 을
    // 오르게 되고, 그것은 논문의 손실이 아니다.
    viewChoice = config.multiView() ? neuralRandomBelow(noiseStream, step + 1, config.views) : 0;
    setSadaWeights(critic, agent.critic.sadaWeight, criticValues, viewChoice, config.sadaAlpha);
    std::copy(batch.actions, batch.actions + actionTotal, tensorValues(critic, agent.critic.action, criticValues));
    std::copy(batch.rewards, batch.rewards + config.batch, tensorValues(critic, agent.critic.reward, criticValues));
    std::copy(
        batch.discounts, batch.discounts + config.batch, tensorValues(critic, agent.critic.discount, criticValues));

    // 타깃 정책 평활화 잡음. 걸음 번호에서 다시 만들 수 있으므로 따로 들고 다니지 않는다.
    float* noise = tensorValues(critic, agent.critic.noise, criticValues);
    float clip = std::abs(settings.noiseClip);
    for (size_t i = 0; i < actionTotal; ++i) {
        float sample = settings.targetNoise * neuralGaussian(noiseStream + step + 1, i);
        noise[i] = std::clamp(sample, -clip, clip);
    }

    forward(critic, parameters.data(), criticValues);
    stats.criticLoss = *tensorValues(critic, agent.critic.loss, criticValues);
    // buildAgent 가 validate 를 지났으므로 여기서 거짓이 나올 수 없다.
    // ponytail: 그래도 돌려주는 값을 버리고 있다. 표를 밖에서 갈아 끼울 수 있게 되면 살펴야 한다.
    backward(critic, parameters.data(), criticValues, gradients.data(), criticActivationGradients.data());
    const ParameterRange& criticRange = agent.parameters.criticUpdate;
    adamStep(settings.critic,
             step + 1,
             criticRange.count(),
             gradients.data() + criticRange.begin,
             criticMoments.data(),
             parameters.data() + criticRange.begin);

    // ---- 액터 한 걸음. 방금 갱신한 크리틱을 본다.
    const Graph& actor = agent.actor.graph;
    float* actorValues = actorActivations.data();
    std::copy(batch.observations,
              batch.observations + observationTotal,
              tensorValues(actor, agent.actor.observation[0], actorValues));
    setSadaWeights(actor, agent.actor.sadaWeight, actorValues, viewChoice, config.sadaAlpha);
    forward(actor, parameters.data(), actorValues);
    // 손실이 가치의 음수 평균이므로 되돌려 담는다.
    stats.value = -*tensorValues(actor, agent.actor.loss, actorValues);
    backward(actor, parameters.data(), actorValues, gradients.data(), actorActivationGradients.data());
    const ParameterRange& actorRange = agent.parameters.actorUpdate;
    adamStep(settings.actor,
             step + 1,
             actorRange.count(),
             gradients.data() + actorRange.begin,
             actorMoments.data(),
             parameters.data() + actorRange.begin);

    updateAgentTarget(agent, settings.tau, parameters.data());
    ++step;
    return stats;
}

bool AgentTrainer::save(const std::string& path) const {
    return saveAgent(agent, parameters.data(), path);
}

bool AgentTrainer::load(const std::string& path) {
    if (!loadAgent(agent, parameters.data(), path)) {
        return false;
    }
    // 최적화기 상태를 되돌린다. 파일에 담기는 것은 가중치뿐이라, 그대로 두면 **읽어 들인 가중치와 아무
    // 상관 없는 모멘트**로 첫 걸음을 밟는다. 편향 보정도 마찬가지라 걸음 수를 0 으로 되돌린다.
    std::fill(criticMoments.begin(), criticMoments.end(), 0.0F);
    std::fill(actorMoments.begin(), actorMoments.end(), 0.0F);
    step = 0;
    return true;
}

} // namespace gfx
