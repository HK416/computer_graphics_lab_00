#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <spdlog/spdlog.h>

#include "gfx/context.h"
#include "gfx/headless_compute.h"
#include "gfx/neural.h"
#include "gfx/neural_math.h"

namespace gfx {

namespace {

// GPU 커널이 CPU 기준과 얼마나 어긋나도 되는가.
//
// 원소별 연산은 **정확히 같아야 한다.** 같은 fp32 산술을 같은 순서로 하므로 반올림까지 같다. 그래서
// 허용치가 0 이고, 0 이 아닌 값이 뜨면 그것은 반올림이 아니라 배선이 틀린 것이다.
constexpr float EXACT = 0.0F;
// tanh 만 예외다. GLSL 의 tanh 와 std::tanh 는 서로 다른 근사라 마지막 자리가 갈린다. 실측이 1.64e-7
// 이라 허용치를 그 두 배에 붙여 둔다 — 1e-6 처럼 넉넉히 잡으면 «조금 틀린 미분» 이 그 안에 숨는다.
//
// ponytail: 드라이버가 바뀌면 tanh 의 근사도 바뀌어 이 값을 넘길 수 있다. 그때는 «더 넉넉히» 가 아니라
// 무엇이 달라졌는지 먼저 본다.
constexpr float TANH_TOLERANCE = 5.0e-7F;

struct Case {
    const char* name;
    Graph graph;
    // 흔들 입력 텐서들.
    std::vector<uint32_t> inputs;
    float tolerance = EXACT;
    // 무작위 대신 갈림을 밟는 값(0 과 동점)을 넣는다.
    bool boundary = false;
};

// (배치, 특징) 하나짜리 입력. 그룹 크기(128)의 배수가 **아닌** 크기를 골라 꼬리 스레드가 범위를 넘어
// 쓰지 않는지도 함께 본다.
constexpr uint32_t BATCH = 3;
constexpr uint32_t FEATURES = 91;

// 시험할 연산 **앞에 학습하는 층을 하나 둔다.** 그러지 않으면 입력이 경사를 받지 않는 INPUT 이라 그
// 연산의 역전파가 아무 것도 쓰지 않고, 경사 열이 «씨앗을 씨앗과 견주는» 헛검사가 된다.
uint32_t addStem(GraphBuilder& builder, uint32_t input) {
    uint32_t weight = builder.addParameter(BATCH, FEATURES, 1, 1);
    return builder.addAdd(input, weight);
}

Case makeUnary(const char* name, OpKind kind, float tolerance) {
    Case item;
    item.name = name;
    GraphBuilder builder(item.graph);
    uint32_t input = builder.addInput(BATCH, FEATURES, 1, 1);
    uint32_t stem = addStem(builder, input);
    switch (kind) {
    case OpKind::RELU:
        builder.addRelu(stem);
        break;
    case OpKind::TANH:
        builder.addTanh(stem);
        break;
    default:
        builder.addScale(stem, 0.375F);
        break;
    }
    item.inputs = {input};
    item.tolerance = tolerance;
    return item;
}

// 짝의 한쪽을 **파라미터**로 둔다. 그래야 파라미터 경사까지 견줄 수 있다.
Case makeBinary(const char* name, OpKind kind) {
    Case item;
    item.name = name;
    GraphBuilder builder(item.graph);
    uint32_t input = builder.addInput(BATCH, FEATURES, 1, 1);
    uint32_t stem = addStem(builder, input);
    uint32_t weight = builder.addParameter(BATCH, FEATURES, 1, 1);
    switch (kind) {
    case OpKind::ADD:
        builder.addAdd(stem, weight);
        break;
    case OpKind::MUL:
        builder.addMul(stem, weight);
        break;
    default:
        builder.addMin2(stem, weight);
        break;
    }
    item.inputs = {input};
    return item;
}

// 한 텐서가 두 갈래로 갈렸다가 다시 만난다. **경사 누적**이 여기서 드러난다 — 갈래마다 더하지 않고
// 덮어쓰면 입력 경사가 절반만 남는다.
Case makeBranching() {
    Case item;
    item.name = "branch";
    GraphBuilder builder(item.graph);
    uint32_t input = builder.addInput(BATCH, FEATURES, 1, 1);
    uint32_t stem = addStem(builder, input);
    uint32_t weight = builder.addParameter(BATCH, FEATURES, 1, 1);
    uint32_t left = builder.addRelu(stem);
    uint32_t right = builder.addMul(stem, weight);
    uint32_t merged = builder.addAdd(left, right);
    // 끊어 둔 갈래도 하나 둔다. 경사를 받지 않는 텐서를 건드리면 안 된다.
    uint32_t detached = builder.addScale(stem, 2.0F);
    builder.detach(detached);
    builder.addMin2(merged, builder.addScale(detached, 0.5F));
    item.inputs = {input};
    return item;
}

// **갈림을 일부러 밟는다.** 무작위 값만 넣으면 ReLU 의 x == 0 과 min 의 동점이 확률 0 이라, 두 규약
// («0 은 음수 쪽», «동점이면 첫째») 이 한 번도 검사되지 않는다. 여기서는 입력 절반을 정확히 0 으로,
// 파라미터를 입력과 같은 값으로 두어 두 갈림을 모두 지나게 한다.
Case makeBoundary() {
    Case item;
    item.name = "boundary";
    GraphBuilder builder(item.graph);
    uint32_t input = builder.addInput(BATCH, FEATURES, 1, 1);
    uint32_t weight = builder.addParameter(BATCH, FEATURES, 1, 1);
    // stem = input + 0 이라 stem 이 곧 input 이다. 그래야 아래에서 «파라미터와 정확히 같은 값» 을 만들 수
    // 있다(파라미터를 0 으로 채우는 것은 부르는 쪽 몫이다).
    uint32_t stem = builder.addAdd(input, weight);
    uint32_t left = builder.addRelu(stem);
    uint32_t second = builder.addParameter(BATCH, FEATURES, 1, 1);
    builder.addMin2(left, second);
    item.inputs = {input};
    item.boundary = true;
    return item;
}

// 네 축이 모두 1 이 아닌 모양. 모든 경우가 (배치, 특징, 1, 1) 이면 dims 를 하나 빼먹거나 두 축을 맞바꾼
// 배치 어긋남이 원소 수가 같아 드러나지 않는다.
Case makeFourDimensional() {
    Case item;
    item.name = "4d shape";
    GraphBuilder builder(item.graph);
    uint32_t input = builder.addInput(2, 3, 5, 7);
    uint32_t weight = builder.addParameter(2, 3, 5, 7);
    uint32_t stem = builder.addAdd(input, weight);
    builder.addTanh(stem);
    item.inputs = {input};
    item.tolerance = TANH_TOLERANCE;
    return item;
}

// **별칭 텐서.** addReshape 는 같은 저장소를 다른 모양으로 가리키는 뷰라, 경사가 같은 자리에 쌓인다.
// 원자 연산 없이 더해도 되는 근거가 정확히 «별칭이 오프셋을 공유한다» 이므로 그것을 밟아 둔다. 두 입력이
// 같은 텐서인 연산도 함께 본다(한 스레드가 두 번 더해 2*dy 가 되어야 한다).
Case makeAliased() {
    Case item;
    item.name = "alias";
    GraphBuilder builder(item.graph);
    uint32_t input = builder.addInput(BATCH, FEATURES, 1, 1);
    uint32_t stem = addStem(builder, input);
    uint32_t view = builder.addReshape(stem, 1, BATCH * FEATURES, 1, 1);
    uint32_t doubled = builder.addAdd(view, view);
    uint32_t back = builder.addReshape(doubled, BATCH, FEATURES, 1, 1);
    builder.addMul(back, stem);
    item.inputs = {input};
    return item;
}

// **파라미터 하나를 두 연산이 쓴다.** 그러지 않으면 파라미터 경사를 «더하기» 대신 «덮어쓰기» 로 바꿔도
// 드러나지 않는다 — 지운 뒤 한 번만 쓰면 두 결과가 같기 때문이다. 여기서는 같은 가중치가 ADD 와 MUL 에
// 함께 들어가 두 갈래의 경사가 한 자리에 쌓여야 한다.
Case makeSharedParameter() {
    Case item;
    item.name = "shared w";
    GraphBuilder builder(item.graph);
    uint32_t input = builder.addInput(BATCH, FEATURES, 1, 1);
    uint32_t weight = builder.addParameter(BATCH, FEATURES, 1, 1);
    uint32_t sum = builder.addAdd(input, weight);
    uint32_t product = builder.addMul(sum, weight);
    builder.addRelu(product);
    item.inputs = {input};
    return item;
}

// 선형 층. **세 축을 모두 다르게** 잡는다(배치 5, 입력 7, 출력 3). 전치나 첨자 밀림은 두 축이 같으면
// 우연히 같은 자리를 가리켜 드러나지 않는다.
//
// 뒤에 층을 하나 더 얹어 dx 가 실제로 쓰이게 한다. 마지막 층의 dx 는 아무 파라미터에도 닿지 않아, 층이
// 하나뿐이면 «입력에 대한 경사» 가 통째로 검사되지 않는다.
Case makeLinear() {
    Case item;
    item.name = "linear";
    GraphBuilder builder(item.graph);
    uint32_t input = builder.addInput(5, 7, 1, 1);
    uint32_t weight = builder.addParameter(3, 7, 1, 1);
    uint32_t bias = builder.addParameter(3, 1, 1, 1);
    uint32_t hidden = builder.addLinear(input, weight, bias);
    uint32_t secondWeight = builder.addParameter(2, 3, 1, 1);
    uint32_t secondBias = builder.addParameter(2, 1, 1, 1);
    builder.addLinear(hidden, secondWeight, secondBias);
    item.inputs = {input};
    return item;
}

// 크리틱이 (특징, 행동) 을 잇는 자리. 두 조각의 폭이 달라야 경계가 드러나고, **둘 다 학습하는 층에서
// 와야** 두 조각의 경사가 모두 검사된다.
Case makeConcat() {
    Case item;
    item.name = "concat";
    GraphBuilder builder(item.graph);
    uint32_t input = builder.addInput(4, 6, 1, 1);
    uint32_t leftWeight = builder.addParameter(5, 6, 1, 1);
    uint32_t leftBias = builder.addParameter(5, 1, 1, 1);
    uint32_t left = builder.addLinear(input, leftWeight, leftBias);
    uint32_t rightWeight = builder.addParameter(2, 6, 1, 1);
    uint32_t rightBias = builder.addParameter(2, 1, 1, 1);
    // 근사가 갈리는 tanh 대신 ReLU 를 쓴다. 이음도 «정확히 0» 계약 안에 두려는 것이다.
    uint32_t right = builder.addRelu(builder.addLinear(input, rightWeight, rightBias));
    uint32_t joined = builder.addConcat(left, right);
    uint32_t headWeight = builder.addParameter(3, 7, 1, 1);
    uint32_t headBias = builder.addParameter(3, 1, 1, 1);
    builder.addLinear(joined, headWeight, headBias);
    item.inputs = {input};
    return item;
}

// 합성곱. **보폭·여백·커널을 바꿔 가며** 본다.
//
// 시험할 층 **앞에 1x1 합성곱을 하나 둔다.** 그러지 않으면 그 층의 입력이 경사를 받지 않는 INPUT 이라
// dx 가 두 엔진 모두에서 통째로 건너뛰어지고, 이 단계의 핵심인 모아 읽기 산술(보폭으로 나누어떨어지는지,
// 여백을 더한 자리, ky 를 거꾸로 도는 것)이 보폭 1·여백 0 으로만 검사된다. 그것이 정확히 처음에 빠졌던
// 함정이다 — 앞 층이 있어야 «보폭 2·여백 1 의 dx» 라는 말이 성립한다.
//
// 여백이 0 이어도 dx 의 건너뛰기는 돈다(범위 밖 출력이 생긴다). 보폭이 1 이어도 ky 마다 oy 가 달라
// «거꾸로 돈다» 가 뜻을 갖는다. 그래서 두 축을 따로 흔든다.
Case makeConv(const char* name, uint32_t stride, uint32_t pad, uint32_t kernelHeight, uint32_t kernelWidth) {
    Case item;
    item.name = name;
    GraphBuilder builder(item.graph);
    // 채널 수를 서로 다르게 둔다. 같으면 입력·출력 채널 첨자를 맞바꿔도 드러나지 않는다.
    uint32_t input = builder.addInput(2, 3, 9, 11);
    uint32_t stemWeight = builder.addParameter(3, 3, 1, 1);
    uint32_t stemBias = builder.addParameter(3, 1, 1, 1);
    uint32_t stem = builder.addConv2d(input, stemWeight, stemBias, 1, 0);
    uint32_t weight = builder.addParameter(4, 3, kernelHeight, kernelWidth);
    uint32_t bias = builder.addParameter(4, 1, 1, 1);
    uint32_t hidden = builder.addConv2d(stem, weight, bias, stride, pad);
    // 뒤에도 층을 하나 더 얹는다. 마지막 층의 dx 는 아무 파라미터에도 닿지 않는다.
    uint32_t relu = builder.addRelu(hidden);
    uint32_t secondWeight = builder.addParameter(2, 4, 2, 2);
    uint32_t secondBias = builder.addParameter(2, 1, 1, 1);
    uint32_t second = builder.addConv2d(relu, secondWeight, secondBias, 1, 0);
    if (hidden == gfx::NO_TENSOR || second == gfx::NO_TENSOR) {
        item.graph = Graph{};
    }
    item.inputs = {input};
    return item;
}

// 출력 채널이 작업 그룹(128)보다 많은 합성곱. 편향 경사는 채널마다 스레드 하나라, 채널이 적으면
// 디스패치 수를 줄여도 남는 스레드가 범위 밖이라 티가 나지 않는다. 1x1 커널에 공간을 작게 잡아
// 채널만 늘린다.
Case makeWideConv() {
    Case item;
    item.name = "wide conv";
    GraphBuilder builder(item.graph);
    uint32_t input = builder.addInput(1, 3, 4, 4);
    uint32_t stemWeight = builder.addParameter(3, 3, 1, 1);
    uint32_t stemBias = builder.addParameter(3, 1, 1, 1);
    uint32_t stem = builder.addConv2d(input, stemWeight, stemBias, 1, 0);
    uint32_t weight = builder.addParameter(140, 3, 1, 1);
    uint32_t bias = builder.addParameter(140, 1, 1, 1);
    uint32_t wide = builder.addConv2d(stem, weight, bias, 1, 0);
    if (wide == gfx::NO_TENSOR) {
        item.graph = Graph{};
    }
    item.inputs = {input};
    return item;
}

// **합성곱 하나를 두 입력에 태운다.** 14단계의 다중 뷰 인코더가 정확히 이 모양이고, 검사로서도 두 가지를
// 한꺼번에 밟는다: 가중치·편향 경사가 두 갈래에서 **쌓이므로** «이미 들어 있던 값에서 출발한다» 가
// 비로소 검사되고(한 번만 쓰이면 0 + s 와 s + 0 이 비트까지 같아 증명 불가능하다), 가중치를 216 개로
// 잡아 작업 그룹(128) 하나를 넘기므로 디스패치 수를 줄이는 결함도 드러난다.
Case makeSharedConv() {
    Case item;
    item.name = "shared conv";
    GraphBuilder builder(item.graph);
    uint32_t input = builder.addInput(2, 3, 9, 11);
    uint32_t weight = builder.addParameter(8, 3, 3, 3);
    uint32_t bias = builder.addParameter(8, 1, 1, 1);
    // 같은 가중치를 서로 다른 «뷰» 둘에 태운다.
    uint32_t second = builder.addScale(input, 0.5F);
    uint32_t left = builder.addConv2d(input, weight, bias, 2, 1);
    uint32_t right = builder.addConv2d(second, weight, bias, 2, 1);
    if (builder.addAdd(left, right) == gfx::NO_TENSOR) {
        item.graph = Graph{};
    }
    item.inputs = {input};
    return item;
}

// **작업 그룹 하나보다 큰 층.** 지금까지의 선형 경우는 가중치가 300 개보다 작아 전부 한 그룹에 들어가고,
// 그러면 디스패치 수를 줄여도 남는 스레드가 어차피 범위 밖이라 티가 나지 않는다. 세 커널의 스레드 수
// (배치x출력, 배치x입력, 출력x입력)가 모두 그룹 크기(128)를 넘게 잡는다.
Case makeWideLinear() {
    Case item;
    item.name = "wide linear";
    GraphBuilder builder(item.graph);
    uint32_t input = builder.addInput(9, 20, 1, 1);
    uint32_t weight = builder.addParameter(15, 20, 1, 1);
    uint32_t bias = builder.addParameter(15, 1, 1, 1);
    uint32_t hidden = builder.addRelu(builder.addLinear(input, weight, bias));
    uint32_t headWeight = builder.addParameter(2, 15, 1, 1);
    uint32_t headBias = builder.addParameter(2, 1, 1, 1);
    builder.addLinear(hidden, headWeight, headBias);
    item.inputs = {input};
    return item;
}

// **가중치 하나를 두 선형 층이 나눠 쓴다.** 그러지 않으면 dW·dB 의 «이미 들어 있던 값에서 출발한다» 가
// 검사되지 않는다 — 경사가 0 에서 시작해 한 번만 쓰이면 0 + s 와 s + 0 이 비트까지 같기 때문이다.
// 연산 표를 쓰는 이유 자체가 «같은 가중치에 여러 갈래의 경사를 누적» 이므로 그 자리를 밟아 둔다.
Case makeSharedLinear() {
    Case item;
    item.name = "shared layer";
    GraphBuilder builder(item.graph);
    uint32_t input = builder.addInput(4, 5, 1, 1);
    uint32_t weight = builder.addParameter(3, 5, 1, 1);
    uint32_t bias = builder.addParameter(3, 1, 1, 1);
    uint32_t left = builder.addLinear(input, weight, bias);
    uint32_t shiftWeight = builder.addParameter(5, 5, 1, 1);
    uint32_t shiftBias = builder.addParameter(5, 1, 1, 1);
    uint32_t shifted = builder.addRelu(builder.addLinear(input, shiftWeight, shiftBias));
    // 같은 (weight, bias) 를 다른 입력에 한 번 더 태운다.
    uint32_t right = builder.addLinear(shifted, weight, bias);
    builder.addAdd(left, right);
    item.inputs = {input};
    return item;
}

// **얼려 둔 가중치와 편향.** 층은 살아 있어 경사가 지나가지만 그 파라미터는 갱신하지 않는 경우다. GPU
// 커널의 neuralHasGrad(weight) / neuralHasGrad(bias) 가드가 여기서만 일한다 — 없으면 GPU 는 얼린 자리에
// 쓰고 CPU 는 0 으로 두어 갈린다. 파라미터를 입력으로 받는 층도 함께 둔다(경사가 파라미터 arena 로
// 흘러야 하는 유일한 자리다).
Case makeFrozenLinear() {
    Case item;
    item.name = "frozen w";
    GraphBuilder builder(item.graph);
    uint32_t input = builder.addInput(4, 5, 1, 1);
    uint32_t frozenWeight = builder.addParameter(3, 5, 1, 1);
    uint32_t liveBias = builder.addParameter(3, 1, 1, 1);
    builder.detach(frozenWeight);
    uint32_t first = builder.addLinear(input, frozenWeight, liveBias);

    uint32_t liveWeight = builder.addParameter(2, 3, 1, 1);
    uint32_t frozenBias = builder.addParameter(2, 1, 1, 1);
    builder.detach(frozenBias);
    uint32_t second = builder.addLinear(first, liveWeight, frozenBias);

    // 합성곱 쪽에도 얼린 가중치와 편향을 둔다. 선형 층만으로는 합성곱 커널의 같은 가드가 죽은 채로
    // 남는다 — 커널이 넷씩 따로라 «선형에서 됐으니 합성곱도 될 것» 이 성립하지 않는다.
    uint32_t picture = builder.addInput(2, 3, 7, 7);
    uint32_t frozenKernel = builder.addParameter(4, 3, 3, 3);
    uint32_t liveKernelBias = builder.addParameter(4, 1, 1, 1);
    builder.detach(frozenKernel);
    uint32_t convolved = builder.addConv2d(picture, frozenKernel, liveKernelBias, 2, 1);
    uint32_t liveKernel = builder.addParameter(2, 4, 2, 2);
    uint32_t frozenKernelBias = builder.addParameter(2, 1, 1, 1);
    builder.detach(frozenKernelBias);
    uint32_t deeper = builder.addConv2d(builder.addRelu(convolved), liveKernel, frozenKernelBias, 1, 0);
    if (deeper == gfx::NO_TENSOR) {
        item.graph = Graph{};
    }
    builder.addScale(deeper, 0.25F);

    // 학습하는 상수 벡터를 입력으로 받는 층. dx 가 파라미터 경사 쪽에 쌓인다.
    uint32_t constant = builder.addParameter(1, 4, 1, 1);
    uint32_t constantWeight = builder.addParameter(2, 4, 1, 1);
    uint32_t constantBias = builder.addParameter(2, 1, 1, 1);
    uint32_t fromConstant = builder.addLinear(constant, constantWeight, constantBias);
    uint32_t view = builder.addReshape(fromConstant, 1, 2, 1, 1);
    uint32_t narrowed = builder.addReshape(second, 4, 2, 1, 1);
    uint32_t headWeight = builder.addParameter(1, 2, 1, 1);
    uint32_t headBias = builder.addParameter(1, 1, 1, 1);
    uint32_t left = builder.addLinear(narrowed, headWeight, headBias);
    uint32_t right = builder.addLinear(view, headWeight, headBias);
    builder.addAdd(left, builder.addReshape(builder.addScale(right, 1.0F), 1, 1, 1, 1));
    item.inputs = {input, picture};
    return item;
}

// 값과 씨앗을 씨앗 번호에서 만든다. 0 근처만 보면 ReLU 와 min 의 갈림을 못 밟는다.
void fill(std::vector<float>& values, uint64_t seed) {
    for (size_t i = 0; i < values.size(); ++i) {
        values[i] = 0.9F * neuralGaussian(seed, i);
    }
}

float compare(const std::vector<float>& expected, const float* measured) {
    float worst = 0.0F;
    for (size_t i = 0; i < expected.size(); ++i) {
        worst = std::max(worst, std::abs(expected[i] - measured[i]));
    }
    return worst;
}

} // namespace

bool runNeuralSelfCheck() {
    Context context(nullptr);
    NeuralExecutor executor(context);
    if (!executor.available()) {
        spdlog::error("신경망 GPU 실행기를 만들지 못했습니다");
        return false;
    }
    HeadlessCompute compute(context);

    std::vector<Case> cases;
    cases.push_back(makeUnary("relu", OpKind::RELU, EXACT));
    cases.push_back(makeUnary("tanh", OpKind::TANH, TANH_TOLERANCE));
    cases.push_back(makeUnary("scale", OpKind::SCALE, EXACT));
    cases.push_back(makeBinary("add", OpKind::ADD));
    cases.push_back(makeBinary("mul", OpKind::MUL));
    cases.push_back(makeBinary("min2", OpKind::MIN2));
    cases.push_back(makeBranching());
    cases.push_back(makeBoundary());
    cases.push_back(makeFourDimensional());
    cases.push_back(makeAliased());
    cases.push_back(makeSharedParameter());
    cases.push_back(makeLinear());
    cases.push_back(makeConcat());
    cases.push_back(makeWideLinear());
    cases.push_back(makeConv("conv s1 p0", 1, 0, 3, 3));
    cases.push_back(makeConv("conv s2 p1", 2, 1, 3, 3));
    cases.push_back(makeConv("conv k2x3 s2", 2, 2, 2, 3));
    cases.push_back(makeSharedConv());
    cases.push_back(makeWideConv());
    cases.push_back(makeSharedLinear());
    cases.push_back(makeFrozenLinear());

    std::printf("신경망 자기 검사 (CPU 기준 대 GPU)\n");
    std::printf("  %-12s %10s %10s %10s\n", "연산", "활성", "파라미터 경사", "활성 경사");
    bool ok = true;
    for (Case& item : cases) {
        if (!executor.build(item.graph)) {
            std::printf("  %-12s 표를 올리지 못했습니다\n", item.name);
            ok = false;
            continue;
        }
        if (executor.unsupportedOps() != 0) {
            std::printf("  %-12s GPU 커널이 없는 연산 %u 개\n", item.name, executor.unsupportedOps());
            ok = false;
            continue;
        }

        const Graph& graph = item.graph;
        std::vector<float> parameters(graph.parameterCount, 0.0F);
        std::vector<float> activations(graph.activationCount, 0.0F);
        std::vector<float> parameterGradients(graph.parameterCount, 0.0F);
        std::vector<float> activationGradients(graph.activationCount, 0.0F);
        fill(parameters, 11);
        for (uint32_t tensor : item.inputs) {
            const Tensor& shape = graph.tensors[tensor];
            for (uint32_t i = 0; i < shape.count(); ++i) {
                activations[shape.offset + i] = 0.9F * neuralGaussian(13, shape.offset + i);
            }
        }
        if (item.boundary) {
            // 입력의 절반을 정확히 0 으로 두고 파라미터를 0 으로 채운다. 그러면 ReLU 가 x == 0 을 지나고
            // (파라미터가 0 이라 stem == input), min 의 두 짝이 정확히 같은 값이 되는 자리도 생긴다.
            std::fill(parameters.begin(), parameters.end(), 0.0F);
            const Tensor& shape = graph.tensors[item.inputs[0]];
            for (uint32_t i = 0; i < shape.count(); i += 2) {
                activations[shape.offset + i] = 0.0F;
            }
            // min 의 둘째 짝(마지막 파라미터 텐서)을 relu 결과와 같게 둔다. 첫째가 이기는지 둘째가
            // 이기는지가 여기서 갈린다.
            std::vector<uint32_t> tensorsOfParameters = graph.parameterTensors();
            const Tensor& partner = graph.tensors[tensorsOfParameters.back()];
            for (uint32_t i = 0; i < partner.count(); ++i) {
                parameters[partner.offset + i] = std::max(activations[shape.offset + i], 0.0F);
            }
        }
        // **활성 경사 배열 전체**를 씨앗으로 채운다. 마지막 출력만 채우면 끊어 둔 텐서의 경사 칸이 0 이라,
        // «경사를 받지 않는 출력의 연산을 건너뛴다» 가 0 을 더하는 것과 구별되지 않는다. 두 엔진 모두
        // backwardFrom 이 지우지 않고 그 위에 더하므로 결과도 같아야 한다.
        for (size_t i = 0; i < activationGradients.size(); ++i) {
            activationGradients[i] = 0.7F * neuralGaussian(17, i) + 0.3F;
        }

        // GPU 는 CPU 가 손대기 전의 값을 그대로 받아야 한다.
        std::vector<float> seedActivations = activations;
        std::vector<float> seedGradients = activationGradients;

        forward(graph, parameters.data(), activations.data());
        backwardFrom(
            graph, parameters.data(), activations.data(), parameterGradients.data(), activationGradients.data());

        std::copy(parameters.begin(), parameters.end(), executor.parameterStaging());
        std::copy(seedActivations.begin(), seedActivations.end(), executor.activationStaging());
        std::copy(seedGradients.begin(), seedGradients.end(), executor.activationGradientStaging());
        auto run = [&]() {
            compute.submit([&](VkCommandBuffer commandBuffer, uint64_t) {
                // 지우고 나서 올린다. 씨앗이 활성 경사 버퍼에 들어가므로 순서가 뒤바뀌면 지워진다.
                executor.recordClearGradients(commandBuffer);
                executor.recordUpload(commandBuffer);
                executor.recordUploadGradientSeed(commandBuffer);
                executor.recordForward(commandBuffer);
                executor.recordBackward(commandBuffer);
                executor.recordDownload(commandBuffer);
            });
            executor.invalidateReadback();
        };
        run();
        // **두 번 돌려도 같아야 한다.** 경사를 지우지 않으면 두 번째가 첫 번째 위에 쌓여 값이 배가 된다.
        std::vector<float> once(executor.parameterGradientResult(),
                                executor.parameterGradientResult() + graph.parameterCount);
        run();
        bool repeatable = std::equal(once.begin(), once.end(), executor.parameterGradientResult());

        // **경사가 실제로 흘렀는지 먼저 본다.** 흐르지 않으면 아래 비교가 0 과 0 을 견주는 헛검사다.
        bool parameterMoved = false;
        for (float value : parameterGradients) {
            parameterMoved = parameterMoved || value != 0.0F;
        }
        bool gradientMoved = false;
        for (size_t i = 0; i < activationGradients.size(); ++i) {
            gradientMoved = gradientMoved || activationGradients[i] != seedGradients[i];
        }
        if (!repeatable) {
            std::printf("  %-12s 두 번 돌리니 결과가 달라집니다(경사를 지우지 않는다)\n", item.name);
            ok = false;
            continue;
        }
        if (!parameterMoved || !gradientMoved) {
            std::printf("  %-12s 경사가 흐르지 않습니다. 검사가 헛돕니다\n", item.name);
            ok = false;
            continue;
        }

        float activationError = compare(activations, executor.activationResult());
        float parameterError = compare(parameterGradients, executor.parameterGradientResult());
        float gradientError = compare(activationGradients, executor.activationGradientResult());
        std::printf("  %-12s %10.3e %10.3e %10.3e\n",
                    item.name,
                    static_cast<double>(activationError),
                    static_cast<double>(parameterError),
                    static_cast<double>(gradientError));
        float worst = std::max({activationError, parameterError, gradientError});
        if (!(worst <= item.tolerance)) {
            std::printf("      허용치 %.3e 를 넘었습니다\n", static_cast<double>(item.tolerance));
            ok = false;
        }
    }
    std::fflush(stdout);
    return ok;
}

} // namespace gfx
