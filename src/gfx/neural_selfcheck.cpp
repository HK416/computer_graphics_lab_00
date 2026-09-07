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
// layernorm 도 예외다. **GPU 의 나눗셈과 sqrt 는 정확 반올림이 아니다** — Vulkan 규격이 정확 반올림을
// 요구하는 것은 덧셈·뺄셈·곱셈·FMA 뿐이고, OpFDiv 는 2.5 ULP, Sqrt 는 3.0 ULP 까지 허용한다. layernorm 은
// 그 둘을 모두 지나므로(평균과 분산을 특징 수로 나누고, 1/sqrt(분산+eps) 를 낸다) 어느 GPU 에서도 CPU
// 기준과 비트로 같을 수 없다.
//
// 실측: 특징 7 에서 4.8e-7, 특징 8(2의 거듭제곱이라 나눗셈이 정확하다)에서 2.4e-7. 나눗셈과 sqrt 가 각각
// 절반씩 낸다는 뜻이다. 이득 경사는 행마다 쌓여 조금 더 크다(실측 1.9e-6).
constexpr float LAYERNORM_TOLERANCE = 5.0e-6F;
// 나눗셈이 하나 끼면 1 ULP 가 날 수 있다. 접는 연산이 모두 원소 수로 나누므로 여기 해당한다.
//
// 값에 따라 정확히 맞기도 한다(mse·huber 의 어떤 판은 0.000e+00 이다). 그것은 그 값들에서 GPU 의 나눗셈이
// 마침 CPU 와 같게 반올림된 것이지 보장이 아니므로, 접는 연산에는 전부 이 허용치를 준다. 그래도 누산
// 순서를 뒤집는 돌연변이는 4.8e-7 을 내 이 안에 들지 못한다.
constexpr float DIVIDE_TOLERANCE = 2.0e-7F;
// 협력 행렬 변종의 허용치. 이 열은 **앞 세 열과 성격이 다르다** — 재는 것이 «비트까지 같은가» 가 아니라
// «가속을 켜도 같은 곳을 가리키는가» 다. fp32 A/B 는 캐스팅이 없어 k 를 타일로 접는 덧셈 순서 차이만
// 남고, fp16 A/B 는 입력을 반정밀도(상대 오차 약 5e-4)로 눌러 두 자릿수가 더 든다.
//
// **이 열만 상대 오차다.** fp16 이 내는 오차는 값의 크기에 비례하므로 절대값으로 재면 층이 넓어지거나
// 씨앗이 바뀔 때마다 허용치를 손봐야 한다. 기준 활성의 최대 크기로 나눈다(1 아래로는 나누지 않는다).
constexpr float COOP_FLOAT32_TOLERANCE = 1.0e-5F;
constexpr float COOP_FLOAT16_TOLERANCE = 2.0e-2F;
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
    // 0 이 아니면 Huber 가 **정확히 꺾이는 지점**을 밟도록 값을 짓는다.
    float kink = 0.0F;
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

// **협력 행렬 타일을 실제로 밟는 모양.** 앞의 선형 검사들은 배치·출력·입력이 모두 16 보다 작아, 가속
// 변종을 켜도 전부 «가장자리 타일» 로 떨어져 스칼라 갈래만 돈다 — 네 번째 열이 0 으로 보이지만 아무
// 것도 검사하지 않은 것이다. 그래서 세 축을 일부러 어긋나게 잡는다.
//
//   배치 20  = 16(온전한 타일) + 4(가장자리)
//   출력 21  = 16(온전한 타일) + 5(가장자리)
//   입력 37  = 16 x 2(협력 행렬이 접는 부분) + 5(스칼라로 미리 세는 꼬리)
//
// 세 갈래(온전한 타일, 꼬리, 가장자리)가 한 표 안에서 모두 돈다.
Case makeCoopTiles() {
    Case item;
    item.name = "coop tiles";
    GraphBuilder builder(item.graph);
    // 앞에 층을 하나 둔다. 뒤 층의 입력이 경사를 받는 활성이라야 dx 갈래가 돌고, 그래야 이 표가
    // 네 열을 모두 검사한다(입력 텐서를 바로 물리면 dx 가 한 번도 돌지 않는다).
    uint32_t input = builder.addInput(20, 5, 1, 1);
    uint32_t stemWeight = builder.addParameter(37, 5, 1, 1);
    uint32_t stemBias = builder.addParameter(37, 1, 1, 1);
    // 이 층은 입력 폭이 5 라 온전한 k 블록이 하나도 없다 — 꼬리만으로 도는 갈래를 밟는다.
    uint32_t stem = builder.addLinear(input, stemWeight, stemBias);
    uint32_t weight = builder.addParameter(21, 37, 1, 1);
    uint32_t bias = builder.addParameter(21, 1, 1, 1);
    if (builder.addLinear(stem, weight, bias) == gfx::NO_TENSOR) {
        item.graph = Graph{};
    }
    item.inputs = {input};
    return item;
}

// 온전한 타일이 **여럿** 나오는 모양. 위 coop tiles 는 온전한 타일이 하나뿐이라 타일 번호를 행·열로
// 푸는 산술이 사실상 검사되지 않는다(0 은 어떤 나눗셈으로도 0 이다).
//
//   배치 33  = 16 x 2 + 1(가장자리)
//   출력 48  = 16 x 3, 가장자리 없음
//   입력 64  = 16 x 4, 꼬리 없음
Case makeCoopWide() {
    Case item;
    item.name = "coop wide";
    GraphBuilder builder(item.graph);
    uint32_t input = builder.addInput(33, 5, 1, 1);
    uint32_t stemWeight = builder.addParameter(64, 5, 1, 1);
    uint32_t stemBias = builder.addParameter(64, 1, 1, 1);
    uint32_t stem = builder.addLinear(input, stemWeight, stemBias);
    uint32_t weight = builder.addParameter(48, 64, 1, 1);
    uint32_t bias = builder.addParameter(48, 1, 1, 1);
    if (builder.addLinear(stem, weight, bias) == gfx::NO_TENSOR) {
        item.graph = Graph{};
    }
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

// layernorm. 이득·편향이 파라미터이고, 앞에 학습하는 층을 두어 dx 가 실제로 돈다.
//
// **이득 하나를 두 layernorm 이 나눠 쓴다.** 그래야 이득·편향 경사의 «이미 들어 있던 값에서 출발한다» 가
// 검사된다(한 번만 쓰이면 0 + s 와 s + 0 이 비트까지 같다). 얼린 이득도 함께 둔다.
Case makeLayerNorm() {
    Case item;
    item.name = "layernorm";
    GraphBuilder builder(item.graph);
    uint32_t input = builder.addInput(5, 9, 1, 1);
    uint32_t stemWeight = builder.addParameter(7, 9, 1, 1);
    uint32_t stemBias = builder.addParameter(7, 1, 1, 1);
    uint32_t stem = builder.addLinear(input, stemWeight, stemBias);
    uint32_t gain = builder.addParameter(7, 1, 1, 1);
    uint32_t shift = builder.addParameter(7, 1, 1, 1);
    // 셋을 **나란히** 둔다. 이어 쌓으면 1 ULP 차이가 층마다 증폭돼(실측 3e-5) 허용치가 무엇을 재는지
    // 알 수 없게 된다. 나란히 두면 누적과 얼림은 그대로 검사하면서 오차는 한 층 몫으로 남는다.
    uint32_t first = builder.addLayerNorm(stem, gain, shift);
    // 같은 이득·편향을 다른 입력에 한 번 더. 이득 경사가 두 갈래에서 쌓인다.
    uint32_t second = builder.addLayerNorm(builder.addRelu(stem), gain, shift);
    uint32_t frozenGain = builder.addParameter(7, 1, 1, 1);
    uint32_t liveShift = builder.addParameter(7, 1, 1, 1);
    builder.detach(frozenGain);
    uint32_t frozen = builder.addLayerNorm(stem, frozenGain, liveShift);
    if (builder.addAdd(builder.addAdd(first, second), frozen) == gfx::NO_TENSOR) {
        item.graph = Graph{};
    }
    item.inputs = {input};
    item.tolerance = LAYERNORM_TOLERANCE;
    return item;
}

// **작업 그룹(128)보다 큰 layernorm.** dx 는 행마다, 이득·편향은 특징마다 디스패치되므로 두 축을 각각
// 넘겨야 한다. 그러지 않으면 디스패치 수를 1 로 줄여도 남는 스레드가 범위 밖이라 티가 나지 않는다.
Case makeWideLayerNorm() {
    Case item;
    item.name = "wide lnorm";
    GraphBuilder builder(item.graph);
    uint32_t input = builder.addInput(130, 5, 1, 1);
    uint32_t stemWeight = builder.addParameter(140, 5, 1, 1);
    uint32_t stemBias = builder.addParameter(140, 1, 1, 1);
    uint32_t stem = builder.addLinear(input, stemWeight, stemBias);
    uint32_t gain = builder.addParameter(140, 1, 1, 1);
    uint32_t shift = builder.addParameter(140, 1, 1, 1);
    if (builder.addLayerNorm(stem, gain, shift) == gfx::NO_TENSOR) {
        item.graph = Graph{};
    }
    item.inputs = {input};
    // 좁은 판(5x7)보다 허용치가 크다. 행 130 개 x 특징 140 개라 이득 경사에 쌓이는 항이 그만큼 많고,
    // 나눗셈·sqrt 의 1 ULP 가 그 길이만큼 자란다. 허용치를 «재 보고 붙이는» 것이라 실측을 적어 둔다:
    // 좁은 판 1.9e-6, 이 판 7.6e-6.
    item.tolerance = 2.0e-5F;
    return item;
}

// 접는 원소 수가 작업 그룹보다 많은 손실. 역전파가 원소마다 스레드 하나라 그쪽 디스패치 수를 건다.
Case makeWideReduce() {
    Case item;
    item.name = "wide mse";
    GraphBuilder builder(item.graph);
    uint32_t input = builder.addInput(40, 5, 1, 1);
    uint32_t weight = builder.addParameter(7, 5, 1, 1);
    uint32_t bias = builder.addParameter(7, 1, 1, 1);
    uint32_t prediction = builder.addLinear(input, weight, bias);
    uint32_t targetWeight = builder.addParameter(7, 5, 1, 1);
    uint32_t targetBias = builder.addParameter(7, 1, 1, 1);
    uint32_t target = builder.addLinear(input, targetWeight, targetBias);
    if (builder.addMse(prediction, target) == gfx::NO_TENSOR) {
        item.graph = Graph{};
    }
    item.inputs = {input};
    item.tolerance = DIVIDE_TOLERANCE;
    return item;
}

// 손실과 평균. 스칼라 하나로 접는 연산들이다.
//
// **목표도 학습하는 층에서 온다.** 목표가 INPUT 이면 손실의 «목표 쪽 경사» 갈래가 죽은 코드로 남는다.
Case makeReduce(const char* name, OpKind kind, float delta) {
    Case item;
    item.name = name;
    GraphBuilder builder(item.graph);
    uint32_t input = builder.addInput(6, 4, 1, 1);
    uint32_t weight = builder.addParameter(3, 4, 1, 1);
    uint32_t bias = builder.addParameter(3, 1, 1, 1);
    uint32_t prediction = builder.addLinear(input, weight, bias);
    uint32_t targetWeight = builder.addParameter(3, 4, 1, 1);
    uint32_t targetBias = builder.addParameter(3, 1, 1, 1);
    // 목표는 **학습하는 층**에서 온다(그래야 손실의 목표 쪽 경사 갈래가 산다). tanh 를 쓰지 않는 것이
    // 요점이다 — 근사가 갈리는 연산을 끼우면 손실 자체를 «정확히 0» 으로 걸 수 없다.
    uint32_t target = builder.addLinear(input, targetWeight, targetBias);
    uint32_t loss = NO_TENSOR;
    switch (kind) {
    case OpKind::MSE:
        loss = builder.addMse(prediction, target);
        break;
    case OpKind::HUBER:
        loss = builder.addHuber(prediction, target, delta);
        break;
    default:
        loss = builder.addMean(builder.addMin2(prediction, target));
        break;
    }
    // 손실 뒤에 배율을 얹어 접는 연산이 받는 상류 경사를 1 이 아니게 만든다.
    if (builder.addScale(loss, 0.375F) == gfx::NO_TENSOR) {
        item.graph = Graph{};
    }
    item.inputs = {input};
    item.tolerance = DIVIDE_TOLERANCE;
    return item;
}

// **Huber 의 꺾이는 지점을 정확히 밟는다.** 무작위 값으로는 |오차| == delta 가 확률 0 이라, «그 지점에서는
// 안쪽 식을 쓴다» 는 규약(<= 인가 < 인가)이 한 번도 검사되지 않는다. 예측을 정확히 0 으로 만들고 목표를
// -delta 로 두면 오차가 자리마다 정확히 delta 다.
Case makeHuberKink(float delta) {
    Case item;
    item.name = "huber kink";
    item.kink = delta;
    GraphBuilder builder(item.graph);
    uint32_t input = builder.addInput(4, 3, 1, 1);
    uint32_t stemWeight = builder.addParameter(3, 3, 1, 1);
    uint32_t stemBias = builder.addParameter(3, 1, 1, 1);
    // 가중치와 편향이 0 이라 예측이 정확히 0 이다. 경사는 그대로 흐른다(dW = dy*x).
    uint32_t prediction = builder.addLinear(input, stemWeight, stemBias);
    uint32_t offset = builder.addParameter(4, 3, 1, 1);
    if (builder.addHuber(prediction, offset, delta) == gfx::NO_TENSOR) {
        item.graph = Graph{};
    }
    item.inputs = {input};
    item.tolerance = DIVIDE_TOLERANCE;
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

// ---- polyak 의 갈래들
//
// 열 걸음 검사가 tau 1 / 0 / 0.3 을 모두 부르는데도 잡지 못하는 자리가 셋 있다.
//
//   - **자르기**: 세 값 모두 이미 [0, 1] 안이라 clamp 가 무동작이다. 범위 밖 tau 를 줘야 산다.
//   - **tau == 1 의 대입**: a + 1*(b - a) 는 두 값의 크기가 비슷하면 b 와 정확히 같다. 크기 차가
//     클 때만 갈린다(CPU 쪽 testPolyak 이 1e30 으로 그것을 본다).
//   - **tau == 0 의 이른 반환**: 값으로는 어차피 같다. 남는 뜻은 NaN 을 곱하지 않는다는 것뿐이라
//     여기서도 잡히지 않는다. 동치 돌연변이로 둔다.
//
// 그래서 극단적인 크기와 범위 밖 tau 로 따로 본다. 표는 버퍼를 잡으려고 두는 껍데기다.
bool runPolyakEdges(NeuralExecutor& executor, HeadlessCompute& compute) {
    constexpr uint32_t SPAN = 40;
    Graph graph;
    GraphBuilder builder(graph);
    uint32_t block = builder.addParameter(SPAN * 8, 1, 1, 1);
    uint32_t input = builder.addInput(1, 1, 1, 1);
    // 표가 유효하려면 연산이 하나는 있어야 한다. 파라미터는 버퍼 크기를 잡으려고 둔 것이다.
    if (builder.addScale(input, 1.0F) == NO_TENSOR || !validateForward(graph)) {
        return false;
    }
    (void)block;
    if (!executor.build(graph)) {
        return false;
    }

    std::vector<float> parameters(graph.parameterCount, 0.0F);
    for (size_t i = 0; i < parameters.size(); ++i) {
        // 크기를 극단적으로 벌린다. a + (b - a) 가 b 가 되지 않는 자리를 만들려는 것이다.
        parameters[i] = (i % 2 == 0) ? 1.0e30F * neuralGaussian(41, i) : 1.0e-8F * neuralGaussian(43, i);
    }
    std::copy(parameters.begin(), parameters.end(), executor.parameterStaging());
    std::fill_n(executor.activationStaging(), graph.activationCount, 0.0F);
    std::fill_n(executor.activationGradientStaging(), graph.activationCount, 0.0F);

    // (원본, 목적지, tau) 넷. 범위 밖 둘이 자르기를 밟는다.
    struct Move {
        uint32_t from;
        uint32_t to;
        float tau;
    };
    const Move MOVES[] = {
        {0, SPAN * 4, 1.0F}, {SPAN, SPAN * 5, 0.0F}, {SPAN * 2, SPAN * 6, 1.75F}, {SPAN * 3, SPAN * 7, -0.5F}};

    compute.submit([&](VkCommandBuffer commandBuffer, uint64_t) {
        executor.recordClearGradients(commandBuffer);
        executor.recordUpload(commandBuffer);
        for (const Move& move : MOVES) {
            executor.recordPolyak(commandBuffer, move.from, move.to, SPAN, move.tau);
        }
        executor.recordDownload(commandBuffer);
    });
    executor.invalidateReadback();

    for (const Move& move : MOVES) {
        polyakStep(move.tau, SPAN, parameters.data() + move.from, parameters.data() + move.to);
    }
    float worst = 0.0F;
    bool exact = true;
    for (size_t i = 0; i < parameters.size(); ++i) {
        float measured = executor.parameterResult()[i];
        exact = exact && measured == parameters[i];
        if (std::isfinite(parameters[i]) && std::isfinite(measured)) {
            worst = std::max(worst, std::abs(parameters[i] - measured));
        }
    }
    std::printf("  polyak 갈래(tau 1 / 0 / 1.75 / -0.5): %s\n", exact ? "비트까지 같음" : "갈림");
    if (!exact) {
        std::printf("      최대 차 %.3e\n", static_cast<double>(worst));
    }
    return exact;
}

// ---- 열 걸음 등가 검사
//
// **이 스택의 관문이다.** 지금까지는 연산 하나하나를 한 걸음씩 견줬다. 여기서는 CPU 기준과 GPU 가 같은
// 고정 배치로 **열 걸음을 학습하고** 가중치가 같은지 본다. 순전파만 맞고 역전파나 최적화가 틀린 경우,
// 그리고 걸음 사이에 상태가 어긋나는 경우(모멘트, 편향 보정, 지우기)가 여기서만 드러난다.
//
// 열 걸음이 **명령 버퍼 하나**에 들어간다. 실제 학습도 그 꼴이라 배리어 배치까지 그대로 검사된다.
bool runTenSteps(NeuralExecutor& executor, HeadlessCompute& compute) {
    // 에이전트와 같은 부품을 한 표에 모은다: 합성곱, 펴기, 선형, layernorm, tanh, ReLU, 그리고 손실.
    Graph graph;
    GraphBuilder builder(graph);
    uint32_t input = builder.addInput(4, 2, 8, 8);
    uint32_t convWeight = builder.addParameter(5, 2, 3, 3);
    uint32_t convBias = builder.addParameter(5, 1, 1, 1);
    uint32_t features = builder.addRelu(builder.addConv2d(input, convWeight, convBias, 2, 1));
    uint32_t flat = builder.addReshape(features, 4, builder.featureCount(features), 1, 1);
    uint32_t trunkWeight = builder.addParameter(6, builder.featureCount(flat), 1, 1);
    uint32_t trunkBias = builder.addParameter(6, 1, 1, 1);
    uint32_t gain = builder.addParameter(6, 1, 1, 1);
    uint32_t shift = builder.addParameter(6, 1, 1, 1);
    uint32_t trunk =
        builder.addTanh(builder.addLayerNorm(builder.addLinear(flat, trunkWeight, trunkBias), gain, shift));
    uint32_t headWeight = builder.addParameter(1, 6, 1, 1);
    uint32_t headBias = builder.addParameter(1, 1, 1, 1);
    uint32_t prediction = builder.addLinear(trunk, headWeight, headBias);
    uint32_t target = builder.addInput(4, 1, 1, 1);
    uint32_t loss = builder.addMse(prediction, target);
    if (loss == NO_TENSOR || !validate(graph)) {
        std::printf("  열 걸음: 표를 짓지 못했습니다\n");
        return false;
    }
    if (!executor.build(graph) || executor.unsupportedOps() != 0) {
        std::printf("  열 걸음: GPU 커널이 없는 연산이 있습니다\n");
        return false;
    }

    // 파라미터 배열을 여러 구간으로 나눈다. **최적화기를 둘 두고 둘 다 0 이 아닌 자리에서 시작하게**
    // 하는 것이 요점이다 — 하나만, 그것도 0 부터 두면 rangeBegin 이 죽은 값이 되어 «구간을 밟는다» 가
    // 검사되지 않는다(에이전트에서는 크리틱·액터가 서로 다른 자리에서 시작한다).
    auto count = static_cast<uint32_t>(graph.parameterCount);
    uint32_t quarter = count / 4;
    if (quarter == 0) {
        std::printf("  열 걸음: 구간을 나눌 수 없습니다\n");
        return false;
    }
    // [0, q) 는 아무 최적화기도 밟지 않는다(polyak 의 원본이다). [q, 2q) 와 [2q, 3q) 를 최적화기 둘이
    // 각각 맡고, [3q, 4q) 는 polyak 의 목적지다.
    uint32_t firstAdamBegin = quarter;
    uint32_t secondAdamBegin = quarter * 2;
    uint32_t polyakSource = 0;
    uint32_t polyakTarget = quarter * 3;
    if (!executor.reserveMoments(0, adamMomentCount(quarter)) ||
        !executor.reserveMoments(1, adamMomentCount(quarter))) {
        return false;
    }

    std::vector<float> parameters(graph.parameterCount, 0.0F);
    std::vector<float> activations(graph.activationCount, 0.0F);
    std::vector<float> parameterGradients(graph.parameterCount, 0.0F);
    std::vector<float> activationGradients(graph.activationCount, 0.0F);
    std::vector<float> firstMoments(adamMomentCount(quarter), 0.0F);
    std::vector<float> secondMoments(adamMomentCount(quarter), 0.0F);
    initializeParameters(graph, 29, parameters.data());
    for (uint32_t tensor : {input, target}) {
        const Tensor& shape = graph.tensors[tensor];
        for (uint32_t i = 0; i < shape.count(); ++i) {
            activations[shape.offset + i] = 0.9F * neuralGaussian(31, shape.offset + i);
        }
    }

    // GPU 는 CPU 가 손대기 전의 값을 받는다. 경사 씨앗은 «손실 자리에 1, 나머지는 0» 이라 CPU 의
    // backward 가 하는 일과 같다.
    std::copy(parameters.begin(), parameters.end(), executor.parameterStaging());
    std::copy(activations.begin(), activations.end(), executor.activationStaging());
    std::fill_n(executor.activationGradientStaging(), graph.activationCount, 0.0F);
    executor.activationGradientStaging()[graph.tensors[loss].offset] = 1.0F;

    // 최적화기 둘을 **다르게** 설정한다. 같으면 둘을 맞바꿔도 결과가 같다.
    AdamSettings first;
    first.learningRate = 5.0e-3F;
    AdamSettings second;
    second.learningRate = 1.5e-3F;
    second.beta1 = 0.8F;
    constexpr uint32_t STEPS = 10;
    // polyak 의 세 갈래를 모두 밟는다. tau 1 은 대입, 0 은 아무 것도, 그 사이는 보간이다. 목적지 구간을
    // 셋으로 나눠 각각 다른 tau 로 끌어당긴다.
    uint32_t third = quarter / 3;
    if (third == 0) {
        std::printf("  열 걸음: polyak 구간을 셋으로 나눌 수 없습니다\n");
        return false;
    }

    compute.submit([&](VkCommandBuffer commandBuffer, uint64_t) {
        for (uint32_t step = 1; step <= STEPS; ++step) {
            executor.recordClearGradients(commandBuffer);
            if (step == 1) {
                // 가중치와 입력은 한 번만 올린다. 걸음마다 올리면 학습한 것을 도로 덮어쓴다.
                executor.recordUpload(commandBuffer);
                // **모멘트를 일부러 더럽힌 뒤 지운다.** 갓 잡은 장치 메모리의 내용은 정해져 있지 않은데
                // Adam 이 첫 걸음에서 그것을 읽는다. 이 드라이버는 새 페이지를 0 으로 주므로 지우기를
                // 빼도 티가 나지 않는다 — 검사가 그 고침을 볼 수 있으려면 먼저 더럽혀야 한다.
                executor.recordClearMoments(commandBuffer, 0, 1.0F);
                executor.recordClearMoments(commandBuffer, 1, -2.5F);
                executor.recordClearMoments(commandBuffer, 0);
                executor.recordClearMoments(commandBuffer, 1);
            }
            executor.recordUploadGradientSeed(commandBuffer);
            executor.recordForward(commandBuffer);
            executor.recordBackward(commandBuffer);
            executor.recordAdam(commandBuffer, 0, firstAdamBegin, quarter, first, step);
            executor.recordAdam(commandBuffer, 1, secondAdamBegin, quarter, second, step);
            executor.recordPolyak(commandBuffer, polyakSource, polyakTarget, third, 1.0F);
            executor.recordPolyak(commandBuffer, polyakSource + third, polyakTarget + third, third, 0.0F);
            executor.recordPolyak(commandBuffer, polyakSource + third * 2, polyakTarget + third * 2, third, 0.3F);
        }
        executor.recordDownload(commandBuffer);
    });
    executor.invalidateReadback();

    float firstLoss = 0.0F;
    float lastLoss = 0.0F;
    for (uint32_t step = 1; step <= STEPS; ++step) {
        forward(graph, parameters.data(), activations.data());
        lastLoss = activations[graph.tensors[loss].offset];
        if (step == 1) {
            firstLoss = lastLoss;
        }
        backward(graph, parameters.data(), activations.data(), parameterGradients.data(), activationGradients.data());
        adamStep(first,
                 step,
                 quarter,
                 parameterGradients.data() + firstAdamBegin,
                 firstMoments.data(),
                 parameters.data() + firstAdamBegin);
        adamStep(second,
                 step,
                 quarter,
                 parameterGradients.data() + secondAdamBegin,
                 secondMoments.data(),
                 parameters.data() + secondAdamBegin);
        polyakStep(1.0F, third, parameters.data() + polyakSource, parameters.data() + polyakTarget);
        polyakStep(0.0F, third, parameters.data() + polyakSource + third, parameters.data() + polyakTarget + third);
        polyakStep(
            0.3F, third, parameters.data() + polyakSource + third * 2, parameters.data() + polyakTarget + third * 2);
    }

    // **성한 수인지 먼저 본다.** std::max(x, NaN) 은 x 라, NaN 이 섞이면 «최대 차 0» 으로 보인다.
    auto allFinite = [](const float* values, size_t total) {
        for (size_t i = 0; i < total; ++i) {
            if (!std::isfinite(values[i])) {
                return false;
            }
        }
        return true;
    };
    if (!allFinite(parameters.data(), parameters.size()) || !allFinite(executor.parameterResult(), parameters.size()) ||
        !allFinite(activations.data(), activations.size()) ||
        !allFinite(executor.activationResult(), activations.size()) || !std::isfinite(lastLoss)) {
        std::printf("      성하지 않은 수가 나왔습니다\n");
        return false;
    }

    // **셋을 따로 잰다.** 파라미터만 보면 어느 최적화기도 밟지 않는 구간이 거저 맞고, 잔차가 어디서
    // 오는지도 알 수 없다.
    float worstParameter = compare(parameters, executor.parameterResult());
    float worstActivation = compare(activations, executor.activationResult());
    float worstGradient = compare(parameterGradients, executor.parameterGradientResult());
    std::printf("  열 걸음 뒤 최대 차 — 가중치 %.3e, 활성 %.3e, 경사 %.3e  (손실 %.5f -> %.5f, 가중치 %zu 개)\n",
                static_cast<double>(worstParameter),
                static_cast<double>(worstActivation),
                static_cast<double>(worstGradient),
                static_cast<double>(firstLoss),
                static_cast<double>(lastLoss),
                parameters.size());
    // 학습이 실제로 일어났는지 먼저 본다. 아무 것도 안 움직였으면 «같다» 가 뜻이 없다.
    if (!(lastLoss < firstLoss)) {
        std::printf("      열 걸음 동안 손실이 줄지 않았습니다\n");
        return false;
    }
    // 비트로 같을 수는 없다. layernorm 의 나눗셈·sqrt 가 있고, Adam 의 갱신 줄도 나눗셈을 지난다. 그
    // 잔차가 열 걸음 동안 얼마나 자라는지가 여기서 재는 값이다.
    float worst = std::max({worstParameter, worstActivation, worstGradient});
    if (!(worst < 1.0e-4F)) {
        std::printf("      허용치 1.0e-04 를 넘었습니다\n");
        return false;
    }
    return true;
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
    cases.push_back(makeCoopTiles());
    cases.push_back(makeCoopWide());
    cases.push_back(makeConv("conv s1 p0", 1, 0, 3, 3));
    cases.push_back(makeConv("conv s2 p1", 2, 1, 3, 3));
    cases.push_back(makeConv("conv k2x3 s2", 2, 2, 2, 3));
    cases.push_back(makeSharedConv());
    cases.push_back(makeWideConv());
    cases.push_back(makeLayerNorm());
    cases.push_back(makeReduce("mse", OpKind::MSE, 0.0F));
    cases.push_back(makeReduce("huber wide", OpKind::HUBER, 4.0F));
    cases.push_back(makeReduce("huber narrow", OpKind::HUBER, 0.05F));
    cases.push_back(makeReduce("mean", OpKind::MEAN, 0.0F));
    cases.push_back(makeWideLayerNorm());
    cases.push_back(makeWideReduce());
    cases.push_back(makeHuberKink(0.75F));
    cases.push_back(makeSharedLinear());
    cases.push_back(makeFrozenLinear());

    std::printf("신경망 자기 검사 (CPU 기준 대 GPU)\n");
    // 네 번째 열은 협력 행렬(텐서 코어) 변종이다. **비트로 같지 않은 것이 정상이다** — k 를 타일로 접어
    // 덧셈 순서가 다르고, fp16 모양은 A/B 를 반정밀도로 낮춘다. 그래서 허용치가 앞 세 열과 따로 논다.
    if (executor.cooperativeAvailable()) {
        std::printf("  협력 행렬 %ux%ux%u (%s A/B, fp32 누산기), 허용치 %.1e\n",
                    context.caps.coopM,
                    context.caps.coopN,
                    context.caps.coopK,
                    context.caps.coopFloat32 ? "fp32" : "fp16",
                    static_cast<double>(context.caps.coopFloat32 ? COOP_FLOAT32_TOLERANCE : COOP_FLOAT16_TOLERANCE));
    } else {
        std::printf("  협력 행렬: 없음 (네 번째 열은 비운다)\n");
    }
    std::printf("  %-12s %10s %10s %10s %12s\n", "연산", "활성", "파라미터 경사", "활성 경사", "협력 행렬(상대)");
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
        if (item.kink != 0.0F) {
            // 예측이 0 이 되도록 앞 층을 0 으로 두고, 목표를 -delta 로 채운다. 오차가 정확히 +delta 다.
            std::fill(parameters.begin(), parameters.end(), 0.0F);
            const Tensor& offset = graph.tensors[graph.parameterTensors().back()];
            for (uint32_t i = 0; i < offset.count(); ++i) {
                parameters[offset.offset + i] = -item.kink;
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

        // 협력 행렬 변종은 **선형 층 순전파만** 갈아 끼운다. 그래서 그 연산이 없는 표는 견줄 것이 없다.
        bool hasLinear = std::ranges::any_of(graph.ops, [](const Op& op) { return op.kind == OpKind::LINEAR; });
        float coopError = 0.0F;
        bool coopRan = false;
        if (executor.cooperativeAvailable() && hasLinear) {
            executor.setCooperative(true);
            std::copy(parameters.begin(), parameters.end(), executor.parameterStaging());
            std::copy(seedActivations.begin(), seedActivations.end(), executor.activationStaging());
            std::copy(seedGradients.begin(), seedGradients.end(), executor.activationGradientStaging());
            // **역전파까지 돌린다.** 순전파만 돌리면 «가속 변종을 역전파에도 잘못 쓴다» 는 결함이 드러나지
            // 않는다 — 그 커널은 flags 를 보지 않고 언제나 순전파를 하므로, 잘못 불리면 경사가 씨앗
            // 그대로 남는다. 역전파 자체는 FMA 경로 그대로라, 경사는 순전파가 낸 차이만큼만 흔들려야 한다.
            compute.submit([&](VkCommandBuffer commandBuffer, uint64_t) {
                executor.recordClearGradients(commandBuffer);
                executor.recordUpload(commandBuffer);
                executor.recordUploadGradientSeed(commandBuffer);
                executor.recordForward(commandBuffer);
                executor.recordBackward(commandBuffer);
                executor.recordDownload(commandBuffer);
            });
            executor.invalidateReadback();
            // 기준의 최대 크기로 나눈다. fp16 오차는 값에 비례하므로 절대값으로 재면 층 폭에 따라
            // 허용치가 떠다닌다. 활성과 파라미터 경사를 각각 제 크기로 재고 큰 쪽을 쓴다.
            auto relative = [](const std::vector<float>& expected, const float* measured) {
                float scale = 1.0F;
                for (float value : expected) {
                    scale = std::max(scale, std::abs(value));
                }
                return compare(expected, measured) / scale;
            };
            coopError = std::max(relative(activations, executor.activationResult()),
                                 relative(parameterGradients, executor.parameterGradientResult()));
            coopRan = true;
            executor.setCooperative(false);
        }

        if (coopRan) {
            std::printf("  %-12s %10.3e %10.3e %10.3e %12.3e\n",
                        item.name,
                        static_cast<double>(activationError),
                        static_cast<double>(parameterError),
                        static_cast<double>(gradientError),
                        static_cast<double>(coopError));
        } else {
            std::printf("  %-12s %10.3e %10.3e %10.3e %12s\n",
                        item.name,
                        static_cast<double>(activationError),
                        static_cast<double>(parameterError),
                        static_cast<double>(gradientError),
                        "-");
        }
        float worst = std::max({activationError, parameterError, gradientError});
        if (!(worst <= item.tolerance)) {
            std::printf("      허용치 %.3e 를 넘었습니다\n", static_cast<double>(item.tolerance));
            ok = false;
        }
        float coopTolerance = context.caps.coopFloat32 ? COOP_FLOAT32_TOLERANCE : COOP_FLOAT16_TOLERANCE;
        if (coopRan && !(coopError <= coopTolerance)) {
            std::printf("      협력 행렬이 허용치 %.3e 를 넘었습니다\n", static_cast<double>(coopTolerance));
            ok = false;
        }
    }
    // 마지막이 이 스택의 관문이다 — polyak 의 갈래들과 열 걸음 등가 검사. 둘 다 표를 다시 올리므로
    // 위 경우들 뒤에 둔다.
    ok = runPolyakEdges(executor, compute) && ok;
    ok = runTenSteps(executor, compute) && ok;
    std::fflush(stdout);
    return ok;
}

} // namespace gfx
