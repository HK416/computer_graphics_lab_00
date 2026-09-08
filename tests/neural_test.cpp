// 신경망의 순수 계산. Vulkan 없이 CPU 기준 구현만 본다.
//
// 이 파일이 이 저장소에서 «연산 하나하나의 역전파가 맞다» 를 말하는 근거다. GPU 커널은 여기서 검증한
// CPU 기준과 대조해 맞춘다. 그래서 검사는 전부 **중앙 유한차분**이다(gradient_check.h) — 손으로 유도한
// 수식을 손으로 유도한 수식과 견주면 같은 실수를 두 번 하게 된다.
//
// 연산을 엮어 만든 **에이전트 그래프** 는 rl_agent_test 가 같은 방식으로 본다.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "gfx/neural_math.h"
#include "gradient_check.h"

namespace {

// ---- 연산별 검사. 모양은 일부러 비대칭으로 잡아 전치와 첨자 밀림이 드러나게 한다.
//
// 모든 그래프에 규칙이 둘 있다.
//
// 1. **시험할 연산 앞에 학습하는 층을 하나 둔다.** 그러지 않으면 그 연산의 «입력에 대한 경사»(dX)가
//    아무 파라미터에도 닿지 않아 통째로 검사되지 않는다. 앞 층의 가중치 경사가 dX 를 지나야만 나온다.
// 2. **머리의 출력이 둘 이상이다.** 출력이 하나면 전치한 첨자가 우연히 같은 자리를 가리켜 전치 버그가
//    드러나지 않는다.

// 배치 B, 특징 F 를 내는 학습하는 앞단. 시험할 연산의 dX 가 여기 가중치 경사로 흘러야 한다.
uint32_t addStem(gfx::GraphBuilder& builder, uint32_t input, uint32_t features) {
    uint32_t weight = builder.addParameter(features, builder.featureCount(input), 1, 1);
    uint32_t bias = builder.addParameter(features, 1, 1, 1);
    return builder.addLinear(input, weight, bias);
}

// 특징을 둘로 줄여 목표와 견주는 머리. 출력이 둘이라 전치가 드러난다.
uint32_t addHead(gfx::GraphBuilder& builder, gfx::Graph& graph, uint32_t feature, uint32_t target) {
    uint32_t weight = builder.addParameter(2, graph.tensors[feature].dims[1], 1, 1);
    uint32_t bias = builder.addParameter(2, 1, 1, 1);
    uint32_t prediction = builder.addLinear(feature, weight, bias);
    return builder.addMse(prediction, target);
}

double checkConv2d(uint32_t stride, uint32_t pad, uint32_t kernelHeight, uint32_t kernelWidth, uint32_t batch) {
    GradientCheck test;
    gfx::GraphBuilder builder(test.graph);
    // 입력 채널 3, 7x9 (정사각이 아니다).
    uint32_t input = builder.addInput(batch, 3, 7, 9);
    // 앞단도 합성곱이다. 시험할 합성곱의 dX 가 이 가중치의 경사로 흘러야 검사된다.
    uint32_t stemWeight = builder.addParameter(3, 3, 3, 3);
    uint32_t stemBias = builder.addParameter(3, 1, 1, 1);
    uint32_t stem = builder.addConv2d(input, stemWeight, stemBias, 1, 1);
    assert(stem != gfx::NO_TENSOR);

    uint32_t weight = builder.addParameter(4, 3, kernelHeight, kernelWidth);
    uint32_t bias = builder.addParameter(4, 1, 1, 1);
    uint32_t conv = builder.addConv2d(stem, weight, bias, stride, pad);
    assert(conv != gfx::NO_TENSOR);

    const gfx::Tensor& convTensor = test.graph.tensors[conv];
    uint32_t features = convTensor.dims[1] * convTensor.dims[2] * convTensor.dims[3];
    uint32_t flat = builder.addReshape(conv, convTensor.dims[0], features, 1, 1);
    assert(flat != gfx::NO_TENSOR);
    uint32_t target = builder.addInput(batch, 2, 1, 1);
    assert(addHead(builder, test.graph, flat, target) != gfx::NO_TENSOR);
    assert(gfx::validate(test.graph));

    test.inputs = {input, target};
    test.allocate();
    return test.check(11, 24);
}

double checkLinear() {
    GradientCheck test;
    gfx::GraphBuilder builder(test.graph);
    // B=5, K=7, M=3. 셋이 모두 달라야 전치가 드러난다.
    uint32_t input = builder.addInput(5, 4, 1, 1);
    uint32_t stem = addStem(builder, input, 7);
    uint32_t weight = builder.addParameter(3, 7, 1, 1);
    uint32_t bias = builder.addParameter(3, 1, 1, 1);
    uint32_t hidden = builder.addLinear(stem, weight, bias);
    uint32_t target = builder.addInput(5, 2, 1, 1);
    assert(addHead(builder, test.graph, hidden, target) != gfx::NO_TENSOR);

    test.inputs = {input, target};
    test.allocate();
    return test.check(23, 32);
}

// 활성 함수와 원소별 연산을 한 그래프에 몰아 넣는다.
double checkElementwise() {
    GradientCheck test;
    gfx::GraphBuilder builder(test.graph);
    uint32_t input = builder.addInput(4, 5, 1, 1);
    uint32_t stem = addStem(builder, input, 6);
    uint32_t weightA = builder.addParameter(6, 6, 1, 1);
    uint32_t biasA = builder.addParameter(6, 1, 1, 1);
    uint32_t branchA = builder.addRelu(builder.addLinear(stem, weightA, biasA));
    uint32_t weightB = builder.addParameter(6, 6, 1, 1);
    uint32_t biasB = builder.addParameter(6, 1, 1, 1);
    uint32_t branchB = builder.addTanh(builder.addLinear(stem, weightB, biasB));
    uint32_t merged = builder.addAdd(branchA, branchB);
    uint32_t scaled = builder.addScale(merged, 0.75F);
    uint32_t smallest = builder.addMin2(scaled, branchB);
    uint32_t target = builder.addInput(4, 2, 1, 1);
    assert(addHead(builder, test.graph, smallest, target) != gfx::NO_TENSOR);

    test.inputs = {input, target};
    test.allocate();
    return test.check(41, 40);
}

// 원소별 곱과 평균. 시간차 목표 y = r + γ(1-끝)·min(Q1', Q2') 과 액터 손실 -mean(Q) 가 쓰는 짝이다.
double checkMulMean() {
    GradientCheck test;
    gfx::GraphBuilder builder(test.graph);
    uint32_t input = builder.addInput(4, 3, 1, 1);
    // 두 갈래가 **서로 다른 값** 을 내야 한다. 대칭이면 dA 와 dB 의 곱하는 짝을 뒤바꿔도 드러나지 않는다.
    uint32_t left = builder.addTanh(addStem(builder, input, 5));
    uint32_t right = builder.addRelu(addStem(builder, input, 5));
    uint32_t product = builder.addMul(left, right);
    // 경사를 받지 않는 짝과도 곱해 본다. 마스크가 그 자리다 — 곱한 쪽만 경사가 흐르고 값은 그대로 든다.
    uint32_t mask = builder.addInput(4, 5, 1, 1);
    uint32_t masked = builder.addMul(product, mask);
    uint32_t merged = builder.addAdd(masked, left);
    // 평균 뒤에 배율을 얹어 MEAN 이 받는 dy 를 1 이 아니게 만든다. 상류 경사를 무시하면 여기서 걸린다.
    uint32_t loss = builder.addScale(builder.addMean(merged), -0.625F);
    assert(loss != gfx::NO_TENSOR);
    assert(test.graph.tensors[loss].count() == 1);

    test.inputs = {input, mask};
    test.allocate();
    return test.check(83, 32);
}

double checkLayerNorm() {
    GradientCheck test;
    gfx::GraphBuilder builder(test.graph);
    uint32_t input = builder.addInput(3, 5, 1, 1);
    uint32_t stem = addStem(builder, input, 8);
    uint32_t gain = builder.addParameter(8, 1, 1, 1);
    uint32_t shift = builder.addParameter(8, 1, 1, 1);
    uint32_t normalized = builder.addLayerNorm(stem, gain, shift);
    uint32_t target = builder.addInput(3, 2, 1, 1);
    assert(addHead(builder, test.graph, normalized, target) != gfx::NO_TENSOR);

    test.inputs = {input, target};
    test.allocate();
    return test.check(59, 40);
}

double checkConcat() {
    GradientCheck test;
    gfx::GraphBuilder builder(test.graph);
    // 크리틱이 (특징, 행동) 을 잇는 자리. 두 조각의 폭이 달라야 경계가 드러나고, **둘 다 학습하는 층에서
    // 와야** 두 조각의 경사가 모두 검사된다.
    uint32_t input = builder.addInput(4, 3, 1, 1);
    uint32_t feature = addStem(builder, input, 5);
    uint32_t actionWeight = builder.addParameter(2, 3, 1, 1);
    uint32_t actionBias = builder.addParameter(2, 1, 1, 1);
    uint32_t action = builder.addTanh(builder.addLinear(input, actionWeight, actionBias));
    uint32_t joined = builder.addConcat(feature, action);
    assert(test.graph.tensors[joined].dims[1] == 7);
    uint32_t weight = builder.addParameter(3, 7, 1, 1);
    uint32_t bias = builder.addParameter(3, 1, 1, 1);
    uint32_t hidden = builder.addRelu(builder.addLinear(joined, weight, bias));
    uint32_t target = builder.addInput(4, 2, 1, 1);
    assert(addHead(builder, test.graph, hidden, target) != gfx::NO_TENSOR);

    test.inputs = {input, target};
    test.allocate();
    return test.check(71, 32);
}

// Huber 손실. delta 안쪽(제곱)과 바깥쪽(눕는 기울기)을 모두 밟아야 한다 — delta 를 작게 잡으면
// 무작위 오차 대부분이 바깥쪽이고, 크게 잡으면 안쪽이다.
double checkHuber(float delta) {
    GradientCheck test;
    gfx::GraphBuilder builder(test.graph);
    uint32_t input = builder.addInput(4, 3, 1, 1);
    uint32_t prediction = addStem(builder, input, 2);
    uint32_t target = builder.addInput(4, 2, 1, 1);
    assert(builder.addHuber(prediction, target, delta) != gfx::NO_TENSOR);
    assert(gfx::validate(test.graph));

    test.inputs = {input, target};
    test.allocate();
    return test.check(113, 24);
}

// Huber 도 손실이 마지막이 아닌 판과 목표가 학습하는 층에서 오는 판이 있어야 한다. MSE 에만 두면
// Huber 의 dy[0] 곱과 dTarget 갈래가 죽은 코드로 남아 부호를 뒤집어도 드러나지 않는다.
double checkHuberBranches(float delta) {
    GradientCheck test;
    gfx::GraphBuilder builder(test.graph);
    uint32_t input = builder.addInput(4, 3, 1, 1);
    uint32_t prediction = addStem(builder, input, 2);
    // 목표도 학습하는 층에서 온다.
    uint32_t targetWeight = builder.addParameter(2, 3, 1, 1);
    uint32_t targetBias = builder.addParameter(2, 1, 1, 1);
    uint32_t target = builder.addTanh(builder.addLinear(input, targetWeight, targetBias));
    uint32_t loss = builder.addHuber(prediction, target, delta);
    assert(loss != gfx::NO_TENSOR);
    // 손실 뒤에 배율을 얹어 Huber 가 받는 dy 를 1 이 아니게 만든다.
    assert(builder.addScale(loss, 0.375F) != gfx::NO_TENSOR);
    assert(gfx::validate(test.graph));

    test.inputs = {input};
    test.allocate();
    return test.check(127, 24);
}

// 손실이 마지막이 아닐 때. MSE 의 역전파가 «위에서 온 경사»(dy[0])를 곱하는 자리가 여기서만 잠긴다 —
// 손실이 늘 마지막이면 dy[0] 이 항상 1 이라 그 곱을 빼먹어도 드러나지 않는다. 쌍둥이 크리틱의 두 손실을
// 더하거나 MAD 의 alpha 로 가중하는 순간 실제로 밟는 자리다.
double checkScaledLoss() {
    GradientCheck test;
    gfx::GraphBuilder builder(test.graph);
    uint32_t input = builder.addInput(3, 4, 1, 1);
    uint32_t stem = addStem(builder, input, 5);
    uint32_t target = builder.addInput(3, 2, 1, 1);
    uint32_t loss = addHead(builder, test.graph, stem, target);
    assert(loss != gfx::NO_TENSOR);
    // 손실 뒤에 배율을 하나 더 얹는다. 이제 MSE 가 받는 dy 는 1 이 아니라 0.375 다.
    assert(builder.addScale(loss, 0.375F) != gfx::NO_TENSOR);

    test.inputs = {input, target};
    test.allocate();
    return test.check(83, 24);
}

// 목표 쪽으로도 경사가 흐르는 경우. 목표가 늘 경사 없는 입력이면 MSE 의 dTarget 갈래가 죽은 코드라
// 부호가 틀려도 보이지 않는다.
double checkLiveTarget() {
    GradientCheck test;
    gfx::GraphBuilder builder(test.graph);
    uint32_t input = builder.addInput(3, 4, 1, 1);
    uint32_t predictionFeature = addStem(builder, input, 2);
    // 목표도 학습하는 층에서 온다.
    uint32_t targetWeight = builder.addParameter(2, 4, 1, 1);
    uint32_t targetBias = builder.addParameter(2, 1, 1, 1);
    uint32_t target = builder.addTanh(builder.addLinear(input, targetWeight, targetBias));
    assert(builder.addMse(predictionFeature, target) != gfx::NO_TENSOR);

    test.inputs = {input};
    test.allocate();
    return test.check(97, 24);
}

// 동점에서 어느 쪽이 이기는지. GLSL 짝과 같은 규칙을 못 박는다 — 난수 float 는 절대 동점이 나지 않아
// 유한차분으로는 잠기지 않는 자리다.
void testTieRules() {
    // min2 의 두 입력이 **서로 다른 텐서인데 값이 같아야** 규칙이 드러난다. 같은 텐서를 두 번 주면
    // 어느 쪽이 이기든 경사가 같은 자리에 쌓여 구별되지 않는다.
    gfx::Graph graph;
    gfx::GraphBuilder builder(graph);
    uint32_t input = builder.addInput(1, 2, 1, 1);
    uint32_t weight = builder.addParameter(2, 2, 1, 1);
    uint32_t bias = builder.addParameter(2, 1, 1, 1);
    uint32_t first = builder.addLinear(input, weight, bias);
    // 배율 1 이라 값은 같고 텐서는 다르다.
    uint32_t second = builder.addScale(first, 1.0F);
    uint32_t smallest = builder.addMin2(first, second);
    uint32_t headWeight = builder.addParameter(1, 2, 1, 1);
    uint32_t headBias = builder.addParameter(1, 1, 1, 1);
    uint32_t prediction = builder.addLinear(smallest, headWeight, headBias);
    uint32_t target = builder.addInput(1, 1, 1, 1);
    assert(builder.addMse(prediction, target) != gfx::NO_TENSOR);
    assert(gfx::validate(graph));

    std::vector<float> parameters(graph.parameterCount, 0.0F);
    std::vector<float> activations(graph.activationCount, 0.0F);
    std::vector<float> parameterGradients(graph.parameterCount, 0.0F);
    std::vector<float> activationGradients(graph.activationCount, 0.0F);
    parameters[graph.tensors[weight].offset] = 1.0F;
    parameters[graph.tensors[weight].offset + 3] = 1.0F;
    parameters[graph.tensors[headWeight].offset] = 1.0F;
    parameters[graph.tensors[headWeight].offset + 1] = 1.0F;
    activations[graph.tensors[input].offset] = 0.5F;
    activations[graph.tensors[input].offset + 1] = -0.25F;
    gfx::forward(graph, parameters.data(), activations.data());
    assert(gfx::backward(
        graph, parameters.data(), activations.data(), parameterGradients.data(), activationGradients.data()));

    // 규칙은 «같으면 첫째가 이긴다» 다. 둘째(SCALE 출력)에는 경사가 하나도 가지 않아야 한다.
    // GLSL 짝이 다른 규칙을 쓰면 두 엔진의 경사가 갈리므로 여기서 못 박는다.
    for (uint32_t i = 0; i < graph.tensors[second].count(); ++i) {
        assert(activationGradients[graph.tensors[second].offset + i] == 0.0F);
    }
    bool firstTouched = false;
    for (uint32_t i = 0; i < graph.tensors[first].count(); ++i) {
        firstTouched = firstTouched || activationGradients[graph.tensors[first].offset + i] != 0.0F;
    }
    assert(firstTouched);

    // ReLU 는 정확히 0 에서 경사를 주지 않는다.
    gfx::Graph relu;
    gfx::GraphBuilder reluBuilder(relu);
    uint32_t reluInput = reluBuilder.addInput(1, 1, 1, 1);
    uint32_t reluWeight = reluBuilder.addParameter(1, 1, 1, 1);
    uint32_t reluBias = reluBuilder.addParameter(1, 1, 1, 1);
    uint32_t activated = reluBuilder.addRelu(reluBuilder.addLinear(reluInput, reluWeight, reluBias));
    uint32_t reluTarget = reluBuilder.addInput(1, 1, 1, 1);
    assert(reluBuilder.addMse(activated, reluTarget) != gfx::NO_TENSOR);
    std::vector<float> reluParameters(relu.parameterCount, 0.0F);
    std::vector<float> reluActivations(relu.activationCount, 0.0F);
    std::vector<float> reluParameterGradients(relu.parameterCount, 0.0F);
    std::vector<float> reluActivationGradients(relu.activationCount, 0.0F);
    // 가중치 1, 편향 0, 입력 0 이라 전활성값이 정확히 0 이다.
    reluParameters[relu.tensors[reluWeight].offset] = 1.0F;
    reluActivations[relu.tensors[reluInput].offset] = 0.0F;
    reluActivations[relu.tensors[reluTarget].offset] = 1.0F;
    gfx::forward(relu, reluParameters.data(), reluActivations.data());
    assert(gfx::backward(relu,
                         reluParameters.data(),
                         reluActivations.data(),
                         reluParameterGradients.data(),
                         reluActivationGradients.data()));
    assert(reluParameterGradients[relu.tensors[reluWeight].offset] == 0.0F);
    assert(reluParameterGradients[relu.tensors[reluBias].offset] == 0.0F);

    // layernorm 의 엡실론은 순전파와 역전파가 같은 값을 쓰므로 유한차분으로는 원리적으로 못 잡는다.
    // GLSL 짝과 묶인 값이라 여기서 값 자체를 못 박는다(마칭 큐브 표를 셰이더와 견주는 것과 같은 이유다).
    assert(gfx::LAYERNORM_EPSILON == 1.0e-5F);
}

// 텐서의 **메모리 배치 규약**을 못 박는다. 유한차분은 순전파와 역전파를 서로 견주는 것이라, 둘이
// 일관되게 다른 배치를 쓰면(예: 합성곱 가중치를 [Cout][Cin][kw][kh] 로) 아무 것도 걸리지 않는다.
// 그런데 그 규약은 GPU 커널·초기화의 팬인·저장된 가중치가 모두 따라야 하는 것이라, 값이 어디에
// 놓이는지를 손으로 계산한 자리와 견줘 잠근다.
void testMemoryLayout() {
    // 합성곱: 값 NCHW = ((n*C + c)*H + h)*W + w, 가중치 [Cout][Cin][kh][kw].
    gfx::Graph graph;
    gfx::GraphBuilder builder(graph);
    constexpr uint32_t IN_CHANNELS = 2;
    constexpr uint32_t HEIGHT = 4;
    constexpr uint32_t WIDTH = 5;
    constexpr uint32_t OUT_CHANNELS = 3;
    constexpr uint32_t KERNEL_HEIGHT = 2;
    constexpr uint32_t KERNEL_WIDTH = 3;
    uint32_t input = builder.addInput(1, IN_CHANNELS, HEIGHT, WIDTH);
    uint32_t weight = builder.addParameter(OUT_CHANNELS, IN_CHANNELS, KERNEL_HEIGHT, KERNEL_WIDTH);
    uint32_t bias = builder.addParameter(OUT_CHANNELS, 1, 1, 1);
    uint32_t conv = builder.addConv2d(input, weight, bias, 1, 0);
    assert(conv != gfx::NO_TENSOR);
    const gfx::Tensor& out = graph.tensors[conv];
    assert(out.dims[1] == OUT_CHANNELS);
    assert(out.dims[2] == HEIGHT - KERNEL_HEIGHT + 1);
    assert(out.dims[3] == WIDTH - KERNEL_WIDTH + 1);

    std::vector<float> parameters(graph.parameterCount, 0.0F);
    std::vector<float> activations(graph.activationCount, 0.0F);
    // 가중치 한 자리만 켠다: 출력 채널 2, 입력 채널 1, 커널 (ky=1, kx=2).
    // (ky, kx) 는 두 규약이 **갈리는** 자리로 고른다. [Cout][Cin][kh][kw] 와 [Cout][Cin][kw][kh] 가
    // 같은 첨자를 내는 자리(예: ky=1, kx=2 는 2x3 커널에서 둘 다 5 다)를 고르면 규약이 잠기지 않는다.
    constexpr uint32_t OC = 2;
    constexpr uint32_t IC = 1;
    constexpr uint32_t KY = 1;
    constexpr uint32_t KX = 0;
    size_t weightIndex = ((static_cast<size_t>(OC) * IN_CHANNELS + IC) * KERNEL_HEIGHT + KY) * KERNEL_WIDTH + KX;
    size_t swappedIndex = ((static_cast<size_t>(OC) * IN_CHANNELS + IC) * KERNEL_WIDTH + KX) * KERNEL_HEIGHT + KY;
    assert(weightIndex != swappedIndex);
    parameters[graph.tensors[weight].offset + weightIndex] = 1.0F;
    // 입력도 한 화소만 켠다: 채널 1, (y=2, x=3).
    constexpr uint32_t IY = 2;
    constexpr uint32_t IX = 2;
    size_t inputIndex = (static_cast<size_t>(IC) * HEIGHT + IY) * WIDTH + IX;
    activations[graph.tensors[input].offset + inputIndex] = 7.0F;

    gfx::forward(graph, parameters.data(), activations.data());
    // 여백 0, 보폭 1 이므로 그 곱은 출력 (oy = IY - KY, ox = IX - KX) 한 자리에만 나타난다.
    constexpr uint32_t OY = IY - KY;
    constexpr uint32_t OX = IX - KX;
    for (uint32_t oc = 0; oc < out.dims[1]; ++oc) {
        for (uint32_t oy = 0; oy < out.dims[2]; ++oy) {
            for (uint32_t ox = 0; ox < out.dims[3]; ++ox) {
                size_t index = ((static_cast<size_t>(oc) * out.dims[2] + oy) * out.dims[3]) + ox;
                float value = activations[out.offset + index];
                float expected = (oc == OC && oy == OY && ox == OX) ? 7.0F : 0.0F;
                assert(value == expected);
            }
        }
    }

    // 선형: 값 [B][F], 가중치 [M][K]. 가중치 한 자리를 켜 어느 입력이 어느 출력으로 가는지 본다.
    gfx::Graph linear;
    gfx::GraphBuilder linearBuilder(linear);
    uint32_t linearInput = linearBuilder.addInput(2, 4, 1, 1);
    uint32_t linearWeight = linearBuilder.addParameter(3, 4, 1, 1);
    uint32_t linearBias = linearBuilder.addParameter(3, 1, 1, 1);
    uint32_t hidden = linearBuilder.addLinear(linearInput, linearWeight, linearBias);
    std::vector<float> linearParameters(linear.parameterCount, 0.0F);
    std::vector<float> linearActivations(linear.activationCount, 0.0F);
    // W[m=2][k=3] = 1 이면 출력 [n][2] 에 입력 [n][3] 이 실린다.
    linearParameters[linear.tensors[linearWeight].offset + 2 * 4 + 3] = 1.0F;
    linearActivations[linear.tensors[linearInput].offset + 1 * 4 + 3] = 5.0F;
    gfx::forward(linear, linearParameters.data(), linearActivations.data());
    for (uint32_t n = 0; n < 2; ++n) {
        for (uint32_t m = 0; m < 3; ++m) {
            float value = linearActivations[linear.tensors[hidden].offset + n * 3 + m];
            assert(value == ((n == 1 && m == 2) ? 5.0F : 0.0F));
        }
    }

    // 이음: 앞 조각이 낮은 첨자다.
    gfx::Graph concat;
    gfx::GraphBuilder concatBuilder(concat);
    uint32_t head = concatBuilder.addInput(2, 3, 1, 1);
    uint32_t tail = concatBuilder.addInput(2, 2, 1, 1);
    uint32_t joined = concatBuilder.addConcat(head, tail);
    std::vector<float> concatParameters;
    std::vector<float> concatActivations(concat.activationCount, 0.0F);
    concatActivations[concat.tensors[head].offset + 1 * 3 + 2] = 1.0F;
    concatActivations[concat.tensors[tail].offset + 1 * 2 + 0] = 2.0F;
    gfx::forward(concat, concatParameters.data(), concatActivations.data());
    const gfx::Tensor& joinedTensor = concat.tensors[joined];
    assert(concatActivations[joinedTensor.offset + 1 * 5 + 2] == 1.0F);
    assert(concatActivations[joinedTensor.offset + 1 * 5 + 3] == 2.0F);
}

// validate 와 backward 가 «손실이 아닌 그래프» 를 조용히 넘기지 않는지.
void testValidate() {
    // 빈 그래프.
    gfx::Graph empty;
    assert(!gfx::validate(empty));

    // 손실이 없는 그래프. 마지막 출력이 스칼라가 아니다.
    gfx::Graph headless;
    gfx::GraphBuilder headlessBuilder(headless);
    uint32_t input = headlessBuilder.addInput(2, 3, 1, 1);
    uint32_t weight = headlessBuilder.addParameter(3, 3, 1, 1);
    uint32_t bias = headlessBuilder.addParameter(3, 1, 1, 1);
    assert(headlessBuilder.addLinear(input, weight, bias) != gfx::NO_TENSOR);
    assert(!gfx::validate(headless));
    std::vector<float> parameters(headless.parameterCount, 0.0F);
    std::vector<float> activations(headless.activationCount, 0.0F);
    std::vector<float> parameterGradients(headless.parameterCount, 1.0F);
    std::vector<float> activationGradients(headless.activationCount, 1.0F);
    assert(!gfx::backward(
        headless, parameters.data(), activations.data(), parameterGradients.data(), activationGradients.data()));

    // 손실 뒤에 입력을 하나 더 잡은 그래프. 아주 그럴듯한 순서인데 마지막 연산이 INPUT 이 된다.
    gfx::Graph trailing;
    gfx::GraphBuilder trailingBuilder(trailing);
    uint32_t trailingInput = trailingBuilder.addInput(2, 3, 1, 1);
    uint32_t trailingWeight = trailingBuilder.addParameter(1, 3, 1, 1);
    uint32_t trailingBias = trailingBuilder.addParameter(1, 1, 1, 1);
    uint32_t prediction = trailingBuilder.addLinear(trailingInput, trailingWeight, trailingBias);
    uint32_t target = trailingBuilder.addInput(2, 1, 1, 1);
    assert(trailingBuilder.addMse(prediction, target) != gfx::NO_TENSOR);
    assert(gfx::validate(trailing));
    trailingBuilder.addInput(1, 1, 1, 1);
    assert(!gfx::validate(trailing));

    // 파라미터는 뷰로 만들 수 없다. 만들 수 있으면 parameterTensors() 가 같은 저장소를 두 번 센다.
    gfx::Graph views;
    gfx::GraphBuilder viewBuilder(views);
    uint32_t parameter = viewBuilder.addParameter(2, 3, 1, 1);
    assert(viewBuilder.addReshape(parameter, 6, 1, 1, 1) == gfx::NO_TENSOR);
    assert(views.parameterTensors().size() == 1);

    // 손으로 지은 표. 빌더는 잘못된 모양을 애초에 거절하므로, validate 자신의 검사는 이렇게만 볼 수 있다.
    // 파일에서 읽은 표가 이 꼴이면 원소별 연산이 범위 밖을 읽는다.
    {
        gfx::Graph handmade;
        gfx::Tensor wide;
        wide.arena = gfx::Arena::ACTIVATION;
        wide.offset = 0;
        wide.dims[0] = 2;
        wide.dims[1] = 3;
        wide.flags = gfx::TENSOR_GRAD;
        gfx::Tensor tall = wide;
        tall.offset = 6;
        tall.dims[0] = 1;
        tall.dims[1] = 6;
        gfx::Tensor result = wide;
        result.offset = 12;
        handmade.tensors = {wide, tall, result};
        handmade.activationCount = 18;
        gfx::Op input0;
        input0.kind = gfx::OpKind::INPUT;
        input0.output = 0;
        gfx::Op input1 = input0;
        input1.output = 1;
        gfx::Op add;
        add.kind = gfx::OpKind::ADD;
        add.inputs[0] = 0;
        add.inputs[1] = 1;
        add.output = 2;
        handmade.ops = {input0, input1, add};
        // 마지막이 스칼라가 아니라서도 거절되므로, 그 조건만 만족시킨 판을 따로 본다.
        assert(!gfx::validate(handmade));

        gfx::Tensor scalar = result;
        scalar.dims[0] = 1;
        scalar.dims[1] = 1;
        scalar.offset = 12;
        handmade.tensors[2] = scalar;
        gfx::Op mse;
        mse.kind = gfx::OpKind::MSE;
        mse.inputs[0] = 0;
        mse.inputs[1] = 1;
        mse.output = 2;
        handmade.ops = {input0, input1, mse};
        // 이제 마지막은 경사를 받는 스칼라다. 남은 결함은 «두 입력의 모양이 다르다» 하나뿐이다.
        assert(!gfx::validate(handmade));
        // 모양을 맞추면 통과한다(위 거절이 «늘 거절» 이 아니라는 확인).
        handmade.tensors[1].dims[0] = 2;
        handmade.tensors[1].dims[1] = 3;
        assert(gfx::validate(handmade));

        // MUL 도 원소별이라 같은 검사를 받아야 한다.
        gfx::Op mul;
        mul.kind = gfx::OpKind::MUL;
        mul.inputs[0] = 0;
        mul.inputs[1] = 1;
        mul.output = 2;
        gfx::Op loss = mse;
        loss.inputs[0] = 2;
        loss.inputs[1] = 2;
        loss.output = 3;
        gfx::Tensor lossTensor = handmade.tensors[2];
        lossTensor.offset = 13;
        handmade.tensors[2].dims[0] = 2;
        handmade.tensors[2].dims[1] = 3;
        handmade.tensors.push_back(lossTensor);
        handmade.activationCount = 19;
        handmade.ops = {input0, input1, mul, loss};
        assert(gfx::validateForward(handmade));
        // 한쪽 모양만 어긋내면 거절이다. 두 입력을 나란히 훑으므로 범위 밖을 읽게 된다.
        handmade.tensors[1].dims[1] = 6;
        handmade.tensors[1].dims[0] = 1;
        assert(!gfx::validateForward(handmade));
        handmade.tensors[1].dims[0] = 2;
        handmade.tensors[1].dims[1] = 3;
        assert(gfx::validateForward(handmade));

        // 접는 연산(MEAN·MSE·HUBER)의 **출력은 스칼라**여야 한다. 아니면 순전파가 y[0] 에만 쓰고
        // 나머지는 지난 값으로 남아 다음 연산이 쓰레기를 읽는다.
        gfx::Op mean;
        mean.kind = gfx::OpKind::MEAN;
        mean.inputs[0] = 0;
        mean.output = 3;
        handmade.ops = {input0, input1, mul, mean};
        assert(gfx::validateForward(handmade));
        handmade.tensors[3].dims[1] = 3;
        assert(!gfx::validateForward(handmade));
        handmade.tensors[3].dims[1] = 1;
        // 손실도 같다.
        handmade.ops = {input0, input1, mul, loss};
        handmade.tensors[3].dims[1] = 3;
        assert(!gfx::validateForward(handmade));
    }

    // 모양은 원소 수가 아니라 네 축을 견준다.
    gfx::Graph shapes;
    gfx::GraphBuilder shapeBuilder(shapes);
    uint32_t wide = shapeBuilder.addInput(2, 3, 1, 1);
    uint32_t tall = shapeBuilder.addInput(1, 6, 1, 1);
    assert(shapeBuilder.addAdd(wide, tall) == gfx::NO_TENSOR);
    assert(shapeBuilder.addMin2(wide, tall) == gfx::NO_TENSOR);
    assert(shapeBuilder.addMse(wide, tall) == gfx::NO_TENSOR);
}

void testGradients() {
    struct Case {
        const char* name;
        double error;
    };
    std::vector<Case> cases{
        {"conv2d s1 p0", checkConv2d(1, 0, 3, 3, 2)},
        {"conv2d s2 p1", checkConv2d(2, 1, 3, 3, 2)},
        {"conv2d s1 p1", checkConv2d(1, 1, 3, 3, 2)},
        // 커널이 정사각이 아니어야 ky/kx 첨자 밀림이 드러난다. 순·역이 같은 식을 쓰므로 유한차분으로는
        // 그것을 잡을 다른 방법이 없다.
        {"conv2d k3x5", checkConv2d(1, 1, 3, 5, 2)},
        {"conv2d k5x3 s2", checkConv2d(2, 2, 5, 3, 2)},
        // 배치 1. 배치 축을 첨자에서 빼먹은 자리가 드러난다.
        {"conv2d batch1", checkConv2d(1, 1, 3, 3, 1)},
        {"linear", checkLinear()},
        {"elementwise", checkElementwise()},
        {"mul mean", checkMulMean()},
        {"layernorm", checkLayerNorm()},
        {"concat", checkConcat()},
        {"scaled loss", checkScaledLoss()},
        {"live target", checkLiveTarget()},
        {"huber wide", checkHuber(4.0F)},
        {"huber narrow", checkHuber(0.05F)},
        {"huber branches", checkHuberBranches(1.0F)},
        {"huber branches L1", checkHuberBranches(0.05F)},
    };
    std::printf("  연산별 최대 상대 오차 (해석 경사 대 중앙 유한차분)\n");
    for (const Case& item : cases) {
        std::printf("    %-14s %.3e\n", item.name, item.error);
    }
    // 표를 먼저 다 찍고 나서 판정한다. 어느 연산이 어긋났는지 보이지 않으면 고칠 수가 없다.
    std::fflush(stdout);
    for (const Case& item : cases) {
        // 실측은 1e-7 ~ 3e-6 이다. 임계를 여유 있게 두면 «수식이 조금 틀린» 경우를 놓치므로 실측의
        // 열 배 남짓으로 조인다. 진짜 구조적 오류(전치·첨자 밀림)는 1e-1 대로 뜬다.
        assert(item.error < 2.0e-5);
    }
}

// 모양 계산과 빌더가 잘못된 모양을 거절하는지.
void testShapes() {
    assert(gfx::convOutputSize(5, 3, 1, 0) == 3);
    assert(gfx::convOutputSize(5, 3, 2, 1) == 3);
    assert(gfx::convOutputSize(84, 3, 2, 0) == 41);
    assert(gfx::convOutputSize(41, 3, 1, 0) == 39);
    // 커널이 입력보다 크면 0 이다.
    assert(gfx::convOutputSize(2, 3, 1, 0) == 0);
    assert(gfx::convOutputSize(5, 3, 0, 0) == 0);

    gfx::Graph graph;
    gfx::GraphBuilder builder(graph);
    uint32_t input = builder.addInput(1, 3, 8, 8);
    // 채널이 맞지 않는 가중치.
    uint32_t wrongWeight = builder.addParameter(4, 5, 3, 3);
    uint32_t wrongBias = builder.addParameter(4, 1, 1, 1);
    assert(builder.addConv2d(input, wrongWeight, wrongBias, 1, 0) == gfx::NO_TENSOR);
    // 편향 길이가 맞지 않는 경우.
    uint32_t weight = builder.addParameter(4, 3, 3, 3);
    uint32_t shortBias = builder.addParameter(3, 1, 1, 1);
    assert(builder.addConv2d(input, weight, shortBias, 1, 0) == gfx::NO_TENSOR);
    // 선형은 평탄한 것만 받는다.
    uint32_t linearWeight = builder.addParameter(2, 3, 1, 1);
    uint32_t linearBias = builder.addParameter(2, 1, 1, 1);
    assert(builder.addLinear(input, linearWeight, linearBias) == gfx::NO_TENSOR);
    // 크기가 다른 덧셈.
    uint32_t small = builder.addInput(1, 2, 1, 1);
    uint32_t large = builder.addInput(1, 3, 1, 1);
    assert(builder.addAdd(small, large) == gfx::NO_TENSOR);
    // 배치가 다른 이음.
    uint32_t otherBatch = builder.addInput(2, 2, 1, 1);
    assert(builder.addConcat(small, otherBatch) == gfx::NO_TENSOR);
}

// 파라미터 arena 의 오프셋이 이어 붙는지, 파라미터 텐서 목록이 순서를 지키는지.
// 뷰는 자리를 새로 잡지 않고 같은 저장소를 다른 모양으로 본다.
void testReshape() {
    gfx::Graph graph;
    gfx::GraphBuilder builder(graph);
    uint32_t input = builder.addInput(2, 3, 4, 5);
    size_t before = graph.activationCount;
    uint32_t flat = builder.addReshape(input, 2, 60, 1, 1);
    assert(flat != gfx::NO_TENSOR);
    assert(graph.activationCount == before);
    assert(graph.tensors[flat].offset == graph.tensors[input].offset);
    assert(graph.tensors[flat].count() == graph.tensors[input].count());
    assert(graph.ops.size() == 1);
    // 원소 수가 다르면 거절한다.
    assert(builder.addReshape(input, 2, 61, 1, 1) == gfx::NO_TENSOR);
}

void testLayout() {
    gfx::Graph graph;
    gfx::GraphBuilder builder(graph);
    uint32_t input = builder.addInput(2, 3, 4, 4);
    uint32_t weight = builder.addParameter(5, 3, 3, 3);
    uint32_t bias = builder.addParameter(5, 1, 1, 1);
    assert(builder.addConv2d(input, weight, bias, 1, 1) != gfx::NO_TENSOR);

    assert(graph.tensors[weight].offset == 0);
    assert(graph.tensors[weight].count() == 5 * 3 * 3 * 3);
    assert(graph.tensors[bias].offset == 5 * 3 * 3 * 3);
    assert(graph.parameterCount == 5 * 3 * 3 * 3 + 5);
    // 입력은 활성 arena 의 앞이고, 합성곱 출력이 그 뒤다.
    assert(graph.tensors[input].arena == gfx::Arena::ACTIVATION);
    assert(graph.tensors[input].offset == 0);
    assert(graph.activationCount == 2 * 3 * 4 * 4 + 2 * 5 * 4 * 4);

    std::vector<uint32_t> parameters = graph.parameterTensors();
    assert(parameters.size() == 2);
    assert(parameters[0] == weight && parameters[1] == bias);
}

// stop-gradient. 끊은 자리 아래로는 경사가 흐르지 않아야 한다.
void testDetach() {
    gfx::Graph graph;
    gfx::GraphBuilder builder(graph);
    uint32_t input = builder.addInput(3, 4, 1, 1);
    uint32_t weight = builder.addParameter(4, 4, 1, 1);
    uint32_t bias = builder.addParameter(4, 1, 1, 1);
    uint32_t hidden = builder.addLinear(input, weight, bias);
    uint32_t headWeight = builder.addParameter(1, 4, 1, 1);
    uint32_t headBias = builder.addParameter(1, 1, 1, 1);
    uint32_t prediction = builder.addLinear(hidden, headWeight, headBias);
    uint32_t target = builder.addInput(3, 1, 1, 1);
    assert(builder.addMse(prediction, target) != gfx::NO_TENSOR);

    std::vector<float> parameters(graph.parameterCount, 0.0F);
    std::vector<float> activations(graph.activationCount, 0.0F);
    std::vector<float> parameterGradients(graph.parameterCount, 0.0F);
    std::vector<float> activationGradients(graph.activationCount, 0.0F);
    for (size_t i = 0; i < parameters.size(); ++i) {
        parameters[i] = 0.5F * gfx::neuralGaussian(3, i);
    }
    for (uint32_t i = 0; i < graph.tensors[input].count(); ++i) {
        activations[graph.tensors[input].offset + i] = gfx::neuralGaussian(5, i);
    }

    gfx::forward(graph, parameters.data(), activations.data());
    gfx::backward(graph, parameters.data(), activations.data(), parameterGradients.data(), activationGradients.data());
    bool touched = false;
    for (uint32_t i = 0; i < graph.tensors[weight].count(); ++i) {
        touched = touched || parameterGradients[graph.tensors[weight].offset + i] != 0.0F;
    }
    assert(touched);

    // 중간 텐서에서 끊으면 그 아래(앞쪽 층)의 가중치 경사가 0 이 된다. 액터 손실이 인코더를 갱신하지
    // 않는 것이 정확히 이 규칙이다.
    builder.detach(hidden);
    assert(!graph.tensors[hidden].receivesGradient());
    gfx::backward(graph, parameters.data(), activations.data(), parameterGradients.data(), activationGradients.data());
    for (uint32_t i = 0; i < graph.tensors[weight].count(); ++i) {
        assert(parameterGradients[graph.tensors[weight].offset + i] == 0.0F);
    }
    // 끊은 자리 위(뒤쪽 층)는 그대로 받는다.
    bool head = false;
    for (uint32_t i = 0; i < graph.tensors[headWeight].count(); ++i) {
        head = head || parameterGradients[graph.tensors[headWeight].offset + i] != 0.0F;
    }
    assert(head);
}

// 경사가 «누적» 인지. 한 텐서가 두 갈래로 갈라지면 두 경사가 더해져야 한다. 다중 뷰 병합과 공유
// 인코더가 통째로 이 성질에 기댄다.
void testAccumulation() {
    gfx::Graph graph;
    gfx::GraphBuilder builder(graph);
    uint32_t input = builder.addInput(2, 3, 1, 1);
    uint32_t weight = builder.addParameter(3, 3, 1, 1);
    uint32_t bias = builder.addParameter(3, 1, 1, 1);
    uint32_t shared = builder.addLinear(input, weight, bias);
    // 같은 텐서를 두 번 쓴다. M = V + V 라 경사가 두 배여야 한다.
    uint32_t doubled = builder.addAdd(shared, shared);
    uint32_t headWeight = builder.addParameter(1, 3, 1, 1);
    uint32_t headBias = builder.addParameter(1, 1, 1, 1);
    uint32_t prediction = builder.addLinear(doubled, headWeight, headBias);
    uint32_t target = builder.addInput(2, 1, 1, 1);
    assert(builder.addMse(prediction, target) != gfx::NO_TENSOR);

    std::vector<float> parameters(graph.parameterCount, 0.0F);
    std::vector<float> activations(graph.activationCount, 0.0F);
    std::vector<float> parameterGradients(graph.parameterCount, 0.0F);
    std::vector<float> activationGradients(graph.activationCount, 0.0F);
    for (size_t i = 0; i < parameters.size(); ++i) {
        parameters[i] = 0.4F * gfx::neuralGaussian(7, i);
    }
    for (uint32_t i = 0; i < graph.tensors[input].count(); ++i) {
        activations[graph.tensors[input].offset + i] = gfx::neuralGaussian(9, i);
    }
    gfx::forward(graph, parameters.data(), activations.data());
    // 순전파도 확인한다 — M = 2V.
    for (uint32_t i = 0; i < graph.tensors[shared].count(); ++i) {
        float single = activations[graph.tensors[shared].offset + i];
        float merged = activations[graph.tensors[doubled].offset + i];
        assert(std::abs(merged - 2.0F * single) < 1.0e-5F);
    }
    gfx::backward(graph, parameters.data(), activations.data(), parameterGradients.data(), activationGradients.data());
    // 갈래가 둘이므로 shared 의 경사는 위쪽 경사의 두 배다.
    const gfx::Tensor& mergedTensor = graph.tensors[doubled];
    const gfx::Tensor& sharedTensor = graph.tensors[shared];
    for (uint32_t i = 0; i < sharedTensor.count(); ++i) {
        double above = activationGradients[mergedTensor.offset + i];
        double below = activationGradients[sharedTensor.offset + i];
        assert(std::abs(below - 2.0 * above) < 1.0e-5);
    }
}

// 난수가 표준 정규인지. 평균·분산만 보면 균등분포도 통과하므로 꼬리 비율까지 본다.
void testGaussian() {
    assert(gfx::neuralGaussian(7, 3) == gfx::neuralGaussian(7, 3));
    assert(gfx::neuralGaussian(7, 3) != gfx::neuralGaussian(7, 4));
    assert(gfx::neuralGaussian(7, 3) != gfx::neuralGaussian(8, 3));

    constexpr uint64_t COUNT = 200000;
    double sum = 0.0;
    double squared = 0.0;
    uint64_t beyondOne = 0;
    uint64_t beyondTwo = 0;
    for (uint64_t i = 0; i < COUNT; ++i) {
        double value = gfx::neuralGaussian(1, i);
        sum += value;
        squared += value * value;
        beyondOne += std::abs(value) > 1.0 ? 1U : 0U;
        beyondTwo += std::abs(value) > 2.0 ? 1U : 0U;
    }
    double mean = sum / static_cast<double>(COUNT);
    double variance = squared / static_cast<double>(COUNT) - mean * mean;
    assert(std::abs(mean) < 0.02);
    assert(std::abs(variance - 1.0) < 0.03);
    // P(|z| > 1) = 0.3173, P(|z| > 2) = 0.0455.
    assert(std::abs(static_cast<double>(beyondOne) / static_cast<double>(COUNT) - 0.3173) < 0.01);
    assert(std::abs(static_cast<double>(beyondTwo) / static_cast<double>(COUNT) - 0.0455) < 0.01);
}

// 초기화가 어느 자리를 채우는지. 가중치만 흩뿌리고 편향은 0, layernorm 의 이득은 1 이다.
void testInitialization() {
    gfx::Graph graph;
    gfx::GraphBuilder builder(graph);
    uint32_t input = builder.addInput(2, 3, 6, 6);
    uint32_t convWeight = builder.addParameter(4, 3, 3, 3);
    uint32_t convBias = builder.addParameter(4, 1, 1, 1);
    uint32_t conv = builder.addConv2d(input, convWeight, convBias, 2, 1);
    const gfx::Tensor& convTensor = graph.tensors[conv];
    uint32_t features = convTensor.dims[1] * convTensor.dims[2] * convTensor.dims[3];
    uint32_t flat = builder.addReshape(conv, convTensor.dims[0], features, 1, 1);
    uint32_t linearWeight = builder.addParameter(5, features, 1, 1);
    uint32_t linearBias = builder.addParameter(5, 1, 1, 1);
    uint32_t hidden = builder.addLinear(flat, linearWeight, linearBias);
    uint32_t gain = builder.addParameter(5, 1, 1, 1);
    uint32_t shift = builder.addParameter(5, 1, 1, 1);
    assert(builder.addLayerNorm(hidden, gain, shift) != gfx::NO_TENSOR);

    std::vector<float> parameters(graph.parameterCount, 7.0F);
    gfx::initializeParameters(graph, 13, parameters.data());

    auto allZero = [&](uint32_t tensor) {
        const gfx::Tensor& item = graph.tensors[tensor];
        for (uint32_t i = 0; i < item.count(); ++i) {
            if (parameters[item.offset + i] != 0.0F) {
                return false;
            }
        }
        return true;
    };
    auto allOne = [&](uint32_t tensor) {
        const gfx::Tensor& item = graph.tensors[tensor];
        for (uint32_t i = 0; i < item.count(); ++i) {
            if (parameters[item.offset + i] != 1.0F) {
                return false;
            }
        }
        return true;
    };
    assert(!allZero(convWeight));
    assert(allZero(convBias));
    assert(!allZero(linearWeight));
    assert(allZero(linearBias));
    assert(allOne(gain));
    assert(allZero(shift));

    // He 정규의 눈금. 표준편차가 sqrt(2 / 팬인) 근처여야 한다. 합성곱의 팬인은 **입력 채널 x 커널
    // 넓이**이지 가중치 전체 개수가 아니다 — 둘을 헷갈리면 눈금이 출력 채널 수만큼 어긋난다.
    auto deviationOf = [&](uint32_t tensor) {
        const gfx::Tensor& item = graph.tensors[tensor];
        double sum = 0.0;
        double squared = 0.0;
        for (uint32_t i = 0; i < item.count(); ++i) {
            double value = parameters[item.offset + i];
            sum += value;
            squared += value * value;
        }
        double mean = sum / item.count();
        return std::sqrt(squared / item.count() - mean * mean);
    };
    double linearExpected = std::sqrt(2.0 / static_cast<double>(features));
    assert(std::abs(deviationOf(linearWeight) - linearExpected) / linearExpected < 0.1);
    double convExpected = std::sqrt(2.0 / static_cast<double>(3 * 3 * 3));
    assert(std::abs(deviationOf(convWeight) - convExpected) / convExpected < 0.15);

    // 같은 씨앗이면 같은 값이다.
    std::vector<float> again(graph.parameterCount, 0.0F);
    gfx::initializeParameters(graph, 13, again.data());
    assert(again == parameters);
    std::vector<float> other(graph.parameterCount, 0.0F);
    gfx::initializeParameters(graph, 14, other.data());
    assert(other != parameters);
}

// Adam 의 첫 걸음은 경사의 «크기» 와 무관하게 학습률 근처다. m 과 v 가 0 에서 시작해 편향 보정을
// 지나면 m / sqrt(v) 가 부호만 남기기 때문이다. 이 성질이 깨지면 학습률을 고르는 감각이 통째로 달라진다.
void testAdamFirstStep() {
    gfx::AdamSettings settings;
    settings.learningRate = 0.01F;
    constexpr size_t COUNT = 4;
    std::vector<float> parameters(COUNT, 0.0F);
    std::vector<float> moments(gfx::adamMomentCount(COUNT), 0.0F);
    // 경사의 크기를 1000 배 차이로 벌려도 걸음은 같아야 한다.
    std::vector<float> gradients{1.0F, -1.0F, 1000.0F, -0.001F};
    gfx::adamStep(settings, 1, COUNT, gradients.data(), moments.data(), parameters.data());
    for (size_t i = 0; i < COUNT; ++i) {
        float expected = gradients[i] > 0.0F ? -settings.learningRate : settings.learningRate;
        assert(std::abs(parameters[i] - expected) < 1.0e-6F);
    }
    // 모멘트 배열의 절반 배치를 못 박는다. 앞 절반이 1차, 뒤 절반이 2차다 — GPU 가 버퍼 하나로 받으므로
    // 여기서 뒤집히면 CPU/GPU 가 갈린다. 첫 걸음이라 m = (1-b1)g, v = (1-b2)g^2 이다.
    for (size_t i = 0; i < COUNT; ++i) {
        float expectedFirst = (1.0F - settings.beta1) * gradients[i];
        float expectedSecond = (1.0F - settings.beta2) * gradients[i] * gradients[i];
        assert(std::abs(moments[i] - expectedFirst) <= 1.0e-4F * std::abs(expectedFirst) + 1.0e-9F);
        assert(std::abs(moments[COUNT + i] - expectedSecond) <= 1.0e-4F * std::abs(expectedSecond) + 1.0e-9F);
    }
    assert(gfx::adamMomentCount(COUNT) == COUNT * 2);

    // 경사가 0 이면 움직이지 않는다.
    std::vector<float> zeroParameters(COUNT, 3.0F);
    std::vector<float> zeroMoments(gfx::adamMomentCount(COUNT), 0.0F);
    std::vector<float> zeroGradients(COUNT, 0.0F);
    gfx::adamStep(settings, 1, COUNT, zeroGradients.data(), zeroMoments.data(), zeroParameters.data());
    for (float value : zeroParameters) {
        assert(value == 3.0F);
    }
    // step 0 은 편향 보정이 0 으로 나누는 자리라 아무 일도 하지 않는다.
    std::vector<float> guarded(COUNT, 5.0F);
    std::vector<float> guardedMoments(gfx::adamMomentCount(COUNT), 0.0F);
    gfx::adamStep(settings, 0, COUNT, gradients.data(), guardedMoments.data(), guarded.data());
    for (float value : guarded) {
        assert(value == 5.0F);
    }
}

// 이차식 최소화. 최적화가 실제로 내려가는지.
void testAdamConvergence() {
    gfx::AdamSettings settings;
    settings.learningRate = 0.05F;
    constexpr size_t COUNT = 3;
    const std::vector<float> target{1.5F, -2.0F, 0.25F};
    std::vector<float> parameters(COUNT, 0.0F);
    std::vector<float> moments(gfx::adamMomentCount(COUNT), 0.0F);
    std::vector<float> gradients(COUNT, 0.0F);

    auto loss = [&]() {
        float sum = 0.0F;
        for (size_t i = 0; i < COUNT; ++i) {
            float error = parameters[i] - target[i];
            sum += error * error;
        }
        return sum;
    };
    float first = loss();
    for (uint32_t step = 1; step <= 2000; ++step) {
        for (size_t i = 0; i < COUNT; ++i) {
            gradients[i] = 2.0F * (parameters[i] - target[i]);
        }
        gfx::adamStep(settings, step, COUNT, gradients.data(), moments.data(), parameters.data());
    }
    assert(loss() < first);
    for (size_t i = 0; i < COUNT; ++i) {
        assert(std::abs(parameters[i] - target[i]) < 1.0e-3F);
    }

    // 같은 씨앗·같은 순서면 결과가 비트로 같다.
    std::vector<float> again(COUNT, 0.0F);
    std::vector<float> againMoments(gfx::adamMomentCount(COUNT), 0.0F);
    for (uint32_t step = 1; step <= 2000; ++step) {
        for (size_t i = 0; i < COUNT; ++i) {
            gradients[i] = 2.0F * (again[i] - target[i]);
        }
        gfx::adamStep(settings, step, COUNT, gradients.data(), againMoments.data(), again.data());
    }
    assert(again == parameters);
}

void testPolyak() {
    constexpr size_t COUNT = 4;
    std::vector<float> online{1.0F, 2.0F, -3.0F, 0.5F};
    std::vector<float> target(COUNT, 0.0F);
    // tau 0 이면 그대로다.
    gfx::polyakStep(0.0F, COUNT, online.data(), target.data());
    for (float value : target) {
        assert(value == 0.0F);
    }
    // tau 1 이면 복사다.
    gfx::polyakStep(1.0F, COUNT, online.data(), target.data());
    assert(target == online);
    // 그 사이는 선형 보간이다.
    std::fill(target.begin(), target.end(), 0.0F);
    gfx::polyakStep(0.25F, COUNT, online.data(), target.data());
    for (size_t i = 0; i < COUNT; ++i) {
        assert(std::abs(target[i] - 0.25F * online[i]) < 1.0e-6F);
    }
    // 범위 밖 tau 는 잘라 낸다. 음수를 그대로 쓰면 타깃이 발산하고 1 보다 크면 진동한다.
    std::fill(target.begin(), target.end(), 1.0F);
    gfx::polyakStep(-0.5F, COUNT, online.data(), target.data());
    for (float value : target) {
        assert(value == 1.0F);
    }
    gfx::polyakStep(2.0F, COUNT, online.data(), target.data());
    assert(target == online);
    // tau 가 1 이면 크기 차가 커도 정확히 복사다(a + (b - a) 는 그렇지 않다).
    std::vector<float> huge{1.0e30F, -1.0e30F, 1.0e30F, -1.0e30F};
    gfx::polyakStep(1.0F, COUNT, online.data(), huge.data());
    assert(huge == online);

    // 되풀이하면 지수적으로 다가간다.
    for (uint32_t i = 0; i < 200; ++i) {
        gfx::polyakStep(0.05F, COUNT, online.data(), target.data());
    }
    for (size_t i = 0; i < COUNT; ++i) {
        assert(std::abs(target[i] - online[i]) < 1.0e-3F);
    }
}

// MEAN 이 «합» 이 아니라 «평균» 인지. 순전파와 역전파가 나란히 합으로 틀리면 유한차분은 자기 일관적이라
// 통과한다 — 값 자체를 단정해야 한다. GPU 커널이 맞춰야 할 규약이기도 하다.
void testMeanValue() {
    gfx::Graph graph;
    gfx::GraphBuilder builder(graph);
    // 배치와 특징 둘 다 1 이 아니어야 «배치마다 평균» 과 «전체 평균» 이 갈린다. MEAN 은 전체다.
    uint32_t input = builder.addInput(2, 3, 1, 1);
    uint32_t mean = builder.addMean(input);
    assert(mean != gfx::NO_TENSOR);
    assert(graph.tensors[mean].count() == 1);

    std::vector<float> parameters;
    std::vector<float> activations(graph.activationCount, 0.0F);
    float values[6] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 9.0F};
    for (uint32_t i = 0; i < 6; ++i) {
        activations[graph.tensors[input].offset + i] = values[i];
    }
    gfx::forward(graph, parameters.data(), activations.data());
    assert(std::abs(activations[graph.tensors[mean].offset] - 4.0F) < 1.0e-6F);

    // 없는 텐서를 접으라면 거절한다. 원소 수는 볼 것이 없다 — 빌더가 축마다 1 을 하한으로 두므로
    // 원소가 0 인 텐서는 애초에 만들어지지 않는다.
    gfx::Graph empty;
    gfx::GraphBuilder emptyBuilder(empty);
    assert(emptyBuilder.addMean(gfx::NO_TENSOR) == gfx::NO_TENSOR);
    assert(emptyBuilder.addInput(0, 0, 0, 0) != gfx::NO_TENSOR);
    assert(empty.tensors[0].count() == 1);
}

// 손실이 없는 그래프는 validateForward 만 지나야 한다. 행동만 내는 정책 망이 그 꼴이다.
void testForwardOnlyGraph() {
    gfx::Graph graph;
    gfx::GraphBuilder builder(graph);
    uint32_t input = builder.addInput(2, 3, 1, 1);
    uint32_t weight = builder.addParameter(4, 3, 1, 1);
    uint32_t bias = builder.addParameter(4, 1, 1, 1);
    uint32_t action = builder.addTanh(builder.addLinear(input, weight, bias));
    assert(action != gfx::NO_TENSOR);
    assert(gfx::validateForward(graph));
    // 마지막이 스칼라가 아니므로 역전파는 못 한다. 조용히 0 을 돌려주지 않고 거짓을 낸다.
    assert(!gfx::validate(graph));
    std::vector<float> parameters(graph.parameterCount, 0.1F);
    std::vector<float> activations(graph.activationCount, 0.0F);
    std::vector<float> parameterGradients(graph.parameterCount, 0.0F);
    std::vector<float> activationGradients(graph.activationCount, 0.0F);
    gfx::forward(graph, parameters.data(), activations.data());
    assert(!gfx::backward(
        graph, parameters.data(), activations.data(), parameterGradients.data(), activationGradients.data()));

    // 반대로 표 자체가 깨지면 둘 다 거짓이다.
    gfx::Graph broken = graph;
    broken.ops.back().inputs[0] = broken.ops.back().output;
    assert(!gfx::validateForward(broken));
    assert(!gfx::validate(broken));
}

// Huber 의 **값**. 유한차분은 경사만 보므로 꺾이는 지점 밖의 상수항(-delta^2/2)이 빠져도 안 걸린다.
// 그 항이 없으면 손실이 그 지점에서 끊기고, 두 크리틱의 손실을 견주는 자리에서 눈금이 어긋난다.
void testHuberValue() {
    gfx::Graph graph;
    gfx::GraphBuilder builder(graph);
    uint32_t prediction = builder.addInput(1, 3, 1, 1);
    uint32_t target = builder.addInput(1, 3, 1, 1);
    constexpr float DELTA = 0.5F;
    uint32_t loss = builder.addHuber(prediction, target, DELTA);
    assert(loss != gfx::NO_TENSOR);
    // delta 가 0 이하면 뜻이 없다.
    assert(builder.addHuber(prediction, target, 0.0F) == gfx::NO_TENSOR);
    assert(builder.addHuber(prediction, target, -1.0F) == gfx::NO_TENSOR);

    std::vector<float> parameters;
    std::vector<float> activations(graph.activationCount, 0.0F);
    // 오차를 꺾이는 지점 안쪽·정확히 그 위·바깥쪽 하나씩 둔다.
    float errors[3] = {0.25F, DELTA, 2.0F};
    for (uint32_t i = 0; i < 3; ++i) {
        activations[graph.tensors[prediction].offset + i] = errors[i];
        activations[graph.tensors[target].offset + i] = 0.0F;
    }
    gfx::forward(graph, parameters.data(), activations.data());
    float expected = 0.5F * errors[0] * errors[0];
    expected += 0.5F * DELTA * DELTA;
    expected += DELTA * (errors[2] - 0.5F * DELTA);
    assert(std::abs(activations[graph.tensors[loss].offset] - expected / 3.0F) < 1.0e-6F);

    // 꺾이는 지점에서 이어져야 한다. 양쪽 식이 같은 값을 내는지 아주 가까운 두 점으로 본다.
    constexpr float EDGE = 1.0e-4F;
    for (uint32_t i = 0; i < 3; ++i) {
        activations[graph.tensors[prediction].offset + i] = DELTA - EDGE;
    }
    gfx::forward(graph, parameters.data(), activations.data());
    float inside = activations[graph.tensors[loss].offset];
    for (uint32_t i = 0; i < 3; ++i) {
        activations[graph.tensors[prediction].offset + i] = DELTA + EDGE;
    }
    gfx::forward(graph, parameters.data(), activations.data());
    float outside = activations[graph.tensors[loss].offset];
    assert(std::abs(outside - inside) < 1.0e-3F);
}

// 저장 -> 적재 왕복과 모양 검사.
void testParameterFile() {
    gfx::Graph graph;
    gfx::GraphBuilder builder(graph);
    uint32_t input = builder.addInput(2, 3, 5, 5);
    uint32_t convWeight = builder.addParameter(4, 3, 3, 3);
    uint32_t convBias = builder.addParameter(4, 1, 1, 1);
    uint32_t conv = builder.addConv2d(input, convWeight, convBias, 2, 1);
    const gfx::Tensor& convTensor = graph.tensors[conv];
    uint32_t features = convTensor.dims[1] * convTensor.dims[2] * convTensor.dims[3];
    uint32_t flat = builder.addReshape(conv, convTensor.dims[0], features, 1, 1);
    uint32_t target = builder.addInput(2, 2, 1, 1);
    assert(addHead(builder, graph, flat, target) != gfx::NO_TENSOR);

    // 배치 요약은 파라미터 텐서마다 dims 넷이다. 뷰는 파라미터가 될 수 없으므로 중복이 없다.
    gfx::ParameterLayout layout = gfx::parameterLayout(graph);
    assert(layout.count == graph.parameterCount);
    assert(layout.shapes.size() == graph.parameterTensors().size() * 4);

    std::vector<float> parameters(graph.parameterCount, 0.0F);
    gfx::initializeParameters(graph, 17, parameters.data());
    const std::string path = "neural_parameters_test.json";
    assert(gfx::saveParameters(graph, parameters.data(), path));

    std::vector<float> loaded(graph.parameterCount, 9.0F);
    assert(gfx::loadParameters(graph, loaded.data(), path));
    assert(loaded == parameters);

    // 파라미터 **총수가 같은데 모양만 다른** 그래프를 거절하는지. 개수만 견주면 통과해 버려 조용히
    // 엉뚱한 자리를 채우게 되므로, 이 짝이 배치 검사의 핵심이다.
    auto makeConvGraph = [](uint32_t channels, uint32_t kernelHeight, uint32_t kernelWidth, gfx::Graph& out) {
        gfx::GraphBuilder convBuilder(out);
        uint32_t convInput = convBuilder.addInput(1, channels, 6, 6);
        uint32_t weight = convBuilder.addParameter(2, channels, kernelHeight, kernelWidth);
        uint32_t bias = convBuilder.addParameter(2, 1, 1, 1);
        assert(convBuilder.addConv2d(convInput, weight, bias, 1, 0) != gfx::NO_TENSOR);
    };
    gfx::Graph shapeA;
    gfx::Graph shapeB;
    makeConvGraph(3, 2, 2, shapeA);
    makeConvGraph(2, 3, 2, shapeB);
    // 2*3*2*2 + 2 = 26 과 2*2*3*2 + 2 = 26. 총수는 같고 모양은 다르다.
    assert(shapeA.parameterCount == shapeB.parameterCount);
    assert(gfx::parameterLayout(shapeA).shapes != gfx::parameterLayout(shapeB).shapes);
    std::vector<float> shapeParameters(shapeA.parameterCount, 1.0F);
    const std::string shapePath = "neural_parameters_shape.json";
    assert(gfx::saveParameters(shapeA, shapeParameters.data(), shapePath));
    std::vector<float> shapeTarget(shapeB.parameterCount, 4.0F);
    std::vector<float> shapeUntouched = shapeTarget;
    assert(!gfx::loadParameters(shapeB, shapeTarget.data(), shapePath));
    assert(shapeTarget == shapeUntouched);
    // 같은 그래프면 그대로 읽힌다(위 거절이 «늘 거절» 이 아니라는 확인).
    std::vector<float> shapeSame(shapeA.parameterCount, 4.0F);
    assert(gfx::loadParameters(shapeA, shapeSame.data(), shapePath));
    assert(shapeSame == shapeParameters);

    // 그래프 해시. 파라미터 배치가 **완전히 같은데** 그래프가 다른 두 경우를 가려야 한다.
    {
        // (가) 보폭·여백만 다르다. 9 에서 커널 3 으로 (보폭 1, 여백 0) 과 (보폭 2, 여백 3) 은 **출력이
        //      똑같이 7** 이라 텐서 모양이 전부 같다 — 해시가 연산의 정수 인자까지 보지 않으면 못 가른다.
        auto makeStrided = [](uint32_t stride, uint32_t pad, gfx::Graph& out) {
            gfx::GraphBuilder strideBuilder(out);
            uint32_t strideInput = strideBuilder.addInput(1, 3, 9, 9);
            uint32_t weight = strideBuilder.addParameter(4, 3, 3, 3);
            uint32_t bias = strideBuilder.addParameter(4, 1, 1, 1);
            uint32_t conv = strideBuilder.addConv2d(strideInput, weight, bias, stride, pad);
            assert(conv != gfx::NO_TENSOR);
            assert(out.tensors[conv].dims[2] == 7 && out.tensors[conv].dims[3] == 7);
        };
        gfx::Graph slow;
        gfx::Graph fast;
        makeStrided(1, 0, slow);
        makeStrided(2, 3, fast);
        assert(gfx::parameterLayout(slow) == gfx::parameterLayout(fast));
        // 텐서 표까지 완전히 같다. 남은 차이는 연산의 정수 인자뿐이다.
        assert(slow.tensors.size() == fast.tensors.size());
        for (size_t i = 0; i < slow.tensors.size(); ++i) {
            for (uint32_t axis = 0; axis < 4; ++axis) {
                assert(slow.tensors[i].dims[axis] == fast.tensors[i].dims[axis]);
            }
            assert(slow.tensors[i].offset == fast.tensors[i].offset);
        }
        assert(gfx::graphHash(slow) != gfx::graphHash(fast));

        std::vector<float> slowParameters(slow.parameterCount, 0.0F);
        gfx::initializeParameters(slow, 5, slowParameters.data());
        const std::string stridePath = "neural_parameters_stride.json";
        assert(gfx::saveParameters(slow, slowParameters.data(), stridePath));
        std::vector<float> fastParameters(fast.parameterCount, 2.0F);
        std::vector<float> fastUntouched = fastParameters;
        assert(!gfx::loadParameters(fast, fastParameters.data(), stridePath));
        assert(fastParameters == fastUntouched);
        // 같은 그래프면 읽힌다.
        std::vector<float> slowAgain(slow.parameterCount, 2.0F);
        assert(gfx::loadParameters(slow, slowAgain.data(), stridePath));
        assert(slowAgain == slowParameters);

        // (나) 파라미터를 지은 순서만 바꿨다. 쌍둥이 크리틱 둘의 가중치가 뒤바뀌는 경우다.
        auto makeSwapped = [](bool swap, gfx::Graph& out) {
            gfx::GraphBuilder swapBuilder(out);
            uint32_t swapInput = swapBuilder.addInput(1, 4, 1, 1);
            uint32_t firstWeight = swapBuilder.addParameter(4, 4, 1, 1);
            uint32_t secondWeight = swapBuilder.addParameter(4, 4, 1, 1);
            uint32_t bias = swapBuilder.addParameter(4, 1, 1, 1);
            uint32_t a = swapBuilder.addLinear(swapInput, swap ? secondWeight : firstWeight, bias);
            uint32_t b = swapBuilder.addLinear(swapInput, swap ? firstWeight : secondWeight, bias);
            assert(swapBuilder.addAdd(a, b) != gfx::NO_TENSOR);
        };
        gfx::Graph plain;
        gfx::Graph swapped;
        makeSwapped(false, plain);
        makeSwapped(true, swapped);
        assert(gfx::parameterLayout(plain) == gfx::parameterLayout(swapped));
        assert(gfx::graphHash(plain) != gfx::graphHash(swapped));
    }

    // NaN 이 섞인 가중치는 저장하지 않는다. JSON 이 null 로 찍어 되읽을 수 없는 파일이 되기 때문이다.
    {
        std::vector<float> broken = parameters;
        broken[3] = std::numeric_limits<float>::quiet_NaN();
        assert(!gfx::saveParameters(graph, broken.data(), "neural_parameters_nan.json"));
        broken[3] = std::numeric_limits<float>::infinity();
        assert(!gfx::saveParameters(graph, broken.data(), "neural_parameters_nan.json"));
    }

    // 없는 파일과 깨진 파일.
    assert(!gfx::loadParameters(graph, loaded.data(), "없는파일.json"));
    {
        std::ofstream broken("neural_parameters_broken.json", std::ios::binary);
        broken << "{ this is not json";
    }
    assert(!gfx::loadParameters(graph, loaded.data(), "neural_parameters_broken.json"));
    {
        // 구문은 맞지만 타입이 다르다. value() 가 예외를 던지는 자리다.
        std::ofstream typed("neural_parameters_typed.json", std::ios::binary);
        typed << R"({"count": "many", "shapes": 3, "weights": null})";
    }
    assert(!gfx::loadParameters(graph, loaded.data(), "neural_parameters_typed.json"));
    {
        std::ofstream array("neural_parameters_array.json", std::ios::binary);
        array << "[1, 2, 3]";
    }
    assert(!gfx::loadParameters(graph, loaded.data(), "neural_parameters_array.json"));

    // 배치는 맞는데 가중치 배열이 짧은 파일. 여기서 거절하지 않으면 나머지가 옛 값으로 남는다.
    {
        gfx::ParameterLayout shortLayout = gfx::parameterLayout(graph);
        std::ofstream truncated("neural_parameters_short.json", std::ios::binary);
        truncated << "{\"count\": " << shortLayout.count << ", \"shapes\": [";
        for (size_t i = 0; i < shortLayout.shapes.size(); ++i) {
            truncated << (i > 0 ? "," : "") << shortLayout.shapes[i];
        }
        truncated << "], \"weights\": [1.0, 2.0, 3.0]}";
    }
    assert(!gfx::loadParameters(graph, loaded.data(), "neural_parameters_short.json"));
    assert(loaded == parameters);
}

// 표 여럿을 한 표로 합치기. 파라미터는 그대로 두고 활성만 미는 것이 요점이다 — 셋이 파라미터 하나를
// 나눠 쓰는 것이 이 합치기의 존재 이유이므로, 파라미터까지 밀면 학습이 세 갈래로 갈라진다.
void testMergeGraphs() {
    // 표 둘. 같은 파라미터 배치를 쓰는 것처럼 꾸민다(파라미터를 먼저 잡아 오프셋을 맞춘다).
    gfx::Graph first;
    gfx::Graph second;
    uint32_t firstWeight = gfx::NO_TENSOR;
    uint32_t secondWeight = gfx::NO_TENSOR;
    uint32_t firstOut = gfx::NO_TENSOR;
    uint32_t secondOut = gfx::NO_TENSOR;
    {
        gfx::GraphBuilder builder(first);
        firstWeight = builder.addParameter(3, 4, 1, 1);
        uint32_t bias = builder.addParameter(3, 1, 1, 1);
        uint32_t input = builder.addInput(2, 4, 1, 1);
        firstOut = builder.addLinear(input, firstWeight, bias);
    }
    {
        gfx::GraphBuilder builder(second);
        secondWeight = builder.addParameter(3, 4, 1, 1);
        uint32_t bias = builder.addParameter(3, 1, 1, 1);
        uint32_t input = builder.addInput(5, 4, 1, 1);
        secondOut = builder.addRelu(builder.addLinear(input, secondWeight, bias));
    }
    assert(first.parameterCount == second.parameterCount);

    std::vector<const gfx::Graph*> sources{&first, &second};
    gfx::MergedGraph merged;
    assert(gfx::mergeGraphs(sources, merged));
    assert(merged.graph.parameterCount == first.parameterCount);
    assert(merged.graph.activationCount == first.activationCount + second.activationCount);
    assert(merged.graph.tensors.size() == first.tensors.size() + second.tensors.size());
    assert(merged.graph.ops.size() == first.ops.size() + second.ops.size());
    assert(merged.opBegin[0] == 0 && merged.opCount[0] == first.ops.size());
    assert(merged.opBegin[1] == first.ops.size() && merged.opCount[1] == second.ops.size());

    // **파라미터 오프셋은 그대로다.** 둘째 표의 가중치가 첫째와 같은 자리를 가리켜야 한다.
    uint32_t mergedFirstWeight = gfx::mergedTensor(merged, 0, firstWeight);
    uint32_t mergedSecondWeight = gfx::mergedTensor(merged, 1, secondWeight);
    assert(merged.graph.tensors[mergedFirstWeight].arena == gfx::Arena::PARAMETER);
    assert(merged.graph.tensors[mergedFirstWeight].offset == first.tensors[firstWeight].offset);
    assert(merged.graph.tensors[mergedSecondWeight].offset == second.tensors[secondWeight].offset);
    assert(merged.graph.tensors[mergedFirstWeight].offset == merged.graph.tensors[mergedSecondWeight].offset);

    // **활성은 밀린다.** 둘째 표의 출력이 첫째 표의 활성 뒤에 앉아야 한다.
    uint32_t mergedSecondOut = gfx::mergedTensor(merged, 1, secondOut);
    assert(merged.graph.tensors[mergedSecondOut].offset == second.tensors[secondOut].offset + first.activationCount);
    uint32_t mergedFirstOut = gfx::mergedTensor(merged, 0, firstOut);
    // 범위를 벗어나면 NO_TENSOR 다. 밖에서 표 목록을 다시 세지 않으므로 여기서 걸린다.
    assert(gfx::mergedTensor(merged, 2, 0) == gfx::NO_TENSOR);
    assert(gfx::mergedTensor(merged, 0, static_cast<uint32_t>(first.tensors.size())) == gfx::NO_TENSOR);
    assert(gfx::mergedTensor(merged, 0, gfx::NO_TENSOR) == gfx::NO_TENSOR);
    assert(merged.graph.tensors[mergedFirstOut].offset == first.tensors[firstOut].offset);

    // 합친 표가 그대로 돌아야 한다.
    assert(gfx::validateForward(merged.graph));

    // **합친 표의 순전파가 표를 따로 돌린 것과 같아야 한다.**
    std::vector<float> parameters(merged.graph.parameterCount, 0.0F);
    for (size_t i = 0; i < parameters.size(); ++i) {
        parameters[i] = gfx::neuralGaussian(3, i) * 0.4F;
    }
    std::vector<float> mergedActivations(merged.graph.activationCount, 0.0F);
    std::vector<float> firstActivations(first.activationCount, 0.0F);
    std::vector<float> secondActivations(second.activationCount, 0.0F);
    for (size_t i = 0; i < mergedActivations.size(); ++i) {
        mergedActivations[i] = gfx::neuralGaussian(9, i);
    }
    std::copy(mergedActivations.begin(),
              mergedActivations.begin() + static_cast<long>(first.activationCount),
              firstActivations.begin());
    std::copy(mergedActivations.begin() + static_cast<long>(first.activationCount),
              mergedActivations.end(),
              secondActivations.begin());
    gfx::forward(merged.graph, parameters.data(), mergedActivations.data());
    gfx::forward(first, parameters.data(), firstActivations.data());
    gfx::forward(second, parameters.data(), secondActivations.data());
    for (size_t i = 0; i < first.activationCount; ++i) {
        assert(mergedActivations[i] == firstActivations[i]);
    }
    for (size_t i = 0; i < second.activationCount; ++i) {
        assert(mergedActivations[first.activationCount + i] == secondActivations[i]);
    }

    // 파라미터 수가 다르면 거절한다. 같은 에이전트에서 나온 표가 아니라는 뜻이다.
    gfx::Graph odd;
    {
        gfx::GraphBuilder builder(odd);
        uint32_t weight = builder.addParameter(2, 2, 1, 1);
        uint32_t input = builder.addInput(1, 2, 1, 1);
        builder.addMul(input, input);
        (void)weight;
    }
    std::vector<const gfx::Graph*> mixed{&first, &odd};
    gfx::MergedGraph rejected;
    assert(!gfx::mergeGraphs(mixed, rejected));
    std::vector<const gfx::Graph*> empty;
    assert(!gfx::mergeGraphs(empty, rejected));
    std::printf("표 합치기 통과\n");
}

// 리플레이 링의 첨자 규칙. **여기가 11단계에서 가장 틀리기 쉬운 자리다** — 링이 되감기고, 에피소드가
// 경계를 긋고, 창이 앞에서 잘려 나가는 셋이 한 식에서 만난다.
void testReplayIndex() {
    // 아직 한 바퀴 돌지 않은 링. 살아 있는 창은 [0, count) 다.
    gfx::ReplayWindow window;
    window.capacity = 8;
    window.count = 5;
    window.cursor = 5;
    assert(window.oldest() == 0);
    assert(window.depthBack(4) == 4);
    assert(window.depthBack(0) == 0);

    // 에피소드 한가운데(걸음 3)에서는 그냥 거슬러 올라간다.
    assert(gfx::replayStackIndex(window, 4, 3, 0) == 4);
    assert(gfx::replayStackIndex(window, 4, 3, 1) == 3);
    assert(gfx::replayStackIndex(window, 4, 3, 2) == 2);
    // **에피소드 첫 걸음에서는 과거가 없다.** 같은 판이 세 채널에 겹쳐 들어가는 것이 옳다.
    assert(gfx::replayStackIndex(window, 4, 0, 1) == 4);
    assert(gfx::replayStackIndex(window, 4, 0, 2) == 4);
    // 둘째 걸음이면 한 칸까지만.
    assert(gfx::replayStackIndex(window, 4, 1, 1) == 3);
    assert(gfx::replayStackIndex(window, 4, 1, 2) == 3);
    // 걸음 수가 남아 있어도 **창이 없으면** 못 간다. 첨자 1 은 뒤에 한 칸뿐이다.
    assert(gfx::replayStackIndex(window, 1, 9, 2) == 0);

    // 한 바퀴 돈 링. 창이 cursor 에서 시작해 되감긴다.
    gfx::ReplayWindow wrapped;
    wrapped.capacity = 8;
    wrapped.count = 8;
    wrapped.cursor = 3;
    assert(wrapped.oldest() == 3);
    assert(wrapped.depthBack(3) == 0);
    assert(wrapped.depthBack(2) == 7);
    // 첨자 0 에서 두 칸 거슬러 오르면 되감겨 6 이다.
    assert(gfx::replayStackIndex(wrapped, 0, 9, 2) == 6);
    // 창의 맨 앞(3)에서는 더 갈 데가 없다.
    assert(gfx::replayStackIndex(wrapped, 3, 9, 2) == 3);
    assert(gfx::replayStackIndex(wrapped, 4, 9, 2) == 3);

    // 유효 표본. 에피소드 번호가 갈리는 자리에서 끊긴다.
    std::vector<uint32_t> episodes(8, 0);
    episodes[2] = 1;
    episodes[3] = 1;
    episodes[4] = 1;
    gfx::ReplayWindow full;
    full.capacity = 8;
    full.count = 6;
    full.cursor = 6;
    assert(gfx::replaySampleValid(full, 0, episodes.data()));
    // 1 -> 2 는 에피소드가 갈린다.
    assert(!gfx::replaySampleValid(full, 1, episodes.data()));
    assert(gfx::replaySampleValid(full, 2, episodes.data()));
    // 4 -> 5 도 갈린다(4 가 마지막 에피소드 칸).
    assert(!gfx::replaySampleValid(full, 4, episodes.data()));
    // 5 는 **마지막으로 쓴 칸**이라 다음이 없다.
    assert(!gfx::replaySampleValid(full, 5, episodes.data()));
    // 6, 7 은 아직 안 쓰였다.
    assert(!gfx::replaySampleValid(full, 6, episodes.data()));
    assert(!gfx::replaySampleValid(full, 7, episodes.data()));

    // 칸이 하나뿐이면 전이가 없다.
    gfx::ReplayWindow single;
    single.capacity = 8;
    single.count = 1;
    single.cursor = 1;
    assert(!gfx::replaySampleValid(single, 0, episodes.data()));

    // 빈 링에서 아무 것도 죽지 않는다.
    gfx::ReplayWindow empty;
    assert(gfx::replayStackIndex(empty, 0, 0, 2) == 0);
    assert(!gfx::replaySampleValid(empty, 0, episodes.data()));
    std::printf("리플레이 첨자 규칙 통과\n");
}

// 무작위 이동 증강의 변위. 범위와 «표본마다 다르다» 를 본다.
void testReplayShift() {
    constexpr int32_t PADDING = 4;
    int32_t histogram[2 * PADDING + 1] = {};
    for (uint64_t i = 0; i < 4000; ++i) {
        int32_t shift = gfx::replayShift(11, i, PADDING);
        assert(shift >= -PADDING && shift <= PADDING);
        histogram[shift + PADDING] += 1;
    }
    // 아홉 값이 모두 나와야 한다. 하나라도 비면 나머지 산술이 어딘가 잘린 것이다.
    for (int32_t bucket : histogram) {
        assert(bucket > 4000 / (2 * PADDING + 1) / 2);
    }
    // padding 0 이면 증강이 꺼진다.
    for (uint64_t i = 0; i < 32; ++i) {
        assert(gfx::replayShift(11, i, 0) == 0);
    }
    // **같은 씨앗·첨자면 같은 값이다.** 그래야 자기 검사가 CPU 와 GPU 를 바이트로 견줄 수 있다.
    assert(gfx::replayShift(11, 7, PADDING) == gfx::replayShift(11, 7, PADDING));
    assert(gfx::replayShift(11, 7, PADDING) != gfx::replayShift(12, 7, PADDING) ||
           gfx::replayShift(11, 8, PADDING) != gfx::replayShift(12, 8, PADDING));
    std::printf("무작위 이동 증강 통과\n");
}

} // namespace

int main() {
    testGaussian();
    testShapes();
    testLayout();
    testReshape();
    testInitialization();
    testValidate();
    testMemoryLayout();
    testDetach();
    testAccumulation();
    testTieRules();
    testAdamFirstStep();
    testAdamConvergence();
    testPolyak();
    testHuberValue();
    testMeanValue();
    testForwardOnlyGraph();
    testParameterFile();
    testMergeGraphs();
    testReplayIndex();
    testReplayShift();
    testGradients();
    std::printf("신경망 순수 계산 테스트 통과\n");
    return 0;
}
