// 신경망의 순수 계산. Vulkan 없이 CPU 기준 구현만 본다.
//
// 이 파일이 이 저장소에서 «역전파가 맞다» 를 말하는 유일한 근거다. GPU 커널은 여기서 검증한 CPU 기준과
// 대조해 맞춘다. 그래서 검사는 전부 **중앙 유한차분**이다 — 손으로 유도한 수식을 손으로 유도한 수식과
// 견주면 같은 실수를 두 번 하게 된다.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "gfx/neural_math.h"

namespace {

// 해석 경사와 유한차분 경사의 상대 오차. 눈금은 «이 그래프에서 경사가 대개 얼마나 큰가»로 잡는다.
// 자리마다 자기 크기로 나누면, 우연히 0 에 가까운 자리 하나가 반올림만으로 1e-2 를 찍어 구조적
// 버그(전치·첨자 밀림)와 구별할 수 없게 된다.
double relativeError(double a, double b, double typical) {
    double scale = std::max({std::abs(a), std::abs(b), 0.05 * typical, 1.0e-6});
    return std::abs(a - b) / scale;
}

// 그래프 하나를 세워 «해석 경사» 와 «유한차분 경사» 를 견준다. 흔드는 것은 파라미터와 입력 둘 다이며,
// 어느 자리를 흔들지는 씨앗으로 고른다(전부 흔들면 큰 그래프에서 너무 느리다).
struct GradientCheck {
    gfx::Graph graph;
    // 해석 경사는 **float** 로 낸다(실제로 쓰는 경로다). 유한차분은 **double** 로 잰다 — float 로 두
    // 손실의 차를 내면 반올림이 그 차이를 통째로 먹는다.
    std::vector<float> parameters;
    std::vector<float> activations;
    std::vector<float> parameterGradients;
    std::vector<float> activationGradients;
    std::vector<double> wideParameters;
    std::vector<double> wideActivations;
    // 흔들어 볼 입력 텐서들.
    std::vector<uint32_t> inputs;

    void allocate() {
        parameters.assign(graph.parameterCount, 0.0F);
        activations.assign(graph.activationCount, 0.0F);
        parameterGradients.assign(graph.parameterCount, 0.0F);
        activationGradients.assign(graph.activationCount, 0.0F);
        wideParameters.assign(graph.parameterCount, 0.0);
        wideActivations.assign(graph.activationCount, 0.0);
    }

    // 파라미터와 입력을 흩뿌린다. 0 근처만 보면 ReLU 의 갈림과 min 의 갈림을 못 밟는다.
    void randomize(uint64_t seed) {
        for (size_t i = 0; i < parameters.size(); ++i) {
            parameters[i] = 0.7F * gfx::neuralGaussian(seed, i);
            wideParameters[i] = parameters[i];
        }
        for (uint32_t tensor : inputs) {
            const gfx::Tensor& item = graph.tensors[tensor];
            for (uint32_t i = 0; i < item.count(); ++i) {
                activations[item.offset + i] = 0.9F * gfx::neuralGaussian(seed + 977, item.offset + i);
                wideActivations[item.offset + i] = activations[item.offset + i];
            }
        }
    }

    double loss() {
        gfx::forward(graph, wideParameters.data(), wideActivations.data());
        return wideActivations[graph.tensors[graph.ops.back().output].offset];
    }

    void analytic() {
        gfx::forward(graph, parameters.data(), activations.data());
        gfx::backward(
            graph, parameters.data(), activations.data(), parameterGradients.data(), activationGradients.data());
    }

    // value 를 흔들어 중앙 유한차분을 잰다. double 이라 h 를 작게 잡을 수 있고, 그만큼 절단 오차가
    // 사라진다(h 의 제곱에 비례한다).
    double numeric(double& value) {
        constexpr double H = 1.0e-5;
        double original = value;
        value = original + H;
        double plus = loss();
        value = original - H;
        double minus = loss();
        value = original;
        return (plus - minus) / (2.0 * H);
    }

    // 파라미터에서 자리를 골라 견준다. 돌려주는 값은 최대 상대 오차.
    double check(uint64_t seed, uint32_t samples) {
        randomize(seed);
        analytic();

        // 이 그래프에서 경사가 «대개 얼마나 큰가». 상대 오차의 눈금이 된다.
        double typical = 0.0;
        for (float value : parameterGradients) {
            typical = std::max(typical, static_cast<double>(std::abs(value)));
        }
        assert(typical > 0.0);

        double worst = 0.0;
        uint32_t checked = 0;
        auto compare = [&](size_t index) {
            double expected = parameterGradients[index];
            double measured = numeric(wideParameters[index]);
            worst = std::max(worst, relativeError(expected, measured, typical));
            ++checked;
        };
        // 파라미터 **텐서마다** 몇 자리씩 반드시 훑는다. 평탄한 첨자에 무작위로 때리기만 하면 작은
        // 텐서(편향은 두세 개다)가 체계적으로 밀려나 편향 경사가 통째로 검사되지 않는다.
        for (uint32_t tensor : graph.parameterTensors()) {
            const gfx::Tensor& item = graph.tensors[tensor];
            uint32_t sweep = std::min(item.count(), 4U);
            for (uint32_t i = 0; i < sweep; ++i) {
                compare(item.offset + static_cast<size_t>(i) * (item.count() / sweep));
            }
        }
        // 그 다음 나머지를 무작위로 흩어 본다.
        for (uint32_t s = 0; s < samples; ++s) {
            compare(static_cast<size_t>(std::abs(gfx::neuralGaussian(seed + 31, s)) * 1000.0F) % parameters.size());
        }
        // 입력을 흔들면 손실이 실제로 바뀌는지. 순전파가 그 값을 읽지 않으면 여기서 걸린다.
        for (uint32_t tensor : inputs) {
            const gfx::Tensor& item = graph.tensors[tensor];
            double moved = 0.0;
            for (uint32_t i = 0; i < item.count(); ++i) {
                moved = std::max(moved, std::abs(numeric(wideActivations[item.offset + i])));
            }
            assert(moved > 1.0e-9);
            ++checked;
        }
        assert(checked > 0);
        return worst;
    }
};

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
        {"layernorm", checkLayerNorm()},
        {"concat", checkConcat()},
        {"scaled loss", checkScaledLoss()},
        {"live target", checkLiveTarget()},
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
    testGradients();
    std::printf("신경망 순수 계산 테스트 통과\n");
    return 0;
}
