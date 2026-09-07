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
