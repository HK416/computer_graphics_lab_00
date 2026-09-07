#include "gfx/neural_math.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <system_error>

#include <nlohmann/json.hpp>

namespace gfx {

using nlohmann::json;

namespace {

// splitmix64. 계수 하나를 섞어 균등한 64비트를 낸다. 씨앗이 1 씩 이어져도 결과가 이어지지 않는다.
uint64_t mix(uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

// [0, 1) 균등. 상위 24비트만 쓴다(float 가 그만큼만 담는다).
float uniform(uint64_t value) {
    return static_cast<float>(value >> 40) * (1.0F / 16777216.0F);
}

// 네 축이 모두 같은가. 원소 수만 견주면 (2,3) 과 (1,6) 이 통과해 CPU 는 맞고 GPU 는 디스패치를
// 다르게 잡는 자리가 생긴다.
bool sameShape(const Tensor& a, const Tensor& b) {
    return a.dims[0] == b.dims[0] && a.dims[1] == b.dims[1] && a.dims[2] == b.dims[2] && a.dims[3] == b.dims[3];
}

// 두 텐서가 같은 저장소를 나눠 쓰는가. addReshape 로 만든 뷰가 그렇고, 같은 텐서를 두 번 준 경우도 그렇다.
bool overlaps(const Tensor& a, const Tensor& b) {
    return a.arena == b.arena && a.offset < b.offset + b.count() && b.offset < a.offset + a.count();
}

// 텐서의 값이 사는 배열과 첫 첨자. 경사는 같은 배치의 다른 배열이라 오프셋이 같다.
template <typename T> const T* readTensor(const Tensor& tensor, const T* parameters, const T* activations) {
    return (tensor.arena == Arena::PARAMETER ? parameters : activations) + tensor.offset;
}

// 연산의 출력은 늘 활성 arena 다(파라미터에 쓰는 연산은 없다).
template <typename T> T* writeTensor(const Tensor& tensor, T* activations) {
    return activations + tensor.offset;
}

// 경사 배열에서의 자리. 경사를 받지 않는 텐서는 nullptr 이라 부르는 쪽이 그냥 건너뛴다.
float* tensorGradient(const Tensor& tensor, float* parameterGradients, float* activationGradients) {
    if (!tensor.receivesGradient()) {
        return nullptr;
    }
    return (tensor.arena == Arena::PARAMETER ? parameterGradients : activationGradients) + tensor.offset;
}

// ---- 연산별 순전파. 첨자는 전부 NCHW 로 ((n*C + c)*H + h)*W + w 다.

template <typename T>
void forwardConv2d(const Tensor& in,
                   const Tensor& weight,
                   const Tensor& bias,
                   const Tensor& out,
                   int32_t stride,
                   int32_t pad,
                   const T* x,
                   const T* w,
                   const T* b,
                   T* y) {
    uint32_t batch = in.dims[0];
    uint32_t inChannels = in.dims[1];
    uint32_t inHeight = in.dims[2];
    uint32_t inWidth = in.dims[3];
    uint32_t outChannels = out.dims[1];
    uint32_t outHeight = out.dims[2];
    uint32_t outWidth = out.dims[3];
    uint32_t kernelHeight = weight.dims[2];
    uint32_t kernelWidth = weight.dims[3];
    (void)bias;

    for (uint32_t n = 0; n < batch; ++n) {
        for (uint32_t oc = 0; oc < outChannels; ++oc) {
            for (uint32_t oy = 0; oy < outHeight; ++oy) {
                for (uint32_t ox = 0; ox < outWidth; ++ox) {
                    T sum = b[oc];
                    for (uint32_t ic = 0; ic < inChannels; ++ic) {
                        for (uint32_t ky = 0; ky < kernelHeight; ++ky) {
                            int32_t iy = static_cast<int32_t>(oy) * stride + static_cast<int32_t>(ky) - pad;
                            if (iy < 0 || iy >= static_cast<int32_t>(inHeight)) {
                                continue;
                            }
                            for (uint32_t kx = 0; kx < kernelWidth; ++kx) {
                                int32_t ix = static_cast<int32_t>(ox) * stride + static_cast<int32_t>(kx) - pad;
                                if (ix < 0 || ix >= static_cast<int32_t>(inWidth)) {
                                    continue;
                                }
                                size_t xi = ((static_cast<size_t>(n) * inChannels + ic) * inHeight +
                                             static_cast<uint32_t>(iy)) *
                                                inWidth +
                                            static_cast<uint32_t>(ix);
                                size_t wi =
                                    ((static_cast<size_t>(oc) * inChannels + ic) * kernelHeight + ky) * kernelWidth +
                                    kx;
                                sum += x[xi] * w[wi];
                            }
                        }
                    }
                    y[((static_cast<size_t>(n) * outChannels + oc) * outHeight + oy) * outWidth + ox] = sum;
                }
            }
        }
    }
}

template <typename T>
void forwardLinear(
    const Tensor& in, const Tensor& weight, const Tensor& out, const T* x, const T* w, const T* b, T* y) {
    uint32_t batch = in.dims[0];
    uint32_t inputs = in.dims[1];
    uint32_t outputs = out.dims[1];
    (void)weight;
    for (uint32_t n = 0; n < batch; ++n) {
        for (uint32_t m = 0; m < outputs; ++m) {
            T sum = b[m];
            for (uint32_t k = 0; k < inputs; ++k) {
                sum += w[static_cast<size_t>(m) * inputs + k] * x[static_cast<size_t>(n) * inputs + k];
            }
            y[static_cast<size_t>(n) * outputs + m] = sum;
        }
    }
}

// layernorm 의 행 통계. 역전파도 같은 값을 쓰므로 함수로 뺀다.
template <typename T> void layerNormStats(const T* row, uint32_t features, T& mean, T& inverseDeviation) {
    T sum = T{0};
    for (uint32_t f = 0; f < features; ++f) {
        sum += row[f];
    }
    mean = sum / static_cast<T>(features);
    T variance = T{0};
    for (uint32_t f = 0; f < features; ++f) {
        T centered = row[f] - mean;
        variance += centered * centered;
    }
    variance /= static_cast<T>(features);
    // 분산이 0 인 행(모든 값이 같다)에서 나눗셈이 터지지 않게 한다. GLSL 짝도 같은 값을 쓴다.
    inverseDeviation = T{1} / std::sqrt(variance + static_cast<T>(LAYERNORM_EPSILON));
}

} // namespace

std::vector<uint32_t> Graph::parameterTensors() const {
    std::vector<uint32_t> result;
    for (uint32_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].arena == Arena::PARAMETER) {
            result.push_back(i);
        }
    }
    return result;
}

uint32_t convOutputSize(uint32_t input, uint32_t kernel, uint32_t stride, uint32_t pad) {
    if (stride == 0 || input + 2 * pad < kernel) {
        return 0;
    }
    return (input + 2 * pad - kernel) / stride + 1;
}

uint32_t GraphBuilder::allocate(Arena arena, uint32_t n, uint32_t c, uint32_t h, uint32_t w, uint32_t flags) {
    Tensor tensor;
    tensor.arena = arena;
    tensor.dims[0] = std::max(n, 1U);
    tensor.dims[1] = std::max(c, 1U);
    tensor.dims[2] = std::max(h, 1U);
    tensor.dims[3] = std::max(w, 1U);
    tensor.flags = flags;
    size_t& cursor = arena == Arena::PARAMETER ? graph.parameterCount : graph.activationCount;
    tensor.offset = static_cast<uint32_t>(cursor);
    cursor += tensor.count();
    graph.tensors.push_back(tensor);
    return static_cast<uint32_t>(graph.tensors.size() - 1);
}

uint32_t GraphBuilder::emit(OpKind kind, uint32_t a, uint32_t b, uint32_t c, uint32_t output) {
    Op op;
    op.kind = kind;
    op.inputs[0] = a;
    op.inputs[1] = b;
    op.inputs[2] = c;
    op.output = output;
    graph.ops.push_back(op);
    return output;
}

uint32_t GraphBuilder::addInput(uint32_t n, uint32_t c, uint32_t h, uint32_t w) {
    uint32_t tensor = allocate(Arena::ACTIVATION, n, c, h, w, 0);
    emit(OpKind::INPUT, NO_TENSOR, NO_TENSOR, NO_TENSOR, tensor);
    return tensor;
}

uint32_t GraphBuilder::addParameter(uint32_t n, uint32_t c, uint32_t h, uint32_t w) {
    return allocate(Arena::PARAMETER, n, c, h, w, TENSOR_GRAD);
}

uint32_t GraphBuilder::addConv2d(uint32_t input, uint32_t weight, uint32_t bias, uint32_t stride, uint32_t pad) {
    if (input >= graph.tensors.size() || weight >= graph.tensors.size() || bias >= graph.tensors.size()) {
        return NO_TENSOR;
    }
    const Tensor& in = graph.tensors[input];
    const Tensor& w = graph.tensors[weight];
    if (w.dims[1] != in.dims[1] || bias >= graph.tensors.size() || graph.tensors[bias].count() != w.dims[0]) {
        return NO_TENSOR;
    }
    // 세 입력이 같은 저장소를 나눠 쓰면 안 된다. GPU 는 dx·dw·db 를 배리어 없이 나란히 돌리므로
    // (서로 다른 텐서라는 전제로) 겹쳐 있으면 두 커널이 같은 자리를 동시에 고친다.
    if (overlaps(in, w) || overlaps(in, graph.tensors[bias]) || overlaps(w, graph.tensors[bias])) {
        return NO_TENSOR;
    }
    uint32_t outHeight = convOutputSize(in.dims[2], w.dims[2], stride, pad);
    uint32_t outWidth = convOutputSize(in.dims[3], w.dims[3], stride, pad);
    if (outHeight == 0 || outWidth == 0) {
        return NO_TENSOR;
    }
    uint32_t output = allocate(Arena::ACTIVATION, in.dims[0], w.dims[0], outHeight, outWidth, TENSOR_GRAD);
    emit(OpKind::CONV2D, input, weight, bias, output);
    graph.ops.back().iparams[0] = static_cast<int32_t>(stride);
    graph.ops.back().iparams[1] = static_cast<int32_t>(pad);
    return output;
}

uint32_t GraphBuilder::addLinear(uint32_t input, uint32_t weight, uint32_t bias) {
    if (input >= graph.tensors.size() || weight >= graph.tensors.size() || bias >= graph.tensors.size()) {
        return NO_TENSOR;
    }
    const Tensor& in = graph.tensors[input];
    const Tensor& w = graph.tensors[weight];
    // 선형은 (배치, 특징) 으로만 본다. 합성곱 출력을 이어 붙일 때는 미리 평탄하게 잡아 둔다.
    if (in.dims[2] != 1 || in.dims[3] != 1 || w.dims[1] != in.dims[1] || graph.tensors[bias].count() != w.dims[0]) {
        return NO_TENSOR;
    }
    // 합성곱과 같은 이유로 세 입력이 겹치면 안 된다.
    if (overlaps(in, w) || overlaps(in, graph.tensors[bias]) || overlaps(w, graph.tensors[bias])) {
        return NO_TENSOR;
    }
    uint32_t output = allocate(Arena::ACTIVATION, in.dims[0], w.dims[0], 1, 1, TENSOR_GRAD);
    emit(OpKind::LINEAR, input, weight, bias, output);
    return output;
}

uint32_t GraphBuilder::addRelu(uint32_t input) {
    if (input >= graph.tensors.size()) {
        return NO_TENSOR;
    }
    const Tensor& in = graph.tensors[input];
    uint32_t output = allocate(Arena::ACTIVATION, in.dims[0], in.dims[1], in.dims[2], in.dims[3], TENSOR_GRAD);
    return emit(OpKind::RELU, input, NO_TENSOR, NO_TENSOR, output);
}

uint32_t GraphBuilder::addTanh(uint32_t input) {
    if (input >= graph.tensors.size()) {
        return NO_TENSOR;
    }
    const Tensor& in = graph.tensors[input];
    uint32_t output = allocate(Arena::ACTIVATION, in.dims[0], in.dims[1], in.dims[2], in.dims[3], TENSOR_GRAD);
    return emit(OpKind::TANH, input, NO_TENSOR, NO_TENSOR, output);
}

uint32_t GraphBuilder::addLayerNorm(uint32_t input, uint32_t gain, uint32_t bias) {
    if (input >= graph.tensors.size() || gain >= graph.tensors.size() || bias >= graph.tensors.size()) {
        return NO_TENSOR;
    }
    const Tensor& in = graph.tensors[input];
    if (in.dims[2] != 1 || in.dims[3] != 1 || graph.tensors[gain].count() != in.dims[1] ||
        graph.tensors[bias].count() != in.dims[1]) {
        return NO_TENSOR;
    }
    uint32_t output = allocate(Arena::ACTIVATION, in.dims[0], in.dims[1], 1, 1, TENSOR_GRAD);
    return emit(OpKind::LAYERNORM, input, gain, bias, output);
}

uint32_t GraphBuilder::addAdd(uint32_t a, uint32_t b) {
    if (a >= graph.tensors.size() || b >= graph.tensors.size()) {
        return NO_TENSOR;
    }
    const Tensor& first = graph.tensors[a];
    if (!sameShape(first, graph.tensors[b])) {
        return NO_TENSOR;
    }
    uint32_t output =
        allocate(Arena::ACTIVATION, first.dims[0], first.dims[1], first.dims[2], first.dims[3], TENSOR_GRAD);
    return emit(OpKind::ADD, a, b, NO_TENSOR, output);
}

uint32_t GraphBuilder::addMul(uint32_t a, uint32_t b) {
    if (a >= graph.tensors.size() || b >= graph.tensors.size()) {
        return NO_TENSOR;
    }
    const Tensor& first = graph.tensors[a];
    if (!sameShape(first, graph.tensors[b])) {
        return NO_TENSOR;
    }
    uint32_t output =
        allocate(Arena::ACTIVATION, first.dims[0], first.dims[1], first.dims[2], first.dims[3], TENSOR_GRAD);
    return emit(OpKind::MUL, a, b, NO_TENSOR, output);
}

uint32_t GraphBuilder::addConcat(uint32_t a, uint32_t b) {
    if (a >= graph.tensors.size() || b >= graph.tensors.size()) {
        return NO_TENSOR;
    }
    const Tensor& first = graph.tensors[a];
    const Tensor& second = graph.tensors[b];
    // (배치, 특징) 끼리만 잇는다. 배치가 같아야 한다.
    if (first.dims[0] != second.dims[0] || first.dims[2] != 1 || first.dims[3] != 1 || second.dims[2] != 1 ||
        second.dims[3] != 1) {
        return NO_TENSOR;
    }
    // **두 조각이 같은 저장소를 나눠 쓰면 안 된다.** 이음은 스레드가 «자기 첨자가 아닌 자리» 에 쓰는
    // 유일한 연산이라(f 가 경계를 넘으면 둘째 조각의 다른 첨자가 된다), 겹쳐 있으면 GPU 에서 두 스레드가
    // 같은 칸을 동시에 읽고 써 경사 하나를 잃는다. CPU 는 순서대로 돌아 둘 다 더하므로 답이 갈린다.
    if (overlaps(first, second)) {
        return NO_TENSOR;
    }
    uint32_t output = allocate(Arena::ACTIVATION, first.dims[0], first.dims[1] + second.dims[1], 1, 1, TENSOR_GRAD);
    return emit(OpKind::CONCAT, a, b, NO_TENSOR, output);
}

uint32_t GraphBuilder::addScale(uint32_t input, float factor) {
    if (input >= graph.tensors.size()) {
        return NO_TENSOR;
    }
    const Tensor& in = graph.tensors[input];
    uint32_t output = allocate(Arena::ACTIVATION, in.dims[0], in.dims[1], in.dims[2], in.dims[3], TENSOR_GRAD);
    emit(OpKind::SCALE, input, NO_TENSOR, NO_TENSOR, output);
    graph.ops.back().fparams[0] = factor;
    return output;
}

uint32_t GraphBuilder::addMean(uint32_t input) {
    if (input >= graph.tensors.size()) {
        return NO_TENSOR;
    }
    // 원소 수는 볼 것이 없다. allocate 가 축마다 1 을 하한으로 두므로 count() 는 늘 1 이상이다.
    uint32_t output = allocate(Arena::ACTIVATION, 1, 1, 1, 1, TENSOR_GRAD);
    return emit(OpKind::MEAN, input, NO_TENSOR, NO_TENSOR, output);
}

uint32_t GraphBuilder::addMin2(uint32_t a, uint32_t b) {
    if (a >= graph.tensors.size() || b >= graph.tensors.size()) {
        return NO_TENSOR;
    }
    const Tensor& first = graph.tensors[a];
    if (!sameShape(first, graph.tensors[b])) {
        return NO_TENSOR;
    }
    uint32_t output =
        allocate(Arena::ACTIVATION, first.dims[0], first.dims[1], first.dims[2], first.dims[3], TENSOR_GRAD);
    return emit(OpKind::MIN2, a, b, NO_TENSOR, output);
}

uint32_t GraphBuilder::addMse(uint32_t prediction, uint32_t target) {
    if (prediction >= graph.tensors.size() || target >= graph.tensors.size()) {
        return NO_TENSOR;
    }
    if (!sameShape(graph.tensors[prediction], graph.tensors[target])) {
        return NO_TENSOR;
    }
    uint32_t output = allocate(Arena::ACTIVATION, 1, 1, 1, 1, TENSOR_GRAD);
    return emit(OpKind::MSE, prediction, target, NO_TENSOR, output);
}

uint32_t GraphBuilder::addHuber(uint32_t prediction, uint32_t target, float delta) {
    if (prediction >= graph.tensors.size() || target >= graph.tensors.size()) {
        return NO_TENSOR;
    }
    if (!sameShape(graph.tensors[prediction], graph.tensors[target]) || !(delta > 0.0F)) {
        return NO_TENSOR;
    }
    uint32_t output = allocate(Arena::ACTIVATION, 1, 1, 1, 1, TENSOR_GRAD);
    emit(OpKind::HUBER, prediction, target, NO_TENSOR, output);
    graph.ops.back().fparams[0] = delta;
    return output;
}

uint32_t GraphBuilder::addReshape(uint32_t tensor, uint32_t n, uint32_t c, uint32_t h, uint32_t w) {
    if (tensor >= graph.tensors.size()) {
        return NO_TENSOR;
    }
    Tensor view = graph.tensors[tensor];
    if (view.arena == Arena::PARAMETER) {
        return NO_TENSOR;
    }
    uint32_t dims[4] = {std::max(n, 1U), std::max(c, 1U), std::max(h, 1U), std::max(w, 1U)};
    if (dims[0] * dims[1] * dims[2] * dims[3] != view.count()) {
        return NO_TENSOR;
    }
    // arena·offset·flags 를 그대로 물려받는다. 자리를 새로 잡지 않으므로 arena 크기가 늘지 않는다.
    for (int i = 0; i < 4; ++i) {
        view.dims[i] = dims[i];
    }
    graph.tensors.push_back(view);
    return static_cast<uint32_t>(graph.tensors.size() - 1);
}

size_t GraphBuilder::parameterCount() const {
    return graph.parameterCount;
}

uint32_t GraphBuilder::featureCount(uint32_t tensor) const {
    if (tensor >= graph.tensors.size()) {
        return 0;
    }
    const Tensor& item = graph.tensors[tensor];
    return item.dims[1] * item.dims[2] * item.dims[3];
}

void GraphBuilder::detach(uint32_t tensor) {
    if (tensor < graph.tensors.size()) {
        graph.tensors[tensor].flags &= ~TENSOR_GRAD;
    }
}

namespace {

template <typename T> void forwardImpl(const Graph& graph, const T* parameters, T* activations) {
    // validate 를 통과한 표를 전제한다. 여기서 첨자를 다시 확인하면 «처리한 것처럼 보이지만 바로
    // 다음 줄에서 죽는» 반쪽 가드가 되고, 학습 루프에서 매번 도는 비용도 든다.
    for (const Op& op : graph.ops) {
        const Tensor& out = graph.tensors[op.output];
        T* y = writeTensor(out, activations);
        const T* constActivations = activations;
        if (op.kind == OpKind::INPUT) {
            continue;
        }
        const Tensor& in = graph.tensors[op.inputs[0]];
        const T* x = readTensor(in, parameters, constActivations);

        switch (op.kind) {
        case OpKind::INPUT:
            break;
        case OpKind::CONV2D: {
            const Tensor& weight = graph.tensors[op.inputs[1]];
            const Tensor& bias = graph.tensors[op.inputs[2]];
            forwardConv2d(in,
                          weight,
                          bias,
                          out,
                          op.iparams[0],
                          op.iparams[1],
                          x,
                          readTensor(weight, parameters, constActivations),
                          readTensor(bias, parameters, constActivations),
                          y);
            break;
        }
        case OpKind::LINEAR: {
            const Tensor& weight = graph.tensors[op.inputs[1]];
            const Tensor& bias = graph.tensors[op.inputs[2]];
            forwardLinear(in,
                          weight,
                          out,
                          x,
                          readTensor(weight, parameters, constActivations),
                          readTensor(bias, parameters, constActivations),
                          y);
            break;
        }
        case OpKind::RELU:
            for (uint32_t i = 0; i < out.count(); ++i) {
                y[i] = std::max(x[i], T{0});
            }
            break;
        case OpKind::TANH:
            for (uint32_t i = 0; i < out.count(); ++i) {
                y[i] = std::tanh(x[i]);
            }
            break;
        case OpKind::LAYERNORM: {
            const T* gain = readTensor(graph.tensors[op.inputs[1]], parameters, constActivations);
            const T* bias = readTensor(graph.tensors[op.inputs[2]], parameters, constActivations);
            uint32_t features = in.dims[1];
            for (uint32_t n = 0; n < in.dims[0]; ++n) {
                const T* row = x + static_cast<size_t>(n) * features;
                T mean = T{0};
                T inverseDeviation = T{0};
                layerNormStats(row, features, mean, inverseDeviation);
                for (uint32_t f = 0; f < features; ++f) {
                    y[static_cast<size_t>(n) * features + f] = (row[f] - mean) * inverseDeviation * gain[f] + bias[f];
                }
            }
            break;
        }
        case OpKind::ADD: {
            const T* second = readTensor(graph.tensors[op.inputs[1]], parameters, constActivations);
            for (uint32_t i = 0; i < out.count(); ++i) {
                y[i] = x[i] + second[i];
            }
            break;
        }
        case OpKind::MUL: {
            const T* second = readTensor(graph.tensors[op.inputs[1]], parameters, constActivations);
            for (uint32_t i = 0; i < out.count(); ++i) {
                y[i] = x[i] * second[i];
            }
            break;
        }
        case OpKind::CONCAT: {
            const Tensor& secondTensor = graph.tensors[op.inputs[1]];
            const T* second = readTensor(secondTensor, parameters, constActivations);
            uint32_t firstFeatures = in.dims[1];
            uint32_t secondFeatures = secondTensor.dims[1];
            uint32_t total = firstFeatures + secondFeatures;
            for (uint32_t n = 0; n < out.dims[0]; ++n) {
                std::memcpy(y + static_cast<size_t>(n) * total,
                            x + static_cast<size_t>(n) * firstFeatures,
                            firstFeatures * sizeof(T));
                std::memcpy(y + static_cast<size_t>(n) * total + firstFeatures,
                            second + static_cast<size_t>(n) * secondFeatures,
                            secondFeatures * sizeof(T));
            }
            break;
        }
        case OpKind::SCALE:
            for (uint32_t i = 0; i < out.count(); ++i) {
                y[i] = x[i] * static_cast<T>(op.fparams[0]);
            }
            break;
        case OpKind::MEAN: {
            uint32_t count = in.count();
            // 순서대로 더한다. GLSL 짝은 트리 리덕션이라 반올림이 갈리지만, 유한한 배치에서 그 차이는
            // 자기 검사의 허용치 안이다.
            T sum = T{0};
            for (uint32_t i = 0; i < count; ++i) {
                sum += x[i];
            }
            y[0] = count > 0 ? sum / static_cast<T>(count) : T{0};
            break;
        }
        case OpKind::MIN2: {
            const T* second = readTensor(graph.tensors[op.inputs[1]], parameters, constActivations);
            for (uint32_t i = 0; i < out.count(); ++i) {
                y[i] = std::min(x[i], second[i]);
            }
            break;
        }
        case OpKind::MSE: {
            const T* target = readTensor(graph.tensors[op.inputs[1]], parameters, constActivations);
            uint32_t count = in.count();
            T sum = T{0};
            for (uint32_t i = 0; i < count; ++i) {
                T error = x[i] - target[i];
                sum += error * error;
            }
            y[0] = count > 0 ? sum / static_cast<T>(count) : T{0};
            break;
        }
        case OpKind::HUBER: {
            const T* target = readTensor(graph.tensors[op.inputs[1]], parameters, constActivations);
            uint32_t count = in.count();
            T delta = static_cast<T>(op.fparams[0]);
            T sum = T{0};
            for (uint32_t i = 0; i < count; ++i) {
                T error = std::abs(x[i] - target[i]);
                // 꺾이는 지점에서 값과 기울기가 모두 이어지도록 뒤쪽 식에 delta/2 를 뺀다.
                sum += error <= delta ? T{0.5} * error * error : delta * (error - T{0.5} * delta);
            }
            y[0] = count > 0 ? sum / static_cast<T>(count) : T{0};
            break;
        }
        }
    }
}

} // namespace

bool validateForward(const Graph& graph) {
    if (graph.ops.empty()) {
        return false;
    }
    for (size_t index = 0; index < graph.ops.size(); ++index) {
        const Op& op = graph.ops[index];
        if (op.output >= graph.tensors.size()) {
            return false;
        }
        // 출력은 늘 활성 arena 다. 파라미터에 쓰는 연산은 없다.
        if (graph.tensors[op.output].arena != Arena::ACTIVATION) {
            return false;
        }
        // 필요한 입력 수는 종류가 정한다. 남는 자리는 NO_TENSOR 여야 한다.
        uint32_t required = 0;
        switch (op.kind) {
        case OpKind::INPUT:
            required = 0;
            break;
        case OpKind::CONV2D:
        case OpKind::LINEAR:
        case OpKind::LAYERNORM:
            required = 3;
            break;
        case OpKind::ADD:
        case OpKind::MUL:
        case OpKind::CONCAT:
        case OpKind::MIN2:
        case OpKind::MSE:
        case OpKind::HUBER:
            required = 2;
            break;
        case OpKind::RELU:
        case OpKind::TANH:
        case OpKind::SCALE:
        case OpKind::MEAN:
            required = 1;
            break;
        }
        for (uint32_t i = 0; i < 3; ++i) {
            bool used = i < required;
            if (used == (op.inputs[i] == NO_TENSOR)) {
                return false;
            }
            // 출력 번호가 늘 입력 번호보다 커야 «표를 거꾸로 훑는 것» 이 올바른 위상 정렬이다.
            if (used && (op.inputs[i] >= graph.tensors.size() || op.inputs[i] >= op.output)) {
                return false;
            }
        }
        if (graph.tensors[op.output].offset + graph.tensors[op.output].count() > graph.activationCount) {
            return false;
        }
        // 원소별로 두 입력을 훑는 연산은 모양이 같아야 한다. 빌더는 이미 막지만, 손으로 짓거나 파일에서
        // 읽은 표는 여기서 걸러야 범위 밖을 읽지 않는다.
        switch (op.kind) {
        case OpKind::ADD:
        case OpKind::MUL:
        case OpKind::MIN2:
        case OpKind::MSE:
        case OpKind::HUBER:
            if (!sameShape(graph.tensors[op.inputs[0]], graph.tensors[op.inputs[1]])) {
                return false;
            }
            break;
        default:
            break;
        }
        // 이음의 두 조각이 겹치면 GPU 에서 경사 하나를 잃는다(addConcat 의 주석 참고). 합성곱·선형은
        // dx·dw·db 를 배리어 없이 나란히 돌리므로 세 입력이 겹치면 두 커널이 같은 자리를 동시에 고친다.
        // 빌더는 이미 막지만 손으로 짓거나 파일에서 읽은 표는 여기서 건다.
        if (op.kind == OpKind::CONCAT && overlaps(graph.tensors[op.inputs[0]], graph.tensors[op.inputs[1]])) {
            return false;
        }
        if (op.kind == OpKind::CONV2D || op.kind == OpKind::LINEAR) {
            const Tensor& source = graph.tensors[op.inputs[0]];
            const Tensor& weight = graph.tensors[op.inputs[1]];
            const Tensor& bias = graph.tensors[op.inputs[2]];
            if (overlaps(source, weight) || overlaps(source, bias) || overlaps(weight, bias)) {
                return false;
            }
        }
        // 합성곱의 보폭이 0 이면 GPU 의 모아 읽기가 0 으로 나눈다.
        if (op.kind == OpKind::CONV2D && op.iparams[0] <= 0) {
            return false;
        }
        // 접는 연산은 출력이 스칼라다. 아니면 순전파가 y[0] 에만 쓰고 나머지는 지난 프레임의 값으로
        // 남아, 다음 연산이 쓰레기를 읽는다.
        switch (op.kind) {
        case OpKind::MEAN:
        case OpKind::MSE:
        case OpKind::HUBER:
            if (graph.tensors[op.output].count() != 1) {
                return false;
            }
            break;
        default:
            break;
        }
    }
    return true;
}

bool validate(const Graph& graph) {
    if (!validateForward(graph)) {
        return false;
    }
    // 마지막 연산의 출력이 «경사를 받는 스칼라» 여야 역전파의 씨앗을 심을 수 있다.
    const Tensor& last = graph.tensors[graph.ops.back().output];
    return last.count() == 1 && last.receivesGradient();
}

void forward(const Graph& graph, const float* parameters, float* activations) {
    forwardImpl<float>(graph, parameters, activations);
}

void forward(const Graph& graph, const double* parameters, double* activations) {
    forwardImpl<double>(graph, parameters, activations);
}

namespace {

// ---- 연산별 역전파. 출력 경사 dy 를 받아 입력·가중치 경사에 **더한다**(누적이라 갈래가 여럿이어도 된다).

void backwardConv2d(const Tensor& in,
                    const Tensor& weight,
                    const Tensor& out,
                    int32_t stride,
                    int32_t pad,
                    const float* x,
                    const float* w,
                    const float* dy,
                    float* dx,
                    float* dw,
                    float* db) {
    uint32_t batch = in.dims[0];
    uint32_t inChannels = in.dims[1];
    uint32_t inHeight = in.dims[2];
    uint32_t inWidth = in.dims[3];
    uint32_t outChannels = out.dims[1];
    uint32_t outHeight = out.dims[2];
    uint32_t outWidth = out.dims[3];
    uint32_t kernelHeight = weight.dims[2];
    uint32_t kernelWidth = weight.dims[3];

    for (uint32_t n = 0; n < batch; ++n) {
        for (uint32_t oc = 0; oc < outChannels; ++oc) {
            for (uint32_t oy = 0; oy < outHeight; ++oy) {
                for (uint32_t ox = 0; ox < outWidth; ++ox) {
                    float g = dy[((static_cast<size_t>(n) * outChannels + oc) * outHeight + oy) * outWidth + ox];
                    if (db != nullptr) {
                        db[oc] += g;
                    }
                    for (uint32_t ic = 0; ic < inChannels; ++ic) {
                        for (uint32_t ky = 0; ky < kernelHeight; ++ky) {
                            int32_t iy = static_cast<int32_t>(oy) * stride + static_cast<int32_t>(ky) - pad;
                            if (iy < 0 || iy >= static_cast<int32_t>(inHeight)) {
                                continue;
                            }
                            for (uint32_t kx = 0; kx < kernelWidth; ++kx) {
                                int32_t ix = static_cast<int32_t>(ox) * stride + static_cast<int32_t>(kx) - pad;
                                if (ix < 0 || ix >= static_cast<int32_t>(inWidth)) {
                                    continue;
                                }
                                size_t xi = ((static_cast<size_t>(n) * inChannels + ic) * inHeight +
                                             static_cast<uint32_t>(iy)) *
                                                inWidth +
                                            static_cast<uint32_t>(ix);
                                size_t wi =
                                    ((static_cast<size_t>(oc) * inChannels + ic) * kernelHeight + ky) * kernelWidth +
                                    kx;
                                if (dw != nullptr) {
                                    dw[wi] += g * x[xi];
                                }
                                if (dx != nullptr) {
                                    dx[xi] += g * w[wi];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void backwardLinear(const Tensor& in,
                    const Tensor& out,
                    const float* x,
                    const float* w,
                    const float* dy,
                    float* dx,
                    float* dw,
                    float* db) {
    uint32_t batch = in.dims[0];
    uint32_t inputs = in.dims[1];
    uint32_t outputs = out.dims[1];
    for (uint32_t n = 0; n < batch; ++n) {
        for (uint32_t m = 0; m < outputs; ++m) {
            float g = dy[static_cast<size_t>(n) * outputs + m];
            if (db != nullptr) {
                db[m] += g;
            }
            for (uint32_t k = 0; k < inputs; ++k) {
                if (dw != nullptr) {
                    dw[static_cast<size_t>(m) * inputs + k] += g * x[static_cast<size_t>(n) * inputs + k];
                }
                if (dx != nullptr) {
                    dx[static_cast<size_t>(n) * inputs + k] += g * w[static_cast<size_t>(m) * inputs + k];
                }
            }
        }
    }
}

} // namespace

bool backward(const Graph& graph,
              const float* parameters,
              const float* activations,
              float* parameterGradients,
              float* activationGradients) {
    std::fill(parameterGradients, parameterGradients + graph.parameterCount, 0.0F);
    std::fill(activationGradients, activationGradients + graph.activationCount, 0.0F);
    if (graph.ops.empty()) {
        return false;
    }
    // 마지막 연산의 출력이 스칼라 손실이다. d(손실)/d(손실) = 1 을 심고 표를 거꾸로 훑는다. 손실이
    // 아니면(모양이 안 맞아 addMse 가 NO_TENSOR 를 냈거나, 손실 뒤에 입력을 하나 더 잡았거나) 여기서
    // 거짓을 돌려준다 — 그러지 않으면 경사가 전부 0 인 채로 «학습이 안 되는데 이유를 모르는» 자리가 된다.
    const Tensor& lastTensor = graph.tensors[graph.ops.back().output];
    if (lastTensor.count() != 1 || !lastTensor.receivesGradient()) {
        return false;
    }
    float* seed = tensorGradient(lastTensor, parameterGradients, activationGradients);
    seed[0] = 1.0F;
    backwardFrom(graph, parameters, activations, parameterGradients, activationGradients);
    return true;
}

void backwardFrom(const Graph& graph,
                  const float* parameters,
                  const float* activations,
                  float* parameterGradients,
                  float* activationGradients) {
    for (size_t index = graph.ops.size(); index-- > 0;) {
        const Op& op = graph.ops[index];
        if (op.kind == OpKind::INPUT) {
            continue;
        }
        const Tensor& out = graph.tensors[op.output];
        const float* dy = tensorGradient(out, parameterGradients, activationGradients);
        if (dy == nullptr) {
            continue;
        }
        const float* y = readTensor(out, parameters, activations);
        const Tensor& in = graph.tensors[op.inputs[0]];
        const float* x = readTensor(in, parameters, activations);
        float* dx = tensorGradient(in, parameterGradients, activationGradients);

        switch (op.kind) {
        case OpKind::INPUT:
            break;
        case OpKind::CONV2D: {
            const Tensor& weight = graph.tensors[op.inputs[1]];
            const Tensor& bias = graph.tensors[op.inputs[2]];
            backwardConv2d(in,
                           weight,
                           out,
                           op.iparams[0],
                           op.iparams[1],
                           x,
                           readTensor(weight, parameters, activations),
                           dy,
                           dx,
                           tensorGradient(weight, parameterGradients, activationGradients),
                           tensorGradient(bias, parameterGradients, activationGradients));
            break;
        }
        case OpKind::LINEAR: {
            const Tensor& weight = graph.tensors[op.inputs[1]];
            const Tensor& bias = graph.tensors[op.inputs[2]];
            backwardLinear(in,
                           out,
                           x,
                           readTensor(weight, parameters, activations),
                           dy,
                           dx,
                           tensorGradient(weight, parameterGradients, activationGradients),
                           tensorGradient(bias, parameterGradients, activationGradients));
            break;
        }
        case OpKind::RELU:
            if (dx != nullptr) {
                // 전활성값의 부호로 마스킹한다. y = max(x, 0) 이라 출력으로 판정해도 결과는 같지만,
                // 순전파가 어차피 전활성값을 남겨 두므로 정의를 그대로 읽는 쪽이 낫다.
                for (uint32_t i = 0; i < out.count(); ++i) {
                    dx[i] += x[i] > 0.0F ? dy[i] : 0.0F;
                }
            }
            break;
        case OpKind::TANH:
            if (dx != nullptr) {
                for (uint32_t i = 0; i < out.count(); ++i) {
                    dx[i] += dy[i] * (1.0F - y[i] * y[i]);
                }
            }
            break;
        case OpKind::LAYERNORM: {
            const Tensor& gainTensor = graph.tensors[op.inputs[1]];
            const float* gain = readTensor(gainTensor, parameters, activations);
            float* dGain = tensorGradient(gainTensor, parameterGradients, activationGradients);
            float* dBias = tensorGradient(graph.tensors[op.inputs[2]], parameterGradients, activationGradients);
            uint32_t features = in.dims[1];
            for (uint32_t n = 0; n < in.dims[0]; ++n) {
                const float* row = x + static_cast<size_t>(n) * features;
                const float* rowGrad = dy + static_cast<size_t>(n) * features;
                float mean = 0.0F;
                float inverseDeviation = 0.0F;
                layerNormStats(row, features, mean, inverseDeviation);
                // 정규화된 값 x̂ 에 대한 경사를 모아 두 리덕션(Σ dx̂, Σ dx̂·x̂)으로 행 전체를 되민다.
                float sumGrad = 0.0F;
                float sumGradNormalized = 0.0F;
                for (uint32_t f = 0; f < features; ++f) {
                    float normalized = (row[f] - mean) * inverseDeviation;
                    float gradNormalized = rowGrad[f] * gain[f];
                    sumGrad += gradNormalized;
                    sumGradNormalized += gradNormalized * normalized;
                    if (dGain != nullptr) {
                        dGain[f] += rowGrad[f] * normalized;
                    }
                    if (dBias != nullptr) {
                        dBias[f] += rowGrad[f];
                    }
                }
                if (dx == nullptr) {
                    continue;
                }
                float inverseFeatures = 1.0F / static_cast<float>(features);
                for (uint32_t f = 0; f < features; ++f) {
                    float normalized = (row[f] - mean) * inverseDeviation;
                    float gradNormalized = rowGrad[f] * gain[f];
                    dx[static_cast<size_t>(n) * features + f] +=
                        inverseDeviation *
                        (gradNormalized - inverseFeatures * sumGrad - inverseFeatures * normalized * sumGradNormalized);
                }
            }
            break;
        }
        case OpKind::ADD: {
            float* dSecond = tensorGradient(graph.tensors[op.inputs[1]], parameterGradients, activationGradients);
            for (uint32_t i = 0; i < out.count(); ++i) {
                if (dx != nullptr) {
                    dx[i] += dy[i];
                }
                if (dSecond != nullptr) {
                    dSecond[i] += dy[i];
                }
            }
            break;
        }
        case OpKind::MUL: {
            const Tensor& secondTensor = graph.tensors[op.inputs[1]];
            const float* second = readTensor(secondTensor, parameters, activations);
            float* dSecond = tensorGradient(secondTensor, parameterGradients, activationGradients);
            for (uint32_t i = 0; i < out.count(); ++i) {
                // 짝의 **값** 을 곱한다. 한쪽이 경사를 받지 않아도 다른 쪽 경사에는 그 값이 든다.
                if (dx != nullptr) {
                    dx[i] += dy[i] * second[i];
                }
                if (dSecond != nullptr) {
                    dSecond[i] += dy[i] * x[i];
                }
            }
            break;
        }
        case OpKind::CONCAT: {
            const Tensor& secondTensor = graph.tensors[op.inputs[1]];
            float* dSecond = tensorGradient(secondTensor, parameterGradients, activationGradients);
            uint32_t firstFeatures = in.dims[1];
            uint32_t secondFeatures = secondTensor.dims[1];
            uint32_t total = firstFeatures + secondFeatures;
            for (uint32_t n = 0; n < out.dims[0]; ++n) {
                for (uint32_t f = 0; f < firstFeatures; ++f) {
                    if (dx != nullptr) {
                        dx[static_cast<size_t>(n) * firstFeatures + f] += dy[static_cast<size_t>(n) * total + f];
                    }
                }
                for (uint32_t f = 0; f < secondFeatures; ++f) {
                    if (dSecond != nullptr) {
                        dSecond[static_cast<size_t>(n) * secondFeatures + f] +=
                            dy[static_cast<size_t>(n) * total + firstFeatures + f];
                    }
                }
            }
            break;
        }
        case OpKind::SCALE:
            if (dx != nullptr) {
                for (uint32_t i = 0; i < out.count(); ++i) {
                    dx[i] += dy[i] * op.fparams[0];
                }
            }
            break;
        case OpKind::MEAN: {
            uint32_t count = in.count();
            if (dx == nullptr || count == 0) {
                break;
            }
            float scale = dy[0] / static_cast<float>(count);
            for (uint32_t i = 0; i < count; ++i) {
                dx[i] += scale;
            }
            break;
        }
        case OpKind::MIN2: {
            const Tensor& secondTensor = graph.tensors[op.inputs[1]];
            const float* second = readTensor(secondTensor, parameters, activations);
            float* dSecond = tensorGradient(secondTensor, parameterGradients, activationGradients);
            for (uint32_t i = 0; i < out.count(); ++i) {
                // 같은 값이면 첫째가 이긴다. GLSL 짝도 같은 규칙이라 두 엔진의 결과가 갈리지 않는다.
                bool firstWins = x[i] <= second[i];
                if (firstWins && dx != nullptr) {
                    dx[i] += dy[i];
                }
                if (!firstWins && dSecond != nullptr) {
                    dSecond[i] += dy[i];
                }
            }
            break;
        }
        case OpKind::MSE: {
            const Tensor& targetTensor = graph.tensors[op.inputs[1]];
            const float* target = readTensor(targetTensor, parameters, activations);
            float* dTarget = tensorGradient(targetTensor, parameterGradients, activationGradients);
            uint32_t count = in.count();
            if (count == 0) {
                break;
            }
            float scale = 2.0F * dy[0] / static_cast<float>(count);
            for (uint32_t i = 0; i < count; ++i) {
                float error = x[i] - target[i];
                if (dx != nullptr) {
                    dx[i] += scale * error;
                }
                if (dTarget != nullptr) {
                    dTarget[i] -= scale * error;
                }
            }
            break;
        }
        case OpKind::HUBER: {
            const Tensor& targetTensor = graph.tensors[op.inputs[1]];
            const float* target = readTensor(targetTensor, parameters, activations);
            float* dTarget = tensorGradient(targetTensor, parameterGradients, activationGradients);
            uint32_t count = in.count();
            if (count == 0) {
                break;
            }
            float delta = op.fparams[0];
            float scale = dy[0] / static_cast<float>(count);
            for (uint32_t i = 0; i < count; ++i) {
                float error = x[i] - target[i];
                // 꺾이는 지점 밖에서는 기울기가 delta 로 눕는다. 정확히 그 지점에서는 안쪽 식을 쓴다.
                float slope = std::abs(error) <= delta ? error : (error > 0.0F ? delta : -delta);
                if (dx != nullptr) {
                    dx[i] += scale * slope;
                }
                if (dTarget != nullptr) {
                    dTarget[i] -= scale * slope;
                }
            }
            break;
        }
        }
    }
}

float neuralGaussian(uint64_t seed, uint64_t index) {
    // Box-Muller. 균등 둘을 표준 정규 하나로 바꾼다. u1 이 0 이면 로그가 발산하므로 바닥을 둔다.
    uint64_t state = mix(seed * 0x9E3779B97F4A7C15ULL + index);
    float u1 = std::max(uniform(state), 1.0e-7F);
    float u2 = uniform(mix(state));
    return std::sqrt(-2.0F * std::log(u1)) * std::cos(6.283185307179586F * u2);
}

void initializeParameters(const Graph& graph, uint64_t seed, float* parameters) {
    std::fill(parameters, parameters + graph.parameterCount, 0.0F);
    // 편향은 0 으로 남기고 가중치만 채운다. layernorm 의 이득만 1 이다. 어느 파라미터가 무엇인지는
    // 연산 표가 말해 준다 — 텐서만 봐서는 가중치와 편향을 가를 수 없다.
    uint64_t stream = 0;
    for (const Op& op : graph.ops) {
        auto fill = [&](uint32_t tensorIndex, uint32_t fanIn) {
            if (tensorIndex >= graph.tensors.size()) {
                return;
            }
            const Tensor& tensor = graph.tensors[tensorIndex];
            if (tensor.arena != Arena::PARAMETER) {
                return;
            }
            // He 정규. ReLU 를 지나며 분산이 절반으로 줄어드는 것을 미리 갚아 둔다.
            float deviation = std::sqrt(2.0F / static_cast<float>(std::max(fanIn, 1U)));
            for (uint32_t i = 0; i < tensor.count(); ++i) {
                parameters[tensor.offset + i] = deviation * neuralGaussian(seed + stream, i);
            }
            ++stream;
        };
        switch (op.kind) {
        case OpKind::CONV2D: {
            const Tensor& weight = graph.tensors[op.inputs[1]];
            fill(op.inputs[1], weight.dims[1] * weight.dims[2] * weight.dims[3]);
            break;
        }
        case OpKind::LINEAR:
            fill(op.inputs[1], graph.tensors[op.inputs[1]].dims[1]);
            break;
        case OpKind::LAYERNORM: {
            const Tensor& gain = graph.tensors[op.inputs[1]];
            if (gain.arena == Arena::PARAMETER) {
                std::fill(parameters + gain.offset, parameters + gain.offset + gain.count(), 1.0F);
            }
            break;
        }
        default:
            break;
        }
    }
}

void adamStep(const AdamSettings& settings,
              uint32_t step,
              size_t count,
              const float* gradients,
              float* moments,
              float* parameters) {
    // step 은 1부터 센다. 0 이면 아래 편향 보정이 0 이 되어 어차피 걸리지만, 뜻을 여기서 밝혀 둔다.
    if (step == 0) {
        return;
    }
    // 편향 보정. m 과 v 가 0 에서 시작하므로 초반 몇 걸음은 실제보다 작게 잡힌다. 그만큼 되돌린다.
    float firstCorrection = 1.0F - std::pow(settings.beta1, static_cast<float>(step));
    float secondCorrection = 1.0F - std::pow(settings.beta2, static_cast<float>(step));
    if (firstCorrection <= 0.0F || secondCorrection <= 0.0F) {
        return;
    }
    float* first = moments;
    float* second = moments + count;
    for (size_t i = 0; i < count; ++i) {
        float gradient = gradients[i];
        first[i] = settings.beta1 * first[i] + (1.0F - settings.beta1) * gradient;
        second[i] = settings.beta2 * second[i] + (1.0F - settings.beta2) * gradient * gradient;
        float correctedFirst = first[i] / firstCorrection;
        float correctedSecond = second[i] / secondCorrection;
        parameters[i] -= settings.learningRate * correctedFirst / (std::sqrt(correctedSecond) + settings.epsilon);
    }
}

void polyakStep(float tau, size_t count, const float* online, float* target) {
    tau = std::clamp(tau, 0.0F, 1.0F);
    if (tau == 0.0F) {
        return;
    }
    if (tau == 1.0F) {
        // 대입으로 처리한다. a + (b - a) 는 두 값의 크기 차가 크면 b 가 되지 않는다.
        std::copy(online, online + count, target);
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        target[i] += tau * (online[i] - target[i]);
    }
}

uint64_t graphHash(const Graph& graph) {
    // 값 하나를 섞어 넣는다. splitmix64 를 되풀이해 순서까지 결과에 남긴다.
    uint64_t state = 0x9E3779B97F4A7C15ULL;
    auto fold = [&state](uint64_t value) { state = mix(state ^ mix(value)); };
    for (const Tensor& tensor : graph.tensors) {
        fold(static_cast<uint64_t>(tensor.arena));
        fold(tensor.offset);
        for (uint32_t axis = 0; axis < 4; ++axis) {
            fold(tensor.dims[axis]);
        }
        fold(tensor.flags);
    }
    for (const Op& op : graph.ops) {
        fold(static_cast<uint64_t>(op.kind));
        for (uint32_t i = 0; i < 3; ++i) {
            fold(op.inputs[i]);
        }
        fold(op.output);
        for (uint32_t i = 0; i < 4; ++i) {
            fold(static_cast<uint64_t>(static_cast<int64_t>(op.iparams[i])));
        }
        for (uint32_t i = 0; i < 2; ++i) {
            // 실수는 비트 그대로 섞는다. 값이 조금만 달라도 다른 그래프다.
            uint32_t bits = 0;
            std::memcpy(&bits, &op.fparams[i], sizeof(bits));
            fold(bits);
        }
    }
    fold(graph.parameterCount);
    fold(graph.activationCount);
    return state;
}

ParameterLayout parameterLayout(const Graph& graph) {
    ParameterLayout layout;
    layout.count = graph.parameterCount;
    for (uint32_t tensor : graph.parameterTensors()) {
        const Tensor& item = graph.tensors[tensor];
        for (uint32_t axis = 0; axis < 4; ++axis) {
            layout.shapes.push_back(item.dims[axis]);
        }
    }
    return layout;
}

bool saveParameters(const Graph& graph, const float* parameters, const std::string& path) {
    ParameterLayout layout = parameterLayout(graph);
    for (size_t i = 0; i < layout.count; ++i) {
        // JSON 은 NaN·무한을 null 로 찍고, 그 파일은 되읽을 때 타입이 어긋나 100% 실패한다. 저장이
        // «성공» 한 척하느니 여기서 걸러 학습이 발산한 것을 알린다.
        if (!std::isfinite(parameters[i])) {
            return false;
        }
    }
    json document;
    document["count"] = layout.count;
    document["shapes"] = layout.shapes;
    document["graph"] = graphHash(graph);
    document["weights"] = std::vector<float>(parameters, parameters + layout.count);
    // 처음 저장할 때 폴더가 없을 수 있다. ofstream 은 만들어 주지 않는다.
    std::error_code error;
    std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, error);
    }
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    file << document.dump(1, ' ') << "\n";
    return static_cast<bool>(file);
}

bool loadParameters(const Graph& graph, float* parameters, const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    json document = json::parse(file, nullptr, false);
    // 구문이 깨진 것은 is_discarded 로 걸리지만, 구문은 맞고 «키의 타입이 다른» 것은 value() 가 예외를
    // 던진다. 손으로 주는 경로라 그대로 두면 크래시가 된다.
    if (document.is_discarded() || !document.is_object()) {
        return false;
    }
    ParameterLayout expected = parameterLayout(graph);
    std::vector<float> weights;
    try {
        ParameterLayout stored;
        stored.count = document.value("count", size_t{0});
        stored.shapes = document.value("shapes", std::vector<uint32_t>{});
        if (!(stored == expected) || document.value("graph", uint64_t{0}) != graphHash(graph)) {
            return false;
        }
        weights = document.value("weights", std::vector<float>{});
    } catch (const json::exception&) {
        return false;
    }
    if (weights.size() != expected.count) {
        return false;
    }
    std::copy(weights.begin(), weights.end(), parameters);
    return true;
}

} // namespace gfx
