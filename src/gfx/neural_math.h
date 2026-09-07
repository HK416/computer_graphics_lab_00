#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gfx {

// 신경망의 순수 계산. Vulkan 을 끌어오지 않아 테스트가 그대로 링크한다(shadow_math·upscaler_math 와
// 같은 자리다). GPU 실행기(gfx/neural.h)는 **여기서 지은 것과 같은 표**를 읽어 컴퓨트로 돈다.
//
// 왜 일반 테이프가 아니라 «한 번 짓고 되도는 고정 연산 표» 인가:
//
// 다중 뷰 강화 학습은 역전파 사슬이 하나가 아니다 — 공유 인코더 × 뷰 n개, 병합, 액터, 쌍둥이 크리틱,
// 타깃 크리틱, 단일 뷰 증강 가지까지 열 갈래 넘게 나고 **같은 가중치에 여러 갈래의 경사를 누적**해야
// 한다. 손으로 이으면 버그가 정확히 거기 산다. 표로 두면 역전파가 «배열을 거꾸로 훑는 for 문 하나» 가
// 되고, 스텝마다 노드를 할당하지 않으며(프레임당 힙 할당 없음), 무엇보다 **CPU 기준과 GPU 실행기가
// 문자 그대로 같은 표를 읽어** 검증이 «같은 표를 두 엔진에 먹여 견준다» 로 떨어진다.

// 텐서가 사는 곳. 경사는 값과 **같은 배치의 별도 배열**에 있으므로 오프셋을 따로 들지 않는다.
enum class Arena : uint32_t {
    // 학습이 갱신하는 가중치·편향. 한 줄로 이어 두어 최적화가 디스패치 하나로 돈다.
    PARAMETER = 0,
    // 순전파가 남기는 중간값. 배치 크기가 고정이라 오프셋을 그래프를 지을 때 못 박는다.
    ACTIVATION = 1,
};

// 이 텐서가 경사를 받는가. 끄면 그 자리에서 역전파가 멈춘다(stop-gradient) — 액터 손실이 인코더를
// 갱신하지 않는 것과 타깃망이 경사를 받지 않는 것이 이 비트 하나로 끝난다.
inline constexpr uint32_t TENSOR_GRAD = 1U << 0;

// layernorm 이 분산에 더하는 값. 분산이 0 인 행에서 나눗셈이 터지지 않게 한다. **GLSL 짝과 같은
// 값이어야 한다** — 다르면 순전파가 조용히 갈리고 유한차분으로는 잡을 수 없다(순·역이 같은 값을
// 쓰기 때문이다).
inline constexpr float LAYERNORM_EPSILON = 1.0e-5F;

// 텐서 하나. GPU 쪽(shaders/neural_common.glsl 의 Tensor)은 arena·offset 대신 buffer device address
// 두 개를 담지만, 나머지 필드와 뜻은 같다. 배치가 묶인 자리라 한쪽을 고치면 다른 쪽도 고친다.
struct Tensor {
    Arena arena = Arena::ACTIVATION;
    // arena 안의 **float 첨자**(바이트가 아니다).
    uint32_t offset = 0;
    // N, C, H, W. 쓰지 않는 축은 1 이다.
    uint32_t dims[4] = {1, 1, 1, 1};
    uint32_t flags = 0;

    uint32_t count() const { return dims[0] * dims[1] * dims[2] * dims[3]; }
    bool receivesGradient() const { return (flags & TENSOR_GRAD) != 0; }
};

// 연산 종류. 순전파와 역전파가 짝을 이룬다.
enum class OpKind : uint32_t {
    // 밖에서 값을 채우는 자리(관측·행동·목표). 순전파가 아무 일도 하지 않는다.
    INPUT = 0,
    // (입력, 가중치[Cout][Cin][kh][kw], 편향[Cout]) -> 출력. iparams: stride, pad.
    CONV2D,
    // (입력[B][K], 가중치[M][K], 편향[M]) -> 출력[B][M].
    LINEAR,
    RELU,
    TANH,
    // (입력[B][F], 이득[F], 편향[F]) -> 출력[B][F]. 행마다 정규화한다.
    LAYERNORM,
    // 원소별 합. 다중 뷰 병합 M = sum(V_i) 가 이것이라 병합의 역전파는 경사 복사로 끝난다.
    ADD,
    // 특징 축으로 잇는다. 크리틱이 (특징, 행동) 을 함께 받는 자리.
    CONCAT,
    // 출력 = 입력 * fparams[0].
    SCALE,
    // 원소별 최소. 쌍둥이 크리틱의 min(Q1, Q2) 다. 경사는 이긴 쪽만 받는다.
    MIN2,
    // (예측[B][1], 목표[B][1]) -> 손실[1]. 평균 제곱 오차.
    MSE,
};

// 쓰지 않는 입력 자리.
inline constexpr uint32_t NO_TENSOR = 0xFFFFFFFFU;

// 연산 하나. GPU 쪽(shaders/neural_common.glsl 의 Op)과 배치가 같아야 한다.
struct Op {
    OpKind kind = OpKind::INPUT;
    uint32_t inputs[3] = {NO_TENSOR, NO_TENSOR, NO_TENSOR};
    uint32_t output = NO_TENSOR;
    // CONV2D 는 [0] stride, [1] pad.
    int32_t iparams[4] = {0, 0, 0, 0};
    // SCALE 은 [0] 배율.
    float fparams[2] = {0.0F, 0.0F};
    uint32_t pad0 = 0;
};

// 지어 둔 그래프. 텐서 표와 연산 표, 그리고 두 arena 의 크기(float 개수)다.
struct Graph {
    std::vector<Tensor> tensors;
    std::vector<Op> ops;
    size_t parameterCount = 0;
    size_t activationCount = 0;

    // 파라미터 텐서만 골라 낸다. 최적화와 저장이 이 순서를 쓴다.
    std::vector<uint32_t> parameterTensors() const;
};

// 그래프를 짓는다. 텐서 자리를 arena 에 이어 붙이고 연산을 표에 쌓는다. 모양이 맞지 않으면
// core::fatal 이 아니라 **NO_TENSOR 를 돌려준다** — 테스트가 잘못된 모양을 확인할 수 있어야 한다.
class GraphBuilder {
public:
    explicit GraphBuilder(Graph& graph) : graph(graph) {}

    // 밖에서 채우는 입력. 경사를 받지 않는다.
    uint32_t addInput(uint32_t n, uint32_t c, uint32_t h, uint32_t w);
    // 학습하는 가중치. 초기화는 initializeParameters 가 한다.
    uint32_t addParameter(uint32_t n, uint32_t c, uint32_t h, uint32_t w);

    // 아래는 출력 텐서를 새로 잡아 그 번호를 돌려준다.
    uint32_t addConv2d(uint32_t input, uint32_t weight, uint32_t bias, uint32_t stride, uint32_t pad);
    uint32_t addLinear(uint32_t input, uint32_t weight, uint32_t bias);
    uint32_t addRelu(uint32_t input);
    uint32_t addTanh(uint32_t input);
    uint32_t addLayerNorm(uint32_t input, uint32_t gain, uint32_t bias);
    uint32_t addAdd(uint32_t a, uint32_t b);
    uint32_t addConcat(uint32_t a, uint32_t b);
    uint32_t addScale(uint32_t input, float factor);
    uint32_t addMin2(uint32_t a, uint32_t b);
    uint32_t addMse(uint32_t prediction, uint32_t target);

    // 같은 저장소를 다른 모양으로 보는 **뷰**를 만든다. 연산을 내지 않고 값도 옮기지 않는다 —
    // 새 텐서가 같은 오프셋을 가리킬 뿐이라 경사도 같은 자리에 쌓인다(항등 함수의 역전파).
    // 합성곱 출력(N, C, H, W)을 선형이 받는 (N, C·H·W) 로 펴는 데 쓴다. 원소 수가 같아야 한다.
    // **활성 텐서만 된다** — 파라미터를 뷰로 만들면 parameterTensors() 가 같은 저장소를 두 번 세어
    // 최적화가 한 자리를 두 번 갱신하고 저장 크기도 어긋난다.
    uint32_t addReshape(uint32_t tensor, uint32_t n, uint32_t c, uint32_t h, uint32_t w);

    // 이 텐서에서 역전파를 끊는다(stop-gradient). 이미 이어 붙인 뒤에도 부를 수 있다.
    void detach(uint32_t tensor);

    // (배치, 특징) 텐서의 특징 수. 다음 층의 가중치 모양을 잡을 때 쓴다.
    uint32_t featureCount(uint32_t tensor) const;

private:
    uint32_t allocate(Arena arena, uint32_t n, uint32_t c, uint32_t h, uint32_t w, uint32_t flags);
    uint32_t emit(OpKind kind, uint32_t a, uint32_t b, uint32_t c, uint32_t output);

    Graph& graph;
};

// CONV2D 의 출력 한 변. 커널·보폭·여백에서 정한다.
uint32_t convOutputSize(uint32_t input, uint32_t kernel, uint32_t stride, uint32_t pad);

// 그래프가 순·역전파를 돌릴 꼴인지. 연산의 입출력 번호가 범위 안이고, 출력 번호가 늘 입력 번호보다
// 커서(빌더가 지키는 불변식) **표를 거꾸로 훑는 것이 올바른 위상 정렬**이며, 마지막 연산의 출력이
// 경사를 받는 스칼라(원소 하나)여야 한다.
//
// 빌더로 지은 그래프는 마지막 조건만 부르는 쪽 몫이다. 손으로 짓거나 파일에서 읽은 표는 여기서 건다.
bool validate(const Graph& graph);

// 순전파. activations 는 graph.activationCount 개, parameters 는 graph.parameterCount 개다.
// 입력 텐서(OpKind::INPUT)는 부르는 쪽이 미리 채워 둔다. **validate 를 통과한 그래프를 전제한다.**
void forward(const Graph& graph, const float* parameters, float* activations);

// 같은 순전파를 double 로 돈다. 유한차분 경사 검사가 쓴다 — float 로 두 손실의 차를 내면 반올림에
// 묻혀, 경사가 작은 자리에서 상대 오차가 1e-3 대로 떠 수식이 맞는지 틀린지를 가릴 수 없다.
void forward(const Graph& graph, const double* parameters, double* activations);

// 역전파. 경사 배열 둘을 **0 으로 지우고** 마지막 연산의 출력에 1 을 심은 뒤 표를 거꾸로 훑는다.
// 그래서 마지막 연산은 스칼라 손실이어야 한다 — 아니면 경사를 지우기만 하고 **거짓을 돌려준다**.
// 조용히 0 을 돌려주면 «학습이 안 되는데 이유를 모르는» 자리가 된다.
bool backward(const Graph& graph,
              const float* parameters,
              const float* activations,
              float* parameterGradients,
              float* activationGradients);

// 표본 번호만으로 정해지는 표준 가우시안. physics::gaussianNoise 와 같은 splitmix64 + Box-Muller 다.
//
// ponytail: 같은 수식이 두 벌이 된다. 여기 것은 GLSL 짝(neural_common.glsl 의 neuralGaussian)이 있어야
// 해서 신경망 쪽에 두었고, physics 쪽은 진화 전략이 쓴다. 한쪽을 고치면 다른 쪽도 고친다.
float neuralGaussian(uint64_t seed, uint64_t index);

// 파라미터를 초기화한다. 합성곱·선형의 가중치는 팬인에 맞춘 He 정규(ReLU 를 전제), 편향은 0,
// layernorm 의 이득은 1 이다. 어느 텐서가 무엇인지는 연산 표를 훑어 정한다.
void initializeParameters(const Graph& graph, uint64_t seed, float* parameters);

} // namespace gfx
