#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
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

// layernorm 이 분산에 더하는 값. 분산이 0 인 행에서 나눗셈이 터지지 않게 한다. **GLSL 커널이 생기면
// 같은 값이어야 한다** — 다르면 순전파가 조용히 갈리고 유한차분으로는 잡을 수 없다(순·역이 같은 값을
// 쓰기 때문이다). 아직 layernorm 커널은 없다(8단계).
inline constexpr float LAYERNORM_EPSILON = 1.0e-5F;

// 텐서 하나. **shaders/neural_common.glsl 의 Tensor 와 배치가 같다** — GPU 도 arena 와 offset 을 그대로
// 읽고, arena 넷의 시작 주소만 푸시 상수로 받는다. 텐서마다 주소를 담지 않는 이유는 그러면 버퍼를 다시
// 잡을 때마다 표를 새로 지어야 하고, 그러면 «두 엔진이 같은 표를 읽는다» 가 거짓이 되기 때문이다.
// 배치가 묶인 자리라 한쪽을 고치면 다른 쪽도 고친다.
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
    // 원소별 곱. 시간차 목표의 γ(1 - 끝) 마스크가 이것이다 — 표본마다 값이 달라 SCALE 로는 안 된다.
    MUL,
    // 특징 축으로 잇는다. 크리틱이 (특징, 행동) 을 함께 받는 자리.
    CONCAT,
    // 출력 = 입력 * fparams[0].
    SCALE,
    // 모든 원소의 평균을 스칼라 하나로 접는다. 액터 손실 -mean(Q) 가 이것이다 — 손실 자리에 손실이
    // 아닌 값(가치)을 놓아야 하므로 MSE·HUBER 로는 대신할 수 없다.
    MEAN,
    // 원소별 최소. 쌍둥이 크리틱의 min(Q1, Q2) 다. 경사는 이긴 쪽만 받는다.
    MIN2,
    // (예측[B][1], 목표[B][1]) -> 손실[1]. 평균 제곱 오차.
    MSE,
    // 같은 자리의 Huber 손실. fparams[0] 이 꺾이는 지점(delta)이고, 그보다 멀어지면 기울기가 상수로
    // 눕는다. 크리틱의 시간차 오차가 이따금 크게 튀어 학습이 흔들릴 때 MSE 대신 쓴다.
    //
    // **눈금이 MSE 의 절반이다.** MSE 는 mean(e^2) 이고 Huber 는 안쪽에서 mean(0.5 e^2) 이라 경사가
    // 절반이다(PyTorch 의 MSELoss / HuberLoss 와 같은 규약). 그대로 바꿔 끼우면 크리틱의 실효 학습률이
    // 조용히 절반이 되므로 학습률을 함께 본다.
    HUBER,
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

// 텐서 하나가 배열 안에서 시작하는 자리. 입력을 채우고 결과를 읽는 데 쓴다. arena 가 맞는 배열을
// 주는 것은 부르는 쪽 몫이다(파라미터 텐서에 활성 배열을 주면 엉뚱한 자리를 가리킨다).
inline float* tensorValues(const Graph& graph, uint32_t tensor, float* array) {
    return array + graph.tensors[tensor].offset;
}

inline const float* tensorValues(const Graph& graph, uint32_t tensor, const float* array) {
    return array + graph.tensors[tensor].offset;
}

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
    uint32_t addMul(uint32_t a, uint32_t b);
    uint32_t addConcat(uint32_t a, uint32_t b);
    uint32_t addScale(uint32_t input, float factor);
    uint32_t addMean(uint32_t input);
    uint32_t addMin2(uint32_t a, uint32_t b);
    uint32_t addMse(uint32_t prediction, uint32_t target);
    uint32_t addHuber(uint32_t prediction, uint32_t target, float delta);

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

    // 지금까지 잡은 파라미터의 float 개수. 구간을 나누며 짓는 쪽(rl_agent)이 경계를 여기서 읽는다.
    size_t parameterCount() const;

private:
    uint32_t allocate(Arena arena, uint32_t n, uint32_t c, uint32_t h, uint32_t w, uint32_t flags);
    uint32_t emit(OpKind kind, uint32_t a, uint32_t b, uint32_t c, uint32_t output);

    Graph& graph;
};

// CONV2D 의 출력 한 변. 커널·보폭·여백에서 정한다.
uint32_t convOutputSize(uint32_t input, uint32_t kernel, uint32_t stride, uint32_t pad);

// 표가 **순전파를 돌릴** 꼴인지. 연산의 입출력 번호가 범위 안이고, 출력 번호가 늘 입력 번호보다
// 커서(빌더가 지키는 불변식) **표를 훑는 순서가 올바른 위상 정렬**이며, 원소별 연산의 두 입력 모양이
// 같은지를 본다.
//
// 손실이 없는 그래프 — 관측에서 행동만 내는 정책 망 — 는 이것으로 충분하다.
bool validateForward(const Graph& graph);

// 그 위에 **역전파의 씨앗을 심을 수 있는지**를 더 본다: 마지막 연산의 출력이 경사를 받는 스칼라(원소
// 하나)여야 한다. backward 를 부를 그래프는 이것을 통과해야 한다.
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

// 같은 역전파지만 **지우지도 심지도 않는다.** 부르는 쪽이 경사 배열을 미리 채워 두고, 표를 거꾸로 훑기만
// 한다. backward 는 «0 으로 지우고 마지막에 1 을 심는» 특수한 경우다.
//
// GPU 실행기가 이 꼴을 쓴다. 손실 커널이 아직 없는 동안에도 씨앗을 호스트가 올려 주면 역전파를 견줄 수
// 있어야 하기 때문이다.
void backwardFrom(const Graph& graph,
                  const float* parameters,
                  const float* activations,
                  float* parameterGradients,
                  float* activationGradients);

// 표본 번호만으로 정해지는 표준 가우시안. physics::gaussianNoise 와 같은 splitmix64 + Box-Muller 다.
//
// ponytail: 같은 수식이 두 벌이 된다. 여기 것과 physics::gaussianNoise 다. 이쪽에 따로 둔 것은 GPU 가
// 잡음을 스스로 만들어야 할 때(무작위 이동 증강, 11단계) GLSL 짝이 생길 자리이기 때문이다. 한쪽을
// 고치면 다른 쪽도 고친다.
float neuralGaussian(uint64_t seed, uint64_t index);

// 파라미터를 초기화한다. 합성곱·선형의 가중치는 팬인에 맞춘 He 정규(ReLU 를 전제), 편향은 0,
// layernorm 의 이득은 1 이다. 어느 텐서가 무엇인지는 연산 표를 훑어 정한다.
void initializeParameters(const Graph& graph, uint64_t seed, float* parameters);

// ---- 최적화. 전부 **평탄한 파라미터 배열 하나**를 훑는다. 텐서 모양을 보지 않으므로 GPU 에서는
// 디스패치 하나로 끝난다.

struct AdamSettings {
    float learningRate = 1.0e-4F;
    float beta1 = 0.9F;
    float beta2 = 0.999F;
    // sqrt(v) 에 **더하는** 값이다(sqrt(v + eps) 가 아니다). PyTorch 와 같은 자리다.
    float epsilon = 1.0e-8F;
};

// Adam 모멘트가 차지하는 float 개수. 앞 절반이 1차(m), 뒤 절반이 2차(v)다. 두 벌을 한 배열에 두어
// GPU 에서 버퍼 하나로 넘긴다.
inline size_t adamMomentCount(size_t parameterCount) {
    return parameterCount * 2;
}

// 한 걸음. step 은 **1부터** 센다(편향 보정이 1 - beta^step 이라 0 이면 0 으로 나눈다).
//
// 첫 걸음의 크기가 경사의 크기와 무관하게 learningRate 근처가 되는 것이 Adam 의 성질이다 —
// m 과 v 가 0 에서 시작해 편향 보정을 지나면 m/sqrt(v) 가 부호만 남기기 때문이다. 다만 |경사| 가
// epsilon 언저리까지 내려가면 그 성질이 깨진다(분모의 epsilon 이 이긴다).
//
// ponytail: NaN·무한 경사를 한 번 먹으면 모멘트가 오염되어 회복하지 못한다. 경사를 자르거나 그런 걸음을
// 건너뛰는 것은 부르는 쪽 몫이다(PyTorch 도 같다).
void adamStep(const AdamSettings& settings,
              uint32_t step,
              size_t count,
              const float* gradients,
              float* moments,
              float* parameters);

// 타깃망을 온라인 쪽으로 조금 끌어당긴다. target = tau * online + (1 - tau) * target.
// tau 가 1 이면 복사, 0 이면 그대로이고, [0, 1] 밖은 잘라 낸다 — 음수면 타깃이 발산하고 1 보다 크면
// 진동한다. 조용히 망가지느니 잘라 두는 편이 낫다.
void polyakStep(float tau, size_t count, const float* online, float* target);

// ---- 직렬화.

// 그래프의 «모양» 을 접은 값. 연산 종류·입출력 번호·정수와 실수 인자, 텐서의 arena·모양·플래그를 모두
// 섞는다.
//
// 파라미터 모양만 견주는 것으로는 모자란다 — 같은 (4,3,3,3) 합성곱을 보폭 1 로 지은 그래프와 보폭 2 로
// 지은 그래프는 파라미터 배치가 **같고**, 쌍둥이 크리틱 둘의 가중치를 지은 순서만 바꾼 그래프도 같다.
// 그런 파일이 조용히 읽히면 학습이 안 되는데 이유를 알 수 없다.
uint64_t graphHash(const Graph& graph);

// 파라미터 배치의 요약. **모양만** 담는다 — «같은 그래프인가» 는 graphHash 가 판정한다. 저장 파일은
// 둘을 함께 담는다.
struct ParameterLayout {
    size_t count = 0;
    // 파라미터 텐서마다 dims 넷을 이어 붙인 것. 순서는 Graph::parameterTensors() 와 같다.
    std::vector<uint32_t> shapes;

    bool operator==(const ParameterLayout&) const = default;
};

ParameterLayout parameterLayout(const Graph& graph);

// 가중치를 JSON 으로 저장하고 읽는다. 배치나 그래프 해시가 맞지 않거나 파일이 깨졌으면 거짓을 돌려주고
// parameters 를 건드리지 않는다(physics::savePolicy 와 같은 규칙).
//
// 저장은 값에 NaN·무한이 있으면 **쓰지 않고 거짓을 돌려준다.** JSON 은 그것을 null 로 찍는데 되읽을 때
// 타입이 어긋나 실패하므로, 그대로 두면 «저장은 성공하고 적재는 100% 실패하는» 체크포인트가 남는다.
// 학습이 발산한 순간을 저장 시점에 알아채는 편이 낫다.
//
// ponytail: 텍스트 JSON 이라 파라미터 200만 개면 43 MB 에 왕복 1 초가 넘는다(float 를 double 로 넓혀
// 17자리로 찍기 때문이다). 정책망 규모에는 넉넉하지만 큰 망에는 이진 형식이 필요하다.
// ponytail: 임시 파일에 쓰고 옮기는 것이 아니라 바로 덮어쓴다. 쓰다 죽으면 옛 체크포인트도 함께 잃는다.
bool saveParameters(const Graph& graph, const float* parameters, const std::string& path);
bool loadParameters(const Graph& graph, float* parameters, const std::string& path);

} // namespace gfx
