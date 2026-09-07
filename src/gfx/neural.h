#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "gfx/neural_math.h"
#include "gfx/resources.h"

namespace gfx {

struct Context;

// 1차원 원소별·선형 커널의 작업 그룹 크기. **셰이더에는 같은 수가 없다** — 특수화 상수로 실어 보내므로
// 여기가 유일한 출처다. 두 벌로 두면 «셰이더 쪽이 더 크다» 는 어긋남이 과다 디스패치로 가려진다.
inline constexpr uint32_t NEURAL_GROUP_SIZE = 128;

// shaders/neural_common.glsl 의 NeuralPushConstants 와 배치가 같아야 한다(scalar).
struct NeuralPushConstants {
    VkDeviceAddress tensors = 0;
    VkDeviceAddress ops = 0;
    VkDeviceAddress parameters = 0;
    VkDeviceAddress activations = 0;
    VkDeviceAddress parameterGradients = 0;
    VkDeviceAddress activationGradients = 0;
    VkDeviceAddress moments = 0;
    uint32_t op = 0;
    uint32_t flags = 0;
    // 아래는 최적화기 커널만 쓴다.
    uint32_t rangeBegin = 0;
    uint32_t rangeCount = 0;
    uint32_t targetBegin = 0;
    float learningRate = 0.0F;
    float beta1 = 0.0F;
    float beta2 = 0.0F;
    float epsilon = 0.0F;
    // 편향 보정. 호스트가 계산해 넘긴다 — GLSL 의 pow 는 std::pow 와 근사가 다르다.
    float firstCorrection = 1.0F;
    float secondCorrection = 1.0F;
    float tau = 0.0F;
    // layernorm 이 분산에 더하는 값. 상수를 GLSL 에 두 벌로 두지 않으려고 실어 보낸다.
    float layerNormEpsilon = LAYERNORM_EPSILON;
};
// 실제 내용은 108 바이트지만 주소 정렬(8) 때문에 뒤가 채워져 112 다. 뒤쪽 채움은 GLSL 이 읽지
// 않으므로 오프셋은 그대로 맞는다.
static_assert(sizeof(NeuralPushConstants) == 112, "신경망 푸시 상수 배치가 셰이더와 어긋난다");
static_assert(sizeof(NeuralPushConstants) <= 128, "푸시 상수는 128 바이트를 넘을 수 없다");

// 역전파를 도는 중. shaders/neural_common.glsl 의 NEURAL_FLAG_BACKWARD 와 같아야 한다.
inline constexpr uint32_t NEURAL_FLAG_BACKWARD = 1U << 0;

// 표를 그대로 memcpy 해 올리므로 배치가 GLSL 짝과 같아야 한다. spirv-dis 로 확인한 오프셋은
// Tensor{0, 4, 8, 24}, Op{0, 4, 16, 20, 36, 44} 다.
static_assert(sizeof(Tensor) == 28, "텐서 배치가 shaders/neural_common.glsl 과 어긋난다");
static_assert(sizeof(Op) == 48, "연산 배치가 shaders/neural_common.glsl 과 어긋난다");

// 연산 표를 컴퓨트로 도는 실행기. **CPU 기준(neural_math)과 문자 그대로 같은 표**를 읽는다 — 텐서가
// 주소 대신 arena·offset 을 들고 다니는 것이 그 때문이다. 그래서 검증이 «같은 표를 두 엔진에 먹여
// 견준다» 로 떨어진다(--neural-selfcheck).
//
// 아직 커널이 있는 연산만 GPU 로 돈다. supported() 가 그것을 말해 주고, 없는 연산은 record 가 건너뛴다 —
// 조용히 건너뛰면 «학습이 안 되는데 이유를 모르는» 자리가 되므로 부르는 쪽이 unsupportedOps() 로 세어
// 알린다.
class NeuralExecutor {
public:
    explicit NeuralExecutor(Context& context);
    ~NeuralExecutor();
    NeuralExecutor(const NeuralExecutor&) = delete;
    NeuralExecutor& operator=(const NeuralExecutor&) = delete;

    // 컴퓨트 파이프라인을 다 만들었는지.
    bool available() const { return ready; }
    // 협력 행렬(텐서 코어) 가속을 쓸 수 있는지. 장치가 모양을 광고하고 그 변종 파이프라인이 만들어진
    // 경우에만 참이다.
    bool cooperativeAvailable() const { return linearCoopPipeline != VK_NULL_HANDLE; }
    // 켜면 선형 층 **순전파**가 협력 행렬 변종으로 돈다. 역전파와 나머지 연산은 그대로 FMA 경로다.
    // 기본은 꺼짐이다 — CPU 기준과 비트로 같다는 계약은 FMA 경로만 진다.
    //
    // ponytail: 지금 이것을 켜는 곳은 자기 검사뿐이다. 학습 경로가 켜는 것은 12단계의 플러그인이고,
    // 그때 «가속을 켜면 같은 곳으로 학습이 가는가» 를 점수로 확인한다.
    //
    // 파이프라인을 만들 때 서브그룹 크기를 못 박는 것과 작업 그룹 크기를 거기에 맞추는 것도 자기 검사가
    // 지켜 주지 못한다(돌연변이로 확인). 이 기기는 서브그룹이 늘 32 이고, 작업 그룹을 128 로 두면
    // 서브그룹 넷이 같은 답을 네 번 쓸 뿐이라 결과가 같다. 규격을 읽어서 지키는 자리다.
    void setCooperative(bool enable) { cooperative = enable && cooperativeAvailable(); }
    bool cooperativeEnabled() const { return cooperative; }

    // 그래프 표를 올리고 버퍼를 잡는다. 다시 부르면 다시 잡는다. validateForward 를 지나지 않는 표는
    // 거짓이다.
    bool build(const Graph& graph);

    // 아직 GPU 커널이 없는 연산 종류.
    static bool supported(OpKind kind);
    // 이 그래프에서 GPU 로 못 도는 연산 수. 0 이 아니면 결과를 믿으면 안 된다.
    uint32_t unsupportedOps() const { return missing; }

    // 호스트가 값을 놓는 자리. build 뒤에 유효하고, recordUpload 가 장치 버퍼로 옮긴다.
    float* parameterStaging() { return stagingFloats; }
    float* activationStaging() { return stagingFloats + parameterCount; }
    // 경사 씨앗을 놓는 자리. 손실 커널이 없는 동안 자기 검사가 여기에 심는다.
    float* activationGradientStaging() { return stagingFloats + parameterCount + activationCount; }

    // recordDownload 뒤에 유효하다.
    const float* activationResult() const { return readbackFloats; }
    const float* parameterGradientResult() const { return readbackFloats + activationCount; }
    const float* activationGradientResult() const { return readbackFloats + activationCount + parameterCount; }
    // 학습이 지나간 뒤의 가중치. 열 걸음 등가 검사가 이것을 견준다.
    const float* parameterResult() const { return readbackFloats + activationCount * 2 + parameterCount; }

    // Adam 의 모멘트 버퍼를 잡는다. 최적화기마다 하나씩(에이전트는 크리틱·액터 둘)이고, 크기는
    // adamMomentCount(구간 길이) 다. build 뒤에 부른다(build 가 이전 것을 버린다).
    bool reserveMoments(uint32_t slot, size_t floats);
    // 모멘트를 채운다. 기본은 0 이고, **첫 걸음 전에 반드시 부른다** — 갓 잡은 장치 메모리의 내용은
    // 정해져 있지 않은데 Adam 이 첫 걸음에서 그것을 읽는다.
    //
    // 값을 받는 이유는 자기 검사가 «지우기가 실제로 일하는가» 를 볼 수 있어야 하기 때문이다. 드라이버가
    // 새 페이지를 0 으로 주면 지우기를 빼도 티가 나지 않으므로, 검사가 일부러 더럽힌 뒤 지운다.
    void recordClearMoments(VkCommandBuffer commandBuffer, uint32_t slot, float value = 0.0F);
    // 파라미터 배열의 [begin, begin+count) 를 Adam 한 걸음 밟는다. step 은 1부터 센다.
    void recordAdam(VkCommandBuffer commandBuffer,
                    uint32_t momentSlot,
                    uint32_t begin,
                    uint32_t count,
                    const AdamSettings& settings,
                    uint32_t step);
    // [onlineBegin, +count) 를 [targetBegin, +count) 로 tau 만큼 끌어당긴다.
    void
    recordPolyak(VkCommandBuffer commandBuffer, uint32_t onlineBegin, uint32_t targetBegin, uint32_t count, float tau);

    // 아래는 모두 명령 버퍼 하나에 이어 기록한다. 사이의 배리어는 각자 안에서 건다.
    void recordUpload(VkCommandBuffer commandBuffer);
    // 경사 씨앗을 활성 경사 버퍼로 옮긴다. **지운 뒤에** 부른다. 손실 커널이 생기면(8단계) 이 대신
    // 손실의 역전파가 씨앗을 심으므로 따로 떼어 두었다.
    void recordUploadGradientSeed(VkCommandBuffer commandBuffer);
    // 경사 배열 둘을 0 으로 지운다. CPU 의 backward 가 맨 앞에서 하는 일이다.
    void recordClearGradients(VkCommandBuffer commandBuffer);
    void recordForward(VkCommandBuffer commandBuffer);
    // 표를 거꾸로 훑는다. 씨앗은 부르는 쪽이 미리 올려 둔다(neural_math 의 backwardFrom 과 같은 규약).
    void recordBackward(VkCommandBuffer commandBuffer);
    void recordDownload(VkCommandBuffer commandBuffer);
    // 되읽기 버퍼를 무효화한다. 제출이 끝난 뒤, 결과를 읽기 전에 한 번 부른다.
    void invalidateReadback();

private:
    void createPipelines();
    void uploadBarrier(VkCommandBuffer commandBuffer);
    // 버퍼 주소만 채운 푸시 상수. 연산 커널과 최적화기 커널이 함께 쓴다.
    NeuralPushConstants basePush() const;
    void dispatch(VkCommandBuffer commandBuffer, uint32_t opIndex, uint32_t flags);
    // 파이프라인 하나를 threads 개 스레드로 돈다. 푸시 상수는 연산 번호와 방향만 다르다.
    void dispatchKernel(
        VkCommandBuffer commandBuffer, VkPipeline pipeline, uint32_t opIndex, uint32_t flags, uint32_t threads);
    void dispatchGroups(
        VkCommandBuffer commandBuffer, VkPipeline pipeline, uint32_t opIndex, uint32_t flags, uint32_t groups);
    void barrier(VkCommandBuffer commandBuffer);
    void clearBarrier(VkCommandBuffer commandBuffer);

    Context& context;
    bool ready = false;
    bool cooperative = false;
    uint32_t missing = 0;

    Graph graph;
    size_t parameterCount = 0;
    size_t activationCount = 0;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline elementwisePipeline = VK_NULL_HANDLE;
    // 선형 층은 방향마다 커널이 다르다. 순전파는 출력 원소마다, 역전파는 dx·dw·db 셋이 각자의 출력
    // 원소마다 스레드 하나씩이라 디스패치 크기가 전부 다르기 때문이다.
    VkPipeline linearPipeline = VK_NULL_HANDLE;
    // 협력 행렬 변종. 만들지 못해도 실행기는 돈다.
    VkPipeline linearCoopPipeline = VK_NULL_HANDLE;
    VkPipeline linearDxPipeline = VK_NULL_HANDLE;
    VkPipeline linearDwPipeline = VK_NULL_HANDLE;
    VkPipeline biasGradPipeline = VK_NULL_HANDLE;
    // 합성곱도 방향마다 커널이 다르다. dx 는 «모아 읽기» 라 입력 원소마다, dw 는 필터 탭마다다.
    VkPipeline convPipeline = VK_NULL_HANDLE;
    VkPipeline convDxPipeline = VK_NULL_HANDLE;
    VkPipeline convDwPipeline = VK_NULL_HANDLE;
    VkPipeline convDbPipeline = VK_NULL_HANDLE;
    VkPipeline layerNormPipeline = VK_NULL_HANDLE;
    VkPipeline layerNormDxPipeline = VK_NULL_HANDLE;
    VkPipeline layerNormDparamPipeline = VK_NULL_HANDLE;
    VkPipeline reducePipeline = VK_NULL_HANDLE;
    VkPipeline reduceDxPipeline = VK_NULL_HANDLE;
    VkPipeline adamPipeline = VK_NULL_HANDLE;
    VkPipeline polyakPipeline = VK_NULL_HANDLE;

    Buffer tensorBuffer;
    Buffer opBuffer;
    Buffer parameterBuffer;
    Buffer parameterGradientBuffer;
    Buffer activationBuffer;
    Buffer activationGradientBuffer;
    // 호스트가 쓰는 자리: [파라미터 | 활성 | 활성 경사 씨앗].
    Buffer staging;
    // 호스트가 읽는 자리: [활성 | 파라미터 경사 | 활성 경사 | 파라미터].
    Buffer readback;
    // 최적화기 모멘트. 에이전트가 크리틱·액터 둘을 쓴다.
    Buffer moments[2];
    // 슬롯마다 잡아 둔 float 수. recordAdam 이 «구간 둘이 버퍼 안에 들어가는가» 를 확인하는 데 쓴다.
    size_t momentFloats[2] = {0, 0};
    float* stagingFloats = nullptr;
    float* readbackFloats = nullptr;
};

// --neural-selfcheck 의 본체. 창 없는 컴퓨트 장치를 만들어 CPU 기준과 GPU 를 견주고, 연산 종류마다 최대
// 오차를 표로 찍는다. 하나라도 허용치를 넘으면 거짓이다.
bool runNeuralSelfCheck();

} // namespace gfx
