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
    uint32_t op = 0;
    uint32_t flags = 0;
};
static_assert(sizeof(NeuralPushConstants) == 56, "신경망 푸시 상수 배치가 셰이더와 어긋난다");

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
    void dispatch(VkCommandBuffer commandBuffer, uint32_t opIndex, uint32_t flags);
    // 파이프라인 하나를 threads 개 스레드로 돈다. 푸시 상수는 연산 번호와 방향만 다르다.
    void dispatchKernel(
        VkCommandBuffer commandBuffer, VkPipeline pipeline, uint32_t opIndex, uint32_t flags, uint32_t threads);
    void barrier(VkCommandBuffer commandBuffer);

    Context& context;
    bool ready = false;
    uint32_t missing = 0;

    Graph graph;
    size_t parameterCount = 0;
    size_t activationCount = 0;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline elementwisePipeline = VK_NULL_HANDLE;
    // 선형 층은 방향마다 커널이 다르다. 순전파는 출력 원소마다, 역전파는 dx·dw·db 셋이 각자의 출력
    // 원소마다 스레드 하나씩이라 디스패치 크기가 전부 다르기 때문이다.
    VkPipeline linearPipeline = VK_NULL_HANDLE;
    VkPipeline linearDxPipeline = VK_NULL_HANDLE;
    VkPipeline linearDwPipeline = VK_NULL_HANDLE;
    VkPipeline biasGradPipeline = VK_NULL_HANDLE;
    // 합성곱도 방향마다 커널이 다르다. dx 는 «모아 읽기» 라 입력 원소마다, dw 는 필터 탭마다다.
    VkPipeline convPipeline = VK_NULL_HANDLE;
    VkPipeline convDxPipeline = VK_NULL_HANDLE;
    VkPipeline convDwPipeline = VK_NULL_HANDLE;
    VkPipeline convDbPipeline = VK_NULL_HANDLE;

    Buffer tensorBuffer;
    Buffer opBuffer;
    Buffer parameterBuffer;
    Buffer parameterGradientBuffer;
    Buffer activationBuffer;
    Buffer activationGradientBuffer;
    // 호스트가 쓰는 자리: [파라미터 | 활성 | 활성 경사 씨앗].
    Buffer staging;
    // 호스트가 읽는 자리: [활성 | 파라미터 경사 | 활성 경사].
    Buffer readback;
    float* stagingFloats = nullptr;
    float* readbackFloats = nullptr;
};

// --neural-selfcheck 의 본체. 창 없는 컴퓨트 장치를 만들어 CPU 기준과 GPU 를 견주고, 연산 종류마다 최대
// 오차를 표로 찍는다. 하나라도 허용치를 넘으면 거짓이다.
bool runNeuralSelfCheck();

} // namespace gfx
