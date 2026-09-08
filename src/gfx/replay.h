#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "gfx/neural_math.h"
#include "gfx/observation.h"
#include "gfx/resources.h"

namespace gfx {

struct Context;

// **한 번에 하나씩 제출하는 것을 전제한다.** 링의 메타·행동·표본 버퍼가 한 벌뿐이라, 프레임을 겹쳐
// 돌리면 도는 표집이 읽는 칸을 다음 프레임의 beginSlot 이 호스트에서 덮는다. 12단계에서 프레임 루프에
// 붙일 때 rigid_body_gpu 처럼 프레임 슬롯을 두어야 한다.
//   ponytail: 지금 부르는 곳이 자기 검사뿐이고 그쪽은 제출마다 기다린다.

// 무작위 이동 증강이 덧대는 화소 수. DrQ-v2 가 84 화소 판에 4 를 쓴다.
inline constexpr int32_t REPLAY_SHIFT_PADDING = 4;

// 링은 화소 넷을 uint32 하나에 묶는다. 한 판의 화소 수가 4 의 배수가 아니면 담을 때와 읽을 때의 워드
// 첨자가 칸마다 어긋나 조용히 남의 그림이 나온다. 변이 짝수면 늘 성립하지만 컴파일러가 잡아 주지 않는다.
static_assert(OBSERVATION_SIZE * OBSERVATION_SIZE % 4 == 0, "관측 한 판의 화소 수가 4 의 배수여야 한다");

// shaders/replay_common.glsl 의 ReplaySlot 과 배치가 같아야 한다.
struct GpuReplaySlot {
    uint32_t episode = 0;
    uint32_t stepInEpisode = 0;
    float reward = 0.0F;
    // **다음 관측이 종료 상태면 0 이다.** 칸 i 가 담는 것은 전이 (o_i, a_i, r_i, discount_i, o_{i+1}) 이고,
    // discount_i 가 0 이면 y = r + discount * minQ' 에서 부트스트랩이 끊긴다.
    //
    // 그래서 감가 0 은 에피소드의 **마지막 칸이 아니라 그 앞 칸**에 붙는다. 마지막 칸은 종료 관측을
    // 담을 뿐이고 뒤가 없어 표본이 되지 않는다(replaySampleValid 가 떨어뜨린다). 여기를 한 칸 밀면
    // 종료 상태의 가치를 물고 들어가면서도 검사는 통과하므로, 규약을 글로 못 박아 둔다.
    float discount = 0.0F;
};

// shaders/replay_common.glsl 의 ReplaySample 과 배치가 같아야 한다.
struct GpuReplaySample {
    uint32_t index = 0;
    int32_t shiftX = 0;
    int32_t shiftY = 0;
    int32_t nextShiftX = 0;
    int32_t nextShiftY = 0;
    uint32_t pad0 = 0;
};

// shaders/neural_replay_store.comp 의 동명 블록과 배치가 같아야 한다.
struct ReplayStorePushConstants {
    VkDeviceAddress features = 0;
    VkDeviceAddress frames = 0;
    uint32_t slot = 0;
    uint32_t views = 0;
    uint32_t stack = 0;
    uint32_t size = 0;
};

// shaders/neural_replay_sample.comp 의 동명 블록과 배치가 같아야 한다.
struct ReplaySamplePushConstants {
    VkDeviceAddress frames = 0;
    VkDeviceAddress slots = 0;
    VkDeviceAddress actions = 0;
    VkDeviceAddress samples = 0;
    VkDeviceAddress state = 0;
    VkDeviceAddress nextState = 0;
    VkDeviceAddress batchAction = 0;
    VkDeviceAddress batchReward = 0;
    VkDeviceAddress batchDiscount = 0;
    uint32_t capacity = 0;
    uint32_t count = 0;
    uint32_t cursor = 0;
    uint32_t views = 0;
    uint32_t stack = 0;
    uint32_t size = 0;
    uint32_t actionCount = 0;
    uint32_t batch = 0;
};

// 표집 한 벌이 쓸 자리. 신경망 그래프 안의 텐서 주소를 그대로 받는다 — 리플레이가 자기 버퍼에 담고
// 다시 옮기면 배치 하나마다 관측 두 벌을 헛되이 복사하게 된다.
struct ReplayBatchTargets {
    VkDeviceAddress state = 0;
    VkDeviceAddress nextState = 0;
    VkDeviceAddress action = 0;
    VkDeviceAddress reward = 0;
    VkDeviceAddress discount = 0;
};

// 관측 전이를 담는 장치 링. **프레임 스택은 담지 않는다** — 전이마다 관측 한 판만 담고, 표집할 때
// 첨자를 거슬러 올라가 스택을 만든다(replay_common.glsl 첫머리).
//
// 값은 uint8 회색이라 전이 하나가 뷰당 7,056 바이트다. 5 만 전이면 뷰당 353 MB 이고, 이것이 이 스택에서
// 가장 큰 메모리 소비처다. 그래서 기동할 때 바이트를 로그에 찍고 예산을 넘으면 스스로 줄인다.
class ReplayBuffer {
public:
    ReplayBuffer(Context& context, uint32_t views, uint32_t actionCount);
    ~ReplayBuffer();
    ReplayBuffer(const ReplayBuffer&) = delete;
    ReplayBuffer& operator=(const ReplayBuffer&) = delete;

    bool available() const { return ready; }

    // 링을 잡는다. 예산의 25% 를 넘으면 용량을 스스로 줄이고 그 사실을 로그에 남긴다(자동 튜닝 철학).
    // 실제로 잡힌 용량을 돌려준다.
    uint32_t reserve(uint32_t capacity, uint32_t batch);

    uint32_t capacity() const { return window.capacity; }
    uint32_t count() const { return window.count; }
    // 지금까지 담은 바이트. 편집기가 보여 준다.
    VkDeviceSize residentBytes() const;

    // 다음에 쓸 칸의 메타를 채운다. **record 하기 전에 부른다** — 그 칸의 에피소드·걸음이 정해져야
    // 표집이 스택을 어디까지 거슬러 올라갈지 안다. 에피소드가 바뀌면 episode 를 올려서 준다.
    void beginSlot(uint32_t episode, uint32_t stepInEpisode, float reward, float discount, const float* action);
    // 담지 못했음을 나타내는 칸 번호. 0 을 실패로 쓰면 0 번 칸에 담은 것과 구별되지 않는다.
    static constexpr uint32_t NO_SLOT = 0xFFFFFFFFU;
    // beginSlot 이 정한 칸에 이번 관측을 담고 커서를 하나 민다. 돌려주는 것은 담은 칸의 번호이고,
    // 담지 못했으면 NO_SLOT 이다.
    uint32_t recordStore(VkCommandBuffer commandBuffer, VkDeviceAddress observationFeatures);

    // 담은 뒤에 메타를 고친다. **행동과 보상은 관측보다 늦게 정해진다** — 행동은 같은 제출의 순전파가
    // 내놓고, 보상은 그 행동으로 물리를 한 걸음 밟은 **다음 프레임**에야 안다. 그래도 순서가 어긋나지
    // 않는 것은, 방금 담은 칸이 뒤가 없어 다음 프레임까지 표본이 되지 못하기 때문이다.
    void patchAction(uint32_t slot, const float* action);
    void patchOutcome(uint32_t slot, float reward, float discount);

    // 표본을 뽑아 targets 에 채운다. 뽑을 것이 모자라면 거짓을 돌려주고 아무 것도 하지 않는다.
    // seed 와 stream 이 같으면 같은 표본이 나온다 — 자기 검사가 CPU 로도 같은 것을 만들어 견준다.
    bool recordSample(VkCommandBuffer commandBuffer, const ReplayBatchTargets& targets, uint64_t seed, uint64_t stream);
    // 표본을 **밖에서 주는** 갈래. 자기 검사가 링의 갈래(에피소드 경계·되감기·창 잘림)를 우연에 맡기지
    // 않고 골라 밟는 데 쓴다. 첨자가 유효한지는 부르는 쪽이 본다.
    bool recordSampleWith(VkCommandBuffer commandBuffer,
                          const ReplayBatchTargets& targets,
                          const std::vector<GpuReplaySample>& chosen);

    // 마지막 recordSample 이 고른 표본과 그때의 창. 자기 검사가 같은 것을 CPU 로 만들어 견준다.
    const std::vector<GpuReplaySample>& lastSamples() const { return samples; }
    const ReplayWindow& windowState() const { return window; }
    const std::vector<GpuReplaySlot>& hostSlots() const { return slotMirror; }

private:
    void createPipelines();

    Context& context;
    uint32_t views = 1;
    uint32_t actionCount = 1;
    uint32_t batchLimit = 0;
    bool ready = false;

    ReplayWindow window;
    // 다음 store 가 쓸 칸의 메타. beginSlot 이 채운다.
    GpuReplaySlot pendingSlot;
    bool pendingReady = false;

    VkPipelineLayout storeLayout = VK_NULL_HANDLE;
    VkPipeline storePipeline = VK_NULL_HANDLE;
    VkPipelineLayout sampleLayout = VK_NULL_HANDLE;
    VkPipeline samplePipeline = VK_NULL_HANDLE;

    Buffer frames;
    Buffer slots;
    Buffer actions;
    Buffer sampleBuffer;

    // 메타의 호스트 사본. 표본을 고를 때 에피소드를 견주는 데 쓴다 — 칸마다 16 바이트뿐이라 되읽는
    // 것보다 싸다. **관측 판은 사본을 두지 않는다**(5 만 전이면 뷰당 353 MB 다).
    std::vector<GpuReplaySlot> slotMirror;
    std::vector<uint32_t> episodeMirror;
    std::vector<GpuReplaySample> samples;
    std::vector<uint32_t> validIndices;
};

} // namespace gfx
