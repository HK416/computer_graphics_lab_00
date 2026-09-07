#include "gfx/replay.h"

#include <algorithm>

#include <spdlog/spdlog.h>

#include "gfx/context.h"
#include "gfx/observation.h"
#include "gfx/vk_check.h"

namespace gfx {
namespace {

// 링이 장치 예산에서 차지해도 되는 몫. 넘으면 용량을 스스로 줄인다. 자동 튜닝이 렌더 배율을 내리는
// 것과 같은 철학이다 — 예산을 넘겨 죽는 대신 조금 나쁜 설정으로 돈다.
constexpr double BUDGET_SHARE = 0.25;
// 이보다 작게는 줄이지 않는다. 배치 하나도 못 채우는 링은 학습이 되지 않으므로 차라리 로그로 알린다.
constexpr uint32_t MINIMUM_CAPACITY = 1024;
constexpr uint32_t STORE_GROUP = 64;
constexpr uint32_t SAMPLE_GROUP = 8;

VkDeviceSize frameBytesPerSlot(uint32_t views) {
    return static_cast<VkDeviceSize>(views) * OBSERVATION_SIZE * OBSERVATION_SIZE;
}

Buffer createStorage(Context& context, VkDeviceSize bytes, MemoryLocation location, const char* name) {
    return createBuffer(context,
                        std::max<VkDeviceSize>(bytes, 4),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        location,
                        name);
}

VkPipeline createComputePipeline(Context& context, VkPipelineLayout layout, const char* shaderName) {
    VkShaderModule module = tryCreateShaderModule(context.device, shaderName);
    if (module == VK_NULL_HANDLE) {
        return VK_NULL_HANDLE;
    }
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = shaderStage(VK_SHADER_STAGE_COMPUTE_BIT, module);
    info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
        spdlog::warn("리플레이 컴퓨트 파이프라인을 만들지 못했습니다: {}", shaderName);
        pipeline = VK_NULL_HANDLE;
    }
    vkDestroyShaderModule(context.device, module, nullptr);
    return pipeline;
}

} // namespace

ReplayBuffer::ReplayBuffer(Context& context, uint32_t views, uint32_t actionCount)
    : context(context), views(std::max(views, 1U)), actionCount(std::max(actionCount, 1U)) {
    createPipelines();
}

ReplayBuffer::~ReplayBuffer() {
    destroyBuffer(context, frames);
    destroyBuffer(context, slots);
    destroyBuffer(context, actions);
    destroyBuffer(context, sampleBuffer);
    vkDestroyPipeline(context.device, storePipeline, nullptr);
    vkDestroyPipeline(context.device, samplePipeline, nullptr);
    vkDestroyPipelineLayout(context.device, storeLayout, nullptr);
    vkDestroyPipelineLayout(context.device, sampleLayout, nullptr);
}

void ReplayBuffer::createPipelines() {
    // 디스크립터 집합을 쓰지 않는다. 버퍼는 전부 주소로 실어 보낸다(저장소 규약).
    VkPushConstantRange storeRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ReplayStorePushConstants)};
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &storeRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &storeLayout));

    VkPushConstantRange sampleRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ReplaySamplePushConstants)};
    layoutInfo.pPushConstantRanges = &sampleRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &sampleLayout));

    storePipeline = createComputePipeline(context, storeLayout, "neural_replay_store.comp.spv");
    samplePipeline = createComputePipeline(context, sampleLayout, "neural_replay_sample.comp.spv");
    ready = storePipeline != VK_NULL_HANDLE && samplePipeline != VK_NULL_HANDLE;
    if (!ready) {
        spdlog::warn("리플레이 버퍼를 만들지 못했습니다. 픽셀 학습 경로를 끕니다");
    }
}

VkDeviceSize ReplayBuffer::residentBytes() const {
    return frames.size + slots.size + actions.size + sampleBuffer.size;
}

uint32_t ReplayBuffer::reserve(uint32_t capacity, uint32_t batch) {
    if (!ready || capacity == 0 || batch == 0) {
        return 0;
    }
    // 예산을 먼저 본다. 링이 이 스택에서 가장 큰 소비처라 여기서 걸러야 뒤에서 안 죽는다.
    VkDeviceSize perSlot =
        frameBytesPerSlot(views) + sizeof(GpuReplaySlot) + static_cast<VkDeviceSize>(actionCount) * sizeof(float);
    Context::MemoryBudget budget = context.deviceMemoryBudget();
    VkDeviceSize allowed =
        budget.budget > budget.usage
            ? static_cast<VkDeviceSize>(static_cast<double>(budget.budget - budget.usage) * BUDGET_SHARE)
            : 0;
    // **예산이 이미 바닥이면 allowed 가 0 이다.** 그것은 «줄일 필요 없음» 이 아니라 «최소로 줄여라» 다.
    // 갈래를 건너뛰면 게이트가 있어야 할 바로 그 상황에서 요청한 용량이 그대로 잡혀 VK_CHECK 로 죽는다.
    uint32_t requested = capacity;
    auto fits = static_cast<uint32_t>(std::min<VkDeviceSize>(allowed / perSlot, 0xFFFFFFFFULL));
    // 바닥을 두는 것은 «배치 하나도 못 채우는 링은 학습이 되지 않는다» 는 뜻이고, 그래서 이 경우
    // 실제 할당량이 예산 몫을 **넘는다.** 로그가 그 사실을 그대로 말해야 다음 사람이 헷갈리지 않는다.
    capacity = std::min(capacity, std::max(fits, MINIMUM_CAPACITY));
    if (capacity != requested || fits < capacity) {
        spdlog::warn("리플레이 용량 {} -> {} 전이 (예산 몫 {:.0f}% 로는 {} 개, 바닥 {} 개, 필요 {} MB)",
                     requested,
                     capacity,
                     BUDGET_SHARE * 100.0,
                     fits,
                     MINIMUM_CAPACITY,
                     (static_cast<VkDeviceSize>(capacity) * perSlot) / (1024ULL * 1024ULL));
    }

    // **맡긴다.** 즉시 파괴하면 도는 프레임이 읽는 중일 수 있다(12단계에서 프레임 루프에 붙는다).
    context.retireBuffer(frames);
    context.retireBuffer(slots);
    context.retireBuffer(actions);
    context.retireBuffer(sampleBuffer);

    frames = createStorage(
        context, static_cast<VkDeviceSize>(capacity) * frameBytesPerSlot(views), MemoryLocation::DEVICE, "리플레이 판");
    // 메타와 행동은 호스트가 칸마다 채우고 셰이더가 읽기만 한다. 스테이징을 거칠 이유가 없다.
    slots = createStorage(context,
                          static_cast<VkDeviceSize>(capacity) * sizeof(GpuReplaySlot),
                          MemoryLocation::HOST_WRITE,
                          "리플레이 메타");
    actions = createStorage(context,
                            static_cast<VkDeviceSize>(capacity) * actionCount * sizeof(float),
                            MemoryLocation::HOST_WRITE,
                            "리플레이 행동");
    sampleBuffer = createStorage(context,
                                 static_cast<VkDeviceSize>(batch) * sizeof(GpuReplaySample),
                                 MemoryLocation::HOST_WRITE,
                                 "리플레이 표본");

    window = ReplayWindow{};
    window.capacity = capacity;
    batchLimit = batch;
    pendingReady = false;
    slotMirror.assign(capacity, GpuReplaySlot{});
    episodeMirror.assign(capacity, 0);
    samples.clear();
    validIndices.clear();
    validIndices.reserve(capacity);

    spdlog::info("리플레이 링: 전이 {} 개, 뷰 {}, 행동 {} 차원, {} MB",
                 capacity,
                 views,
                 actionCount,
                 residentBytes() / (1024ULL * 1024ULL));
    return capacity;
}

void ReplayBuffer::beginSlot(
    uint32_t episode, uint32_t stepInEpisode, float reward, float discount, const float* action) {
    if (!ready || window.capacity == 0) {
        return;
    }
    pendingSlot.episode = episode;
    pendingSlot.stepInEpisode = stepInEpisode;
    pendingSlot.reward = reward;
    pendingSlot.discount = discount;
    uint32_t slot = window.cursor;
    static_cast<GpuReplaySlot*>(slots.mapped)[slot] = pendingSlot;
    slotMirror[slot] = pendingSlot;
    episodeMirror[slot] = episode;
    auto* target = static_cast<float*>(actions.mapped) + static_cast<size_t>(slot) * actionCount;
    for (uint32_t i = 0; i < actionCount; ++i) {
        target[i] = action != nullptr ? action[i] : 0.0F;
    }
    // **플러시한다.** createBuffer 는 coherent 를 «선호» 만 할 뿐 보장하지 않는다. non-coherent 타입이
    // 뽑히는 기기에서는 메타와 행동이 GPU 에 안 보여 관측이 남의 칸에서 나온다. coherent 면 무동작이다.
    vmaFlushAllocation(context.allocator, slots.allocation, 0, VK_WHOLE_SIZE);
    vmaFlushAllocation(context.allocator, actions.allocation, 0, VK_WHOLE_SIZE);
    pendingReady = true;
}

void ReplayBuffer::recordStore(VkCommandBuffer commandBuffer, VkDeviceAddress observationFeatures) {
    if (!ready || window.capacity == 0 || !pendingReady || observationFeatures == 0) {
        return;
    }
    // **앞에도 배리어를 건다.** 인코드가 features 를 다 썼어야 하고(RAW), 앞의 표집이 frames 를 다
    // 읽었어야 한다(WAR). 뒤에만 걸면 한 명령 버퍼에 [담기][표집][담기] 가 오는 순간 뒤 담기가 앞
    // 표집이 읽는 칸을 덮는다.
    memoryBarrier(commandBuffer,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    ReplayStorePushConstants push{};
    push.features = observationFeatures;
    push.frames = frames.address;
    push.slot = window.cursor;
    push.views = views;
    push.stack = OBSERVATION_STACK;
    push.size = OBSERVATION_SIZE;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, storePipeline);
    vkCmdPushConstants(commandBuffer, storeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    auto words = static_cast<uint32_t>(frameBytesPerSlot(views) / 4);
    vkCmdDispatch(commandBuffer, (words + STORE_GROUP - 1) / STORE_GROUP, 1, 1);
    memoryBarrier(commandBuffer,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    // 커서를 민다. 한 바퀴 돌면 가장 오래된 칸부터 덮인다.
    window.cursor = (window.cursor + 1) % window.capacity;
    window.count = std::min(window.count + 1, window.capacity);
    pendingReady = false;
}

bool ReplayBuffer::recordSample(VkCommandBuffer commandBuffer,
                                const ReplayBatchTargets& targets,
                                uint64_t seed,
                                uint64_t stream) {
    if (!ready || window.capacity == 0 || batchLimit == 0) {
        return false;
    }
    // 유효한 칸을 모은다. 규칙은 순수 함수 하나가 쥐고 있고 테스트가 그것을 본다.
    validIndices.clear();
    for (uint32_t index = 0; index < window.capacity; ++index) {
        if (replaySampleValid(window, index, episodeMirror.data())) {
            validIndices.push_back(index);
        }
    }
    if (validIndices.size() < batchLimit) {
        return false;
    }

    // **호스트가 고른다.** 그래야 자기 검사가 같은 표본을 CPU 로 만들어 바이트로 견줄 수 있다. 씨앗과
    // 흐름이 같으면 언제나 같은 것이 나온다.
    samples.resize(batchLimit);
    for (uint32_t i = 0; i < batchLimit; ++i) {
        uint64_t draw = stream * batchLimit + i;
        uint32_t pick = neuralRandomBelow(seed, draw, static_cast<uint32_t>(validIndices.size()));
        GpuReplaySample& item = samples[i];
        item = GpuReplaySample{};
        item.index = validIndices[pick];
        // 상태와 다음 상태의 변위를 따로 뽑는다(DrQ-v2 규약). 흐름을 어긋나게 해 두 값이 같아지지 않게 한다.
        item.shiftX = replayShift(seed ^ 0x51ED2701ULL, draw * 2, REPLAY_SHIFT_PADDING);
        item.shiftY = replayShift(seed ^ 0x51ED2701ULL, draw * 2 + 1, REPLAY_SHIFT_PADDING);
        item.nextShiftX = replayShift(seed ^ 0x9E37B1C3ULL, draw * 2, REPLAY_SHIFT_PADDING);
        item.nextShiftY = replayShift(seed ^ 0x9E37B1C3ULL, draw * 2 + 1, REPLAY_SHIFT_PADDING);
    }
    return recordSampleWith(commandBuffer, targets, samples);
}

bool ReplayBuffer::recordSampleWith(VkCommandBuffer commandBuffer,
                                    const ReplayBatchTargets& targets,
                                    const std::vector<GpuReplaySample>& chosen) {
    if (!ready || window.capacity == 0 || chosen.size() != batchLimit) {
        return false;
    }
    if (&chosen != &samples) {
        samples = chosen;
    }
    std::copy(samples.begin(), samples.end(), static_cast<GpuReplaySample*>(sampleBuffer.mapped));
    vmaFlushAllocation(context.allocator, sampleBuffer.allocation, 0, VK_WHOLE_SIZE);

    ReplaySamplePushConstants push{};
    push.frames = frames.address;
    push.slots = slots.address;
    push.actions = actions.address;
    push.samples = sampleBuffer.address;
    push.state = targets.state;
    push.nextState = targets.nextState;
    push.batchAction = targets.action;
    push.batchReward = targets.reward;
    push.batchDiscount = targets.discount;
    push.capacity = window.capacity;
    push.count = window.count;
    push.cursor = window.cursor;
    push.views = views;
    push.stack = OBSERVATION_STACK;
    push.size = OBSERVATION_SIZE;
    push.actionCount = actionCount;
    push.batch = batchLimit;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, samplePipeline);
    vkCmdPushConstants(commandBuffer, sampleLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    uint32_t groups = (OBSERVATION_SIZE + SAMPLE_GROUP - 1) / SAMPLE_GROUP;
    vkCmdDispatch(commandBuffer, groups, groups, batchLimit * views);
    memoryBarrier(commandBuffer,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);
    return true;
}

} // namespace gfx
