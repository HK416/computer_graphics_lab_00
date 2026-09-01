#include "gfx/profiler.h"

#include <chrono>

#include <spdlog/spdlog.h>

#include "gfx/context.h"
#include "gfx/vk_check.h"

namespace gfx {
namespace {

uint64_t nowNanoseconds() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

} // namespace

GpuProfiler::GpuProfiler(Context& context, uint32_t frameCount) : context(context) {
    gpuSupported = context.caps.timestamps;
    period = context.properties.limits.timestampPeriod;
    validBits = context.queueFamilies.graphicsTimestampBits;

    pending.assign(frameCount, 0);
    pendingQueries.assign(frameCount, 0);
    frameZones.resize(frameCount);
    pendingZones.resize(frameCount);
    if (!gpuSupported) {
        spdlog::info("GPU 프로파일러: 타임스탬프 미지원 장치라 CPU 구간만 잽니다");
        return;
    }

    pools.resize(frameCount, VK_NULL_HANDLE);
    for (VkQueryPool& pool : pools) {
        VkQueryPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        poolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        poolInfo.queryCount = MAX_PROFILER_ZONES * 2;
        VK_CHECK(vkCreateQueryPool(context.device, &poolInfo, nullptr, &pool));
        // hostQueryReset 이 필수 기능이라 커맨드 버퍼에 리셋을 넣지 않고 호스트에서 되돌린다.
        vkResetQueryPool(context.device, pool, 0, MAX_PROFILER_ZONES * 2);
    }
}

GpuProfiler::~GpuProfiler() {
    for (VkQueryPool pool : pools) {
        vkDestroyQueryPool(context.device, pool, nullptr);
    }
}

void GpuProfiler::collect() {
    if (!enabled) {
        return;
    }

    uint32_t frameSlot = currentSlot;
    uint32_t count = pending[frameSlot];
    pending[frameSlot] = 0;
    if (count == 0) {
        if (gpuSupported) {
            vkResetQueryPool(context.device, pools[frameSlot], 0, MAX_PROFILER_ZONES * 2);
        }
        return;
    }

    const std::array<Zone, MAX_PROFILER_ZONES>& recorded = pendingZones[frameSlot];
    uint32_t queries = pendingQueries[frameSlot];
    pendingQueries[frameSlot] = 0;
    bool haveGpu = false;
    if (gpuSupported && queries > 0) {
        // 대기 없이 읽는다. 아직 안 끝났으면 이번 프레임 GPU 값은 버린다.
        VkResult result = vkGetQueryPoolResults(context.device,
                                                pools[frameSlot],
                                                0,
                                                queries * 2,
                                                sizeof(uint64_t) * queries * 2,
                                                results.data(),
                                                sizeof(uint64_t),
                                                VK_QUERY_RESULT_64_BIT);
        haveGpu = result == VK_SUCCESS;
    }

    display.resize(count);
    for (uint32_t i = 0; i < count; ++i) {
        ProfilerZone& zone = display[i];
        // 구간 구성이 바뀌면(경로 추적을 켜는 등) 평활값을 이어 쓰면 안 된다.
        if (zone.name != recorded[i].name) {
            zone = ProfilerZone{};
        }
        zone.name = recorded[i].name;
        zone.depth = recorded[i].depth;
        zone.cpuMilliseconds = smoothMilliseconds(zone.cpuMilliseconds, recorded[i].cpuMilliseconds, smoothing);
        uint32_t query = recorded[i].query;
        zone.hasGpu = haveGpu && query != INVALID_PROFILER_ZONE;
        if (zone.hasGpu) {
            float sample = timestampMilliseconds(results[query * 2], results[query * 2 + 1], period, validBits);
            zone.gpuMilliseconds = smoothMilliseconds(zone.gpuMilliseconds, sample, smoothing);
        }
    }

    if (gpuSupported) {
        vkResetQueryPool(context.device, pools[frameSlot], 0, MAX_PROFILER_ZONES * 2);
    }
}

void GpuProfiler::beginFrame(uint32_t frameSlot) {
    currentSlot = frameSlot;
    zoneCount = 0;
    queryCount = 0;
    depth = 0;
    if (!enabled) {
        // 껐다 켜는 사이의 낡은 결과를 읽지 않도록 대기 목록을 비운다.
        pending[frameSlot] = 0;
        pendingQueries[frameSlot] = 0;
    }
}

void GpuProfiler::endFrame() {
    if (!enabled) {
        pending[currentSlot] = 0;
        pendingQueries[currentSlot] = 0;
        return;
    }
    pendingZones[currentSlot] = frameZones[currentSlot];
    pending[currentSlot] = zoneCount;
    pendingQueries[currentSlot] = queryCount;
}

uint32_t GpuProfiler::begin(const char* name, VkCommandBuffer commandBuffer) {
    if (!enabled || zoneCount >= MAX_PROFILER_ZONES) {
        return INVALID_PROFILER_ZONE;
    }

    uint32_t index = zoneCount++;
    Zone& zone = frameZones[currentSlot][index];
    zone.name = name;
    zone.depth = depth++;
    zone.cpuBegin = nowNanoseconds();
    zone.query = INVALID_PROFILER_ZONE;
    if (gpuSupported && commandBuffer != VK_NULL_HANDLE) {
        zone.query = queryCount++;
        // 파이프라인 처음과 끝에서만 찍는다. 다른 단계는 GPU 가 작업을 겹쳐 실행해 신뢰할 수 없다.
        vkCmdWriteTimestamp2(commandBuffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, pools[currentSlot], zone.query * 2);
    }
    return index;
}

void GpuProfiler::end(uint32_t zoneIndex, VkCommandBuffer commandBuffer) {
    if (zoneIndex == INVALID_PROFILER_ZONE) {
        return;
    }

    Zone& zone = frameZones[currentSlot][zoneIndex];
    zone.cpuMilliseconds = static_cast<float>(static_cast<double>(nowNanoseconds() - zone.cpuBegin) / 1.0e6);
    if (zone.query != INVALID_PROFILER_ZONE) {
        vkCmdWriteTimestamp2(
            commandBuffer, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, pools[currentSlot], zone.query * 2 + 1);
    }
    if (depth > 0) {
        --depth;
    }
}

} // namespace gfx
