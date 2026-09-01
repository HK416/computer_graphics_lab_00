#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "gfx/profiler_math.h"

namespace gfx {

struct Context;

// 한 프레임에 잴 수 있는 구간 수. 고정 배열이라 켜고 끄는 데 할당이 없다.
inline constexpr uint32_t MAX_PROFILER_ZONES = 32;
inline constexpr uint32_t INVALID_PROFILER_ZONE = 0xFFFFFFFFU;

// 화면에 보여줄 구간 하나.
struct ProfilerZone {
    // 리터럴 포인터만 담는다. 복사도 비교도 하지 않는다.
    const char* name = nullptr;
    uint32_t depth = 0;
    float cpuMilliseconds = 0.0F;
    float gpuMilliseconds = 0.0F;
    bool hasGpu = false;
};

// GPU 타임스탬프와 CPU 스코프를 같은 이름으로 재는 프로파일러.
//
// 꺼져 있으면 begin/end 가 분기 하나로 돌아온다. Vulkan 호출도, 할당도, 문자열 비교도 없다.
// 결과는 같은 프레임 슬롯을 다시 쓸 때 읽으므로(이미 타임라인 세마포어를 기다린 뒤다) 대기가 없다.
class GpuProfiler {
public:
    GpuProfiler(Context& context, uint32_t frameCount);
    ~GpuProfiler();
    GpuProfiler(const GpuProfiler&) = delete;
    GpuProfiler& operator=(const GpuProfiler&) = delete;

    // 장치가 타임스탬프 쿼리를 지원하는지. 아니면 CPU 구간만 잰다.
    bool gpuAvailable() const { return gpuSupported; }

    bool enabled = false;
    float smoothing = 0.15F;

    // 프레임 맨 앞. CPU 구간이 GPU 보다 먼저 기록되므로 여기서 슬롯을 열어야 같은 프레임에 담긴다.
    void beginFrame(uint32_t frameSlot);
    // 이 슬롯의 지난 결과를 읽고 쿼리 풀을 되돌린다. 타임라인 대기를 통과한 뒤에 불러야
    // 기다리지 않고 읽을 수 있다.
    void collect();
    // 커맨드 버퍼 기록이 끝난 뒤. 이번 프레임 구간을 결과 대기 목록으로 옮긴다.
    void endFrame();

    // commandBuffer 가 널이면 CPU 만 잰다. 이름은 수명이 프로그램 전체인 리터럴이어야 한다.
    uint32_t begin(const char* name, VkCommandBuffer commandBuffer = VK_NULL_HANDLE);
    void end(uint32_t zone, VkCommandBuffer commandBuffer = VK_NULL_HANDLE);

    const std::vector<ProfilerZone>& zones() const { return display; }

private:
    struct Zone {
        const char* name = nullptr;
        uint32_t depth = 0;
        uint64_t cpuBegin = 0;
        float cpuMilliseconds = 0.0F;
        // 쿼리 번호는 GPU 구간만 따로 센다. CPU 전용 구간에 번호를 내주면 기록되지 않은 쿼리가
        // 사이에 섞여 vkGetQueryPoolResults 가 통째로 VK_NOT_READY 를 돌려준다.
        uint32_t query = INVALID_PROFILER_ZONE;
    };

    Context& context;
    std::vector<VkQueryPool> pools;
    // 슬롯마다 결과를 기다리는 구간 수. 0 이면 읽을 것이 없다.
    std::vector<uint32_t> pending;
    std::vector<uint32_t> pendingQueries;
    // 기록 중인 구간과, 결과를 기다리는 구간의 사본. 기록이 대기 중인 것을 덮어쓰면 안 된다.
    std::vector<std::array<Zone, MAX_PROFILER_ZONES>> frameZones;
    std::vector<std::array<Zone, MAX_PROFILER_ZONES>> pendingZones;
    std::vector<ProfilerZone> display;
    std::array<uint64_t, MAX_PROFILER_ZONES * 2> results{};
    uint32_t currentSlot = 0;
    uint32_t zoneCount = 0;
    uint32_t queryCount = 0;
    uint32_t depth = 0;
    bool gpuSupported = false;
    float period = 0.0F;
    uint32_t validBits = 0;
};

// RAII 로 감싸는 구간.
class ProfilerScope {
public:
    ProfilerScope(GpuProfiler& profiler, const char* name, VkCommandBuffer commandBuffer = VK_NULL_HANDLE)
        : profiler(profiler), commandBuffer(commandBuffer), zone(profiler.begin(name, commandBuffer)) {}
    ~ProfilerScope() { profiler.end(zone, commandBuffer); }
    ProfilerScope(const ProfilerScope&) = delete;
    ProfilerScope& operator=(const ProfilerScope&) = delete;

private:
    GpuProfiler& profiler;
    VkCommandBuffer commandBuffer;
    uint32_t zone;
};

} // namespace gfx
