#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include "app/plugin.h"

namespace app {

// 구간 계측. --profile 로 켜고, 편집기 «프로파일러» 절과 창을 그리며, 종료할 때 결과를 로그로 남긴다(창을 못 보는
// 스크린샷·CI 실행에서도 결과가 남는다). 계측기 자체(gfx::GpuProfiler)는 커맨드 버퍼 기록과 묶여 Renderer 에 있다.
//
// 창은 평활 전 표본을 프레임마다 링에 모아 추이 그래프로 보여 주고, 그 링을 CSV 로 저장한다.
class ProfilerPlugin : public Plugin {
public:
    ~ProfilerPlugin() override;
    const char* name() const override { return "프로파일러"; }
    void build(Services& services) override;
    void update(Services& services, float deltaSeconds) override;
    void ui(Services& services) override;
    void window(Services& services) override;

private:
    // 링에 담는 프레임 수. 60 FPS 에서 4 초.
    static constexpr uint32_t HISTORY_FRAMES = 240;

    struct History {
        const char* name = nullptr;
        uint32_t depth = 0;
        bool hasGpu = false;
        std::array<float, HISTORY_FRAMES> cpu{};
        std::array<float, HISTORY_FRAMES> gpu{};
    };

    void saveCsv() const;

    gfx::Renderer* renderer = nullptr;
    gfx::Context* context = nullptr;
    bool showWindow = false;
    // ponytail: 구간 색인 위치로 이름을 맞춘다. 구성이 바뀌면(경로 추적을 켜는 등) 그 자리만 비우므로
    // 나타났다 사라지는 구간은 링에 빈칸으로 남는다.
    std::vector<History> history;
    std::array<float, HISTORY_FRAMES> frameCpu{};
    std::array<float, HISTORY_FRAMES> frameGpu{};
    uint32_t cursor = 0;
    uint32_t filled = 0;
    // GPU 메모리 예산은 VMA 조회라 프레임마다 묻지 않고 한 번에 얼마간 쓴다.
    uint64_t memoryUsage = 0;
    uint64_t memoryBudget = 0;
    uint64_t memoryDevice = 0;
    uint32_t memoryCountdown = 0;
};

} // namespace app
