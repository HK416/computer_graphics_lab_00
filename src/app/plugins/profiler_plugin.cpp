#include "app/plugins/profiler_plugin.h"

#include <algorithm>
#include <cfloat>
#include <ctime>
#include <format>
#include <fstream>
#include <string>
#include <vector>

#include <imgui.h>
#include <spdlog/spdlog.h>

#include "app/application.h"
#include "gfx/profiler.h"

namespace app {

namespace {

// 편집기 기본 배치가 같은 이름으로 도킹 자리를 잡는다(editor_internal.h 의 WINDOW_PROFILER).
constexpr const char* WINDOW_NAME = "프로파일러";
constexpr double MEGABYTE = 1024.0 * 1024.0;

} // namespace

ProfilerPlugin::~ProfilerPlugin() {
    if (renderer == nullptr || !renderer->profiler().enabled) {
        return;
    }
    spdlog::info("구간 계측 결과 (CPU / GPU, ms)");
    for (const gfx::ProfilerZone& zone : renderer->profiler().zones()) {
        spdlog::info("  {:<28} {:7.3f}  {:>7}",
                     std::string(zone.depth * 2, ' ') + zone.name,
                     zone.cpuMilliseconds,
                     zone.hasGpu ? std::format("{:.3f}", zone.gpuMilliseconds) : std::string{"-"});
    }
}

void ProfilerPlugin::build(Services& services) {
    renderer = services.renderer;
    context = services.context;
    if (renderer != nullptr) {
        renderer->profiler().enabled = services.options.profile;
        // --profile 로 시작했으면 창도 바로 연다.
        showWindow = services.options.profile;
    }
}

void ProfilerPlugin::update(Services& services, float deltaSeconds) {
    (void)services;
    (void)deltaSeconds;
    if (renderer == nullptr || !renderer->profiler().enabled) {
        return;
    }
    const std::vector<gfx::ProfilerZone>& zones = renderer->profiler().zones();
    if (zones.empty()) {
        return;
    }
    history.resize(zones.size());
    float cpuTotal = 0.0F;
    float gpuTotal = 0.0F;
    // GPU 합은 GPU 를 재는 가장 바깥 구간들만 더한다. 최상위 구간은 CPU 만 재는 프레임 래퍼일 수 있고,
    // 안쪽 GPU 구간을 다 더하면 겹쳐서 두 번 센다. 구간은 시작 순서라 깊이만 보면 조상을 안다.
    uint32_t gpuCoverDepth = UINT32_MAX;
    for (size_t i = 0; i < zones.size(); ++i) {
        const gfx::ProfilerZone& zone = zones[i];
        History& entry = history[i];
        if (entry.name != zone.name) {
            entry = History{};
            entry.name = zone.name;
        }
        entry.depth = zone.depth;
        entry.hasGpu = zone.hasGpu;
        entry.cpu[cursor] = zone.cpuSampleMilliseconds;
        entry.gpu[cursor] = zone.gpuSampleMilliseconds;
        if (zone.depth == 0) {
            cpuTotal += zone.cpuSampleMilliseconds;
        }
        if (zone.depth <= gpuCoverDepth) {
            gpuCoverDepth = UINT32_MAX;
        }
        if (zone.hasGpu && gpuCoverDepth == UINT32_MAX) {
            gpuTotal += zone.gpuSampleMilliseconds;
            gpuCoverDepth = zone.depth;
        }
    }
    frameCpu[cursor] = cpuTotal;
    frameGpu[cursor] = gpuTotal;
    cursor = (cursor + 1) % HISTORY_FRAMES;
    filled = std::min(filled + 1, HISTORY_FRAMES);

    if (context != nullptr && memoryCountdown == 0) {
        memoryCountdown = 30;
        gfx::Context::MemoryBudget budget = context->deviceMemoryBudget();
        memoryUsage = budget.usage;
        memoryBudget = budget.budget;
        memoryDevice = context->deviceLocalMemoryBytes();
    }
    if (memoryCountdown > 0) {
        --memoryCountdown;
    }
}

void ProfilerPlugin::ui(Services& services) {
    if (!services.editor->settingsSection("프로파일러")) {
        return;
    }
    gfx::GpuProfiler& profiler = services.renderer->profiler();
    ImGui::Checkbox("구간 계측", &profiler.enabled);
    if (!profiler.gpuAvailable()) {
        ImGui::SameLine();
        ImGui::TextDisabled("(GPU 타임스탬프 미지원, CPU 만)");
    }
    if (!profiler.enabled) {
        return;
    }
    ImGui::Checkbox("프로파일러 창", &showWindow);
    ImGui::SliderFloat("평활", &profiler.smoothing, 0.01F, 1.0F, "%.2f");
}

void ProfilerPlugin::window(Services& services) {
    if (!showWindow) {
        return;
    }
    if (!ImGui::Begin(WINDOW_NAME, &showWindow)) {
        ImGui::End();
        return;
    }
    gfx::GpuProfiler& profiler = services.renderer->profiler();
    if (!profiler.enabled) {
        ImGui::TextDisabled("렌더 설정의 «구간 계측» 을 켜면 잰다.");
        ImGui::End();
        return;
    }
    if (history.empty()) {
        ImGui::TextDisabled("측정 중...");
        ImGui::End();
        return;
    }

    // 링은 cursor 가 가장 오래된 칸이므로 거기서부터 그리면 시간순이 된다.
    int offset = static_cast<int>(cursor);
    int count = static_cast<int>(HISTORY_FRAMES);
    float cpuMax = *std::max_element(frameCpu.begin(), frameCpu.end());
    float gpuMax = *std::max_element(frameGpu.begin(), frameGpu.end());
    float scaleMax = std::max(cpuMax, gpuMax) * 1.1F + 0.01F;
    uint32_t last = (cursor + HISTORY_FRAMES - 1) % HISTORY_FRAMES;
    std::string cpuLabel = std::format("CPU {:.2f} ms (최대 {:.2f})", frameCpu[last], cpuMax);
    ImGui::PlotLines(
        "##frameCpu", frameCpu.data(), count, offset, cpuLabel.c_str(), 0.0F, scaleMax, ImVec2{-1.0F, 70.0F});
    if (profiler.gpuAvailable()) {
        std::string gpuLabel = std::format("GPU {:.2f} ms (최대 {:.2f})", frameGpu[last], gpuMax);
        ImGui::PlotLines(
            "##frameGpu", frameGpu.data(), count, offset, gpuLabel.c_str(), 0.0F, scaleMax, ImVec2{-1.0F, 70.0F});
    }
    if (context != nullptr) {
        ImGui::Text("GPU 메모리 %.0f / 예산 %.0f MB (장치 %.0f MB)",
                    static_cast<double>(memoryUsage) / MEGABYTE,
                    static_cast<double>(memoryBudget) / MEGABYTE,
                    static_cast<double>(memoryDevice) / MEGABYTE);
    }
    if (ImGui::Button("CSV 저장")) {
        saveCsv();
    }
    ImGui::SameLine();
    ImGui::TextDisabled("최근 %u 프레임", filled);

    if (ImGui::BeginTable("구간", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("구간", ImGuiTableColumnFlags_None, 3.0F);
        ImGui::TableSetupColumn("CPU", ImGuiTableColumnFlags_None, 1.0F);
        ImGui::TableSetupColumn("GPU", ImGuiTableColumnFlags_None, 1.0F);
        ImGui::TableSetupColumn("추이 (CPU)", ImGuiTableColumnFlags_None, 3.0F);
        ImGui::TableHeadersRow();
        const std::vector<gfx::ProfilerZone>& zones = profiler.zones();
        for (size_t i = 0; i < history.size() && i < zones.size(); ++i) {
            const gfx::ProfilerZone& zone = zones[i];
            History& entry = history[i];
            ImGui::TableNextRow();
            ImGui::TableNextColumn();
            // 중첩된 구간은 들여써서 상위 구간과 구분한다.
            ImGui::Text("%*s%s", static_cast<int>(zone.depth) * 2, "", zone.name);
            ImGui::TableNextColumn();
            ImGui::Text("%.3f", static_cast<double>(zone.cpuMilliseconds));
            ImGui::TableNextColumn();
            if (zone.hasGpu) {
                ImGui::Text("%.3f", static_cast<double>(zone.gpuMilliseconds));
            } else {
                ImGui::TextDisabled("-");
            }
            ImGui::TableNextColumn();
            ImGui::PushID(static_cast<int>(i));
            ImGui::PlotLines("##spark", entry.cpu.data(), count, offset, nullptr, 0.0F, FLT_MAX, ImVec2{-1.0F, 18.0F});
            ImGui::PopID();
        }
        ImGui::EndTable();
    }
    ImGui::End();
}

void ProfilerPlugin::saveCsv() const {
    // Apple libc++ 에는 std::chrono::zoned_time(시간대 DB)이 없어 C 시간 함수로 현지 시각을 찍는다.
    std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    char stamp[32];
    std::strftime(stamp, sizeof(stamp), "%Y%m%d_%H%M%S", &local);
    std::string path = std::format("profiler_{}.csv", stamp);
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        spdlog::error("프로파일러 CSV 를 열지 못함: {}", path);
        return;
    }
    file << "frame,zone,depth,cpu_ms,gpu_ms\n";
    // 오래된 프레임부터 0 번으로 센다.
    uint32_t start = (cursor + HISTORY_FRAMES - filled) % HISTORY_FRAMES;
    for (uint32_t frame = 0; frame < filled; ++frame) {
        uint32_t slot = (start + frame) % HISTORY_FRAMES;
        for (const History& entry : history) {
            if (entry.name == nullptr) {
                continue;
            }
            file << std::format("{},{},{},{:.4f},", frame, entry.name, entry.depth, entry.cpu[slot]);
            if (entry.hasGpu) {
                file << std::format("{:.4f}", entry.gpu[slot]);
            }
            file << '\n';
        }
    }
    spdlog::info("프로파일러 CSV 저장: {} ({} 프레임)", path, filled);
}

} // namespace app
