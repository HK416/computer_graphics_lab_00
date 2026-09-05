#include "app/plugins/profiler_plugin.h"

#include <format>
#include <string>
#include <vector>

#include <imgui.h>
#include <spdlog/spdlog.h>

#include "app/application.h"
#include "gfx/profiler.h"

namespace app {

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
    renderer = &services.renderer;
    renderer->profiler().enabled = services.options.profile;
}

void ProfilerPlugin::ui(Services& services) {
    if (!services.editor.settingsSection("프로파일러")) {
        return;
    }
    gfx::GpuProfiler& profiler = services.renderer.profiler();
    ImGui::Checkbox("구간 계측", &profiler.enabled);
    if (!profiler.gpuAvailable()) {
        ImGui::SameLine();
        ImGui::TextDisabled("(GPU 타임스탬프 미지원, CPU 만)");
    }
    if (!profiler.enabled) {
        return;
    }
    ImGui::SliderFloat("평활", &profiler.smoothing, 0.01F, 1.0F, "%.2f");
    const std::vector<gfx::ProfilerZone>& zones = profiler.zones();
    if (zones.empty()) {
        ImGui::TextDisabled("측정 중...");
        return;
    }
    if (ImGui::BeginTable("구간", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
        ImGui::TableSetupColumn("구간");
        ImGui::TableSetupColumn("CPU");
        ImGui::TableSetupColumn("GPU");
        ImGui::TableHeadersRow();
        for (const gfx::ProfilerZone& zone : zones) {
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
        }
        ImGui::EndTable();
    }
}

} // namespace app
