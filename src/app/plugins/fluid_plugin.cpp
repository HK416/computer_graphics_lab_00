#include "app/plugins/fluid_plugin.h"

#include <imgui.h>

#include "app/application.h"
#include "gfx/fluid.h"

namespace app {

void FluidPlugin::build(Services& services) {
    // 자동 튜닝이 꺼져 있으면 기본값을 그대로 둔다(applyHardwareProfile 과 같은 규칙).
    if (services.options.autoTune != gfx::AutoTune::OFF) {
        services.settings.fluidParticleLimit = services.profile.fluidParticleLimit;
    }
}

void FluidPlugin::ui(Services& services) {
    if (!services.editor->settingsSection("유체")) {
        return;
    }
    gfx::Renderer& renderer = *services.renderer;
    auto limit = static_cast<int>(services.settings.fluidParticleLimit);
    if (ImGui::SliderInt("입자 상한", &limit, 1024, static_cast<int>(gfx::FLUID_MAX_PARTICLES))) {
        services.settings.fluidParticleLimit = static_cast<uint32_t>(limit);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("장면의 모든 유체 부품이 함께 쓰는 GPU 입자 수 상한. 자동 튜닝이 기기 등급에 맞춰 정한다");
    }
    ImGui::TextDisabled("GPU 백엔드 %s, 표면 컴퓨트 %s",
                        renderer.fluidGpuAvailable() ? "사용 가능" : "미지원",
                        renderer.fluidSurfaceAvailable() ? "사용 가능" : "미지원");
}

} // namespace app
