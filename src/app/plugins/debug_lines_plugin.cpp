#include "app/plugins/debug_lines_plugin.h"

#include <imgui.h>

#include "app/application.h"

namespace app {

void DebugLinesPlugin::build(Services& services) {
    services.settings.showColliders = services.options.showColliders;
}

void DebugLinesPlugin::ui(Services& services) {
    if (!services.editor->settingsSection("콜라이더 표시")) {
        return;
    }
    ImGui::Checkbox("콜라이더 표시", &services.settings.showColliders);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("강체 콜라이더는 초록, 유체 용기는 청록, 방출 상자는 노랑으로 덧그린다");
    }
    // Path Tracing은 깊이 버퍼를 채우지 않아 가림을 판정할 수 없다. 그 모드에서는 늘 보인다.
    bool depthAvailable = !services.settings.usePathTracing;
    ImGui::BeginDisabled(!services.settings.showColliders || !depthAvailable);
    ImGui::SameLine();
    ImGui::Checkbox("가림 판정", &services.settings.colliderOcclusion);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(depthAvailable ? "끄면 물체에 가려도 선이 그대로 보인다"
                                         : "Path Tracing은 깊이 버퍼를 채우지 않아 늘 보인다");
    }
    ImGui::EndDisabled();
}

} // namespace app
