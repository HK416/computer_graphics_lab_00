#include "app/plugins/physics_plugin.h"

#include <algorithm>
#include <optional>

#include <imgui.h>

#include "gfx/profiler.h"
#include "gfx/render_graph.h"
#include "gfx/renderer.h"
#include "physics/rigid_body.h"

namespace app {

void PhysicsPlugin::build(Services& services) {
    // 헤드리스에는 GPU 솔버가 없다. GPU 백엔드 강체는 CPU 솔버가 건너뛰므로 그대로 멈춰 있다.
    if (services.context == nullptr) {
        return;
    }
    rigid = std::make_unique<gfx::RigidBodySimulator>(*services.context, *services.bindless);
    // GPU 솔버는 변형 정점 뒤, 유체 앞에서 돈다. 아무 것도 읽지 않으므로 순서는 자유롭지만, 되읽기 복사가
    // 프레임 앞쪽에 있어야 큐가 비는 동안 옮겨진다. prepare 는 그래프를 짤 때(그리기 명령 구성 뒤) 장면에서
    // 강체를 모으고, 노드는 스텝이 있거나 올릴 것이 있을 때만 기록한다.
    services.renderer->addPass([this](gfx::RenderGraph& graph, const gfx::Renderer::FrameInfo& info) {
        rigid->prepare(info.scene, steps, STEP_SECONDS);
        graph.addAfter("스킨",
                       gfx::RenderNode{"강체",
                                       "강체",
                                       [this] { return rigid->bodyCount() > 0; },
                                       {},
                                       {},
                                       {},
                                       [this, &info](VkCommandBuffer cmd) { rigid->record(cmd, info.frameIndex); }});
    });
}

void PhysicsPlugin::update(Services& services, float deltaSeconds) {
    scene::Scene& scene = services.scenes.active();
    gfx::Renderer* renderer = services.renderer;

    if (rigid != nullptr) {
        // 재생을 켜고 끌 때 GPU 에 남은 상태를 버리고 장면 값으로 다시 시작한다. 편집기가 지난 프레임에 바꾼
        // 것을 여기서 알아채므로 아직 읽지 않은 옛 되읽기도 함께 버려진다.
        if (scene.simulating != wasSimulating) {
            rigid->invalidate();
            wasSimulating = scene.simulating;
        }
        // GPU 솔버가 끝낸 결과를 먼저 장면에 되쓴다. 뒤의 scene.refresh 가 이 값으로 세계 변환을 다시 만든다.
        rigid->applyReadback(scene, renderer->completedFrames());
    }

    steps = 0;
    if (scene.simulating) {
        // 프로파일러는 렌더러 것이라 헤드리스에는 없다. 구간을 열지 않고 같은 루프를 돈다.
        std::optional<gfx::ProfilerScope> scope;
        if (renderer != nullptr) {
            scope.emplace(renderer->profiler(), "강체 물리");
        }
        accumulator = std::min(accumulator + deltaSeconds, STEP_SECONDS * static_cast<float>(MAX_STEPS_PER_FRAME));
        while (accumulator >= STEP_SECONDS) {
            physics::stepRigidBodies(scene, STEP_SECONDS, &services.jobs);
            accumulator -= STEP_SECONDS;
            ++steps;
        }
    } else {
        accumulator = 0.0F;
    }

    // 인스펙터가 «지금 도는 백엔드»를 보여 주는 데 쓴다. editor 는 app 을 보지 않으므로 값으로 넘긴다.
    if (services.editor != nullptr && rigid != nullptr) {
        services.editor->rigidStatus.gpuAvailable = rigid->available();
        services.editor->rigidStatus.gpuBodies = rigid->bodyCount();
    }
}

void PhysicsPlugin::ui(Services& services) {
    if (!services.editor->settingsSection("물리")) {
        return;
    }
    ImGui::Text(
        "고정 간격 %.2f ms, 프레임당 최대 %u 스텝", static_cast<double>(STEP_SECONDS * 1000.0F), MAX_STEPS_PER_FRAME);
    ImGui::TextDisabled("GPU 강체 솔버 %s, 지난 프레임 GPU 강체 %u 개",
                        rigid->available() ? "사용 가능" : "미지원",
                        rigid->bodyCount());
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("백엔드는 강체 부품마다 고른다. GPU 는 Jacobi 라 CPU 와 수치가 다르고 결과가 몇 프레임 늦다");
    }
}

} // namespace app
