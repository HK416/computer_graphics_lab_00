#include "app/plugins/robot_plugin.h"

#include <algorithm>
#include <chrono>
#include <string>

#include <imgui.h>
#include <spdlog/spdlog.h>

#include "app/application.h"

namespace app {

void RobotPlugin::build(Services& services) {
    scene::Scene& scene = services.scenes.active();
    rebuildLayout(scene);
    if (layout.empty()) {
        if (services.options.train) {
            spdlog::warn("--train: 장면에 몰 수 있는 경첩 모터가 없어 학습할 것이 없다");
        }
        return;
    }
    physics::TrainingSettings settings;
    settings.generations = services.options.generations;
    settings.population = services.options.population;
    settings.rollout.frames = services.options.rolloutFrames;
    policy = std::make_unique<physics::Policy>(layout.observationCount(), settings.hidden, layout.actionCount());

    std::string path = services.options.policyPath.generic_string();
    if (!services.options.train) {
        if (path.empty()) {
            policy.reset();
            return;
        }
        if (!physics::loadPolicy(*policy, path)) {
            spdlog::warn("정책 파일을 읽지 못했다(모양이 다르거나 없는 파일): {}", path);
            policy.reset();
        } else {
            spdlog::info("정책 적재: {} (관측 {}, 행동 {})", path, layout.observationCount(), layout.actionCount());
        }
        return;
    }

    uint32_t gpuActuators = 0;
    for (const physics::Actuator& actuator : layout.actuators) {
        const scene::RigidBody* body = scene.component<scene::RigidBody>(actuator.object);
        gpuActuators += body != nullptr && body->backend == scene::SimulationBackend::GPU ? 1U : 0U;
    }
    if (gpuActuators > 0) {
        // 롤아웃은 physics::stepRigidBodies 만 부른다. GPU 백엔드 강체는 그 안에서 건너뛰므로 장면이
        // 한 걸음도 움직이지 않고, 점수가 전부 같아져 학습이 무작위 걸음이 된다.
        spdlog::warn("--train: 액추에이터 {} 개가 GPU 백엔드다. 롤아웃은 CPU 솔버만 타므로 그 관절은 "
                     "움직이지 않는다 — 부품의 백엔드를 CPU 로 두고 학습해라",
                     gpuActuators);
    }
    spdlog::info("정책 학습 시작: 액추에이터 {} 개, 세대 {}, 표본 {}, 롤아웃 {} 프레임",
                 layout.actionCount(),
                 settings.generations,
                 settings.population,
                 settings.rollout.frames);
    auto started = std::chrono::steady_clock::now();
    // 롤아웃은 시작 장면의 사본 위에서 돈다. 지금 도는 장면은 건드리지 않는다.
    scene::Scene initial = scene;
    uint32_t reportEvery = std::max(settings.generations / 20U, 1U);
    *policy =
        physics::trainRobot(initial, layout, settings, &services.jobs, [&](const physics::TrainingProgress& progress) {
            lastScore = progress.center;
            lastGenerations = progress.generation;
            if (progress.generation % reportEvery == 0 || progress.generation == settings.generations) {
                spdlog::info("세대 {}/{}: 중심 {:.3f}, 최고 {:.3f}, 평균 {:.3f}",
                             progress.generation,
                             settings.generations,
                             static_cast<double>(progress.center),
                             static_cast<double>(progress.best),
                             static_cast<double>(progress.mean));
            }
        });
    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    spdlog::info("학습 끝: {:.1f} 초, 점수 {:.3f}", elapsed, static_cast<double>(lastScore));
    if (!path.empty() && physics::savePolicy(*policy, path)) {
        spdlog::info("정책 저장: {}", path);
    } else if (!path.empty()) {
        spdlog::warn("정책을 저장하지 못했다: {}", path);
    }
}

// 자리를 다시 만든다. 정책이 몰던 장면이면 저작한 모터 목표가 이미 «마지막 행동» 으로 덮여 있으므로,
// 되돌린 뒤에 읽어야 한다. 그러지 않으면 상한이 프레임마다 조금씩 줄어 로봇이 조용히 힘을 잃는다.
void RobotPlugin::rebuildLayout(scene::Scene& scene) {
    if (!layout.empty()) {
        physics::restoreAuthoredMotors(scene, layout);
    }
    layout = physics::buildRobotLayout(scene);
    layoutSceneId = scene.id;
    layoutComponents = scene.componentRevision();
    observation.assign(layout.observationCount(), 0.0F);
    action.assign(layout.actionCount(), 0.0F);
}

void RobotPlugin::applyPolicy(scene::Scene& scene) {
    physics::observe(scene, layout, observation.data());
    policy->evaluate(observation.data(), action.data());
    physics::act(scene, layout, action.data());
}

void RobotPlugin::update(Services& services, float deltaSeconds) {
    (void)deltaSeconds;
    scene::Scene& scene = services.scenes.active();
    // 장면이 바뀌었으면(탭 전환, 부품 추가·삭제) 자리를 다시 만든다. 액추에이터 수가 달라지면 정책의
    // 모양이 맞지 않으므로 그대로 놓는다.
    if (scene.id != layoutSceneId || scene.componentRevision() != layoutComponents) {
        rebuildLayout(scene);
        shapeMismatch = false;
    }
    if (!driving || policy == nullptr || !scene.simulating || layout.empty()) {
        return;
    }
    if (policy->inputCount() != layout.observationCount() || policy->outputCount() != layout.actionCount()) {
        // 장면이 바뀌어 액추에이터 수가 달라졌다. 조용히 멈추면 «몰고 있다» 고 오해하므로 한 번 적는다.
        if (!shapeMismatch) {
            spdlog::warn("정책의 모양이 장면과 다르다(관측 {} 대 {}, 행동 {} 대 {}). 몰지 않는다",
                         policy->inputCount(),
                         layout.observationCount(),
                         policy->outputCount(),
                         layout.actionCount());
            shapeMismatch = true;
        }
        return;
    }
    applyPolicy(scene);
}

void RobotPlugin::ui(Services& services) {
    if (!services.editor->settingsSection("로봇")) {
        return;
    }
    ImGui::Text("액추에이터 %u 개 (모터가 달린 경첩)", layout.actionCount());
    if (policy == nullptr) {
        ImGui::TextDisabled("정책 없음. --train 으로 학습하거나 --policy 로 읽는다");
        return;
    }
    if (shapeMismatch) {
        ImGui::TextDisabled("정책의 모양이 장면과 달라 몰지 않는다");
    }
    if (ImGui::Checkbox("정책이 몬다", &driving) && !driving) {
        // 끄면 저작한 모터 목표를 돌려놓는다. 그러지 않으면 마지막 행동이 그대로 남는다.
        physics::restoreAuthoredMotors(services.scenes.active(), layout);
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("끄면 장면에 적힌 모터 목표가 그대로 쓰인다");
    }
    if (lastGenerations > 0) {
        ImGui::TextDisabled("학습 %u 세대, 점수 %.3f", lastGenerations, static_cast<double>(lastScore));
    }
    if (!action.empty()) {
        ImGui::TextDisabled("행동 [0] %.3f", static_cast<double>(action[0]));
    }
}

} // namespace app
