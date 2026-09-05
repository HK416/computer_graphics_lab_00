#include "app/plugins/physics_plugin.h"

#include <algorithm>

#include "gfx/profiler.h"
#include "physics/rigid_body.h"

namespace app {

void PhysicsPlugin::update(Services& services, float deltaSeconds) {
    scene::Scene& scene = services.scenes.active();
    gfx::Renderer& renderer = services.renderer;

    // GPU 솔버가 끝낸 결과를 먼저 장면에 되쓴다. 뒤의 scene.refresh 가 이 값으로 세계 변환을 다시 만든다.
    renderer.applyRigidBodyReadback(scene);
    uint32_t steps = 0;
    if (scene.simulating) {
        gfx::ProfilerScope scope(renderer.profiler(), "강체 물리");
        accumulator = std::min(accumulator + deltaSeconds, STEP_SECONDS * static_cast<float>(MAX_STEPS_PER_FRAME));
        while (accumulator >= STEP_SECONDS) {
            physics::stepRigidBodies(scene, STEP_SECONDS, &services.jobs);
            accumulator -= STEP_SECONDS;
            ++steps;
        }
    } else {
        accumulator = 0.0F;
    }
    // GPU 백엔드 강체는 같은 간격으로 렌더러가 푼다.
    renderer.setRigidBodySteps(steps, STEP_SECONDS);
}

} // namespace app
