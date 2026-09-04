#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "core/job_system.h"
#include "physics/fluid_sph.h"

namespace {

// 용기 안에 유체 하나만 있는 장면. 강체는 인자로 넣는다.
//
// 부품 기본값을 그대로 쓴다. 커널 반지름 0.1 이 입자 간격(2r = 0.05)의 두 배인데, 그 비율이
// 이 저장소가 맞춰 둔 값이다. 커널을 간격까지 좁히면 입자 하나의 자기 밀도가 기준 밀도를 넘어
// 가만히 있어도 압력이 생기고 물이 튀어 오른다.
scene::Scene makeScene() {
    scene::Scene scene;
    scene::Object emitter;
    emitter.name = "유체";
    emitter.transform.position = glm::vec3{0.0F, 1.0F, 0.0F};
    scene.objects.push_back(std::move(emitter));
    scene.attachFluid(0);
    scene.refresh();
    return scene;
}

physics::FluidParams paramsFor(const scene::Scene& scene, uint32_t count) {
    return physics::deriveFluidParams(scene.fluids[0], scene.world(0), count, 1024, scene);
}

bool insideContainer(const physics::FluidSolver& solver, const physics::FluidParams& params) {
    for (const glm::vec4& particle : solver.particles()) {
        for (int axis = 0; axis < 3; ++axis) {
            // 벽에서 입자 반지름만큼 안쪽에 붙는다. 부동소수 오차만큼 여유를 둔다.
            if (particle[axis] < params.containerMin[axis] - 1e-3F ||
                particle[axis] > params.containerMax[axis] + 1e-3F) {
                return false;
            }
        }
    }
    return true;
}

} // namespace

int main() {
    // 워커를 넉넉히 둔다. 결정성 검사는 스레드가 많을수록 인터리빙이 다양해져 잘 깨진다.
    core::JobSystem jobs(8);

    // ---- 방출은 상자 안에 격자로 놓는다 ----
    {
        scene::Scene scene = makeScene();
        scene.fluids[0].emitterHalfExtents = glm::vec3{0.4F};
        scene.refresh();
        physics::FluidParams params = paramsFor(scene, 512);

        physics::FluidSolver solver;
        solver.emit(params, 512);
        assert(solver.particleCount() == 512);
        // 격자 반쪽 크기. x, y 는 상자 안이고 남는 입자는 z 로 이어 쌓이므로 z 만 넘칠 수 있다.
        glm::vec3 half = glm::vec3{params.lattice - 1U} * params.spacing * 0.5F;
        for (const glm::vec4& particle : solver.particles()) {
            glm::vec3 local = glm::vec3{particle} - glm::vec3{0.0F, 1.0F, 0.0F};
            assert(std::abs(local.x) <= half.x + 1e-4F && "방출 격자는 x 로 상자 안이다");
            assert(std::abs(local.y) <= half.y + 1e-4F && "방출 격자는 y 로 상자 안이다");
        }
        // 밀도는 기준 밀도로 시작한다. 첫 스텝이 그것을 다시 잰다.
        assert(std::abs(solver.particles()[0].w - params.restDensity) < 1e-3F);
        // 지난 위치는 처음에 지금 위치와 같다. 그래야 모션 벡터가 튀지 않는다. assert 는 매크로라
        // 중괄호 안의 쉼표가 인자 구분으로 읽힌다. 괄호를 하나 더 씌운다.
        assert((glm::vec3{solver.previousParticles()[0]} == glm::vec3{solver.particles()[0]}));
    }

    // ---- 입자는 용기 밖으로 나가지 않는다 ----
    {
        scene::Scene scene = makeScene();
        scene.refresh();
        physics::FluidParams params = paramsFor(scene, 512);

        physics::FluidSolver solver;
        solver.emit(params, 512);
        for (int frame = 0; frame < 60; ++frame) {
            solver.step(params, 1.0F / 60.0F, &jobs);
        }
        assert(insideContainer(solver, params) && "용기 벽이 입자를 붙잡아야 한다");

        // 가라앉은 뒤에는 바닥 근처에 모인다. 중력이 -Y 라 위쪽이 비어야 한다.
        float highest = -1000.0F;
        for (const glm::vec4& particle : solver.particles()) {
            highest = std::max(highest, particle.y);
        }
        assert(highest < 0.6F && "1 초면 바닥에 가라앉는다");
    }

    // ---- 눌린 물은 기준 밀도 언저리로 수렴한다 ----
    {
        scene::Scene scene = makeScene();
        // 좁은 용기에 촘촘히 채워 바닥에 쌓이게 한다.
        scene.fluids[0].emitterHalfExtents = glm::vec3{0.2F};
        scene.fluids[0].containerMin = glm::vec3{-0.25F, 0.0F, -0.25F};
        scene.fluids[0].containerMax = glm::vec3{0.25F, 1.0F, 0.25F};
        scene.objects[0].transform.position = glm::vec3{0.0F, 0.5F, 0.0F};
        scene.refresh();
        physics::FluidParams params = paramsFor(scene, 1024);

        physics::FluidSolver solver;
        solver.emit(params, 1024);
        for (int frame = 0; frame < 120; ++frame) {
            solver.step(params, 1.0F / 60.0F, &jobs);
        }

        // 바닥에 잠긴 입자만 본다. 표면 입자는 이웃이 없어 밀도가 낮게 나오는 것이 정상이다.
        double sum = 0.0;
        uint32_t counted = 0;
        for (const glm::vec4& particle : solver.particles()) {
            if (particle.y < 0.2F) {
                sum += particle.w;
                ++counted;
            }
        }
        assert(counted > 0 && "바닥에 쌓인 입자가 있어야 한다");
        double average = sum / counted;
        assert(average > params.restDensity * 0.6 && "잠긴 물의 밀도가 기준 밀도 언저리여야 한다");
        assert(average < params.restDensity * 1.6);
    }

    // ---- 같은 시작에서 같은 결과가 나온다 ----
    {
        scene::Scene scene = makeScene();
        scene.refresh();
        physics::FluidParams params = paramsFor(scene, 1024);

        physics::FluidSolver first;
        physics::FluidSolver second;
        first.emit(params, 1024);
        second.emit(params, 1024);
        // 갈리려면 시간이 걸린다. 예전 구현은 스물몇 프레임째부터 어긋났으므로 넉넉히 돌린다.
        for (int frame = 0; frame < 150; ++frame) {
            first.step(params, 1.0F / 60.0F, &jobs);
            // 두 번째는 워커 없이 돌린다. 결과가 같아야 스레드 수가 그림을 바꾸지 않는다.
            second.step(params, 1.0F / 60.0F, nullptr);
        }
        for (uint32_t i = 0; i < first.particleCount(); ++i) {
            // 마지막 비트까지 같아야 한다. SPH 는 혼돈계라 «비슷하다» 는 몇십 프레임이면 «다르다» 가 된다.
            assert(first.particles()[i] == second.particles()[i] && "워커 수가 결과를 바꾸면 안 된다");
        }
    }

    // ---- 강체 콜라이더가 입자를 밀어낸다 ----
    {
        scene::Scene scene = makeScene();
        scene::Object ball;
        ball.name = "공";
        ball.transform.position = glm::vec3{0.0F, 0.5F, 0.0F};
        scene.objects.push_back(std::move(ball));
        scene::RigidBody body;
        body.shape = scene::ColliderShape::SPHERE;
        body.radius = 0.3F;
        scene.attachRigidBody(1, body);
        scene.refresh();

        physics::FluidParams params = paramsFor(scene, 512);
        assert(params.colliderCount == 1 && "강체가 콜라이더로 넘어와야 한다");
        assert(params.colliders[0].shape == scene::ColliderShape::SPHERE);
        assert(std::abs(params.colliders[0].radius - 0.3F) < 1e-5F);

        physics::FluidSolver solver;
        solver.emit(params, 512);
        for (int frame = 0; frame < 60; ++frame) {
            solver.step(params, 1.0F / 60.0F, &jobs);
        }
        for (const glm::vec4& particle : solver.particles()) {
            float distance = glm::distance(glm::vec3{particle}, glm::vec3{0.0F, 0.5F, 0.0F});
            assert(distance > 0.3F - 1e-2F && "입자가 구 안으로 들어가면 안 된다");
        }
    }

    // ---- 상수 유도 ----
    {
        scene::Scene scene = makeScene();
        scene.fluids[0].particleRadius = 0.05F;
        scene.fluids[0].restDensity = 1000.0F;
        // 커널이 간격보다 작으면 이웃을 못 찾는다. 유도가 그것을 올려 줘야 한다.
        scene.fluids[0].smoothingRadius = 0.01F;
        scene.refresh();
        physics::FluidParams params = paramsFor(scene, 100);
        assert(std::abs(params.spacing - 0.1F) < 1e-6F);
        assert(std::abs(params.particleMass - 1000.0F * 0.001F) < 1e-4F && "질량은 간격 세제곱의 물이다");
        assert(params.smoothingRadius >= params.spacing && "커널은 간격보다 작을 수 없다");
        assert(params.cellCount == 1024);
    }

    // ---- 서브스텝은 1 과 8 사이다 ----
    {
        physics::FluidParams params;
        assert(physics::fluidSubsteps(params, 0.0F) == 1);
        params.stiffness = 100000.0F;
        assert(physics::fluidSubsteps(params, 1.0F / 60.0F) == 8 && "강성이 크면 잘게 나눈다");
        params.stiffness = 1.0F;
        params.smoothingRadius = 10.0F;
        assert(physics::fluidSubsteps(params, 1.0F / 60.0F) == 1 && "여유가 있으면 한 번이면 된다");
        // 프레임이 길어도 정해진 만큼만 따라잡는다.
        assert(physics::fluidSubsteps(params, 100.0F) <= 8);
    }

    std::printf("유체 SPH 자체 점검 통과\n");
    return 0;
}
