#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

#include "core/job_system.h"
#include "physics/rigid_body.h"
#include "scene/scene.h"

namespace {

constexpr float STEP = 1.0F / 120.0F;

// 원점을 지나는 바닥 평면(법선 +Y) 하나.
uint32_t addFloor(scene::Scene& scene) {
    scene::Object floor;
    floor.name = "바닥";
    scene.objects.push_back(std::move(floor));
    auto index = static_cast<uint32_t>(scene.objects.size() - 1);
    scene::RigidBody body;
    body.shape = scene::ColliderShape::PLANE;
    scene.attachRigidBody(index, body);
    return index;
}

uint32_t addSphere(scene::Scene& scene, glm::vec3 position, float radius, float restitution) {
    scene::Object object;
    object.name = "구";
    object.transform.position = position;
    scene.objects.push_back(std::move(object));
    auto index = static_cast<uint32_t>(scene.objects.size() - 1);
    scene::RigidBody body;
    body.radius = radius;
    body.restitution = restitution;
    scene.attachRigidBody(index, body);
    return index;
}

void simulate(scene::Scene& scene, float seconds, core::JobSystem* jobs) {
    auto steps = static_cast<int>(seconds / STEP);
    for (int i = 0; i < steps; ++i) {
        physics::stepRigidBodies(scene, STEP, jobs);
    }
}

} // namespace

int main() {
    core::JobSystem jobs(2);

    // 구가 바닥에 떨어져 반지름 높이에서 쉰다.
    {
        scene::Scene scene;
        addFloor(scene);
        uint32_t ball = addSphere(scene, glm::vec3{0.0F, 3.0F, 0.0F}, 0.5F, 0.0F);
        simulate(scene, 3.0F, &jobs);
        float y = scene.objects[ball].transform.position.y;
        assert(std::abs(y - 0.5F) < 0.02F && "구는 바닥 위 반지름 높이에서 멈춰야 한다");
        assert(std::abs(scene.rigidBodies[scene.objects[ball].rigidBody].velocity.y) < 0.05F &&
               "쉬는 구는 거의 서 있어야 한다");
        assert(scene.objects[0].transform.position.y == 0.0F && "평면은 움직이지 않는다");
    }

    // 반발이 없으면 닿은 뒤 다시 튀어 오르지 않는다.
    {
        scene::Scene scene;
        addFloor(scene);
        uint32_t ball = addSphere(scene, glm::vec3{0.0F, 2.0F, 0.0F}, 0.5F, 0.0F);
        bool touched = false;
        float highestAfterTouch = 0.0F;
        for (int i = 0; i < 360; ++i) {
            physics::stepRigidBodies(scene, STEP, nullptr);
            float y = scene.objects[ball].transform.position.y;
            if (y < 0.52F) {
                touched = true;
            }
            if (touched) {
                highestAfterTouch = std::max(highestAfterTouch, y);
            }
        }
        assert(touched && "3초면 바닥에 닿아야 한다");
        assert(highestAfterTouch < 0.53F && "반발 0 이면 튀지 않아야 한다");
    }

    // 운동학 물체는 중력에도 제자리다. 그 위에 떨어진 구는 그 위에서 멈춘다.
    {
        scene::Scene scene;
        scene::Object platform;
        platform.transform.position = glm::vec3{0.0F, 1.0F, 0.0F};
        scene.objects.push_back(std::move(platform));
        scene::RigidBody box;
        box.shape = scene::ColliderShape::BOX;
        box.kinematic = true;
        box.halfExtents = glm::vec3{2.0F, 0.1F, 2.0F};
        scene.attachRigidBody(0, box);
        uint32_t ball = addSphere(scene, glm::vec3{0.5F, 3.0F, 0.0F}, 0.25F, 0.0F);
        simulate(scene, 3.0F, &jobs);
        assert(std::abs(scene.objects[0].transform.position.y - 1.0F) < 1e-6F && "운동학 상자는 그대로다");
        assert(std::abs(scene.objects[ball].transform.position.y - 1.35F) < 0.02F && "구는 상자 윗면에서 멈춘다");
    }

    // 같은 질량의 정면 충돌은 운동량을 보존한다.
    {
        scene::Scene scene;
        uint32_t left = addSphere(scene, glm::vec3{-1.0F, 0.0F, 0.0F}, 0.25F, 1.0F);
        uint32_t right = addSphere(scene, glm::vec3{1.0F, 0.0F, 0.0F}, 0.25F, 1.0F);
        scene.rigidBodies[scene.objects[left].rigidBody].useGravity = false;
        scene.rigidBodies[scene.objects[right].rigidBody].useGravity = false;
        scene.rigidBodies[scene.objects[left].rigidBody].velocity = glm::vec3{4.0F, 0.0F, 0.0F};
        simulate(scene, 1.0F, nullptr);
        glm::vec3 momentum = scene.rigidBodies[scene.objects[left].rigidBody].velocity +
                             scene.rigidBodies[scene.objects[right].rigidBody].velocity;
        assert(std::abs(momentum.x - 4.0F) < 0.05F && "운동량 합이 그대로여야 한다");
        assert(scene.rigidBodies[scene.objects[right].rigidBody].velocity.x > 3.0F && "오른쪽 구가 밀려 나가야 한다");
    }

    std::printf("강체 물리 자체 점검 통과\n");
    return 0;
}
