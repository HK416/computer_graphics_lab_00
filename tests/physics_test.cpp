#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>

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

// 임의 모양의 동적 강체. 원기둥·캡슐은 radius 와 halfExtents.y(반높이)를 쓴다.
uint32_t addBody(scene::Scene& scene,
                 scene::ColliderShape shape,
                 glm::vec3 position,
                 glm::quat rotation,
                 float radius,
                 glm::vec3 halfExtents) {
    scene::Object object;
    object.name = "강체";
    object.transform.position = position;
    object.transform.rotation = rotation;
    scene.objects.push_back(std::move(object));
    auto index = static_cast<uint32_t>(scene.objects.size() - 1);
    scene::RigidBody body;
    body.shape = shape;
    body.radius = radius;
    body.halfExtents = halfExtents;
    body.restitution = 0.0F;
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

    // 쉬는 물체는 떨지 않아야 한다. 위치 보정이 과하면 매 스텝 튀어 올랐다 도로 떨어진다.
    {
        scene::Scene scene;
        addFloor(scene);
        uint32_t ball = addSphere(scene, glm::vec3{0.0F, 1.0F, 0.0F}, 0.5F, 0.0F);
        simulate(scene, 3.0F, &jobs);
        float lowest = 1.0e9F;
        float highest = -1.0e9F;
        for (int i = 0; i < 120; ++i) {
            physics::stepRigidBodies(scene, STEP, &jobs);
            float y = scene.objects[ball].transform.position.y;
            lowest = std::min(lowest, y);
            highest = std::max(highest, y);
        }
        std::printf("  쉬는 구 진폭 %.6f\n", static_cast<double>(highest - lowest));
        assert(highest - lowest < 0.002F && "쉬는 구는 떨지 않아야 한다");
    }

    // 동적 상자를 쌓으면 서로 위에 선다. 상자 대 평면과 상자 대 상자를 함께 본다.
    {
        scene::Scene scene;
        addFloor(scene);
        std::array<uint32_t, 3> boxes{};
        for (int i = 0; i < 3; ++i) {
            scene::Object object;
            object.name = "상자";
            object.transform.position = glm::vec3{0.0F, 0.3F + static_cast<float>(i) * 0.55F, 0.0F};
            scene.objects.push_back(std::move(object));
            auto index = static_cast<uint32_t>(scene.objects.size() - 1);
            scene::RigidBody body;
            body.shape = scene::ColliderShape::BOX;
            body.halfExtents = glm::vec3{0.25F};
            body.restitution = 0.05F;
            scene.attachRigidBody(index, body);
            boxes[static_cast<size_t>(i)] = index;
        }
        simulate(scene, 3.0F, &jobs);
        for (int i = 0; i < 3; ++i) {
            float y = scene.objects[boxes[static_cast<size_t>(i)]].transform.position.y;
            float expected = 0.25F + static_cast<float>(i) * 0.5F;
            std::printf("  상자 %d: y=%.3f (기대 %.3f)\n", i, static_cast<double>(y), static_cast<double>(expected));
            assert(std::abs(y - expected) < 0.06F && "쌓은 상자는 서로 위에 서 있어야 한다");
        }
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

    // 서 있는 원기둥은 뚜껑 전체로 바닥에 닿아 넘어지지 않고 반높이에서 쉰다.
    {
        scene::Scene scene;
        addFloor(scene);
        uint32_t cylinder = addBody(scene,
                                    scene::ColliderShape::CYLINDER,
                                    glm::vec3{0.0F, 1.5F, 0.0F},
                                    glm::quat{1.0F, 0.0F, 0.0F, 0.0F},
                                    0.3F,
                                    glm::vec3{0.3F, 0.5F, 0.3F});
        simulate(scene, 3.0F, &jobs);
        const scene::Transform& transform = scene.objects[cylinder].transform;
        std::printf("  서 있는 원기둥 y=%.3f\n", static_cast<double>(transform.position.y));
        assert(std::abs(transform.position.y - 0.5F) < 0.03F && "원기둥은 반높이에서 멈춘다");
        glm::vec3 axis = transform.rotation * glm::vec3{0.0F, 1.0F, 0.0F};
        assert(axis.y > 0.99F && "서 있는 원기둥은 넘어지지 않는다");
    }

    // 누운 원기둥은 옆면(반지름)에서 쉰다. 상자 위에 놓아 원기둥 대 상자도 함께 본다.
    {
        scene::Scene scene;
        scene::Object platform;
        scene.objects.push_back(std::move(platform));
        scene::RigidBody box;
        box.shape = scene::ColliderShape::BOX;
        box.kinematic = true;
        box.halfExtents = glm::vec3{2.0F, 0.5F, 2.0F};
        scene.attachRigidBody(0, box);
        glm::quat onSide = glm::angleAxis(glm::half_pi<float>(), glm::vec3{0.0F, 0.0F, 1.0F});
        uint32_t cylinder = addBody(scene,
                                    scene::ColliderShape::CYLINDER,
                                    glm::vec3{0.0F, 1.5F, 0.0F},
                                    onSide,
                                    0.25F,
                                    glm::vec3{0.25F, 0.6F, 0.25F});
        simulate(scene, 3.0F, &jobs);
        float y = scene.objects[cylinder].transform.position.y;
        std::printf("  누운 원기둥 y=%.3f\n", static_cast<double>(y));
        assert(std::abs(y - 0.75F) < 0.03F && "누운 원기둥은 상자 윗면 위 반지름 높이에서 멈춘다");
    }

    // 캡슐은 서면 반높이 + 반지름, 누우면 반지름에서 쉰다.
    {
        scene::Scene scene;
        addFloor(scene);
        uint32_t standing = addBody(scene,
                                    scene::ColliderShape::CAPSULE,
                                    glm::vec3{-2.0F, 2.0F, 0.0F},
                                    glm::quat{1.0F, 0.0F, 0.0F, 0.0F},
                                    0.25F,
                                    glm::vec3{0.25F, 0.5F, 0.25F});
        glm::quat onSide = glm::angleAxis(glm::half_pi<float>(), glm::vec3{1.0F, 0.0F, 0.0F});
        uint32_t lying = addBody(scene,
                                 scene::ColliderShape::CAPSULE,
                                 glm::vec3{2.0F, 2.0F, 0.0F},
                                 onSide,
                                 0.25F,
                                 glm::vec3{0.25F, 0.5F, 0.25F});
        simulate(scene, 3.0F, &jobs);
        float standingY = scene.objects[standing].transform.position.y;
        float lyingY = scene.objects[lying].transform.position.y;
        std::printf("  캡슐 서서 y=%.3f, 누워서 y=%.3f\n", static_cast<double>(standingY), static_cast<double>(lyingY));
        assert(std::abs(standingY - 0.75F) < 0.03F && "선 캡슐은 반높이 + 반지름에서 멈춘다");
        assert(std::abs(lyingY - 0.25F) < 0.03F && "누운 캡슐은 반지름에서 멈춘다");
    }

    // 구가 캡슐 위에 얹히면 두 반지름 합만큼 떨어져 멈춘다(구 대 캡슐은 표본 기반 접촉이다).
    {
        scene::Scene scene;
        scene::Object post;
        scene.objects.push_back(std::move(post));
        scene::RigidBody capsule;
        capsule.shape = scene::ColliderShape::CAPSULE;
        capsule.kinematic = true;
        capsule.radius = 0.5F;
        capsule.halfExtents = glm::vec3{0.5F, 0.5F, 0.5F};
        scene.attachRigidBody(0, capsule);
        uint32_t ball = addSphere(scene, glm::vec3{0.0F, 3.0F, 0.0F}, 0.25F, 0.0F);
        simulate(scene, 3.0F, &jobs);
        float y = scene.objects[ball].transform.position.y;
        std::printf("  캡슐 위의 구 y=%.3f\n", static_cast<double>(y));
        assert(std::abs(y - 1.25F) < 0.03F && "구는 캡슐 꼭대기에서 멈춘다");
    }

    // 메쉬 콜라이더. 두 삼각형으로 만든 정사각형 위에 구와 상자가 앉는다. 메쉬는 늘 운동학이다.
    {
        std::vector<scene::ColliderMesh> meshes(1);
        scene::ColliderMesh& quad = meshes[0];
        quad.positions = {glm::vec3{-2.0F, 0.0F, -2.0F},
                          glm::vec3{2.0F, 0.0F, -2.0F},
                          glm::vec3{2.0F, 0.0F, 2.0F},
                          glm::vec3{-2.0F, 0.0F, 2.0F}};
        // 위(+Y)에서 보아 반시계가 앞면이다.
        quad.indices = {0, 3, 2, 0, 2, 1};
        quad.boundsRadius = 3.0F;

        scene::Scene scene;
        scene.colliderMeshes = &meshes;
        scene::Object ground;
        ground.transform.position = glm::vec3{0.0F, 1.0F, 0.0F};
        scene.objects.push_back(std::move(ground));
        scene.attachMeshRenderer(0, 0);
        scene::RigidBody meshBody;
        meshBody.shape = scene::ColliderShape::MESH;
        meshBody.mass = 5.0F;
        scene.attachRigidBody(0, meshBody);

        uint32_t ball = addSphere(scene, glm::vec3{0.5F, 3.0F, 0.5F}, 0.25F, 0.0F);
        uint32_t box = addBody(scene,
                               scene::ColliderShape::BOX,
                               glm::vec3{-0.8F, 3.0F, 0.0F},
                               glm::quat{1.0F, 0.0F, 0.0F, 0.0F},
                               0.5F,
                               glm::vec3{0.25F});
        simulate(scene, 3.0F, &jobs);
        float ballY = scene.objects[ball].transform.position.y;
        float boxY = scene.objects[box].transform.position.y;
        std::printf("  메쉬 위의 구 y=%.3f, 상자 y=%.3f\n", static_cast<double>(ballY), static_cast<double>(boxY));
        assert(std::abs(scene.objects[0].transform.position.y - 1.0F) < 1e-6F && "메쉬 콜라이더는 밀리지 않는다");
        assert(std::abs(ballY - 1.25F) < 0.03F && "구는 메쉬 위 반지름 높이에서 멈춘다");
        assert(std::abs(boxY - 1.25F) < 0.03F && "상자는 메쉬 위 반쪽 크기 높이에서 멈춘다");
    }

    // 광역 격자가 O(n²) 참조와 같은 짝을 내고, 워커 수와 무관하게 같은 순서인지.
    {
        std::mt19937 random(1234);
        std::uniform_real_distribution<float> position(-20.0F, 20.0F);
        std::uniform_real_distribution<float> radius(0.1F, 1.0F);
        std::vector<physics::RigidBodyState> bodies;
        for (int i = 0; i < 300; ++i) {
            physics::RigidBodyState body;
            body.shape = i % 3 == 0 ? scene::ColliderShape::SPHERE : scene::ColliderShape::BOX;
            body.position = glm::vec3{position(random), position(random), position(random)};
            body.boundingRadius = radius(random);
            // 열에 하나는 정적이다. 정적끼리는 짝이 아니다.
            body.inverseMass = i % 10 == 0 ? 0.0F : 1.0F;
            bodies.push_back(body);
        }
        physics::RigidBodyState plane;
        plane.shape = scene::ColliderShape::PLANE;
        bodies.push_back(plane);

        std::vector<std::pair<uint32_t, uint32_t>> reference;
        for (uint32_t i = 0; i < bodies.size(); ++i) {
            for (uint32_t j = i + 1; j < bodies.size(); ++j) {
                const physics::RigidBodyState& a = bodies[i];
                const physics::RigidBodyState& b = bodies[j];
                if (!physics::isDynamic(a) && !physics::isDynamic(b)) {
                    continue;
                }
                bool planePair = a.shape == scene::ColliderShape::PLANE || b.shape == scene::ColliderShape::PLANE;
                float reach = a.boundingRadius + b.boundingRadius;
                if (!planePair && glm::dot(a.position - b.position, a.position - b.position) > reach * reach) {
                    continue;
                }
                reference.emplace_back(i, j);
            }
        }
        std::vector<std::pair<uint32_t, uint32_t>> parallel;
        std::vector<std::pair<uint32_t, uint32_t>> serial;
        physics::collectPairs(bodies, &jobs, parallel);
        physics::collectPairs(bodies, nullptr, serial);
        assert(!reference.empty());
        assert(parallel == reference);
        assert(serial == reference);
        std::printf("  광역 격자 짝 %zu 개 (참조와 같음)\n", parallel.size());
    }

    // 상자 더미 192개. 격자가 짝을 놓치면 관통하거나 바닥을 뚫는다.
    {
        scene::Scene scene;
        addFloor(scene);
        std::vector<uint32_t> boxes;
        for (int x = 0; x < 8; ++x) {
            for (int y = 0; y < 3; ++y) {
                for (int z = 0; z < 8; ++z) {
                    boxes.push_back(addBody(scene,
                                            scene::ColliderShape::BOX,
                                            glm::vec3{static_cast<float>(x) * 1.1F - 3.85F,
                                                      0.5F + static_cast<float>(y) * 1.05F,
                                                      static_cast<float>(z) * 1.1F - 3.85F},
                                            glm::quat{1.0F, 0.0F, 0.0F, 0.0F},
                                            0.5F,
                                            glm::vec3{0.5F}));
                }
            }
        }
        simulate(scene, 2.0F, &jobs);
        for (uint32_t box : boxes) {
            assert(scene.objects[box].transform.position.y > 0.4F);
        }
        for (size_t a = 0; a < boxes.size(); ++a) {
            for (size_t b = a + 1; b < boxes.size(); ++b) {
                glm::vec3 delta =
                    scene.objects[boxes[a]].transform.position - scene.objects[boxes[b]].transform.position;
                // 한 변 1 인 상자끼리 중심 거리가 0.9 아래면 서로 깊이 파고든 것이다.
                assert(glm::dot(delta, delta) > 0.9F * 0.9F);
            }
        }
        std::printf("  상자 192개 더미 2초 뒤 관통 없음\n");
    }

    std::printf("강체 물리 자체 점검 통과\n");
    return 0;
}
