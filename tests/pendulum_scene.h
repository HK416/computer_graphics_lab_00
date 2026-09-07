#pragma once

// 진자 장면. **두 학습법이 문자 그대로 같은 장면을 봐야** 점수를 견줄 수 있어서 한 곳에 둔다 —
// robot_test 의 진화 전략과 rl_agent_test 의 DDPG 가 이것을 함께 쓴다. 한쪽에만 두면 «같은 장면» 이라는
// 말이 주석으로만 남고, 막대 길이나 최대 토크가 갈리는 순간 두 점수는 견줄 수 없는 것이 된다.
#include <cstdint>

#include <glm/glm.hpp>

#include "scene/scene.h"

namespace {

// 세계에 경첩으로 매달린 막대 하나. 축은 +Z 라 XY 평면 안에서 돈다. 각이 0 이면 막대가 +X 로 눕고,
// 자세를 주지 않으면 중력에 끌려 내려간다.
uint32_t addBar(scene::Scene& scene, glm::vec3 pivot, float halfLength, scene::JointMotor motor) {
    scene::Object object;
    object.name = "막대";
    object.transform.position = pivot + glm::vec3{halfLength, 0.0F, 0.0F};
    scene.objects.push_back(std::move(object));
    auto index = static_cast<uint32_t>(scene.objects.size() - 1);

    scene::RigidBody body;
    body.shape = scene::ColliderShape::BOX;
    body.halfExtents = glm::vec3{halfLength, 0.05F, 0.05F};
    body.mass = 1.0F;
    scene.attachRigidBody(index, body);

    scene::Joint joint;
    joint.type = scene::JointType::HINGE;
    joint.other = -1;
    joint.anchorA = glm::vec3{-halfLength, 0.0F, 0.0F};
    joint.anchorB = pivot;
    joint.axis = glm::vec3{0.0F, 0.0F, 1.0F};
    joint.motor = motor;
    joint.targetSpeed = 360.0F;
    joint.maxTorque = 4.0F;
    // 거꾸로 선 자세가 목표다. 보상이 이 값을 본다.
    joint.targetAngle = 180.0F;
    scene.attachJoint(index, joint);
    return index;
}

scene::Scene makePendulum(scene::JointMotor motor = scene::JointMotor::VELOCITY) {
    scene::Scene scene;
    addBar(scene, glm::vec3{0.0F, 2.0F, 0.0F}, 0.5F, motor);
    scene.refresh();
    return scene;
}

} // namespace
