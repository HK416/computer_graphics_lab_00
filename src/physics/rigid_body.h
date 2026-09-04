#pragma once

#include <cstdint>
#include <vector>

#include <glm/gtc/quaternion.hpp>
#include <glm/vec3.hpp>

#include "scene/scene.h"

namespace core {
class JobSystem;
} // namespace core

namespace physics {

// CPU 솔버와 GPU 솔버가 **함께 쓰는** 상수. 한쪽만 고치면 백엔드를 바꿀 때 거동이 달라진다.
// GPU 쪽은 gfx::RigidBodySimulator 가 이 값을 푸시 상수에 실어 보낸다.
inline constexpr float GRAVITY = -9.81F;
// Baumgarte 위치 보정 비율과 허용 침투. 너무 크면 튀고 너무 작으면 서서히 가라앉는다.
inline constexpr float POSITION_CORRECTION = 0.2F;
inline constexpr float PENETRATION_SLOP = 0.005F;
// 이보다 느리게 닿으면 반발을 주지 않는다. 쉬고 있는 물체가 미세하게 떨리는 것을 막는다.
inline constexpr float RESTITUTION_THRESHOLD = 1.0F;
// 위치 보정을 몇 번 도는지. 한 번만 돌면 쌓인 물체가 서서히 가라앉는다.
inline constexpr uint32_t POSITION_ITERATIONS = 8;
// 한 짝이 낼 수 있는 접촉 점 수. 나란히 놓인 상자의 면 접촉이 네 점이다. GLSL 의 RIGID_MAX_MANIFOLD
// 와 같아야 두 백엔드가 같은 접촉을 본다.
inline constexpr size_t MAX_MANIFOLD_POINTS = 4;

// 강체 하나의 세계 공간 상태. 부품과 오브젝트 변환에서 펴낸 것이라 솔버는 이것만 본다. CPU 솔버와
// GPU 솔버가 같은 함수로 만든 같은 값을 봐야 백엔드를 바꿔도 같은 물체가 나온다.
struct RigidBodyState {
    uint32_t object = 0;
    scene::ColliderShape shape = scene::ColliderShape::SPHERE;
    glm::vec3 position{0.0F};
    glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
    // 오브젝트의 배율. 되쓸 때 그대로 돌려준다.
    glm::vec3 scale{1.0F};
    glm::vec3 velocity{0.0F};
    glm::vec3 angularVelocity{0.0F};
    float inverseMass = 0.0F;
    // 지역 축의 관성 역수. 세계 공간에서는 회전으로 감싼다.
    glm::vec3 inverseInertia{0.0F};
    float radius = 0.5F;
    glm::vec3 halfExtents{0.5F};
    float restitution = 0.3F;
    float friction = 0.5F;
    // 광역 검사용 경계 반지름. 평면은 무한이라 따로 다룬다.
    float boundingRadius = 0.0F;
    bool useGravity = true;
};

// 힘을 받는 물체인지. 운동학 물체와 평면은 밀리지 않는다.
inline bool isDynamic(const RigidBodyState& body) {
    return body.inverseMass > 0.0F;
}

// backend 가 붙은 강체 부품을 모아 세계 공간 상태로 편다. AUTO 는 하드웨어가 정한 기본값이므로
// 부르는 쪽이 CPU 또는 GPU 로 풀어 넘긴다.
void collectRigidBodies(const scene::Scene& scene, scene::SimulationBackend backend, std::vector<RigidBodyState>& out);

// 세계 상태를 오브젝트의 지역 변환과 부품 속도로 되돌려 쓴다. 부모 변환을 읽으므로 직렬이다.
void writeBackRigidBodies(scene::Scene& scene, const std::vector<RigidBodyState>& bodies);

// 강체 부품이 붙은 오브젝트를 dt 만큼 진행시킨다. 세계 공간에서 적분하고 충돌을 풀어 오브젝트의 지역
// 변환에 되돌려 쓴다. 고정 간격으로 나눠 부르는 것은 부르는 쪽 몫이다. jobs 가 있으면 적분·광역
// 검사·되돌려 쓰기를 워커에 나누고, 접촉 목록은 원자 카운터로 모아 잠금이 없다.
//
// 구·상자·평면 콜라이더, 순차 임펄스 접촉 해결(반발·마찰) 뒤 Baumgarte 위치 보정. 상자끼리는 여섯
// 면 축의 분리축 검사로 접촉을 만든다. 관절이나 슬립은 없다.
// GPU 백엔드로 표시된 강체는 건너뛴다. 그쪽은 gfx::RigidBodySimulator 가 푼다.
void stepRigidBodies(scene::Scene& scene, float dt, core::JobSystem* jobs);

} // namespace physics
