#include "physics/rigid_body.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <functional>
#include <vector>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/norm.hpp>

#include "core/job_system.h"
#include "scene/scene.h"

namespace physics {
namespace {

constexpr float GRAVITY = -9.81F;
constexpr uint32_t SOLVER_ITERATIONS = 8;
// Baumgarte 위치 보정 비율과 허용 침투. 너무 크면 튀고 너무 작으면 서서히 가라앉는다.
constexpr float POSITION_CORRECTION = 0.2F;
constexpr float PENETRATION_SLOP = 0.005F;
// 이보다 느리게 닿으면 반발을 주지 않는다. 쉬고 있는 물체가 미세하게 떨리는 것을 막는다.
constexpr float RESTITUTION_THRESHOLD = 1.0F;
constexpr uint32_t GRANULARITY = 16;

// 시뮬레이션 동안의 세계 공간 상태. 오브젝트와 부품은 끝날 때 한 번만 건드린다.
struct Body {
    uint32_t object = 0;
    scene::ColliderShape shape = scene::ColliderShape::SPHERE;
    glm::vec3 position{0.0F};
    glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
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

// 접촉 하나. 법선은 a 에서 b 를 향한다.
struct Contact {
    uint32_t a = 0;
    uint32_t b = 0;
    glm::vec3 point{0.0F};
    glm::vec3 normal{0.0F, 1.0F, 0.0F};
    float penetration = 0.0F;
    float normalImpulse = 0.0F;
    // 풀기 전에 잰 접근 속도로 정한 반발 목표 속도. 반복마다 다시 재면 첫 반복이 튀긴 뒤 두 번째가 도로
    // 당겨 와 반발이 사라진다.
    float restitutionTarget = 0.0F;
};

void forRange(core::JobSystem* jobs, uint32_t count, const std::function<void(uint32_t, uint32_t)>& body) {
    if (jobs != nullptr && count > GRANULARITY) {
        jobs->parallelFor(count, GRANULARITY, body);
    } else if (count > 0) {
        body(0, count);
    }
}

glm::mat3 worldInverseInertia(const Body& body) {
    glm::mat3 rotation = glm::mat3_cast(body.rotation);
    return rotation *
           glm::mat3{body.inverseInertia.x,
                     0.0F,
                     0.0F,
                     0.0F,
                     body.inverseInertia.y,
                     0.0F,
                     0.0F,
                     0.0F,
                     body.inverseInertia.z} *
           glm::transpose(rotation);
}

bool isDynamic(const Body& body) {
    return body.inverseMass > 0.0F;
}

// 상자 지역 공간에서 점까지의 가장 가까운 표면점과 침투. 점이 안이면 가장 가까운 면으로 밀어낸다.
// 돌려주는 값: 침투 깊이(양수면 겹침), point 와 normal 은 세계 공간.
bool closestOnBox(const Body& box, const glm::vec3& worldPoint, float radius, Contact& contact) {
    glm::quat inverse = glm::conjugate(box.rotation);
    glm::vec3 local = inverse * (worldPoint - box.position);
    glm::vec3 clamped = glm::clamp(local, -box.halfExtents, box.halfExtents);
    glm::vec3 delta = local - clamped;
    float distanceSq = glm::length2(delta);
    if (distanceSq > 1.0e-10F) {
        // 바깥 점.
        float distance = std::sqrt(distanceSq);
        if (distance >= radius) {
            return false;
        }
        glm::vec3 normal = box.rotation * (delta / distance);
        contact.normal = normal;
        contact.point = box.position + box.rotation * clamped;
        contact.penetration = radius - distance;
        return true;
    }
    // 안쪽 점. 가장 얕은 면으로 나간다.
    glm::vec3 gap = box.halfExtents - glm::abs(local);
    int axis = gap.x < gap.y ? (gap.x < gap.z ? 0 : 2) : (gap.y < gap.z ? 1 : 2);
    glm::vec3 localNormal{0.0F};
    localNormal[axis] = local[axis] >= 0.0F ? 1.0F : -1.0F;
    glm::vec3 surface = local;
    surface[axis] = localNormal[axis] * box.halfExtents[axis];
    contact.normal = box.rotation * localNormal;
    contact.point = box.position + box.rotation * surface;
    contact.penetration = gap[axis] + radius;
    return true;
}

// 상자 여덟 꼭짓점.
void boxCorners(const Body& box, glm::vec3 corners[8]) {
    for (int i = 0; i < 8; ++i) {
        glm::vec3 local{(i & 1) != 0 ? box.halfExtents.x : -box.halfExtents.x,
                        (i & 2) != 0 ? box.halfExtents.y : -box.halfExtents.y,
                        (i & 4) != 0 ? box.halfExtents.z : -box.halfExtents.z};
        corners[i] = box.position + box.rotation * local;
    }
}

// 평면의 법선(+Y)과 그 위의 한 점.
glm::vec3 planeNormal(const Body& plane) {
    return plane.rotation * glm::vec3{0.0F, 1.0F, 0.0F};
}

void collideSphereSphere(const Body& a, const Body& b, uint32_t ia, uint32_t ib, std::vector<Contact>& out) {
    glm::vec3 delta = b.position - a.position;
    float distanceSq = glm::length2(delta);
    float reach = a.radius + b.radius;
    if (distanceSq >= reach * reach) {
        return;
    }
    float distance = std::sqrt(std::max(distanceSq, 1.0e-12F));
    Contact contact;
    contact.a = ia;
    contact.b = ib;
    contact.normal = distance > 1.0e-6F ? delta / distance : glm::vec3{0.0F, 1.0F, 0.0F};
    contact.point = a.position + contact.normal * a.radius;
    contact.penetration = reach - distance;
    out.push_back(contact);
}

void collideSphereBox(const Body& sphere, const Body& box, uint32_t is, uint32_t ib, std::vector<Contact>& out) {
    Contact contact;
    if (!closestOnBox(box, sphere.position, sphere.radius, contact)) {
        return;
    }
    // closestOnBox 의 법선은 상자에서 구를 향한다. a=구, b=상자 이므로 뒤집는다.
    contact.a = is;
    contact.b = ib;
    contact.normal = -contact.normal;
    out.push_back(contact);
}

void collideSpherePlane(const Body& sphere, const Body& plane, uint32_t is, uint32_t ip, std::vector<Contact>& out) {
    glm::vec3 normal = planeNormal(plane);
    float distance = glm::dot(sphere.position - plane.position, normal);
    if (distance >= sphere.radius) {
        return;
    }
    Contact contact;
    contact.a = is;
    contact.b = ip;
    contact.normal = -normal;
    contact.point = sphere.position - normal * sphere.radius;
    contact.penetration = sphere.radius - distance;
    out.push_back(contact);
}

void collideBoxPlane(const Body& box, const Body& plane, uint32_t ib, uint32_t ip, std::vector<Contact>& out) {
    glm::vec3 normal = planeNormal(plane);
    glm::vec3 corners[8];
    boxCorners(box, corners);
    for (const glm::vec3& corner : corners) {
        float distance = glm::dot(corner - plane.position, normal);
        if (distance >= 0.0F) {
            continue;
        }
        Contact contact;
        contact.a = ib;
        contact.b = ip;
        contact.normal = -normal;
        contact.point = corner;
        contact.penetration = -distance;
        out.push_back(contact);
    }
}

// ponytail: 상자끼리는 서로의 꼭짓점이 상대 안에 들어왔는지로만 본다. 모서리끼리 걸치는 경우를
// 놓치므로 정확히 하려면 분리축(SAT) 검사로 바꾼다.
void collideBoxBox(const Body& a, const Body& b, uint32_t ia, uint32_t ib, std::vector<Contact>& out) {
    glm::vec3 corners[8];
    boxCorners(a, corners);
    for (const glm::vec3& corner : corners) {
        Contact contact;
        if (closestOnBox(b, corner, 0.0F, contact)) {
            contact.a = ia;
            contact.b = ib;
            contact.normal = -contact.normal;
            contact.point = corner;
            out.push_back(contact);
        }
    }
    boxCorners(b, corners);
    for (const glm::vec3& corner : corners) {
        Contact contact;
        if (closestOnBox(a, corner, 0.0F, contact)) {
            contact.a = ia;
            contact.b = ib;
            contact.point = corner;
            out.push_back(contact);
        }
    }
}

void collide(const std::vector<Body>& bodies, uint32_t ia, uint32_t ib, std::vector<Contact>& out) {
    using scene::ColliderShape;
    const Body& a = bodies[ia];
    const Body& b = bodies[ib];
    // 평면끼리는 만나지 않는다. 짝은 항상 (구/상자, 무엇이든) 순서로 맞춘다.
    if (a.shape == ColliderShape::PLANE) {
        if (b.shape == ColliderShape::PLANE) {
            return;
        }
        collide(bodies, ib, ia, out);
        return;
    }
    if (a.shape == ColliderShape::SPHERE) {
        switch (b.shape) {
        case ColliderShape::SPHERE:
            collideSphereSphere(a, b, ia, ib, out);
            break;
        case ColliderShape::BOX:
            collideSphereBox(a, b, ia, ib, out);
            break;
        case ColliderShape::PLANE:
            collideSpherePlane(a, b, ia, ib, out);
            break;
        }
        return;
    }
    // a 는 상자.
    switch (b.shape) {
    case ColliderShape::SPHERE:
        collideSphereBox(b, a, ib, ia, out);
        break;
    case ColliderShape::BOX:
        collideBoxBox(a, b, ia, ib, out);
        break;
    case ColliderShape::PLANE:
        collideBoxPlane(a, b, ia, ib, out);
        break;
    }
}

void solveContacts(std::vector<Body>& bodies, std::vector<Contact>& contacts, float dt) {
    for (Contact& contact : contacts) {
        const Body& a = bodies[contact.a];
        const Body& b = bodies[contact.b];
        glm::vec3 velocityA = a.velocity + glm::cross(a.angularVelocity, contact.point - a.position);
        glm::vec3 velocityB = b.velocity + glm::cross(b.angularVelocity, contact.point - b.position);
        float closing = glm::dot(velocityB - velocityA, contact.normal);
        float restitution = std::min(a.restitution, b.restitution);
        contact.restitutionTarget = closing < -RESTITUTION_THRESHOLD ? -restitution * closing : 0.0F;
    }
    for (uint32_t iteration = 0; iteration < SOLVER_ITERATIONS; ++iteration) {
        for (Contact& contact : contacts) {
            Body& a = bodies[contact.a];
            Body& b = bodies[contact.b];
            float totalInverseMass = a.inverseMass + b.inverseMass;
            if (totalInverseMass <= 0.0F) {
                continue;
            }
            glm::vec3 armA = contact.point - a.position;
            glm::vec3 armB = contact.point - b.position;
            glm::mat3 inertiaA = worldInverseInertia(a);
            glm::mat3 inertiaB = worldInverseInertia(b);

            auto relativeVelocity = [&]() {
                glm::vec3 velocityA = a.velocity + glm::cross(a.angularVelocity, armA);
                glm::vec3 velocityB = b.velocity + glm::cross(b.angularVelocity, armB);
                return velocityB - velocityA;
            };
            auto effectiveMass = [&](const glm::vec3& direction) {
                glm::vec3 torqueA = glm::cross(armA, direction);
                glm::vec3 torqueB = glm::cross(armB, direction);
                float angular = glm::dot(torqueA, inertiaA * torqueA) + glm::dot(torqueB, inertiaB * torqueB);
                return 1.0F / (totalInverseMass + angular);
            };
            auto applyImpulse = [&](const glm::vec3& impulse) {
                a.velocity -= impulse * a.inverseMass;
                a.angularVelocity -= inertiaA * glm::cross(armA, impulse);
                b.velocity += impulse * b.inverseMass;
                b.angularVelocity += inertiaB * glm::cross(armB, impulse);
            };

            // 법선 임펄스. 접근 속도를 없애고, 빠르게 닿았으면 반발 목표까지 더한다. 침투는 속도에 편향을
            // 섞지 않고 아래의 위치 보정이 따로 밀어낸다. 편향을 속도에 넣으면 쉬고 있는 물체가 그만큼의
            // 속도를 계속 들고 있어 미세하게 떨린다.
            float closing = glm::dot(relativeVelocity(), contact.normal);
            float lambda = effectiveMass(contact.normal) * (-closing + contact.restitutionTarget);
            float accumulated = std::max(contact.normalImpulse + lambda, 0.0F);
            lambda = accumulated - contact.normalImpulse;
            contact.normalImpulse = accumulated;
            applyImpulse(contact.normal * lambda);

            // 마찰. 접선 속도를 줄이되 쿨롱 원뿔(μ·법선 임펄스) 안에서만.
            glm::vec3 relative = relativeVelocity();
            glm::vec3 tangent = relative - contact.normal * glm::dot(relative, contact.normal);
            float tangentSpeed = glm::length(tangent);
            if (tangentSpeed > 1.0e-6F) {
                tangent /= tangentSpeed;
                float friction = std::sqrt(a.friction * b.friction);
                float tangentLambda = -effectiveMass(tangent) * tangentSpeed;
                float limit = friction * contact.normalImpulse;
                tangentLambda = std::clamp(tangentLambda, -limit, limit);
                applyImpulse(tangent * tangentLambda);
            }
        }
    }

    // 남은 침투를 위치로 밀어낸다. 속도는 건드리지 않으므로 쉬는 물체가 떨지 않는다. 질량 역수 비율로
    // 나눠 무거운 쪽이 덜 밀린다.
    for (const Contact& contact : contacts) {
        Body& a = bodies[contact.a];
        Body& b = bodies[contact.b];
        float totalInverseMass = a.inverseMass + b.inverseMass;
        float depth = contact.penetration - PENETRATION_SLOP;
        if (totalInverseMass <= 0.0F || depth <= 0.0F) {
            continue;
        }
        glm::vec3 correction = contact.normal * (POSITION_CORRECTION * depth / totalInverseMass);
        a.position -= correction * a.inverseMass;
        b.position += correction * b.inverseMass;
    }
}

} // namespace

void stepRigidBodies(scene::Scene& scene, float dt, core::JobSystem* jobs) {
    if (dt <= 0.0F) {
        return;
    }
    // 1) 부품이 붙은 오브젝트를 모아 세계 공간 상태로 편다.
    std::vector<Body> bodies;
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        int32_t slot = scene.objects[index].rigidBody;
        if (slot < 0 || static_cast<size_t>(slot) >= scene.rigidBodies.size()) {
            continue;
        }
        Body body;
        body.object = index;
        bodies.push_back(body);
    }
    if (bodies.empty()) {
        return;
    }

    forRange(jobs, static_cast<uint32_t>(bodies.size()), [&](uint32_t begin, uint32_t end) {
        for (uint32_t i = begin; i < end; ++i) {
            Body& body = bodies[i];
            const scene::RigidBody& component =
                scene.rigidBodies[static_cast<size_t>(scene.objects[body.object].rigidBody)];
            scene::Transform world = scene::Transform::fromMatrix(scene.worldMatrix(body.object));
            body.shape = component.shape;
            body.position = world.position;
            body.rotation = glm::normalize(world.rotation);
            body.scale = world.scale;
            body.velocity = component.velocity;
            body.angularVelocity = component.angularVelocity;
            body.restitution = component.restitution;
            body.friction = component.friction;
            body.useGravity = component.useGravity;
            float scaleMax = std::max({world.scale.x, world.scale.y, world.scale.z});
            body.radius = component.radius * scaleMax;
            body.halfExtents = component.halfExtents * world.scale;
            bool fixed =
                component.kinematic || component.shape == scene::ColliderShape::PLANE || component.mass <= 0.0F;
            body.inverseMass = fixed ? 0.0F : 1.0F / component.mass;
            if (!fixed) {
                if (component.shape == scene::ColliderShape::SPHERE) {
                    float inertia = 0.4F * component.mass * body.radius * body.radius;
                    body.inverseInertia = glm::vec3{1.0F / inertia};
                } else {
                    glm::vec3 size = body.halfExtents * 2.0F;
                    glm::vec3 inertia{component.mass / 12.0F * (size.y * size.y + size.z * size.z),
                                      component.mass / 12.0F * (size.x * size.x + size.z * size.z),
                                      component.mass / 12.0F * (size.x * size.x + size.y * size.y)};
                    body.inverseInertia = 1.0F / glm::max(inertia, glm::vec3{1.0e-6F});
                }
            }
            body.boundingRadius =
                component.shape == scene::ColliderShape::SPHERE ? body.radius : glm::length(body.halfExtents);

            // 2) 반암시적 오일러 적분. 힘은 중력뿐이다.
            if (isDynamic(body)) {
                if (body.useGravity) {
                    body.velocity.y += GRAVITY * dt;
                }
                body.position += body.velocity * dt;
                glm::quat spin{0.0F, body.angularVelocity.x, body.angularVelocity.y, body.angularVelocity.z};
                body.rotation = glm::normalize(body.rotation + 0.5F * dt * (spin * body.rotation));
            }
        }
    });

    // 3) 광역: 경계 구가 겹치는 짝을 모은다. 미리 잡은 배열에 원자 카운터로 밀어 넣어 잠금이 없다.
    // ponytail: O(n²) 검사. 수백 개를 넘으면 정렬·스윕으로 바꾼다.
    auto count = static_cast<uint32_t>(bodies.size());
    std::vector<std::pair<uint32_t, uint32_t>> pairs(static_cast<size_t>(count) * (count - 1) / 2);
    std::atomic<uint32_t> pairCount{0};
    forRange(jobs, count, [&](uint32_t begin, uint32_t end) {
        for (uint32_t i = begin; i < end; ++i) {
            const Body& a = bodies[i];
            for (uint32_t j = i + 1; j < count; ++j) {
                const Body& b = bodies[j];
                if (!isDynamic(a) && !isDynamic(b)) {
                    continue;
                }
                bool plane = a.shape == scene::ColliderShape::PLANE || b.shape == scene::ColliderShape::PLANE;
                if (!plane) {
                    float reach = a.boundingRadius + b.boundingRadius;
                    if (glm::length2(a.position - b.position) > reach * reach) {
                        continue;
                    }
                }
                pairs[pairCount.fetch_add(1, std::memory_order_relaxed)] = {i, j};
            }
        }
    });
    pairs.resize(pairCount.load(std::memory_order_relaxed));

    // 4) 협역: 짝마다 접촉을 만든다. 짝 수가 적어 직렬이다.
    std::vector<Contact> contacts;
    for (const auto& [i, j] : pairs) {
        collide(bodies, i, j, contacts);
    }

    // 5) 순차 임펄스로 속도를 고친다.
    solveContacts(bodies, contacts, dt);

    // 6) 세계 상태를 오브젝트의 지역 변환과 부품 속도로 되돌려 쓴다. 부모 변환을 읽으므로 직렬이다.
    // ponytail: 강체 안에 강체가 달린 계층은 부모가 뒤에 처리되면 자식이 한 스텝 어긋난다. 그런 배치는
    // 드물어 위상 순서 정렬은 두지 않았다.
    for (const Body& body : bodies) {
        scene::Object& object = scene.objects[body.object];
        scene::RigidBody& component = scene.rigidBodies[static_cast<size_t>(object.rigidBody)];
        component.velocity = body.velocity;
        component.angularVelocity = body.angularVelocity;
        if (!isDynamic(body)) {
            continue;
        }
        scene::Transform world;
        world.position = body.position;
        world.rotation = body.rotation;
        world.scale = body.scale;
        glm::mat4 parentWorld =
            object.parent >= 0 ? scene.worldMatrix(static_cast<uint32_t>(object.parent)) : glm::mat4{1.0F};
        scene::Transform local = scene::Transform::fromMatrix(glm::inverse(parentWorld) * world.matrix());
        object.transform.position = local.position;
        object.transform.rotation = local.rotation;
    }
}

} // namespace physics
