#include "physics/rigid_body.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/norm.hpp>

#include "core/job_system.h"
#include "physics/collider_shapes.h"
#include "physics/spatial_hash.h"
#include "scene/scene.h"

namespace physics {
namespace {

// GPU 솔버는 접촉을 Jacobi 로 풀어 수렴이 느리므로 더 많이 돈다(gfx::RIGID_SOLVER_ITERATIONS).
// 나머지 상수는 rigid_body.h 에서 두 백엔드가 함께 쓴다.
constexpr uint32_t SOLVER_ITERATIONS = 8;
// 경첩 각을 한 바퀴로 접을 때 쓴다.
constexpr float TWO_PI = 6.283185307179586F;
constexpr uint32_t GRANULARITY = 16;

// 솔버 안에서 쓰는 짧은 이름.
using Body = RigidBodyState;

// 접촉 하나. 법선은 a 에서 b 를 향한다.
struct Contact {
    uint32_t a = 0;
    uint32_t b = 0;
    glm::vec3 point{0.0F};
    glm::vec3 normal{0.0F, 1.0F, 0.0F};
    float penetration = 0.0F;
    float normalImpulse = 0.0F;
    // 위치 보정으로 이미 밀어낸 거리. 접촉을 다시 만들지 않고 보정을 여러 번 돌기 위한 것이다.
    float appliedCorrection = 0.0F;
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

// 한 짝의 접촉은 가장 깊은 MAX_MANIFOLD_POINTS 개만 남긴다. GLSL 의 addPoint 와 같은 규칙이어야
// 두 백엔드가 같은 접촉을 본다. begin 은 이 짝의 첫 접촉이 들어간 자리다.
void addManifoldPoint(std::vector<Contact>& out, size_t begin, const Contact& contact) {
    if (out.size() - begin < MAX_MANIFOLD_POINTS) {
        out.push_back(contact);
        return;
    }
    size_t shallowest = begin;
    for (size_t i = begin + 1; i < out.size(); ++i) {
        if (out[i].penetration < out[shallowest].penetration) {
            shallowest = i;
        }
    }
    if (out[shallowest].penetration < contact.penetration) {
        out[shallowest] = contact;
    }
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
    size_t begin = out.size();
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
        addManifoldPoint(out, begin, contact);
    }
}

// 축에 투영한 상자의 반지름.
float boxProjection(const glm::mat3& rotation, const glm::vec3& extent, const glm::vec3& axis) {
    return std::abs(glm::dot(rotation[0], axis)) * extent.x + std::abs(glm::dot(rotation[1], axis)) * extent.y +
           std::abs(glm::dot(rotation[2], axis)) * extent.z;
}

// 상자 대 상자. 면 축 여섯과 모서리 축 아홉(두 상자 축의 외적)의 분리축 검사로 가장 얕게 겹치는 축을 찾는다.
// 면 축이면 그 축에서 겨루는 면 안에 든 상대 꼭짓점을 접촉으로 낸다 — 나란히 놓인 상자면 네 점이 나와
// 넘어지지 않는다. 모서리 축이면 두 지지 모서리의 최근접점 중점 하나를 낸다 — 비스듬히 걸친 상자를 면 축으로만
// 재면 침투가 과대 평가되어 튀거나 가라앉는다. GLSL 의 collideBoxBox 와 축 순서·규칙이 같아야 한다.
//
// 서로의 꼭짓점이 상대 안에 들어왔는지로만 보면 크기가 같은 상자를 쌓았을 때 꼭짓점이 옆면에 딱
// 붙어 «가장 얕은 면»이 옆으로 잡히고 침투가 0 이 되어 서로를 그대로 통과한다.
//
// ponytail: 면·모서리 겹침이 거의 같을 때 프레임마다 축이 갈려 떨리면 면 쪽 5 % 편향을 넣는다(상수는
// rigid_body.h 에 두고 GPU 에는 푸시 상수로).
void collideBoxBox(const Body& a, const Body& b, uint32_t ia, uint32_t ib, std::vector<Contact>& out) {
    glm::mat3 rotationA = glm::mat3_cast(a.rotation);
    glm::mat3 rotationB = glm::mat3_cast(b.rotation);
    glm::vec3 center = b.position - a.position;

    float bestOverlap = std::numeric_limits<float>::max();
    glm::vec3 bestNormal{0.0F, 1.0F, 0.0F};
    bool referenceIsA = true;
    int bestAxis = 1;
    // 0~2 A 의 면, 3~5 B 의 면, 6~14 모서리 (6 + i*3 + j = cross(A_i, B_j)).
    int bestIndex = 1;
    for (int index = 0; index < 15; ++index) {
        glm::vec3 axis;
        if (index < 6) {
            axis = index < 3 ? rotationA[index] : rotationB[index - 3];
        } else {
            axis = glm::cross(rotationA[(index - 6) / 3], rotationB[(index - 6) % 3]);
            // 거의 평행한 축의 외적은 방향이 뜻이 없다. 그 경우 면 축이 이미 가른다.
            if (glm::length2(axis) < 1.0e-6F) {
                continue;
            }
            axis = glm::normalize(axis);
        }
        float distance = glm::dot(center, axis);
        float overlap = boxProjection(rotationA, a.halfExtents, axis) + boxProjection(rotationB, b.halfExtents, axis) -
                        std::abs(distance);
        if (overlap <= 0.0F) {
            // 이 축으로 떨어져 있다.
            return;
        }
        if (overlap < bestOverlap) {
            bestOverlap = overlap;
            // 법선은 a 에서 b 를 향해야 한다.
            bestNormal = distance < 0.0F ? -axis : axis;
            bestIndex = index;
            if (index < 6) {
                referenceIsA = index < 3;
                bestAxis = index < 3 ? index : index - 3;
            }
        }
    }

    if (bestIndex >= 6) {
        // 모서리끼리 걸침. 법선 쪽으로 가장 나간 A 의 모서리와 반대쪽으로 가장 나간 B 의 모서리를 잡아 두 직선의
        // 최근접점을 구하고 그 중점을 접촉으로 낸다.
        int edgeA = (bestIndex - 6) / 3;
        int edgeB = (bestIndex - 6) % 3;
        glm::vec3 pointA = a.position;
        glm::vec3 pointB = b.position;
        for (int k = 0; k < 3; ++k) {
            if (k != edgeA) {
                pointA +=
                    rotationA[k] * (glm::dot(rotationA[k], bestNormal) >= 0.0F ? a.halfExtents[k] : -a.halfExtents[k]);
            }
            if (k != edgeB) {
                pointB +=
                    rotationB[k] * (glm::dot(rotationB[k], -bestNormal) >= 0.0F ? b.halfExtents[k] : -b.halfExtents[k]);
            }
        }
        glm::vec3 directionA = rotationA[edgeA];
        glm::vec3 directionB = rotationB[edgeB];
        glm::vec3 offset = pointA - pointB;
        float along = glm::dot(directionA, directionB);
        float alongA = glm::dot(directionA, offset);
        float alongB = glm::dot(directionB, offset);
        float denominator = 1.0F - along * along;
        float s = glm::clamp((along * alongB - alongA) / denominator, -a.halfExtents[edgeA], a.halfExtents[edgeA]);
        float t = glm::clamp((alongB - along * alongA) / denominator, -b.halfExtents[edgeB], b.halfExtents[edgeB]);
        Contact contact;
        contact.a = ia;
        contact.b = ib;
        contact.normal = bestNormal;
        contact.point = ((pointA + directionA * s) + (pointB + directionB * t)) * 0.5F;
        contact.penetration = bestOverlap;
        out.push_back(contact);
        return;
    }

    // 겨루는 면을 가진 쪽이 기준, 반대쪽이 입사다.
    const Body& reference = referenceIsA ? a : b;
    const Body& incident = referenceIsA ? b : a;
    const glm::mat3& referenceRotation = referenceIsA ? rotationA : rotationB;
    float axisSign = glm::dot(referenceRotation[bestAxis], bestNormal) < 0.0F ? -1.0F : 1.0F;
    float direction = referenceIsA ? 1.0F : -1.0F;

    glm::vec3 corners[8];
    boxCorners(incident, corners);
    glm::quat inverse = glm::conjugate(reference.rotation);
    size_t begin = out.size();
    for (const glm::vec3& corner : corners) {
        glm::vec3 local = inverse * (corner - reference.position);
        float depth = reference.halfExtents[bestAxis] - direction * axisSign * local[bestAxis];
        if (depth <= 0.0F) {
            continue;
        }
        bool inside = true;
        for (int axis = 0; axis < 3; ++axis) {
            if (axis != bestAxis && std::abs(local[axis]) > reference.halfExtents[axis] + PENETRATION_SLOP) {
                inside = false;
            }
        }
        if (!inside) {
            continue;
        }
        Contact contact;
        contact.a = ia;
        contact.b = ib;
        contact.normal = bestNormal;
        contact.point = corner;
        contact.penetration = std::min(depth, bestOverlap);
        addManifoldPoint(out, begin, contact);
    }
    if (out.size() > begin) {
        return;
    }
    // 면 안에 든 꼭짓점이 없으면(모서리끼리 걸침) 가장 깊은 지지점 하나로 대신한다.
    Contact contact;
    contact.a = ia;
    contact.b = ib;
    contact.normal = bestNormal;
    contact.penetration = bestOverlap;
    contact.point = b.position;
    for (int axis = 0; axis < 3; ++axis) {
        float side = glm::dot(rotationB[axis], bestNormal) < 0.0F ? 1.0F : -1.0F;
        contact.point += rotationB[axis] * (side * b.halfExtents[axis]);
    }
    out.push_back(contact);
}

ColliderLocal asCollider(const Body& body) {
    return ColliderLocal{body.shape, body.radius, body.halfExtents};
}

// 상대 표면에서 자기 중심에 가장 가까운 점을 자기 지역 공간으로. probePointLocal 의 힌트다.
glm::vec3 probeHint(const Body& self, const Body& other) {
    SurfacePoint near = closestOnCollider(asCollider(other), other.position, other.rotation, self.position);
    return glm::conjugate(self.rotation) * (near.point - self.position);
}

// 원기둥·캡슐이 낀 짝. a 의 표본점을 b 표면에, b 의 표본점을 a 표면에 찔러 깊이가 양수인 것을 접촉으로
// 낸다. GLSL 의 rigidCollideGeneric 과 같은 규칙이다.
//
// ponytail: 볼록 형상 일반 해법(GJK/EPA)이 아니라 표본 기반이다. 표본 사이(원기둥 테두리 22.5도)로
// 파고드는 얇은 모서리는 놓칠 수 있다.
void collideGeneric(const Body& a, const Body& b, uint32_t ia, uint32_t ib, std::vector<Contact>& out) {
    size_t begin = out.size();
    ColliderLocal colliderA = asCollider(a);
    ColliderLocal colliderB = asCollider(b);
    glm::vec3 hintA = probeHint(a, b);
    for (int i = 0, n = probeCount(a.shape); i < n; ++i) {
        float radius = 0.0F;
        glm::vec3 local = probePointLocal(colliderA, hintA, i, radius);
        glm::vec3 world = a.position + a.rotation * local;
        SurfacePoint surface = closestOnCollider(colliderB, b.position, b.rotation, world);
        float depth = radius - surface.distance;
        if (depth <= 0.0F) {
            continue;
        }
        Contact contact;
        contact.a = ia;
        contact.b = ib;
        // b 의 바깥 법선은 b 에서 a 를 향하므로 뒤집는다.
        contact.normal = -surface.normal;
        contact.point = surface.point;
        contact.penetration = depth;
        addManifoldPoint(out, begin, contact);
    }
    glm::vec3 hintB = probeHint(b, a);
    for (int i = 0, n = probeCount(b.shape); i < n; ++i) {
        float radius = 0.0F;
        glm::vec3 local = probePointLocal(colliderB, hintB, i, radius);
        glm::vec3 world = b.position + b.rotation * local;
        SurfacePoint surface = closestOnCollider(colliderA, a.position, a.rotation, world);
        float depth = radius - surface.distance;
        if (depth <= 0.0F) {
            continue;
        }
        Contact contact;
        contact.a = ia;
        contact.b = ib;
        contact.normal = surface.normal;
        contact.point = surface.point;
        contact.penetration = depth;
        addManifoldPoint(out, begin, contact);
    }
}

// a 의 표본점을 메쉬 b 의 삼각형마다 본다. 메쉬는 운동학이라 찌르지 않는다. 힌트는 세계 아래쪽이라
// 지형 위에 놓인 원기둥이 아래 테두리를 표본으로 고른다.
//
// ponytail: 삼각형을 전부 훑는다. ColliderMesh::MAX_TRIANGLES 가 상한이라 감당하지만 BVH 가 있으면
// 0단계 LOD 를 쓸 수 있다.
void collideMesh(const Body& a,
                 const Body& mesh,
                 uint32_t ia,
                 uint32_t ib,
                 const std::vector<Triangle>& triangles,
                 std::vector<Contact>& out) {
    size_t begin = out.size();
    ColliderLocal colliderA = asCollider(a);
    glm::vec3 hint = glm::conjugate(a.rotation) * glm::vec3{0.0F, -a.boundingRadius, 0.0F};
    // 뒷면으로 이만큼까지 들어간 점은 앞면으로 밀어 올린다. 물체 크기에 비례해야 큰 상자의 꼭짓점이
    // 한 스텝에 깊이 들어와도 놓치지 않는다.
    float thickness = 0.5F * a.boundingRadius;
    for (int i = 0, n = probeCount(a.shape); i < n; ++i) {
        float radius = 0.0F;
        glm::vec3 local = probePointLocal(colliderA, hint, i, radius);
        glm::vec3 world = a.position + a.rotation * local;
        for (uint32_t t = 0; t < mesh.triangleCount; ++t) {
            const Triangle& triangle = triangles[mesh.triangleOffset + t];
            SurfacePoint surface;
            if (!closestOnTriangleSurface(triangle.a, triangle.b, triangle.c, world, radius, thickness, surface)) {
                continue;
            }
            Contact contact;
            contact.a = ia;
            contact.b = ib;
            contact.normal = -surface.normal;
            contact.point = surface.point;
            contact.penetration = radius - surface.distance;
            addManifoldPoint(out, begin, contact);
        }
    }
}

void collide(const std::vector<Body>& bodies,
             uint32_t ia,
             uint32_t ib,
             const std::vector<Triangle>& triangles,
             std::vector<Contact>& out) {
    using scene::ColliderShape;
    const Body& a = bodies[ia];
    const Body& b = bodies[ib];
    auto isStatic = [](ColliderShape shape) { return shape == ColliderShape::PLANE || shape == ColliderShape::MESH; };
    // 평면·메쉬끼리는 만나지 않는다. 짝은 항상 (움직이는 모양, 무엇이든) 순서로 맞춘다.
    if (isStatic(a.shape)) {
        if (isStatic(b.shape)) {
            return;
        }
        collide(bodies, ib, ia, triangles, out);
        return;
    }
    if (b.shape == ColliderShape::MESH) {
        collideMesh(a, b, ia, ib, triangles, out);
        return;
    }
    if (a.shape == ColliderShape::CYLINDER || a.shape == ColliderShape::CAPSULE || b.shape == ColliderShape::CYLINDER ||
        b.shape == ColliderShape::CAPSULE) {
        collideGeneric(a, b, ia, ib, out);
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
        default:
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
    default:
        break;
    }
}

void solveContacts(std::vector<Body>& bodies, std::vector<Contact>& contacts) {
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
    //
    // 같은 짝의 여러 점은 «가장 깊은 것 하나»로만 민다. 나란히 놓인 상자는 네 점이 나오는데 넷을 다
    // 밀면 네 배로 튀어 오른다. 접촉은 짝 단위로 이어 붙어 있으므로 구간을 훑는다.
    for (uint32_t iteration = 0; iteration < POSITION_ITERATIONS; ++iteration) {
        size_t index = 0;
        while (index < contacts.size()) {
            size_t end = index;
            size_t deepest = index;
            while (end < contacts.size() && contacts[end].a == contacts[index].a &&
                   contacts[end].b == contacts[index].b) {
                if (contacts[end].penetration > contacts[deepest].penetration) {
                    deepest = end;
                }
                ++end;
            }
            Contact& contact = contacts[deepest];
            Body& a = bodies[contact.a];
            Body& b = bodies[contact.b];
            float totalInverseMass = a.inverseMass + b.inverseMass;
            // 이미 밀어낸 만큼은 빼고 남은 침투만 다시 민다. 접촉을 다시 만들지 않고도 반복할 수 있다.
            float depth = contact.penetration - contact.appliedCorrection - PENETRATION_SLOP;
            if (totalInverseMass > 0.0F && depth > 0.0F) {
                float push = POSITION_CORRECTION * depth;
                contact.appliedCorrection += push;
                glm::vec3 correction = contact.normal * (push / totalInverseMass);
                a.position -= correction * a.inverseMass;
                b.position += correction * b.inverseMass;
            }
            index = end;
        }
    }
}

glm::mat3 skewMatrix(const glm::vec3& v) {
    // skew(v) * x == cross(v, x). glm 은 열 우선이다.
    return glm::mat3{0.0F, v.z, -v.y, -v.z, 0.0F, v.x, v.y, -v.x, 0.0F};
}

// 관절 한쪽의 앵커·팔·속도. B 가 고정점이면 질량 역수와 관성이 0 이고 위치만 있다.
struct JointSide {
    glm::vec3 anchor{0.0F};
    glm::vec3 arm{0.0F};
    glm::vec3 velocity{0.0F};
    glm::vec3 angularVelocity{0.0F};
    float inverseMass = 0.0F;
    glm::mat3 inverseInertia{0.0F};
};

JointSide jointSideOf(const Body& body, const glm::vec3& local) {
    JointSide side;
    side.arm = glm::mat3_cast(body.rotation) * local;
    side.anchor = body.position + side.arm;
    side.velocity = body.velocity + glm::cross(body.angularVelocity, side.arm);
    side.angularVelocity = body.angularVelocity;
    side.inverseMass = body.inverseMass;
    side.inverseInertia = worldInverseInertia(body);
    return side;
}

JointSide jointStaticSide(const glm::vec3& worldAnchor) {
    JointSide side;
    side.anchor = worldAnchor;
    return side;
}

// B 를 기준으로 A 가 경첩 축 둘레로 돈 각(라디안). 부품이 A 에 붙어 있으니 «이 오브젝트가 얼마나
// 돌았는가» 가 편집기에서 자연스러운 뜻이고, 축의 오른손 방향이 양이다. 두 자세가 같을 때 0 이고
// (-pi, pi] 에 든다. 상대 회전에서 축 성분만 남긴 «비틀림» 사원수의 각을 뒤집은 것이다(경첩은 두
// 물체의 축이 나란하므로 뒤집기만 하면 기준이 바뀐다). GLSL 의 rigidHingeAngle 과 같은 식이다.
float hingeAngle(const glm::quat& rotationA, const glm::quat& rotationB, const glm::vec3& localAxis) {
    glm::quat relative = glm::conjugate(rotationA) * rotationB;
    float twist = glm::dot(glm::vec3{relative.x, relative.y, relative.z}, localAxis);
    float real = relative.w;
    // 사원수는 q 와 -q 가 같은 회전이다. 한쪽으로 모아야 각이 (-pi, pi] 에 들어온다.
    if (real < 0.0F) {
        real = -real;
        twist = -twist;
    }
    return -2.0F * std::atan2(twist, real);
}

// 관절 하나가 이번 스텝에서 축 둘레로 쌓은 임펄스. 순차 임펄스는 누적값을 클램프해야 모터의 토크
// 상한과 한계각의 «한쪽으로만 민다» 는 성질이 반복 횟수와 무관해진다.
struct JointAccumulator {
    float motor = 0.0F;
    float limit = 0.0F;
};

// 관절의 속도 행. 거리는 앵커 사이 방향 하나, 볼·경첩은 앵커를 한 점으로 묶는 3×3 유효 질량이고 경첩은 축에
// 수직한 각속도 차를 두 행으로 없앤다. 경첩은 그 뒤에 축 둘레의 모터 행과 한계각 행을 더 푼다.
// GPU(rigid_common.glsl 의 rigidJointImpulse)와 같은 식이다.
void solveJoint(std::vector<Body>& bodies, const JointState& joint, JointAccumulator& accumulator, float dt) {
    Body& a = bodies[joint.bodyA];
    Body* b = joint.bodyB >= 0 ? &bodies[static_cast<size_t>(joint.bodyB)] : nullptr;
    JointSide sideA = jointSideOf(a, joint.localA);
    JointSide sideB = b != nullptr ? jointSideOf(*b, joint.localB) : jointStaticSide(joint.worldAnchorB);
    float totalInverseMass = sideA.inverseMass + sideB.inverseMass;
    if (totalInverseMass <= 0.0F) {
        return;
    }
    glm::vec3 relative = sideB.velocity - sideA.velocity;
    glm::vec3 impulse{0.0F};
    if (joint.type == scene::JointType::DISTANCE) {
        glm::vec3 delta = sideB.anchor - sideA.anchor;
        float span = glm::length(delta);
        if (span < 1.0e-6F) {
            return;
        }
        glm::vec3 normal = delta / span;
        glm::vec3 torqueA = glm::cross(sideA.arm, normal);
        glm::vec3 torqueB = glm::cross(sideB.arm, normal);
        float k = totalInverseMass + glm::dot(torqueA, sideA.inverseInertia * torqueA) +
                  glm::dot(torqueB, sideB.inverseInertia * torqueB);
        impulse = normal * (-glm::dot(relative, normal) / k);
    } else {
        glm::mat3 skewA = skewMatrix(sideA.arm);
        glm::mat3 skewB = skewMatrix(sideB.arm);
        glm::mat3 k =
            glm::mat3{totalInverseMass} - skewA * sideA.inverseInertia * skewA - skewB * sideB.inverseInertia * skewB;
        impulse = glm::inverse(k) * (-relative);
    }
    a.velocity -= impulse * sideA.inverseMass;
    a.angularVelocity -= sideA.inverseInertia * glm::cross(sideA.arm, impulse);
    if (b != nullptr) {
        b->velocity += impulse * sideB.inverseMass;
        b->angularVelocity += sideB.inverseInertia * glm::cross(sideB.arm, impulse);
    }

    if (joint.type == scene::JointType::HINGE) {
        glm::vec3 axis = glm::mat3_cast(a.rotation) * joint.localAxis;
        float axisLength = glm::length(axis);
        if (axisLength < 1.0e-6F) {
            return;
        }
        axis /= axisLength;
        glm::vec3 helper = std::abs(axis.y) < 0.9F ? glm::vec3{0.0F, 1.0F, 0.0F} : glm::vec3{1.0F, 0.0F, 0.0F};
        glm::vec3 u = glm::normalize(glm::cross(axis, helper));
        glm::vec3 v = glm::cross(axis, u);
        for (const glm::vec3& t : {u, v}) {
            glm::vec3 omega = (b != nullptr ? b->angularVelocity : glm::vec3{0.0F}) - a.angularVelocity;
            float k = glm::dot(t, (sideA.inverseInertia + sideB.inverseInertia) * t);
            if (k <= 1.0e-9F) {
                continue;
            }
            float lambda = -glm::dot(omega, t) / k;
            a.angularVelocity -= sideA.inverseInertia * (t * lambda);
            if (b != nullptr) {
                b->angularVelocity += sideB.inverseInertia * (t * lambda);
            }
        }

        // 모터와 한계각은 축 둘레 한 행씩이다. 각과 마찬가지로 양의 임펄스가 «A 가 B 보다 축 방향으로
        // 빨라지는» 쪽이라 hingeAngle 을 늘린다.
        float axisMass = glm::dot(axis, (sideA.inverseInertia + sideB.inverseInertia) * axis);
        if (axisMass <= 1.0e-9F) {
            return;
        }
        glm::quat rotationB = b != nullptr ? b->rotation : glm::quat{1.0F, 0.0F, 0.0F, 0.0F};
        float angle = hingeAngle(a.rotation, rotationB, joint.localAxis / axisLength);
        float rate = glm::dot(a.angularVelocity - (b != nullptr ? b->angularVelocity : glm::vec3{0.0F}), axis);
        auto applyAxis = [&](float lambda) {
            a.angularVelocity += sideA.inverseInertia * (axis * lambda);
            if (b != nullptr) {
                b->angularVelocity -= sideB.inverseInertia * (axis * lambda);
            }
            // 다음 행이 방금 고친 속도를 보게 한다. 임펄스 하나가 각속도 차를 axisMass 배만큼 바꾼다.
            rate += axisMass * lambda;
        };

        if (joint.motor != scene::JointMotor::NONE && joint.maxTorque > 0.0F) {
            float target = joint.targetSpeed;
            if (joint.motor == scene::JointMotor::POSITION) {
                float error = joint.targetAngle - angle;
                if (!joint.useLimit) {
                    // 각이 (-pi, pi] 로 접혀 나오므로 오차도 접어야 짧은 쪽으로 돈다. 접지 않으면 목표가
                    // 경계 건너에 있을 때 한 바퀴 가까이 먼 길로 간다. 한계각이 켜져 있으면 그 짧은 길이
                    // 막혀 있을 수 있으므로 접지 않고 허용 범위 안을 그대로 지난다.
                    error -= TWO_PI * std::round(error / TWO_PI);
                }
                float limit = std::abs(joint.targetSpeed);
                target = std::clamp(joint.motorStiffness * error, -limit, limit);
            }
            float wanted = (target - rate) / axisMass;
            // 토크 상한은 «이번 스텝에 낼 수 있는 각운동량» 이라 dt 를 곱한다.
            float maxImpulse = joint.maxTorque * dt;
            float total = std::clamp(accumulator.motor + wanted, -maxImpulse, maxImpulse);
            applyAxis(total - accumulator.motor);
            accumulator.motor = total;
        }

        if (joint.useLimit) {
            // 어긴 각을 속도 편향으로 바꿔 되민다. 한쪽 한계는 한쪽 방향으로만 밀 수 있으므로 누적
            // 임펄스의 부호를 묶는다.
            //
            // ponytail: 접힌 각으로 재므로 한계는 (-180, 180] 안에서만 뜻이 있다. 그 경계를 걸치는
            // 범위는 적을 수 없고, 한 스텝에 180 도 넘게 어기면 부호가 뒤집혀 반대쪽 한계로 넘어간다.
            float excess = angle < joint.lowerAngle   ? angle - joint.lowerAngle
                           : angle > joint.upperAngle ? angle - joint.upperAngle
                                                      : 0.0F;
            if (excess == 0.0F) {
                accumulator.limit = 0.0F;
            } else {
                float bias =
                    std::clamp(excess * (POSITION_CORRECTION / dt), -JOINT_LIMIT_MAX_SPEED, JOINT_LIMIT_MAX_SPEED);
                float wanted = -(rate + bias) / axisMass;
                float total = excess < 0.0F ? std::max(accumulator.limit + wanted, 0.0F)
                                            : std::min(accumulator.limit + wanted, 0.0F);
                applyAxis(total - accumulator.limit);
                accumulator.limit = total;
            }
        }
    }
}

// 관절의 위치 보정. 앵커 오차(거리는 목표와의 차)를 질량 역수 비율로 나눠 민다. 회전은 고치지 않는다.
//
// ponytail: 경첩의 축 어긋남(각도 오차)은 보정하지 않아 오래 돌면 조금 기운다. 각도 위치 행을 더하면 된다.
void correctJoint(std::vector<Body>& bodies, const JointState& joint) {
    Body& a = bodies[joint.bodyA];
    Body* b = joint.bodyB >= 0 ? &bodies[static_cast<size_t>(joint.bodyB)] : nullptr;
    glm::vec3 anchorA = a.position + glm::mat3_cast(a.rotation) * joint.localA;
    glm::vec3 anchorB = b != nullptr ? b->position + glm::mat3_cast(b->rotation) * joint.localB : joint.worldAnchorB;
    float wa = a.inverseMass;
    float wb = b != nullptr ? b->inverseMass : 0.0F;
    if (wa + wb <= 0.0F) {
        return;
    }
    glm::vec3 error = anchorB - anchorA;
    if (joint.type == scene::JointType::DISTANCE) {
        float span = glm::length(error);
        if (span < 1.0e-6F) {
            return;
        }
        error = error / span * (span - joint.length);
    }
    glm::vec3 push = error * (POSITION_CORRECTION / (wa + wb));
    a.position += push * wa;
    if (b != nullptr) {
        b->position -= push * wb;
    }
}

} // namespace

void collectJoints(const scene::Scene& scene, const std::vector<RigidBodyState>& bodies, std::vector<JointState>& out) {
    out.clear();
    std::vector<int32_t> bodyOf(scene.objects.size(), -1);
    for (uint32_t i = 0; i < bodies.size(); ++i) {
        if (bodies[i].object < bodyOf.size()) {
            bodyOf[bodies[i].object] = static_cast<int32_t>(i);
        }
    }
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        int32_t slot = scene.objects[index].joint;
        if (slot < 0 || static_cast<size_t>(slot) >= scene.joints.size() || bodyOf[index] < 0) {
            continue;
        }
        const scene::Joint& joint = scene.joints[static_cast<size_t>(slot)];
        // 자기 자신과 잇는 관절은 뜻이 없다(장면 파일이 그렇게 적혀 있을 수 있다).
        if (joint.other == static_cast<int32_t>(index)) {
            continue;
        }
        JointState state;
        state.bodyA = static_cast<uint32_t>(bodyOf[index]);
        state.localA = joint.anchorA;
        state.localAxis = joint.axis;
        state.length = joint.length;
        state.type = joint.type;
        state.motor = joint.motor;
        state.useLimit = joint.useLimit;
        // 부품은 편집기에서 읽기 쉬운 도로 적혀 있다. 솔버는 라디안만 본다.
        state.lowerAngle = glm::radians(std::min(joint.lowerAngle, joint.upperAngle));
        state.upperAngle = glm::radians(std::max(joint.lowerAngle, joint.upperAngle));
        state.targetAngle = glm::radians(joint.targetAngle);
        state.targetSpeed = glm::radians(joint.targetSpeed);
        state.motorStiffness = joint.motorStiffness;
        state.maxTorque = joint.maxTorque;
        bool otherValid = joint.other >= 0 && static_cast<size_t>(joint.other) < scene.objects.size();
        state.bodyB = otherValid ? bodyOf[static_cast<size_t>(joint.other)] : -1;
        if (state.bodyB >= 0) {
            state.localB = joint.anchorB;
        } else if (otherValid) {
            // B 오브젝트는 있지만 이 백엔드의 강체가 아니다. 그 오브젝트의 세계 변환에 붙은 고정점으로 본다.
            state.worldAnchorB =
                glm::vec3(scene.worldMatrix(static_cast<uint32_t>(joint.other)) * glm::vec4{joint.anchorB, 1.0F});
        } else {
            state.worldAnchorB = joint.anchorB;
        }
        out.push_back(state);
    }
}

void collectRigidBodies(const scene::Scene& scene,
                        scene::SimulationBackend backend,
                        std::vector<RigidBodyState>& out,
                        std::vector<Triangle>& triangles) {
    out.clear();
    triangles.clear();
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        int32_t slot = scene.objects[index].rigidBody;
        if (slot < 0 || static_cast<size_t>(slot) >= scene.rigidBodies.size()) {
            continue;
        }
        const scene::RigidBody& component = scene.rigidBodies[static_cast<size_t>(slot)];
        // AUTO 는 CPU 다. GPU 솔버는 골라야 돈다.
        bool onGpu = component.backend == scene::SimulationBackend::GPU;
        if ((backend == scene::SimulationBackend::GPU) != onGpu) {
            continue;
        }

        glm::mat4 worldMatrix = scene.worldMatrix(index);
        scene::Transform world = scene::Transform::fromMatrix(worldMatrix);
        // 세계 공간 크기는 편집기의 콜라이더 표시와 같은 함수로 낸다. 어긋나면 보이는 것과
        // 부딪히는 것이 달라진다.
        scene::ColliderPose pose = scene::colliderPose(component, worldMatrix);

        RigidBodyState body;
        body.object = index;
        body.shape = component.shape;
        body.position = pose.position;
        body.rotation = pose.rotation;
        body.scale = world.scale;
        body.velocity = component.velocity;
        body.angularVelocity = component.angularVelocity;
        body.restitution = component.restitution;
        body.friction = component.friction;
        body.useGravity = component.useGravity;
        body.radius = pose.radius;
        body.halfExtents = pose.halfExtents;
        bool fixed = component.kinematic || component.shape == scene::ColliderShape::PLANE ||
                     component.shape == scene::ColliderShape::MESH || component.mass <= 0.0F;
        body.inverseMass = fixed ? 0.0F : 1.0F / component.mass;
        if (!fixed) {
            float mass = component.mass;
            float r2 = body.radius * body.radius;
            switch (component.shape) {
            case scene::ColliderShape::SPHERE:
                body.inverseInertia = glm::vec3{1.0F / (0.4F * mass * r2)};
                break;
            case scene::ColliderShape::CYLINDER:
            case scene::ColliderShape::CAPSULE: {
                // 캡슐은 반구까지 포함한 높이의 원기둥으로 어림한다.
                float height = 2.0F * body.halfExtents.y +
                               (component.shape == scene::ColliderShape::CAPSULE ? 2.0F * body.radius : 0.0F);
                float side = mass / 12.0F * (3.0F * r2 + height * height);
                glm::vec3 inertia{side, 0.5F * mass * r2, side};
                body.inverseInertia = 1.0F / glm::max(inertia, glm::vec3{1.0e-6F});
                break;
            }
            default: {
                glm::vec3 size = body.halfExtents * 2.0F;
                glm::vec3 inertia{mass / 12.0F * (size.y * size.y + size.z * size.z),
                                  mass / 12.0F * (size.x * size.x + size.z * size.z),
                                  mass / 12.0F * (size.x * size.x + size.y * size.y)};
                body.inverseInertia = 1.0F / glm::max(inertia, glm::vec3{1.0e-6F});
                break;
            }
            }
        }
        switch (component.shape) {
        case scene::ColliderShape::SPHERE:
            body.boundingRadius = body.radius;
            break;
        case scene::ColliderShape::CYLINDER:
            body.boundingRadius = std::sqrt(body.radius * body.radius + body.halfExtents.y * body.halfExtents.y);
            break;
        case scene::ColliderShape::CAPSULE:
            body.boundingRadius = body.radius + body.halfExtents.y;
            break;
        case scene::ColliderShape::MESH: {
            // 삼각형을 세계 공간으로 옮겨 이어 붙인다. 경계 구 중심이 원점에서 벗어나 있을 수 있어 그
            // 거리까지 더한 보수적 반지름이다.
            const scene::ColliderMesh* mesh = scene.colliderMesh(index);
            body.triangleOffset = static_cast<uint32_t>(triangles.size());
            if (mesh != nullptr) {
                auto toWorld = [&](uint32_t vertex) {
                    return glm::vec3{worldMatrix * glm::vec4{mesh->positions[vertex], 1.0F}};
                };
                for (size_t t = 0; t + 2 < mesh->indices.size(); t += 3) {
                    triangles.push_back(Triangle{
                        toWorld(mesh->indices[t]), toWorld(mesh->indices[t + 1]), toWorld(mesh->indices[t + 2])});
                }
                float maxScale = std::max({pose.scale.x, pose.scale.y, pose.scale.z});
                body.boundingRadius = (glm::length(mesh->boundsCenter) + mesh->boundsRadius) * maxScale;
            }
            body.triangleCount = static_cast<uint32_t>(triangles.size()) - body.triangleOffset;
            break;
        }
        default:
            body.boundingRadius = glm::length(body.halfExtents);
            break;
        }
        out.push_back(body);
    }
}

void writeBackRigidBodies(scene::Scene& scene, const std::vector<RigidBodyState>& bodies) {
    // ponytail: 강체 안에 강체가 달린 계층은 부모가 뒤에 처리되면 자식이 한 스텝 어긋난다. 그런 배치는
    // 드물어 위상 순서 정렬은 두지 않았다.
    for (const RigidBodyState& body : bodies) {
        scene::Object& object = scene.objects[body.object];
        scene::RigidBody* component = scene.component<scene::RigidBody>(body.object);
        if (component == nullptr) {
            continue;
        }
        component->velocity = body.velocity;
        component->angularVelocity = body.angularVelocity;
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

namespace {

// 반암시적 오일러 적분. 힘은 중력뿐이다. GPU 솔버의 rigid_integrate.comp 과 같아야 한다.
void integrate(RigidBodyState& body, float dt) {
    if (!isDynamic(body)) {
        return;
    }
    if (body.useGravity) {
        body.velocity.y += GRAVITY * dt;
    }
    body.position += body.velocity * dt;
    glm::quat spin{0.0F, body.angularVelocity.x, body.angularVelocity.y, body.angularVelocity.z};
    body.rotation = glm::normalize(body.rotation + 0.5F * dt * (spin * body.rotation));
}

} // namespace

void collectPairs(const std::vector<RigidBodyState>& bodies,
                  core::JobSystem* jobs,
                  std::vector<std::pair<uint32_t, uint32_t>>& pairs) {
    pairs.clear();
    auto count = static_cast<uint32_t>(bodies.size());
    if (count < 2) {
        return;
    }

    // 셀은 가장 큰 물체 둘이 닿는 거리다. 그러면 닿을 수 있는 짝은 반드시 이웃 27 셀 안에 있다.
    // 평면은 반지름이 뜻이 없어 격자 밖 목록으로 뺀다.
    float maxRadius = 0.0F;
    std::vector<uint32_t> planes;
    for (uint32_t i = 0; i < count; ++i) {
        if (bodies[i].shape == scene::ColliderShape::PLANE) {
            planes.push_back(i);
        } else {
            maxRadius = std::max(maxRadius, bodies[i].boundingRadius);
        }
    }
    float cellSize = std::max(2.0F * maxRadius, 1.0e-3F);
    uint32_t cellCount = std::clamp(std::bit_ceil(count * 2U), 64U, 65536U);

    std::vector<glm::ivec3> cellCoord(count);
    std::vector<uint32_t> cellOf(count);
    forRange(jobs, count, [&](uint32_t begin, uint32_t end) {
        for (uint32_t i = begin; i < end; ++i) {
            cellCoord[i] = spatialCell(bodies[i].position, cellSize);
            cellOf[i] = spatialHash(cellCoord[i], cellCount);
        }
    });

    // 계수 정렬. 용량이 없어 넘칠 일이 없고, 번호 오름차순으로 넣어 순서가 워커 수와 무관하다.
    std::vector<uint32_t> cellStart(static_cast<size_t>(cellCount) + 1, 0);
    for (uint32_t i = 0; i < count; ++i) {
        ++cellStart[cellOf[i] + 1];
    }
    for (uint32_t cell = 0; cell < cellCount; ++cell) {
        cellStart[cell + 1] += cellStart[cell];
    }
    std::vector<uint32_t> sorted(count);
    {
        std::vector<uint32_t> cursor(cellStart.begin(), cellStart.end() - 1);
        for (uint32_t i = 0; i < count; ++i) {
            sorted[cursor[cellOf[i]]++] = i;
        }
    }

    // i 의 후보 j 를 순서대로 방문한다. 버킷은 해시 충돌로 다른 셀의 물체도 담으므로 실제 셀을 견줘 자기 셀에서
    // 만 한 번 잡는다(같은 짝이 두 번 나오면 접촉 풀기의 «같은 짝은 연속» 가정이 깨진다).
    auto visit = [&](uint32_t i, auto&& emit) {
        const Body& a = bodies[i];
        if (a.shape == scene::ColliderShape::PLANE) {
            return;
        }
        for (int dz = -1; dz <= 1; ++dz) {
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    glm::ivec3 cell = cellCoord[i] + glm::ivec3{dx, dy, dz};
                    uint32_t bucket = spatialHash(cell, cellCount);
                    for (uint32_t k = cellStart[bucket]; k < cellStart[bucket + 1]; ++k) {
                        uint32_t j = sorted[k];
                        if (j <= i || cellCoord[j] != cell) {
                            continue;
                        }
                        const Body& b = bodies[j];
                        if (b.shape == scene::ColliderShape::PLANE || (!isDynamic(a) && !isDynamic(b))) {
                            continue;
                        }
                        float reach = a.boundingRadius + b.boundingRadius;
                        if (glm::length2(a.position - b.position) > reach * reach) {
                            continue;
                        }
                        emit(j);
                    }
                }
            }
        }
    };

    // 두 번 돈다: 개수를 세어 자리를 나눈 뒤 채운다. 원자 카운터 없이도 병렬이고 결과가 결정적이다.
    std::vector<uint32_t> pairStart(static_cast<size_t>(count) + 1, 0);
    forRange(jobs, count, [&](uint32_t begin, uint32_t end) {
        for (uint32_t i = begin; i < end; ++i) {
            uint32_t found = 0;
            visit(i, [&](uint32_t) { ++found; });
            pairStart[i + 1] = found;
        }
    });
    for (uint32_t i = 0; i < count; ++i) {
        pairStart[i + 1] += pairStart[i];
    }
    pairs.resize(pairStart[count]);
    forRange(jobs, count, [&](uint32_t begin, uint32_t end) {
        for (uint32_t i = begin; i < end; ++i) {
            uint32_t cursor = pairStart[i];
            visit(i, [&](uint32_t j) { pairs[cursor++] = {i, j}; });
        }
    });

    // 평면은 동적 물체 전부와 짝이다.
    for (uint32_t plane : planes) {
        for (uint32_t j = 0; j < count; ++j) {
            if (j == plane || !isDynamic(bodies[j])) {
                continue;
            }
            pairs.emplace_back(std::min(plane, j), std::max(plane, j));
        }
    }
    // 순차 임펄스는 순서에 따라 결과가 갈린다. (i, j) 오름차순으로 고정한다.
    std::sort(pairs.begin(), pairs.end());
}

void stepRigidBodies(scene::Scene& scene, float dt, core::JobSystem* jobs) {
    if (dt <= 0.0F) {
        return;
    }
    // 1) 부품이 붙은 오브젝트를 모아 세계 공간 상태로 편다. GPU 백엔드는 여기서 빠진다.
    std::vector<Body> bodies;
    std::vector<Triangle> triangles;
    collectRigidBodies(scene, scene::SimulationBackend::CPU, bodies, triangles);
    if (bodies.empty()) {
        return;
    }

    // 2) 반암시적 오일러 적분.
    forRange(jobs, static_cast<uint32_t>(bodies.size()), [&](uint32_t begin, uint32_t end) {
        for (uint32_t i = begin; i < end; ++i) {
            integrate(bodies[i], dt);
        }
    });

    // 3) 광역: 해시 격자로 경계 구가 닿을 수 있는 짝을 모은다.
    std::vector<std::pair<uint32_t, uint32_t>> pairs;
    collectPairs(bodies, jobs, pairs);

    // 4) 협역: 짝마다 접촉을 만든다. 짝 수가 적어 직렬이다.
    std::vector<Contact> contacts;
    for (const auto& [i, j] : pairs) {
        collide(bodies, i, j, triangles, contacts);
    }

    // 5) 관절 속도 행을 먼저, 접촉을 순차 임펄스로, 그 뒤 관절 위치 보정.
    std::vector<JointState> joints;
    collectJoints(scene, bodies, joints);
    // 모터·한계각의 누적 임펄스는 반복 사이에 살아 있어야 상한이 반복 횟수와 무관해진다.
    std::vector<JointAccumulator> jointAccumulators(joints.size());
    for (uint32_t iteration = 0; iteration < SOLVER_ITERATIONS; ++iteration) {
        for (size_t index = 0; index < joints.size(); ++index) {
            solveJoint(bodies, joints[index], jointAccumulators[index], dt);
        }
    }
    solveContacts(bodies, contacts);
    for (uint32_t iteration = 0; iteration < POSITION_ITERATIONS; ++iteration) {
        for (const JointState& joint : joints) {
            correctJoint(bodies, joint);
        }
    }

    // 6) 세계 상태를 오브젝트의 지역 변환과 부품 속도로 되돌려 쓴다.
    writeBackRigidBodies(scene, bodies);
}

} // namespace physics
