#pragma once

#include <algorithm>
#include <cmath>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/norm.hpp>
#include <glm/vec3.hpp>

#include "scene/scene.h"

// 콜라이더 모양의 기하. 강체 솔버(CPU)와 유체 CPU 백엔드가 함께 쓰고, GPU 쪽은
// shaders/collider_shapes.glsl 이 같은 규칙을 갖는다. 한쪽을 고치면 다른 쪽도 고친다.
//
// 모두 콜라이더의 **지역 공간**이다. 원기둥·캡슐의 축과 평면의 법선은 +Y 다. 부르는 쪽이 세계 ↔ 지역
// 변환을 맡는다(강체는 쿼터니언, 유체는 행렬).
namespace physics {

// 지역 공간 콜라이더 하나. halfExtents.y 는 원기둥·캡슐의 반높이다.
struct ColliderLocal {
    scene::ColliderShape shape = scene::ColliderShape::SPHERE;
    float radius = 0.0F;
    glm::vec3 halfExtents{0.0F};
};

// 점에서 가장 가까운 표면점. distance 는 부호 있는 거리(안이면 음수), normal 은 표면의 바깥 방향이다.
struct SurfacePoint {
    glm::vec3 point{0.0F};
    glm::vec3 normal{0.0F, 1.0F, 0.0F};
    float distance = 0.0F;
};

inline SurfacePoint closestOnSphereLocal(float radius, const glm::vec3& p) {
    SurfacePoint result;
    float d = glm::length(p);
    result.normal = d > 1.0e-6F ? p / d : glm::vec3{0.0F, 1.0F, 0.0F};
    result.point = result.normal * radius;
    result.distance = d - radius;
    return result;
}

// 상자. 바깥 점은 클램프한 점, 안쪽 점은 가장 얕은 면으로 나간다.
inline SurfacePoint closestOnBoxLocal(const glm::vec3& extent, const glm::vec3& p) {
    SurfacePoint result;
    glm::vec3 clamped = glm::clamp(p, -extent, extent);
    glm::vec3 delta = p - clamped;
    float distanceSq = glm::length2(delta);
    if (distanceSq > 1.0e-10F) {
        float distance = std::sqrt(distanceSq);
        result.normal = delta / distance;
        result.point = clamped;
        result.distance = distance;
        return result;
    }
    glm::vec3 gap = extent - glm::abs(p);
    int axis = gap.x < gap.y ? (gap.x < gap.z ? 0 : 2) : (gap.y < gap.z ? 1 : 2);
    glm::vec3 normal{0.0F};
    normal[axis] = p[axis] >= 0.0F ? 1.0F : -1.0F;
    result.point = p;
    result.point[axis] = normal[axis] * extent[axis];
    result.normal = normal;
    result.distance = -gap[axis];
    return result;
}

inline SurfacePoint closestOnPlaneLocal(const glm::vec3& p) {
    SurfacePoint result;
    result.normal = glm::vec3{0.0F, 1.0F, 0.0F};
    result.point = glm::vec3{p.x, 0.0F, p.z};
    result.distance = p.y;
    return result;
}

// 캡슐: Y 축 선분(±halfHeight)에서 가장 가까운 점을 중심으로 한 구.
inline SurfacePoint closestOnCapsuleLocal(float radius, float halfHeight, const glm::vec3& p) {
    glm::vec3 onSegment{0.0F, std::clamp(p.y, -halfHeight, halfHeight), 0.0F};
    SurfacePoint result = closestOnSphereLocal(radius, p - onSegment);
    result.point += onSegment;
    return result;
}

// 원기둥. 바깥은 축 방향과 반지름 방향을 따로 클램프하고, 안은 뚜껑과 옆면 중 얕은 쪽으로 나간다.
inline SurfacePoint closestOnCylinderLocal(float radius, float halfHeight, const glm::vec3& p) {
    SurfacePoint result;
    glm::vec3 radial{p.x, 0.0F, p.z};
    float d = glm::length(radial);
    glm::vec3 radialDirection = d > 1.0e-6F ? radial / d : glm::vec3{1.0F, 0.0F, 0.0F};
    bool inside = std::abs(p.y) <= halfHeight && d <= radius;
    if (inside) {
        float gapY = halfHeight - std::abs(p.y);
        float gapR = radius - d;
        if (gapY < gapR) {
            float side = p.y >= 0.0F ? 1.0F : -1.0F;
            result.normal = glm::vec3{0.0F, side, 0.0F};
            result.point = glm::vec3{p.x, side * halfHeight, p.z};
            result.distance = -gapY;
        } else {
            result.normal = radialDirection;
            result.point = radialDirection * radius + glm::vec3{0.0F, p.y, 0.0F};
            result.distance = -gapR;
        }
        return result;
    }
    glm::vec3 closest = radialDirection * std::min(d, radius);
    closest.y = std::clamp(p.y, -halfHeight, halfHeight);
    glm::vec3 delta = p - closest;
    float distance = glm::length(delta);
    result.point = closest;
    result.normal = distance > 1.0e-6F ? delta / distance : glm::vec3{0.0F, 1.0F, 0.0F};
    result.distance = distance;
    return result;
}

// 메쉬는 볼록하지 않아 여기 없다. 삼각형 단위로 closestOnTriangle 을 쓴다.
inline SurfacePoint closestOnColliderLocal(const ColliderLocal& collider, const glm::vec3& p) {
    switch (collider.shape) {
    case scene::ColliderShape::SPHERE:
        return closestOnSphereLocal(collider.radius, p);
    case scene::ColliderShape::BOX:
        return closestOnBoxLocal(collider.halfExtents, p);
    case scene::ColliderShape::PLANE:
        return closestOnPlaneLocal(p);
    case scene::ColliderShape::CYLINDER:
        return closestOnCylinderLocal(collider.radius, collider.halfExtents.y, p);
    case scene::ColliderShape::CAPSULE:
        return closestOnCapsuleLocal(collider.radius, collider.halfExtents.y, p);
    case scene::ColliderShape::MESH:
        break;
    }
    return closestOnSphereLocal(0.0F, p);
}

// 세계 공간 편의 함수. rotation 은 지역 → 세계.
inline SurfacePoint closestOnCollider(const ColliderLocal& collider,
                                      const glm::vec3& position,
                                      const glm::quat& rotation,
                                      const glm::vec3& worldPoint) {
    glm::vec3 local = glm::conjugate(rotation) * (worldPoint - position);
    SurfacePoint result = closestOnColliderLocal(collider, local);
    result.point = position + rotation * result.point;
    result.normal = rotation * result.normal;
    return result;
}

// 원기둥 테두리 표본 각도의 순서. 90도 간격 넷을 앞에 두어, 서 있는 원기둥이 뚜껑 전체로 닿아 깊이가 같을
// 때 접촉 넷이 균형 잡히게 남는다(addManifoldPoint 는 같은 깊이를 바꾸지 않는다).
inline constexpr int CYLINDER_RIM_SAMPLES = 8;
inline constexpr float CYLINDER_RIM_ANGLES[CYLINDER_RIM_SAMPLES] = {
    0.0F, 1.5707964F, 3.1415927F, 4.712389F, 0.7853982F, 2.3561945F, 3.9269907F, 5.4977870F};

// 상대와의 접촉을 찾을 때 콜라이더 표면에서 «찔러 보는» 점의 수. 구는 중심 하나(반지름을 가짐),
// 상자는 꼭짓점 여덟, 캡슐은 두 끝과 힌트에 가장 가까운 축 위 점, 원기둥은 힌트 방향의 테두리 두 점 +
// 뚜껑 중심 둘 + 테두리 표본 열여섯이다. 평면과 메쉬는 찌르지 않고 찔리기만 한다.
inline int probeCount(scene::ColliderShape shape) {
    switch (shape) {
    case scene::ColliderShape::SPHERE:
        return 1;
    case scene::ColliderShape::BOX:
        return 8;
    case scene::ColliderShape::CAPSULE:
        return 3;
    case scene::ColliderShape::CYLINDER:
        return 4 + 2 * CYLINDER_RIM_SAMPLES;
    case scene::ColliderShape::PLANE:
    case scene::ColliderShape::MESH:
        break;
    }
    return 0;
}

// index 번째 찌르는 점(지역 공간)과 그 점의 반지름. hint 는 상대 표면에서 이 콜라이더 중심에 가장 가까운
// 점(지역 공간)으로, 캡슐·원기둥이 상대를 향한 표본을 하나씩 더 고르는 데 쓴다.
inline glm::vec3 probePointLocal(const ColliderLocal& collider, const glm::vec3& hint, int index, float& radius) {
    radius = 0.0F;
    switch (collider.shape) {
    case scene::ColliderShape::SPHERE:
        radius = collider.radius;
        return glm::vec3{0.0F};
    case scene::ColliderShape::BOX:
        return glm::vec3{(index & 1) != 0 ? collider.halfExtents.x : -collider.halfExtents.x,
                         (index & 2) != 0 ? collider.halfExtents.y : -collider.halfExtents.y,
                         (index & 4) != 0 ? collider.halfExtents.z : -collider.halfExtents.z};
    case scene::ColliderShape::CAPSULE: {
        radius = collider.radius;
        float h = collider.halfExtents.y;
        float y = index == 0 ? h : (index == 1 ? -h : std::clamp(hint.y, -h, h));
        return glm::vec3{0.0F, y, 0.0F};
    }
    case scene::ColliderShape::CYLINDER: {
        float h = collider.halfExtents.y;
        float r = collider.radius;
        if (index < 2) {
            // 힌트 방향의 테두리 점. 힌트가 축 위면 방향이 없어 +X 를 쓴다.
            glm::vec3 radial{hint.x, 0.0F, hint.z};
            float d = glm::length(radial);
            glm::vec3 direction = d > 1.0e-6F ? radial / d : glm::vec3{1.0F, 0.0F, 0.0F};
            return direction * r + glm::vec3{0.0F, index == 0 ? h : -h, 0.0F};
        }
        if (index < 4) {
            return glm::vec3{0.0F, index == 2 ? h : -h, 0.0F};
        }
        int rim = index - 4;
        float angle = CYLINDER_RIM_ANGLES[rim % CYLINDER_RIM_SAMPLES];
        float y = rim < CYLINDER_RIM_SAMPLES ? h : -h;
        return glm::vec3{std::cos(angle) * r, y, std::sin(angle) * r};
    }
    case scene::ColliderShape::PLANE:
    case scene::ColliderShape::MESH:
        break;
    }
    return glm::vec3{0.0F};
}

// 삼각형 위의 가장 가까운 점(Ericson, Real-Time Collision Detection 5.1.5).
inline glm::vec3 closestOnTriangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, const glm::vec3& p) {
    glm::vec3 ab = b - a;
    glm::vec3 ac = c - a;
    glm::vec3 ap = p - a;
    float d1 = glm::dot(ab, ap);
    float d2 = glm::dot(ac, ap);
    if (d1 <= 0.0F && d2 <= 0.0F) {
        return a;
    }
    glm::vec3 bp = p - b;
    float d3 = glm::dot(ab, bp);
    float d4 = glm::dot(ac, bp);
    if (d3 >= 0.0F && d4 <= d3) {
        return b;
    }
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0F && d1 >= 0.0F && d3 <= 0.0F) {
        float v = d1 / (d1 - d3);
        return a + ab * v;
    }
    glm::vec3 cp = p - c;
    float d5 = glm::dot(ab, cp);
    float d6 = glm::dot(ac, cp);
    if (d6 >= 0.0F && d5 <= d6) {
        return c;
    }
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0F && d2 >= 0.0F && d6 <= 0.0F) {
        float w = d2 / (d2 - d6);
        return a + ac * w;
    }
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0F && (d4 - d3) >= 0.0F && (d5 - d6) >= 0.0F) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + (c - b) * w;
    }
    float denominator = 1.0F / (va + vb + vc);
    float v = vb * denominator;
    float w = vc * denominator;
    return a + ab * v + ac * w;
}

// 반지름 radius 의 점이 삼각형(앞면 CCW)에 닿는지. 앞면 쪽에서는 구 대 삼각형이고, 뒷면으로 thickness
// 안까지 들어간 점은 앞면 법선으로 밀어 올린다(이미 뚫은 점이 계속 뚫고 내려가지 않게). normal 은
// 삼각형에서 점을 향하는 바깥 방향이고, distance 는 부호 있는 거리다.
inline bool closestOnTriangleSurface(const glm::vec3& a,
                                     const glm::vec3& b,
                                     const glm::vec3& c,
                                     const glm::vec3& p,
                                     float radius,
                                     float thickness,
                                     SurfacePoint& result) {
    glm::vec3 faceNormal = glm::cross(b - a, c - a);
    float area = glm::length(faceNormal);
    if (area < 1.0e-12F) {
        return false;
    }
    faceNormal /= area;
    float signedHeight = glm::dot(p - a, faceNormal);
    glm::vec3 closest = closestOnTriangle(a, b, c, p);
    glm::vec3 delta = p - closest;
    float distance = glm::length(delta);
    if (signedHeight >= 0.0F) {
        if (distance >= radius) {
            return false;
        }
        result.point = closest;
        result.normal = distance > 1.0e-6F ? delta / distance : faceNormal;
        result.distance = distance;
        return true;
    }
    // 뒷면. 삼각형 안쪽으로 곧장 투영된 점만(가장 가까운 점이 곧 발이면) 잡고, 너무 깊으면 다른 면의
    // 것이라 보고 놓아 준다.
    if (-signedHeight > thickness || std::abs(distance + signedHeight) > 1.0e-4F * std::max(1.0F, distance)) {
        return false;
    }
    result.point = closest;
    result.normal = faceNormal;
    result.distance = signedHeight;
    return true;
}

} // namespace physics
