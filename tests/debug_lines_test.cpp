#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/trigonometric.hpp>

#include "gfx/debug_lines.h"

namespace {

uint32_t addObject(scene::Scene& scene, glm::vec3 position, glm::vec3 scale) {
    scene::Object object;
    object.transform.position = position;
    object.transform.scale = scale;
    scene.objects.push_back(std::move(object));
    return static_cast<uint32_t>(scene.objects.size() - 1);
}

// 선분 목록의 모든 끝점이 조건을 만족하는지.
template <typename Predicate> bool allVertices(const std::vector<gfx::DebugLineVertex>& lines, Predicate predicate) {
    return std::all_of(lines.begin(), lines.end(), predicate);
}

bool anyVertex(const std::vector<gfx::DebugLineVertex>& lines, uint32_t color) {
    return std::any_of(
        lines.begin(), lines.end(), [color](const gfx::DebugLineVertex& vertex) { return vertex.color == color; });
}

} // namespace

int main() {
    std::vector<gfx::DebugLineVertex> lines;
    gfx::DebugLineOptions options;

    // ---- 구 콜라이더는 세계 공간 반지름의 원 세 개다 ----
    {
        scene::Scene scene;
        uint32_t object = addObject(scene, glm::vec3{2.0F, 1.0F, -3.0F}, glm::vec3{3.0F, 1.0F, 1.0F});
        scene::RigidBody body;
        body.shape = scene::ColliderShape::SPHERE;
        body.radius = 0.5F;
        scene.attachRigidBody(object, body);
        scene.refresh();

        gfx::buildDebugLines(scene, options, lines);
        assert(!lines.empty() && "구 콜라이더는 선을 만든다");
        assert(lines.size() % 2 == 0 && "선분 목록은 짝수 개의 끝점이다");
        // 구는 배율 중 가장 큰 축만 받는다(scene::colliderPose). 여기서는 3 이다.
        float expected = 0.5F * 3.0F;
        bool onSphere = allVertices(lines, [&](const gfx::DebugLineVertex& vertex) {
            float distance = glm::distance(vertex.position, glm::vec3{2.0F, 1.0F, -3.0F});
            return std::abs(distance - expected) < 1e-3F;
        });
        assert(onSphere && "모든 끝점이 콜라이더 표면에 있어야 한다");
        assert(anyVertex(lines, gfx::DEBUG_COLOR_COLLIDER) && "기본은 강체 색이다");

        // 고른 오브젝트만 밝게 그린다.
        options.selected = static_cast<int32_t>(object);
        gfx::buildDebugLines(scene, options, lines);
        assert(anyVertex(lines, gfx::DEBUG_COLOR_COLLIDER_SELECTED) && "고른 오브젝트는 밝은 색이다");
        assert(!anyVertex(lines, gfx::DEBUG_COLOR_COLLIDER));
        options.selected = -1;
    }

    // ---- 상자 콜라이더는 축마다 다른 배율을 받고 열두 모서리를 그린다 ----
    {
        scene::Scene scene;
        uint32_t object = addObject(scene, glm::vec3{0.0F}, glm::vec3{2.0F, 0.5F, 4.0F});
        scene::RigidBody body;
        body.shape = scene::ColliderShape::BOX;
        body.halfExtents = glm::vec3{1.0F, 1.0F, 1.0F};
        scene.attachRigidBody(object, body);
        scene.refresh();

        gfx::buildDebugLines(scene, options, lines);
        assert(lines.size() == 12 * 2 && "상자는 모서리 열둘이다");
        glm::vec3 expected{2.0F, 0.5F, 4.0F};
        bool onCorners = allVertices(lines, [&](const gfx::DebugLineVertex& vertex) {
            for (int axis = 0; axis < 3; ++axis) {
                if (std::abs(std::abs(vertex.position[axis]) - expected[axis]) > 1e-3F) {
                    return false;
                }
            }
            return true;
        });
        assert(onCorners && "모든 끝점이 상자의 꼭짓점이어야 한다");

        // 꼭짓점 개수만 보면 면 대각선을 그려도 통과한다. 모서리는 «한 축에서만 부호가 다른» 짝이다.
        for (size_t i = 0; i + 1 < lines.size(); i += 2) {
            int differing = 0;
            for (int axis = 0; axis < 3; ++axis) {
                if (std::abs(lines[i].position[axis] - lines[i + 1].position[axis]) > 1e-3F) {
                    ++differing;
                }
            }
            assert(differing == 1 && "모서리는 한 축만 다르다");
        }

        // 회전해도 크기는 그대로다. 중심에서 꼭짓점까지의 거리로 본다.
        scene.objects[object].transform.rotation = glm::angleAxis(0.7F, glm::normalize(glm::vec3{1.0F, 2.0F, 3.0F}));
        scene.refresh();
        gfx::buildDebugLines(scene, options, lines);
        float diagonal = glm::length(expected);
        bool sameSize = allVertices(lines, [&](const gfx::DebugLineVertex& vertex) {
            return std::abs(glm::length(vertex.position) - diagonal) < 1e-3F;
        });
        assert(sameSize && "회전은 크기를 바꾸지 않는다");
    }

    // ---- 유체는 용기와 방출 상자를 따로 그린다 ----
    {
        scene::Scene scene;
        uint32_t object = addObject(scene, glm::vec3{0.0F, 1.0F, 0.0F}, glm::vec3{1.0F});
        scene::Fluid fluid;
        fluid.containerMin = glm::vec3{-2.0F, 0.0F, -2.0F};
        fluid.containerMax = glm::vec3{2.0F, 3.0F, 2.0F};
        fluid.emitterHalfExtents = glm::vec3{0.25F};
        scene.attachFluid(object, fluid);
        scene.refresh();

        gfx::buildDebugLines(scene, options, lines);
        assert(lines.size() == 24 * 2 && "용기와 방출 상자로 상자 둘이다");
        assert(anyVertex(lines, gfx::DEBUG_COLOR_FLUID_CONTAINER));
        assert(anyVertex(lines, gfx::DEBUG_COLOR_FLUID_EMITTER));

        // 용기는 월드 공간이라 오브젝트를 옮겨도 그대로다.
        bool containerInPlace = std::any_of(lines.begin(), lines.end(), [](const gfx::DebugLineVertex& vertex) {
            return vertex.color == gfx::DEBUG_COLOR_FLUID_CONTAINER &&
                   glm::distance(vertex.position, glm::vec3{2.0F, 3.0F, 2.0F}) < 1e-3F;
        });
        assert(containerInPlace && "용기 꼭짓점은 설정 그대로여야 한다");

        // 방출 상자는 오브젝트를 따라다닌다.
        bool emitterMoved = std::any_of(lines.begin(), lines.end(), [](const gfx::DebugLineVertex& vertex) {
            return vertex.color == gfx::DEBUG_COLOR_FLUID_EMITTER &&
                   glm::distance(vertex.position, glm::vec3{0.25F, 1.25F, 0.25F}) < 1e-3F;
        });
        assert(emitterMoved && "방출 상자는 오브젝트 지역 공간이다");

        options.fluidBounds = false;
        gfx::buildDebugLines(scene, options, lines);
        assert(lines.empty() && "끄면 아무것도 그리지 않는다");
        options.fluidBounds = true;
    }

    // ---- 무한 평면은 오브젝트의 +Y 를 법선으로 하는 격자 한 장이다 ----
    {
        scene::Scene scene;
        uint32_t object = addObject(scene, glm::vec3{0.0F, 2.0F, 0.0F}, glm::vec3{1.0F});
        // 오브젝트를 90도 눕히면 법선이 +Y 에서 -Z 로 간다.
        scene.objects[object].transform.rotation = glm::angleAxis(glm::radians(90.0F), glm::vec3{1.0F, 0.0F, 0.0F});
        scene::RigidBody body;
        body.shape = scene::ColliderShape::PLANE;
        scene.attachRigidBody(object, body);
        scene.refresh();

        gfx::buildDebugLines(scene, options, lines);
        assert(!lines.empty() && "평면도 선을 만든다");
        assert(lines.size() % 2 == 0);
        // 모든 끝점이 그 평면 위에 있어야 한다. 법선은 오브젝트의 +Y 를 회전한 것이다.
        glm::vec3 normal = scene.objects[object].transform.rotation * glm::vec3{0.0F, 1.0F, 0.0F};
        glm::vec3 origin{0.0F, 2.0F, 0.0F};
        bool onPlane = allVertices(lines, [&](const gfx::DebugLineVertex& vertex) {
            return std::abs(glm::dot(vertex.position - origin, normal)) < 1e-3F;
        });
        assert(onPlane && "격자는 콜라이더 평면 위에 있어야 한다");
        assert(anyVertex(lines, gfx::DEBUG_COLOR_COLLIDER));

        options.colliders = false;
        gfx::buildDebugLines(scene, options, lines);
        assert(lines.empty() && "콜라이더 표시를 끄면 그리지 않는다");
        options.colliders = true;
    }

    // ---- 부모의 변환이 콜라이더에 함께 실린다 ----
    {
        scene::Scene scene;
        uint32_t parent = addObject(scene, glm::vec3{10.0F, 0.0F, 0.0F}, glm::vec3{2.0F});
        uint32_t child = addObject(scene, glm::vec3{1.0F, 0.0F, 0.0F}, glm::vec3{1.0F});
        scene.objects[child].parent = static_cast<int32_t>(parent);
        scene::RigidBody body;
        body.shape = scene::ColliderShape::SPHERE;
        body.radius = 0.5F;
        scene.attachRigidBody(child, body);
        scene.refresh();

        gfx::buildDebugLines(scene, options, lines);
        // 자식의 세계 위치는 (10 + 1*2, 0, 0) 이고 배율은 부모의 2 가 곱해진다.
        bool inWorldSpace = allVertices(lines, [](const gfx::DebugLineVertex& vertex) {
            return std::abs(glm::distance(vertex.position, glm::vec3{12.0F, 0.0F, 0.0F}) - 1.0F) < 1e-3F;
        });
        assert(inWorldSpace && "부모의 이동과 배율이 함께 실려야 한다");
    }

    // ---- 색 상수는 셰이더의 unpackUnorm4x8 과 같은 순서다 ----
    {
        // 0xAABBGGRR 이라 하위 바이트가 빨강이다. 리터럴만 보면 뒤집힌 것처럼 읽힌다.
        auto channel = [](uint32_t color, int index) { return (color >> (index * 8)) & 0xFFU; };
        assert(channel(gfx::DEBUG_COLOR_COLLIDER, 1) > channel(gfx::DEBUG_COLOR_COLLIDER, 0) &&
               channel(gfx::DEBUG_COLOR_COLLIDER, 1) > channel(gfx::DEBUG_COLOR_COLLIDER, 2) && "강체는 초록이다");
        assert(channel(gfx::DEBUG_COLOR_FLUID_CONTAINER, 0) < channel(gfx::DEBUG_COLOR_FLUID_CONTAINER, 2) &&
               "유체 용기는 청록이다");
        assert(channel(gfx::DEBUG_COLOR_FLUID_EMITTER, 2) < channel(gfx::DEBUG_COLOR_FLUID_EMITTER, 0) &&
               "방출 상자는 노랑이다");
    }

    // ---- 부품이 없으면 아무것도 그리지 않는다 ----
    {
        scene::Scene scene;
        addObject(scene, glm::vec3{0.0F}, glm::vec3{1.0F});
        scene.refresh();
        gfx::buildDebugLines(scene, options, lines);
        assert(lines.empty());
    }

    std::printf("디버그 선 자체 점검 통과\n");
    return 0;
}
