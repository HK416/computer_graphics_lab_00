#include "gfx/debug_lines.h"

#include <array>
#include <cmath>
#include <numbers>

#include <glm/gtc/quaternion.hpp>

namespace gfx {

namespace {

constexpr float PI = std::numbers::pi_v<float>;
// 원 하나를 이만큼의 선분으로 그린다. 강체가 수십 개여도 몇천 선분이라 부담이 없다.
constexpr uint32_t CIRCLE_SEGMENTS = 32;
// 무한 평면은 그릴 수 없으니 이만큼 크기의 격자 한 장으로 대신한다.
//
// ponytail: 크기가 고정이라 큰 장면에서는 점처럼 보이고 작은 장면에서는 화면을 덮는다. 장면 경계나
// 오브젝트 배율에 맞추려면 여기로 그 값을 넘겨야 한다.
constexpr float PLANE_EXTENT = 2.0F;
constexpr uint32_t PLANE_DIVISIONS = 4;

void line(std::vector<DebugLineVertex>& out, glm::vec3 from, glm::vec3 to, uint32_t color) {
    out.push_back(DebugLineVertex{from, color});
    out.push_back(DebugLineVertex{to, color});
}

// 중심에서 두 축이 만드는 평면 위의 원.
void circle(std::vector<DebugLineVertex>& out,
            glm::vec3 center,
            glm::vec3 axisU,
            glm::vec3 axisV,
            float radius,
            uint32_t color) {
    glm::vec3 previous = center + axisU * radius;
    for (uint32_t segment = 1; segment <= CIRCLE_SEGMENTS; ++segment) {
        float angle = 2.0F * PI * static_cast<float>(segment) / static_cast<float>(CIRCLE_SEGMENTS);
        glm::vec3 point = center + (axisU * std::cos(angle) + axisV * std::sin(angle)) * radius;
        line(out, previous, point, color);
        previous = point;
    }
}

// 회전한 상자의 열두 모서리.
void box(std::vector<DebugLineVertex>& out,
         glm::vec3 center,
         const glm::quat& rotation,
         glm::vec3 halfExtents,
         uint32_t color) {
    std::array<glm::vec3, 8> corners{};
    for (uint32_t i = 0; i < corners.size(); ++i) {
        glm::vec3 sign{(i & 1U) != 0U ? 1.0F : -1.0F, (i & 2U) != 0U ? 1.0F : -1.0F, (i & 4U) != 0U ? 1.0F : -1.0F};
        corners[i] = center + rotation * (sign * halfExtents);
    }
    // 비트 하나만 다른 꼭짓점끼리가 모서리다. 각 짝을 한 번씩만 잇는다.
    for (uint32_t i = 0; i < corners.size(); ++i) {
        for (uint32_t bit = 1; bit < 8U; bit <<= 1U) {
            uint32_t other = i | bit;
            if (other != i) {
                line(out, corners[i], corners[other], color);
            }
        }
    }
}

// 축 정렬 상자. 유체의 용기가 이 꼴이다.
void axisAlignedBox(std::vector<DebugLineVertex>& out, glm::vec3 minimum, glm::vec3 maximum, uint32_t color) {
    box(out, (minimum + maximum) * 0.5F, glm::quat{1.0F, 0.0F, 0.0F, 0.0F}, (maximum - minimum) * 0.5F, color);
}

} // namespace

void buildDebugLines(const scene::Scene& scene, const DebugLineOptions& options, std::vector<DebugLineVertex>& out) {
    out.clear();
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        const scene::Object& object = scene.objects[index];
        bool selected = static_cast<int32_t>(index) == options.selected;
        glm::mat4 world = scene.world(index);

        if (options.colliders && object.rigidBody >= 0 &&
            static_cast<size_t>(object.rigidBody) < scene.rigidBodies.size()) {
            const scene::RigidBody& body = scene.rigidBodies[static_cast<size_t>(object.rigidBody)];
            scene::ColliderPose pose = scene::colliderPose(body, world);
            uint32_t color = selected ? DEBUG_COLOR_COLLIDER_SELECTED : DEBUG_COLOR_COLLIDER;
            switch (body.shape) {
            case scene::ColliderShape::SPHERE: {
                // 세 축의 원만 그린다. Unity 도 그렇게 그리고, 구는 그것만으로 크기가 읽힌다.
                circle(out, pose.position, {1, 0, 0}, {0, 1, 0}, pose.radius, color);
                circle(out, pose.position, {0, 1, 0}, {0, 0, 1}, pose.radius, color);
                circle(out, pose.position, {0, 0, 1}, {1, 0, 0}, pose.radius, color);
                break;
            }
            case scene::ColliderShape::BOX:
                box(out, pose.position, pose.rotation, pose.halfExtents, color);
                break;
            case scene::ColliderShape::PLANE: {
                // 무한 평면이라 오브젝트의 +Y 를 법선으로 하는 격자 한 장으로 대신한다.
                glm::vec3 right = pose.rotation * glm::vec3{1.0F, 0.0F, 0.0F};
                glm::vec3 forward = pose.rotation * glm::vec3{0.0F, 0.0F, 1.0F};
                for (uint32_t i = 0; i <= PLANE_DIVISIONS; ++i) {
                    float t = static_cast<float>(i) / static_cast<float>(PLANE_DIVISIONS) * 2.0F - 1.0F;
                    glm::vec3 alongRight = right * (t * PLANE_EXTENT);
                    glm::vec3 alongForward = forward * (t * PLANE_EXTENT);
                    line(out,
                         pose.position + alongRight - forward * PLANE_EXTENT,
                         pose.position + alongRight + forward * PLANE_EXTENT,
                         color);
                    line(out,
                         pose.position + alongForward - right * PLANE_EXTENT,
                         pose.position + alongForward + right * PLANE_EXTENT,
                         color);
                }
                break;
            }
            }
        }

        if (options.fluidBounds && object.fluid >= 0 && static_cast<size_t>(object.fluid) < scene.fluids.size()) {
            const scene::Fluid& fluid = scene.fluids[static_cast<size_t>(object.fluid)];
            // 용기는 월드 공간, 방출 상자는 오브젝트 지역 공간이다(scene::Fluid 주석).
            axisAlignedBox(out, fluid.containerMin, fluid.containerMax, DEBUG_COLOR_FLUID_CONTAINER);
            scene::Transform emitter = scene::Transform::fromMatrix(world);
            box(out,
                emitter.position,
                glm::normalize(emitter.rotation),
                fluid.emitterHalfExtents * emitter.scale,
                DEBUG_COLOR_FLUID_EMITTER);
        }
    }
}

} // namespace gfx
