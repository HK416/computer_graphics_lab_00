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

// 원기둥: 두 뚜껑의 원과 네 세로선.
void cylinder(std::vector<DebugLineVertex>& out,
              glm::vec3 center,
              const glm::quat& rotation,
              float radius,
              float halfHeight,
              uint32_t color) {
    glm::vec3 up = rotation * glm::vec3{0.0F, 1.0F, 0.0F};
    glm::vec3 right = rotation * glm::vec3{1.0F, 0.0F, 0.0F};
    glm::vec3 forward = rotation * glm::vec3{0.0F, 0.0F, 1.0F};
    glm::vec3 top = center + up * halfHeight;
    glm::vec3 bottom = center - up * halfHeight;
    circle(out, top, right, forward, radius, color);
    circle(out, bottom, right, forward, radius, color);
    for (const glm::vec3& side : {right, -right, forward, -forward}) {
        line(out, top + side * radius, bottom + side * radius, color);
    }
}

// 캡슐: 원기둥에 두 반구(두 평면의 반원씩)를 씌운다.
void capsule(std::vector<DebugLineVertex>& out,
             glm::vec3 center,
             const glm::quat& rotation,
             float radius,
             float halfHeight,
             uint32_t color) {
    cylinder(out, center, rotation, radius, halfHeight, color);
    glm::vec3 up = rotation * glm::vec3{0.0F, 1.0F, 0.0F};
    glm::vec3 right = rotation * glm::vec3{1.0F, 0.0F, 0.0F};
    glm::vec3 forward = rotation * glm::vec3{0.0F, 0.0F, 1.0F};
    for (float side : {1.0F, -1.0F}) {
        glm::vec3 cap = center + up * (side * halfHeight);
        for (const glm::vec3& axis : {right, forward}) {
            glm::vec3 previous = cap + axis * radius;
            for (uint32_t segment = 1; segment <= CIRCLE_SEGMENTS / 2; ++segment) {
                float angle = PI * static_cast<float>(segment) / static_cast<float>(CIRCLE_SEGMENTS / 2);
                glm::vec3 point = cap + (axis * std::cos(angle) + up * (side * std::sin(angle))) * radius;
                line(out, previous, point, color);
                previous = point;
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
            case scene::ColliderShape::CYLINDER:
                cylinder(out, pose.position, pose.rotation, pose.radius, pose.halfExtents.y, color);
                break;
            case scene::ColliderShape::CAPSULE:
                capsule(out, pose.position, pose.rotation, pose.radius, pose.halfExtents.y, color);
                break;
            case scene::ColliderShape::MESH: {
                // 콜라이더가 실제로 보는 삼각형(굵은 LOD)을 그린다. 메쉬가 없으면 표시할 것이 없다.
                const scene::ColliderMesh* mesh = scene.colliderMesh(index);
                if (mesh == nullptr) {
                    break;
                }
                for (size_t t = 0; t + 2 < mesh->indices.size(); t += 3) {
                    glm::vec3 a{world * glm::vec4{mesh->positions[mesh->indices[t]], 1.0F}};
                    glm::vec3 b{world * glm::vec4{mesh->positions[mesh->indices[t + 1]], 1.0F}};
                    glm::vec3 c{world * glm::vec4{mesh->positions[mesh->indices[t + 2]], 1.0F}};
                    line(out, a, b, color);
                    line(out, b, c, color);
                    line(out, c, a, color);
                }
                break;
            }
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

        // 관절: 앵커 A 와 B 를 잇는 선과 앵커의 십자. B 가 강체가 아니면 그 오브젝트(또는 세계)의 고정점이다.
        if (options.colliders && object.joint >= 0 && static_cast<size_t>(object.joint) < scene.joints.size()) {
            const scene::Joint& joint = scene.joints[static_cast<size_t>(object.joint)];
            glm::vec3 anchorA = glm::vec3(world * glm::vec4{joint.anchorA, 1.0F});
            glm::vec3 anchorB =
                joint.other >= 0 && static_cast<size_t>(joint.other) < scene.objects.size()
                    ? glm::vec3(scene.worldMatrix(static_cast<uint32_t>(joint.other)) * glm::vec4{joint.anchorB, 1.0F})
                    : joint.anchorB;
            line(out, anchorA, anchorB, DEBUG_COLOR_JOINT);
            for (glm::vec3 anchor : {anchorA, anchorB}) {
                constexpr float CROSS = 0.06F;
                line(out,
                     anchor - glm::vec3{CROSS, 0.0F, 0.0F},
                     anchor + glm::vec3{CROSS, 0.0F, 0.0F},
                     DEBUG_COLOR_JOINT);
                line(out,
                     anchor - glm::vec3{0.0F, CROSS, 0.0F},
                     anchor + glm::vec3{0.0F, CROSS, 0.0F},
                     DEBUG_COLOR_JOINT);
                line(out,
                     anchor - glm::vec3{0.0F, 0.0F, CROSS},
                     anchor + glm::vec3{0.0F, 0.0F, CROSS},
                     DEBUG_COLOR_JOINT);
            }
        }

        // 카메라 부품: 앞(-Z)으로 벌어지는 작은 절두체. 활성이면 밝다.
        if (options.fluidBounds && object.cameraComponent >= 0 &&
            static_cast<size_t>(object.cameraComponent) < scene.cameraComponents.size()) {
            const scene::CameraComponent& camera = scene.cameraComponents[static_cast<size_t>(object.cameraComponent)];
            glm::vec3 eye = glm::vec3(world[3]);
            auto unit = [](glm::vec3 v) {
                return glm::length(v) > 1.0e-6F ? glm::normalize(v) : glm::vec3{0.0F, 1.0F, 0.0F};
            };
            glm::vec3 forward = unit(-glm::vec3(world[2]));
            glm::vec3 up = unit(glm::vec3(world[1]));
            glm::vec3 right = unit(glm::vec3(world[0]));
            float depth = 0.5F;
            float halfHeight = depth * std::tan(glm::radians(camera.fovYDegrees) * 0.5F);
            float halfWidth = halfHeight * 16.0F / 9.0F;
            glm::vec3 center = eye + forward * depth;
            std::array<glm::vec3, 4> corners{center + up * halfHeight - right * halfWidth,
                                             center + up * halfHeight + right * halfWidth,
                                             center - up * halfHeight + right * halfWidth,
                                             center - up * halfHeight - right * halfWidth};
            uint32_t color = camera.active ? DEBUG_COLOR_CAMERA : DEBUG_COLOR_COLLIDER;
            for (size_t i = 0; i < corners.size(); ++i) {
                line(out, eye, corners[i], color);
                line(out, corners[i], corners[(i + 1) % corners.size()], color);
            }
        }

        // 카메라 경로: 키를 잇는 Catmull-Rom 곡선과 키마다의 작은 십자.
        if (options.fluidBounds && object.cameraPath >= 0 &&
            static_cast<size_t>(object.cameraPath) < scene.cameraPaths.size()) {
            const scene::CameraPath& path = scene.cameraPaths[static_cast<size_t>(object.cameraPath)];
            if (path.keys.size() >= 2 && path.duration > 0.0F) {
                constexpr uint32_t STEPS_PER_KEY = 12;
                auto steps = static_cast<uint32_t>(path.keys.size()) * STEPS_PER_KEY;
                glm::vec3 previous = scene::evaluateCameraPath(path, 0.0F).position;
                for (uint32_t step = 1; step <= steps; ++step) {
                    float seconds = path.duration * static_cast<float>(step) / static_cast<float>(steps);
                    if (!path.loop && step == steps) {
                        seconds = path.duration;
                    }
                    glm::vec3 point = scene::evaluateCameraPath(path, seconds).position;
                    line(out, previous, point, DEBUG_COLOR_CAMERA);
                    previous = point;
                }
            }
            for (const scene::CameraKey& key : path.keys) {
                constexpr float CROSS = 0.08F;
                line(out,
                     key.position - glm::vec3{CROSS, 0.0F, 0.0F},
                     key.position + glm::vec3{CROSS, 0.0F, 0.0F},
                     DEBUG_COLOR_CAMERA);
                line(out,
                     key.position - glm::vec3{0.0F, CROSS, 0.0F},
                     key.position + glm::vec3{0.0F, CROSS, 0.0F},
                     DEBUG_COLOR_CAMERA);
                line(out,
                     key.position - glm::vec3{0.0F, 0.0F, CROSS},
                     key.position + glm::vec3{0.0F, 0.0F, CROSS},
                     DEBUG_COLOR_CAMERA);
            }
        }

        // DDGI 볼륨. 위치 ± 배율의 축 정렬 상자. 꺼진 것도 그린다(자리를 잡는 중일 수 있다).
        if (options.fluidBounds && object.ddgiVolume >= 0 &&
            static_cast<size_t>(object.ddgiVolume) < scene.ddgiVolumes.size()) {
            glm::vec3 center = glm::vec3(world[3]);
            glm::vec3 half{
                glm::length(glm::vec3(world[0])), glm::length(glm::vec3(world[1])), glm::length(glm::vec3(world[2]))};
            axisAlignedBox(out, center - half, center + half, DEBUG_COLOR_DDGI_VOLUME);
        }

        // 힘 마당. 바람은 앞(-Z) 화살표, 소용돌이는 +Y 축과 둘레 원, 점은 세 축 원. 반지름 0(무한)은 1 로 그린다.
        if (options.fluidBounds && object.forceField >= 0 &&
            static_cast<size_t>(object.forceField) < scene.forceFields.size()) {
            const scene::ForceField& field = scene.forceFields[static_cast<size_t>(object.forceField)];
            glm::vec3 center = glm::vec3(world[3]);
            auto unit = [](glm::vec3 v) {
                return glm::length(v) > 1.0e-6F ? glm::normalize(v) : glm::vec3{0.0F, 1.0F, 0.0F};
            };
            glm::vec3 forward = unit(-glm::vec3(world[2]));
            glm::vec3 up = unit(glm::vec3(world[1]));
            glm::vec3 right = unit(glm::vec3(world[0]));
            float radius = field.radius > 0.0F ? field.radius : 1.0F;
            uint32_t color = DEBUG_COLOR_FORCE_FIELD;
            switch (field.type) {
            case scene::ForceFieldType::WIND: {
                glm::vec3 tip = center + forward * radius;
                line(out, center, tip, color);
                line(out, tip, tip - forward * (0.2F * radius) + right * (0.1F * radius), color);
                line(out, tip, tip - forward * (0.2F * radius) - right * (0.1F * radius), color);
                circle(out, center, right, up, radius, color);
                break;
            }
            case scene::ForceFieldType::VORTEX:
                line(out, center - up * (0.5F * radius), center + up * (0.5F * radius), color);
                circle(out, center, right, forward, radius, color);
                break;
            case scene::ForceFieldType::POINT:
                circle(out, center, right, up, radius, color);
                circle(out, center, right, forward, radius, color);
                circle(out, center, up, forward, radius, color);
                break;
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
