#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>

#include <glm/geometric.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "scene/scene.h"

namespace physics {

// GPU 경로의 shaders/force_field.glsl 과 같은 값·같은 식이어야 한다. 천·유체·입자가 세 백엔드에서 같은 힘을 본다.
inline constexpr uint32_t MAX_FORCE_FIELDS = 8;

// 장면에서 풀어 낸 힘 마당 하나(세계 공간).
struct ForceFieldSample {
    glm::vec3 position{0.0F};
    // 바람의 방향(-Z 앞) 또는 소용돌이의 축(+Y). 점은 쓰지 않는다.
    glm::vec3 axis{0.0F, 0.0F, -1.0F};
    float strength = 0.0F;
    float radius = 0.0F;
    float falloff = 1.0F;
    scene::ForceFieldType type = scene::ForceFieldType::WIND;
};

// 한 점에서의 가속도. shaders/force_field.glsl 의 forceFieldAcceleration 과 같은 식이다.
inline glm::vec3 forceFieldAcceleration(const ForceFieldSample& field, const glm::vec3& position) {
    glm::vec3 offset = position - field.position;
    float distance = glm::length(offset);
    float weight = 1.0F;
    if (field.radius > 0.0F) {
        float t = std::clamp(1.0F - distance / field.radius, 0.0F, 1.0F);
        weight = t <= 0.0F ? 0.0F : std::pow(t, field.falloff);
    }
    if (weight <= 0.0F) {
        return glm::vec3{0.0F};
    }
    switch (field.type) {
    case scene::ForceFieldType::WIND:
        return field.axis * (field.strength * weight);
    case scene::ForceFieldType::VORTEX: {
        glm::vec3 radial = offset - field.axis * glm::dot(offset, field.axis);
        float r = glm::length(radial);
        if (r < 1.0e-4F) {
            return glm::vec3{0.0F};
        }
        return glm::cross(field.axis, radial / r) * (field.strength * weight);
    }
    case scene::ForceFieldType::POINT:
        if (distance < 1.0e-4F) {
            return glm::vec3{0.0F};
        }
        return (-offset / distance) * (field.strength * weight);
    }
    return glm::vec3{0.0F};
}

inline glm::vec3 forceFieldsAcceleration(const std::array<ForceFieldSample, MAX_FORCE_FIELDS>& fields,
                                         uint32_t count,
                                         const glm::vec3& position) {
    glm::vec3 total{0.0F};
    for (uint32_t i = 0; i < count; ++i) {
        total += forceFieldAcceleration(fields[i], position);
    }
    return total;
}

// 장면의 힘 마당 부품(보이는 것만)을 세계 공간으로 푼다. 콜라이더 수집(collectShapeColliders)과 같은 규칙이다.
inline uint32_t collectForceFields(const scene::Scene& scene, std::array<ForceFieldSample, MAX_FORCE_FIELDS>& out) {
    uint32_t count = 0;
    for (uint32_t index = 0; index < scene.objects.size() && count < MAX_FORCE_FIELDS; ++index) {
        int32_t slot = scene.objects[index].forceField;
        if (slot < 0 || static_cast<size_t>(slot) >= scene.forceFields.size() || !scene.visibleCached(index)) {
            continue;
        }
        const scene::ForceField& field = scene.forceFields[static_cast<size_t>(slot)];
        const glm::mat4& world = scene.world(index);
        ForceFieldSample& sample = out[count++];
        sample.position = glm::vec3(world[3]);
        glm::vec3 axis = field.type == scene::ForceFieldType::VORTEX ? glm::vec3(world[1]) : -glm::vec3(world[2]);
        float length = glm::length(axis);
        sample.axis = length > 1.0e-6F ? axis / length : glm::vec3{0.0F, 1.0F, 0.0F};
        sample.strength = field.strength;
        sample.radius = std::max(field.radius, 0.0F);
        sample.falloff = std::max(field.falloff, 0.0F);
        sample.type = field.type;
    }
    return count;
}

} // namespace physics
