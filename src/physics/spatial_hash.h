#pragma once

#include <cstdint>

#include <glm/common.hpp>
#include <glm/vec3.hpp>

namespace physics {

// 균일 격자 해시. 유체 SPH 와 강체 광역 검사가 함께 쓴다. shaders/spatial_hash.glsl 의 같은 이름 함수와
// 결과가 같아야 두 백엔드가 같은 이웃을 본다. cellCount 는 2 의 거듭제곱이어야 한다.
inline glm::ivec3 spatialCell(const glm::vec3& position, float cellSize) {
    return glm::ivec3{glm::floor(position / cellSize)};
}

inline uint32_t spatialHash(const glm::ivec3& cell, uint32_t cellCount) {
    uint32_t hashed = static_cast<uint32_t>(cell.x) * 73856093U ^ static_cast<uint32_t>(cell.y) * 19349663U ^
                      static_cast<uint32_t>(cell.z) * 83492791U;
    return hashed & (cellCount - 1U);
}

} // namespace physics
