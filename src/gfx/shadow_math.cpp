#include "gfx/shadow_math.h"

#include <algorithm>
#include <cmath>

#include <glm/geometric.hpp>

namespace gfx {

uint32_t extractFrustumPlanes(const glm::mat4& viewProjection,
                              std::array<glm::vec4, MAX_FRUSTUM_PLANES>& planes,
                              bool hasFarPlane) {
    glm::vec4 row0{viewProjection[0][0], viewProjection[1][0], viewProjection[2][0], viewProjection[3][0]};
    glm::vec4 row1{viewProjection[0][1], viewProjection[1][1], viewProjection[2][1], viewProjection[3][1]};
    glm::vec4 row2{viewProjection[0][2], viewProjection[1][2], viewProjection[2][2], viewProjection[3][2]};
    glm::vec4 row3{viewProjection[0][3], viewProjection[1][3], viewProjection[2][3], viewProjection[3][3]};

    planes[0] = row3 + row0; // 좌
    planes[1] = row3 - row0; // 우
    planes[2] = row3 + row1; // 하
    planes[3] = row3 - row1; // 상
    planes[4] = row3 - row2; // 근 (reverse-Z 는 row3 - row2 가 근평면이다)

    uint32_t count = 5;
    if (hasFarPlane) {
        planes[5] = row3 + row2;
        count = 6;
    }
    for (uint32_t i = 0; i < count; ++i) {
        float length = std::max(glm::length(glm::vec3(planes[i])), 1e-8F);
        planes[i] /= length;
    }
    return count;
}

bool sphereInFrustum(const std::array<glm::vec4, MAX_FRUSTUM_PLANES>& planes,
                     uint32_t count,
                     glm::vec3 center,
                     float radius) {
    for (uint32_t i = 0; i < count; ++i) {
        if (glm::dot(glm::vec3(planes[i]), center) + planes[i].w < -radius) {
            return false;
        }
    }
    return true;
}

bool sweptSphereInFrustum(const std::array<glm::vec4, MAX_FRUSTUM_PLANES>& planes,
                          uint32_t count,
                          glm::vec3 center,
                          float radius,
                          glm::vec3 lightDirection,
                          float sweep) {
    glm::vec3 swept = center + lightDirection * sweep;
    for (uint32_t i = 0; i < count; ++i) {
        float atCenter = glm::dot(glm::vec3(planes[i]), center);
        float atSwept = glm::dot(glm::vec3(planes[i]), swept);
        // 캡슐 전체가 이 평면 밖일 때만 버린다. 한쪽만 밖이면 걸쳐 있는 것이라 남겨야 한다.
        if (std::max(atCenter, atSwept) + planes[i].w < -radius) {
            return false;
        }
    }
    return true;
}

glm::vec4 transformBoundingSphere(const glm::mat4& model, const glm::vec4& sphere) {
    glm::vec3 center = glm::vec3(model * glm::vec4{glm::vec3(sphere), 1.0F});
    float scale = std::sqrt(std::max({glm::dot(glm::vec3(model[0]), glm::vec3(model[0])),
                                      glm::dot(glm::vec3(model[1]), glm::vec3(model[1])),
                                      glm::dot(glm::vec3(model[2]), glm::vec3(model[2]))}));
    return glm::vec4{center, sphere.w * scale};
}

} // namespace gfx
