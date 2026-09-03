#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <glm/geometric.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

// 정점 속성 압축. shaders/scene_types.glsl 의 encodeUnitVector/decodeUnitVector/encodeTangent/decodeTangent
// 와 같은 계산이어야 CPU 가 넣은 값을 GPU 가 그대로 되돌리고, 스킨 컴퓨트가 다시 넣은 값을 CPU 쪽과
// 같은 규칙으로 읽는다.
namespace asset {

namespace detail {
inline float signNotZero(float v) {
    return v >= 0.0F ? 1.0F : -1.0F;
}

inline uint32_t packSnorm2x16(glm::vec2 v) {
    auto quantize = [](float x) {
        auto q = static_cast<int32_t>(std::lround(std::clamp(x, -1.0F, 1.0F) * 32767.0F));
        return static_cast<uint32_t>(q) & 0xFFFFU;
    };
    return quantize(v.x) | (quantize(v.y) << 16U);
}

inline glm::vec2 unpackSnorm2x16(uint32_t packed) {
    auto expand = [](uint32_t bits) {
        auto q = static_cast<int16_t>(bits & 0xFFFFU);
        return std::clamp(static_cast<float>(q) / 32767.0F, -1.0F, 1.0F);
    };
    return glm::vec2{expand(packed), expand(packed >> 16U)};
}
} // namespace detail

// 단위 벡터를 8진법(octahedral) 매핑으로 [-1, 1]² 에 펴서 snorm16 둘에 담는다. 길이가 0 이면 +Z 다.
inline uint32_t packUnitVector(glm::vec3 n) {
    float sum = std::abs(n.x) + std::abs(n.y) + std::abs(n.z);
    if (sum <= 0.0F) {
        return detail::packSnorm2x16(glm::vec2{0.0F});
    }
    n /= sum;
    glm::vec2 projected{n.x, n.y};
    if (n.z < 0.0F) {
        projected =
            (1.0F - glm::abs(glm::vec2{n.y, n.x})) * glm::vec2{detail::signNotZero(n.x), detail::signNotZero(n.y)};
    }
    return detail::packSnorm2x16(projected);
}

inline glm::vec3 unpackUnitVector(uint32_t packed) {
    glm::vec2 f = detail::unpackSnorm2x16(packed);
    glm::vec3 n{f.x, f.y, 1.0F - std::abs(f.x) - std::abs(f.y)};
    float t = std::clamp(-n.z, 0.0F, 1.0F);
    n.x += n.x >= 0.0F ? -t : t;
    n.y += n.y >= 0.0F ? -t : t;
    return glm::normalize(n);
}

// 탄젠트는 방향을 8진법으로 넣고 손 방향(w 의 부호)을 y 성분의 최하위 비트에 둔다. 1 이면 +, 0 이면 -.
inline constexpr uint32_t TANGENT_SIGN_BIT = 0x10000U;

inline uint32_t packTangent(glm::vec4 tangent) {
    uint32_t packed = packUnitVector(glm::vec3{tangent}) & ~TANGENT_SIGN_BIT;
    return tangent.w >= 0.0F ? packed | TANGENT_SIGN_BIT : packed;
}

inline glm::vec4 unpackTangent(uint32_t packed) {
    return glm::vec4{unpackUnitVector(packed & ~TANGENT_SIGN_BIT), (packed & TANGENT_SIGN_BIT) != 0U ? 1.0F : -1.0F};
}

} // namespace asset
