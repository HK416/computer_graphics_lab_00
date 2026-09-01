#include "gfx/upscaler_math.h"

#include <algorithm>

namespace gfx {
namespace {
// 밑 base 의 반전 기수 수열. 자릿수를 뒤집어 [0, 1) 을 고르게 채운다.
float halton(uint32_t index, uint32_t base) {
    float result = 0.0F;
    float fraction = 1.0F;
    while (index > 0) {
        fraction /= static_cast<float>(base);
        result += fraction * static_cast<float>(index % base);
        index /= base;
    }
    return result;
}
} // namespace

glm::vec2 haltonJitter(uint32_t index) {
    return glm::vec2{halton(index, 2) - 0.5F, halton(index, 3) - 0.5F};
}

uint32_t jitterPhaseCount(uint32_t renderWidth, uint32_t displayWidth) {
    if (renderWidth == 0) {
        return 8;
    }
    float scale = static_cast<float>(displayWidth) / static_cast<float>(renderWidth);
    float count = 8.0F * scale * scale;
    // 배율이 1 이면 8 이고, 아무리 크게 잡아도 수열이 한 바퀴 도는 데 몇 초를 넘지 않게 막는다.
    return std::clamp(static_cast<uint32_t>(count), 8U, 128U);
}

} // namespace gfx
