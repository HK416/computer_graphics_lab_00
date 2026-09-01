#pragma once

#include <cstdint>

#include <glm/vec2.hpp>

namespace gfx {

// Halton(2, 3) 수열의 index 번째 표본을 [-0.5, 0.5) 로 옮긴 값. 화소 안에서 표본 위치를 흩어
// 프레임마다 다른 곳을 보게 한다. index 는 1 부터 센다. 0 은 두 밑에서 모두 0 이라 지터가 없다.
glm::vec2 haltonJitter(uint32_t index);

// 지터 수열을 몇 프레임에 걸쳐 돌릴지. 확대 배율이 클수록 출력 화소 하나를 채우는 데 더 많은
// 프레임이 필요하다. AMD 가 FSR 에 권하는 8 * (표시 폭 / 렌더 폭)^2 을 그대로 쓴다.
uint32_t jitterPhaseCount(uint32_t renderWidth, uint32_t displayWidth);

} // namespace gfx
