#include <cassert>
#include <cmath>
#include <cstdio>
#include <set>
#include <utility>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>

#include "gfx/upscaler_math.h"

namespace {

// 지터는 화소 하나 안에서만 움직여야 한다. 이보다 크면 표본이 이웃 화소로 넘어가 이웃 클리핑
// 범위를 벗어난다.
void testJitterRange() {
    for (uint32_t index = 1; index <= 256; ++index) {
        glm::vec2 jitter = gfx::haltonJitter(index);
        assert(jitter.x >= -0.5F && jitter.x < 0.5F);
        assert(jitter.y >= -0.5F && jitter.y < 0.5F);
    }
}

// 한 바퀴 도는 동안 같은 자리를 두 번 보면 그만큼 표본이 낭비된다.
void testJitterDistinct() {
    std::set<std::pair<int, int>> seen;
    for (uint32_t index = 1; index <= 128; ++index) {
        glm::vec2 jitter = gfx::haltonJitter(index);
        // 1/1024 화소까지 같은 자리를 같은 것으로 본다.
        auto key = std::pair<int, int>{static_cast<int>(jitter.x * 1024.0F), static_cast<int>(jitter.y * 1024.0F)};
        assert(seen.insert(key).second);
    }
}

// 수열이 화소를 고르게 덮어야 시간이 지나면서 표본이 한쪽으로 쏠리지 않는다.
void testJitterBalanced() {
    glm::vec2 sum{0.0F};
    for (uint32_t index = 1; index <= 64; ++index) {
        sum += gfx::haltonJitter(index);
    }
    glm::vec2 mean = sum / 64.0F;
    assert(std::abs(mean.x) < 0.02F);
    assert(std::abs(mean.y) < 0.03F);
}

// 배율이 클수록 출력 화소 하나를 채우는 데 더 많은 프레임이 든다.
void testPhaseCount() {
    assert(gfx::jitterPhaseCount(1920, 1920) == 8);
    assert(gfx::jitterPhaseCount(960, 1920) == 32);
    assert(gfx::jitterPhaseCount(640, 1920) == 72);
    // 0 으로 나누지 않고, 아무리 크게 잡아도 한계를 넘지 않는다.
    assert(gfx::jitterPhaseCount(0, 1920) == 8);
    assert(gfx::jitterPhaseCount(1, 4096) == 128);
}

// 렌더러가 투영에 지터를 넣는 방법과 같은 계산. NDC 평행이동을 앞에 곱하면 클립 좌표의 xy 가
// w 에 비례해 밀리고, 원근 나눗셈 뒤에는 정확히 지터만큼 옮겨져야 한다.
void testJitterShiftsProjectionByExactlyOnePixel() {
    float focal = 1.0F / std::tan(glm::radians(60.0F) * 0.5F);
    glm::mat4 projection(0.0F);
    projection[0][0] = focal;
    projection[1][1] = -focal;
    projection[2][3] = -1.0F;
    projection[3][2] = 0.05F;

    glm::vec2 renderSize{1600.0F, 900.0F};
    glm::vec2 jitterPixels{0.25F, -0.375F};
    glm::vec2 jitterNdc = 2.0F * jitterPixels / renderSize;
    glm::mat4 jittered = glm::translate(glm::mat4{1.0F}, glm::vec3{jitterNdc, 0.0F}) * projection;

    // 깊이가 다른 두 점이 같은 화소 변위를 받아야 한다. 그래야 지터가 장면 내용과 무관하다.
    for (float depth : {1.0F, 37.5F}) {
        glm::vec4 point{0.3F, -0.2F, -depth, 1.0F};
        glm::vec4 plain = projection * point;
        glm::vec4 shifted = jittered * point;
        glm::vec2 delta = (glm::vec2{shifted} / shifted.w - glm::vec2{plain} / plain.w) * 0.5F * renderSize;
        assert(std::abs(delta.x - jitterPixels.x) < 1e-3F);
        assert(std::abs(delta.y - jitterPixels.y) < 1e-3F);
    }
}

} // namespace

int main() {
    testJitterRange();
    testJitterDistinct();
    testJitterBalanced();
    testPhaseCount();
    testJitterShiftsProjectionByExactlyOnePixel();
    std::printf("업스케일 지터 점검 통과\n");
    return 0;
}
