#include <cassert>
#include <cmath>
#include <cstdio>

#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>

#include "asset/vertex_pack.h"

namespace {

// snorm16 둘로 단위 구를 덮으면 최악 오차가 0.01도 근처다. 이보다 크면 8진법 접기가 틀린 것이다.
// acos(dot) 는 1 근처에서 float 정밀도가 0.02도라 쓸 수 없고, 차 벡터의 길이(작은 각에서 라디안과 같다)로 잰다.
constexpr float MAX_ANGLE_RADIANS = glm::radians(0.02F);

void checkRoundTrip(glm::vec3 direction) {
    glm::vec3 unit = glm::normalize(direction);
    glm::vec3 restored = asset::unpackUnitVector(asset::packUnitVector(unit));
    float error = glm::length(unit - restored);
    assert(error < MAX_ANGLE_RADIANS);
    (void)error;
}

} // namespace

int main() {
    // 축과 8진법 접힘 경계(z<0) 를 포함해 구 전체를 훑는다.
    const float STEP = 7.0F;
    for (float pitch = -90.0F; pitch <= 90.0F; pitch += STEP) {
        for (float yaw = 0.0F; yaw < 360.0F; yaw += STEP) {
            float p = glm::radians(pitch);
            float y = glm::radians(yaw);
            checkRoundTrip(glm::vec3{std::cos(p) * std::cos(y), std::sin(p), std::cos(p) * std::sin(y)});
        }
    }
    for (glm::vec3 axis : {glm::vec3{1, 0, 0}, glm::vec3{0, 1, 0}, glm::vec3{0, 0, 1}}) {
        checkRoundTrip(axis);
        checkRoundTrip(-axis);
    }

    // 비트 배치를 고정한다. x 가 하위 16비트, y 가 상위 16비트이고 z<0 은 접혀서 모서리로 간다.
    // 왕복만 검사하면 두 반쪽이 바뀌어도 통과하므로 GLSL packSnorm2x16 과의 짝을 여기서 잡는다.
    assert(asset::packUnitVector(glm::vec3{1.0F, 0.0F, 0.0F}) == 0x00007FFFU);
    assert(asset::packUnitVector(glm::vec3{0.0F, 1.0F, 0.0F}) == 0x7FFF0000U);
    assert(asset::packUnitVector(glm::vec3{0.0F, 0.0F, 1.0F}) == 0U);
    assert(asset::packUnitVector(glm::vec3{0.0F, 0.0F, -1.0F}) == 0x7FFF7FFFU);
    assert((asset::packTangent(glm::vec4{1.0F, 0.0F, 0.0F, 1.0F}) & asset::TANGENT_SIGN_BIT) != 0U);
    assert((asset::packTangent(glm::vec4{1.0F, 0.0F, 0.0F, -1.0F}) & asset::TANGENT_SIGN_BIT) == 0U);

    // 길이 0 은 +Z 로 정해 둔다. NaN 이 나오면 안 된다.
    glm::vec3 zero = asset::unpackUnitVector(asset::packUnitVector(glm::vec3{0.0F}));
    assert(std::abs(zero.z - 1.0F) < 1e-6F);

    // 탄젠트는 방향과 손 방향을 모두 되돌려야 하고, 부호 비트가 방향을 흔들지 않아야 한다.
    for (float handedness : {1.0F, -1.0F}) {
        glm::vec4 tangent{glm::normalize(glm::vec3{0.3F, -0.8F, -0.5F}), handedness};
        glm::vec4 restored = asset::unpackTangent(asset::packTangent(tangent));
        assert(restored.w == handedness);
        float error = glm::length(glm::vec3{tangent} - glm::vec3{restored});
        assert(error < MAX_ANGLE_RADIANS);
        (void)error;
    }
    (void)zero;

    std::puts("vertex_pack: 통과");
    return 0;
}
