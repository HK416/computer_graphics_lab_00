#include <cassert>
#include <cmath>
#include <cstdio>

#include <glm/gtc/matrix_transform.hpp>

#include "gfx/shadow_math.h"

namespace {

// 원점에서 -Z 를 보는 유한 투영. 원평면이 있으므로 평면이 여섯 개 나온다.
glm::mat4 makeCameraViewProjection() {
    glm::mat4 projection = glm::perspectiveRH_ZO(glm::radians(60.0F), 1.0F, 0.1F, 100.0F);
    glm::mat4 view = glm::lookAt(glm::vec3{0.0F}, glm::vec3{0.0F, 0.0F, -1.0F}, glm::vec3{0.0F, 1.0F, 0.0F});
    return projection * view;
}

// 이 저장소의 장면 카메라와 같은 무한 원거리 reverse-Z 투영.
glm::mat4 makeInfiniteViewProjection() {
    float focal = 1.0F / std::tan(glm::radians(60.0F) * 0.5F);
    glm::mat4 projection(0.0F);
    projection[0][0] = focal;
    projection[1][1] = -focal;
    projection[2][3] = -1.0F;
    projection[3][2] = 0.1F;
    return projection * glm::lookAt(glm::vec3{0.0F}, glm::vec3{0.0F, 0.0F, -1.0F}, glm::vec3{0.0F, 1.0F, 0.0F});
}

} // namespace

int main() {
    std::array<glm::vec4, gfx::MAX_FRUSTUM_PLANES> planes{};

    // 무한 원거리 투영은 원평면이 없다.
    assert(gfx::extractFrustumPlanes(makeInfiniteViewProjection(), planes, false) == 5);
    // 유한 투영은 여섯 개.
    uint32_t count = gfx::extractFrustumPlanes(makeCameraViewProjection(), planes, true);
    assert(count == 6);
    for (uint32_t i = 0; i < count; ++i) {
        assert(std::abs(glm::length(glm::vec3(planes[i])) - 1.0F) < 1e-4F && "평면 법선은 정규화되어야 한다");
    }

    // 절두체 안, 밖, 걸친 구.
    assert(gfx::sphereInFrustum(planes, count, glm::vec3{0.0F, 0.0F, -5.0F}, 1.0F));
    assert(!gfx::sphereInFrustum(planes, count, glm::vec3{0.0F, 50.0F, -5.0F}, 1.0F) && "위로 한참 벗어난 구");
    assert(!gfx::sphereInFrustum(planes, count, glm::vec3{0.0F, 0.0F, 5.0F}, 1.0F) && "카메라 뒤의 구");
    assert(gfx::sphereInFrustum(planes, count, glm::vec3{0.0F, 3.0F, -5.0F}, 3.0F) && "걸친 구는 남겨야 한다");

    // 캐스터 스윕. 절두체 밖에 있어도 그림자가 절두체를 지나가면 남겨야 한다.
    // 이 검사를 반대로 하면(양 끝 중 가까운 쪽만 보면) 보이는 그림자가 통째로 사라진다.
    glm::vec3 above{0.0F, 50.0F, -5.0F};
    assert(!gfx::sphereInFrustum(planes, count, above, 1.0F));
    assert(gfx::sweptSphereInFrustum(planes, count, above, 1.0F, glm::vec3{0.0F, -1.0F, 0.0F}, 60.0F) &&
           "아래로 드리운 그림자가 절두체를 가로지르면 캐스터를 남겨야 한다");
    assert(!gfx::sweptSphereInFrustum(planes, count, above, 1.0F, glm::vec3{0.0F, 1.0F, 0.0F}, 60.0F) &&
           "위로 뻗는 그림자는 화면에 닿을 수 없다");

    // 절두체 안의 캐스터는 어느 방향으로 끌어도 남는다.
    glm::vec3 inside{0.0F, 0.0F, -5.0F};
    assert(gfx::sweptSphereInFrustum(planes, count, inside, 1.0F, glm::vec3{0.0F, 1.0F, 0.0F}, 60.0F));
    assert(gfx::sweptSphereInFrustum(planes, count, inside, 1.0F, glm::vec3{0.0F, -1.0F, 0.0F}, 0.0F));

    // 경계 구 변환.
    glm::mat4 moved = glm::translate(glm::mat4{1.0F}, glm::vec3{1.0F, 2.0F, 3.0F});
    glm::vec4 sphere = gfx::transformBoundingSphere(moved, glm::vec4{0.0F, 0.0F, 0.0F, 2.0F});
    assert(sphere.x == 1.0F && sphere.y == 2.0F && sphere.z == 3.0F);
    assert(std::abs(sphere.w - 2.0F) < 1e-5F && "평행이동은 반지름을 바꾸지 않는다");

    glm::mat4 scaled = glm::scale(glm::mat4{1.0F}, glm::vec3{2.0F, 3.0F, 1.0F});
    sphere = gfx::transformBoundingSphere(scaled, glm::vec4{0.0F, 0.0F, 0.0F, 2.0F});
    assert(std::abs(sphere.w - 6.0F) < 1e-4F && "비균등 스케일은 최대 성분으로 보수적으로 잡는다");

    // 회전은 반지름을 바꾸지 않아야 한다.
    glm::mat4 rotated = glm::rotate(glm::mat4{1.0F}, 0.7F, glm::vec3{0.3F, 0.8F, 0.5F});
    sphere = gfx::transformBoundingSphere(rotated, glm::vec4{0.0F, 0.0F, 0.0F, 2.0F});
    assert(std::abs(sphere.w - 2.0F) < 1e-4F);

    std::printf("그림자 컬링 자체 점검 통과\n");
    return 0;
}
