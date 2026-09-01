#include <cassert>
#include <cmath>
#include <cstdio>

#include <glm/geometric.hpp>
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

    // ---- 캐스케이드 분할 ----
    std::array<float, gfx::MAX_SHADOW_CASCADES> splits{};
    gfx::cascadeSplits(0.1F, 100.0F, 4, 0.5F, splits);
    for (uint32_t i = 1; i < 4; ++i) {
        assert(splits[i] > splits[i - 1] && "분할 거리는 단조 증가해야 한다");
    }
    assert(std::abs(splits[3] - 100.0F) < 1e-3F && "마지막 분할은 원평면이다");

    // lambda 0 은 균등 분할이다.
    gfx::cascadeSplits(0.0F, 100.0F, 4, 0.0F, splits);
    assert(std::abs(splits[0] - 25.0F) < 1e-3F);
    assert(std::abs(splits[1] - 50.0F) < 1e-3F);
    // lambda 1 은 로그 분할이라 앞쪽이 훨씬 촘촘하다.
    gfx::cascadeSplits(1.0F, 100.0F, 4, 1.0F, splits);
    assert(splits[0] < 25.0F && "로그 분할은 앞쪽 캐스케이드가 좁다");

    // ---- 캐스케이드 구 피팅 ----
    float fov = glm::radians(60.0F);
    float aspect = 16.0F / 9.0F;
    gfx::CascadeSphere cascade = gfx::fitCascadeSphere(2.0F, 20.0F, fov, aspect);
    // 부분 절두체의 여덟 꼭짓점이 모두 구 안에 들어와야 한다.
    float v = std::tan(fov * 0.5F);
    float h = v * aspect;
    for (float z : {2.0F, 20.0F}) {
        for (float sx : {-1.0F, 1.0F}) {
            for (float sy : {-1.0F, 1.0F}) {
                glm::vec3 corner{sx * h * z, sy * v * z, z};
                float distance = glm::length(corner - glm::vec3{0.0F, 0.0F, cascade.distance});
                assert(distance <= cascade.radius + 1e-3F && "구가 절두체 꼭짓점을 모두 감싸야 한다");
            }
        }
    }

    // ---- 텍셀 스냅 ----
    glm::mat4 lightRotation = glm::lookAt(glm::vec3{0.0F}, glm::vec3{0.3F, -1.0F, 0.2F}, glm::vec3{0.0F, 0.0F, 1.0F});
    float radius = 8.0F;
    uint32_t resolution = 1024;
    float texel = 2.0F * radius / static_cast<float>(resolution);

    // 라이트 공간에서 셀 한가운데를 잡고, 그 안에서 움직여 본다. 셀 경계에서 시작하면 아주 작은
    // 이동에도 격자가 한 칸 넘어가는 것이 정상이라 성질을 확인할 수 없다.
    glm::mat4 inverseRotation = glm::inverse(lightRotation);
    auto worldAt = [&](float x, float y) { return glm::vec3(inverseRotation * glm::vec4{x, y, -50.0F, 1.0F}); };
    glm::vec3 cellCenter = worldAt(texel * 0.5F, texel * 0.5F);

    glm::mat4 base = gfx::snapCascadeMatrix(lightRotation, cellCenter, radius, 0.0F, 100.0F, resolution);
    // 텍셀 안쪽 이동은 행렬을 비트 단위로 그대로 두어야 한다. 그림자 캐싱이 이 성질에 기댄다.
    for (float offset : {-0.4F, -0.1F, 0.1F, 0.4F}) {
        glm::vec3 moved = worldAt(texel * (0.5F + offset), texel * 0.5F);
        glm::mat4 nudged = gfx::snapCascadeMatrix(lightRotation, moved, radius, 0.0F, 100.0F, resolution);
        assert(base == nudged && "텍셀 안쪽 이동은 캐스케이드 행렬을 바꾸지 않아야 한다");
    }

    // 텍셀을 넘겨 움직이면 달라진다.
    glm::mat4 shifted =
        gfx::snapCascadeMatrix(lightRotation, worldAt(texel * 5.5F, texel * 0.5F), radius, 0.0F, 100.0F, resolution);
    assert(base != shifted && "텍셀을 넘겨 움직이면 행렬이 바뀌어야 한다");

    std::printf("그림자 컬링 자체 점검 통과\n");
    return 0;
}
