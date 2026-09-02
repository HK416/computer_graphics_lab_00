#include <cassert>
#include <cmath>
#include <cstdio>

#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>
#include <glm/mat4x4.hpp>
#include <glm/matrix.hpp>

#include "scene/camera.h"

namespace {

// ImGuizmo 의 reversed 판정. 시야 공간 z 가 양수인 두 점을 투영해 깊이가 줄어드는지 본다.
bool reversedProjection(const glm::mat4& projection) {
    glm::vec4 nearPoint = projection * glm::vec4{0.0F, 0.0F, 1.0F, 1.0F};
    glm::vec4 farPoint = projection * glm::vec4{0.0F, 0.0F, 2.0F, 1.0F};
    return nearPoint.z / nearPoint.w > farPoint.z / farPoint.w;
}

// ImGuizmo 의 ComputeCameraRay 와 같은 계산. NDC 의 양 끝 점을 역변환해 광선을 만든다.
// 원거리 평면이 무한대인 투영을 주면 그 점의 w 가 0 이 되어 NaN 이 나온다.
glm::vec3 computeRayDirection(const glm::mat4& view, const glm::mat4& projection, glm::vec2 ndc) {
    glm::mat4 inverse = glm::inverse(projection * view);
    bool reversed = reversedProjection(projection);
    float nearZ = reversed ? 1.0F - 1e-7F : 0.0F;
    float farZ = reversed ? 0.0F : 1.0F - 1e-7F;

    glm::vec4 origin = inverse * glm::vec4{ndc.x, ndc.y, nearZ, 1.0F};
    glm::vec4 end = inverse * glm::vec4{ndc.x, ndc.y, farZ, 1.0F};
    return glm::normalize(glm::vec3{end / end.w} - glm::vec3{origin / origin.w});
}

bool finite(glm::vec3 value) {
    return std::isfinite(value.x) && std::isfinite(value.y) && std::isfinite(value.z);
}

} // namespace

int main() {
    scene::Camera camera;
    camera.position = glm::vec3{0.0F, 1.0F, 3.0F};
    float aspect = 16.0F / 9.0F;

    // 장면 투영은 원거리 평면이 무한대라 광선이 NaN 이 된다. 기즈모에 이 행렬을 주면 안 된다.
    glm::vec3 sceneRay = computeRayDirection(camera.viewMatrix(), camera.projectionMatrix(aspect), {0.3F, -0.2F});
    assert(!finite(sceneRay) && "장면 투영은 무한 원거리라 기즈모 광선을 만들 수 없어야 한다");

    // 기즈모 투영은 화면 어디에서나 유한한 광선을 준다.
    glm::mat4 gizmo = camera.gizmoProjectionMatrix(aspect);
    for (float y = -1.0F; y <= 1.0F; y += 0.5F) {
        for (float x = -1.0F; x <= 1.0F; x += 0.5F) {
            glm::vec3 ray = computeRayDirection(camera.viewMatrix(), gizmo, {x, y});
            if (!finite(ray) || std::abs(glm::length(ray) - 1.0F) > 1e-3F) {
                std::printf(
                    "NDC (%.1f, %.1f) 광선이 유효하지 않습니다\n", static_cast<double>(x), static_cast<double>(y));
                assert(false && "기즈모 투영으로 만든 광선은 항상 유한해야 한다");
            }
        }
    }

    // 화면 중앙의 광선은 카메라 전방 축과 나란해야 한다. 광선 방향은 reversed 판정에 따라 뒤집힌다.
    glm::vec3 center = computeRayDirection(camera.viewMatrix(), gizmo, {0.0F, 0.0F});
    assert(std::abs(glm::dot(center, camera.forward())) > 0.999F && "화면 중앙 광선은 카메라 전방과 나란해야 한다");

    // ---- 궤도 카메라 ----
    {
        scene::Camera orbit;
        orbit.mode = scene::CameraMode::ORBIT;
        orbit.focusOn(glm::vec3{2.0F, 1.0F, -3.0F}, 6.0F);
        assert(std::abs(orbit.distance - 6.0F) < 1e-5F);
        // 위치는 대상에서 시선 반대쪽으로 거리만큼 떨어져 있어야 한다.
        glm::vec3 expected = orbit.target - orbit.forward() * orbit.distance;
        assert(glm::length(orbit.position - expected) < 1e-4F);
        // 그러므로 시선을 따라가면 정확히 대상에 닿는다.
        glm::vec3 hit = orbit.position + orbit.forward() * orbit.distance;
        assert(glm::length(hit - orbit.target) < 1e-4F && "궤도 카메라는 항상 대상을 본다");

        // 모드를 오갔다 와도 화면이 튀면 안 된다.
        glm::vec3 before = orbit.position;
        glm::vec3 direction = orbit.forward();
        orbit.setMode(scene::CameraMode::FLY);
        assert(glm::length(orbit.position - before) < 1e-4F && "자유 모드로 바꿔도 위치는 그대로다");
        orbit.setMode(scene::CameraMode::ORBIT);
        orbit.applyOrbit();
        assert(glm::length(orbit.position - before) < 1e-3F && "궤도로 돌아와도 위치는 그대로다");
        assert(glm::length(orbit.forward() - direction) < 1e-5F);

        // 거리는 0 이 될 수 없다. 회전 중심이 눈 안으로 들어오면 조작이 뒤집힌다.
        orbit.distance = 0.0F;
        orbit.applyOrbit();
        assert(orbit.distance > 0.0F);
    }

    // 클릭 피킹이 쓰는 화면 -> 월드 광선. 부호와 종횡비가 가장 틀리기 쉽다.
    {
        scene::Camera eye;
        eye.position = glm::vec3{0.0F, 0.0F, 0.0F};
        eye.yawDegrees = -90.0F; // -Z 를 본다
        eye.pitchDegrees = 0.0F;
        eye.fovYDegrees = 90.0F;

        // 화면 한가운데는 시선과 같은 방향이다.
        scene::Ray middle = eye.screenToRay(glm::vec2{0.5F, 0.5F}, 1.0F);
        assert(glm::length(middle.origin - eye.position) < 1e-5F);
        assert(glm::length(middle.direction - eye.forward()) < 1e-5F && "가운데는 시선 그대로여야 한다");

        // 화면 위쪽은 위를 향한다. y 를 뒤집지 않으면 여기서 부호가 반대로 나온다.
        scene::Ray top = eye.screenToRay(glm::vec2{0.5F, 0.0F}, 1.0F);
        assert(top.direction.y > 0.0F && "화면 위는 월드에서도 위여야 한다");
        scene::Ray bottom = eye.screenToRay(glm::vec2{0.5F, 1.0F}, 1.0F);
        assert(bottom.direction.y < 0.0F);

        // 시야각 90 도에서 화면 위 끝은 시선과 45 도를 이룬다.
        float cosine = glm::dot(top.direction, eye.forward());
        assert(std::abs(cosine - std::cos(glm::radians(45.0F))) < 1e-4F && "세로 시야각이 반영되어야 한다");

        // 종횡비가 넓어지면 가로로 더 벌어진다.
        scene::Ray narrow = eye.screenToRay(glm::vec2{1.0F, 0.5F}, 1.0F);
        scene::Ray wide = eye.screenToRay(glm::vec2{1.0F, 0.5F}, 2.0F);
        assert(glm::dot(wide.direction, eye.forward()) < glm::dot(narrow.direction, eye.forward()) &&
               "종횡비가 클수록 화면 가장자리 광선이 더 벌어진다");

        // 방향은 단위 벡터여야 한다. 피킹의 거리 비교가 이걸 전제한다.
        assert(std::abs(glm::length(wide.direction) - 1.0F) < 1e-5F);
    }

    std::printf("카메라 자체 점검 통과\n");
    return 0;
}
