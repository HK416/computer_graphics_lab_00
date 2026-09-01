#pragma once

#include <array>
#include <cstdint>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace gfx {

// 그림자 관련 순수 계산. Vulkan 을 끌어오지 않아 테스트가 그대로 링크한다.

inline constexpr uint32_t MAX_FRUSTUM_PLANES = 6;

// Gribb-Hartmann 절두체 평면. 카메라의 무한 원거리 reverse-Z 투영은 원평면이 없어 5개,
// 그림자의 유한 투영은 6개가 나온다. 실제로 채운 개수를 돌려준다.
uint32_t extractFrustumPlanes(const glm::mat4& viewProjection,
                              std::array<glm::vec4, MAX_FRUSTUM_PLANES>& planes,
                              bool hasFarPlane);

// 구가 절두체 안이거나 걸치면 참. 셰이더의 sphereInFrustum 과 같은 수식이다.
bool sphereInFrustum(const std::array<glm::vec4, MAX_FRUSTUM_PLANES>& planes,
                     uint32_t count,
                     glm::vec3 center,
                     float radius);

// 캐스터의 경계 구를 광 진행 방향으로 sweep 만큼 끈 캡슐이 절두체에 닿는지.
//
// 화면 밖 캐스터도 그림자는 화면에 보일 수 있어 카메라 절두체만으로는 버릴 수 없다. 대신 그림자가
// 뻗어 나갈 범위까지 부풀린 볼륨을 카메라 절두체와 비교한다. 평면마다 양 끝 중심의 부호 거리 중
// 작은 쪽만 보면 캡슐 전체를 보수적으로 덮는다.
bool sweptSphereInFrustum(const std::array<glm::vec4, MAX_FRUSTUM_PLANES>& planes,
                          uint32_t count,
                          glm::vec3 center,
                          float radius,
                          glm::vec3 lightDirection,
                          float sweep);

// 캐스케이드 그림자에 쓰는 계산들.

inline constexpr uint32_t MAX_SHADOW_CASCADES = 4;

// PSSM 혼합 분할. lambda 0 이면 균등, 1 이면 로그 분할이다. splits[i] 는 캐스케이드 i 의 끝 거리다.
void cascadeSplits(
    float nearPlane, float farPlane, uint32_t count, float lambda, std::array<float, MAX_SHADOW_CASCADES>& splits);

// 부분 절두체 [near, far] 를 감싸는 최소 구. 중심은 카메라 축 위에 있으므로 거리 하나로 표현된다.
// 반지름이 카메라 회전과 무관해야 캐스케이드가 흔들리지 않는다.
struct CascadeSphere {
    float distance = 0.0F;
    float radius = 0.0F;
};
CascadeSphere fitCascadeSphere(float nearPlane, float farPlane, float fovYRadians, float aspect);

// 텍셀 격자에 맞춘 캐스케이드 시점 행렬.
//
// lightRotation 은 평행이동이 없는 회전만의 행렬이어야 한다. 광원을 카메라 쪽으로 옮겨 가며
// 스냅하면 격자가 카메라를 따라다녀 스냅이 무의미해진다.
glm::mat4 snapCascadeMatrix(const glm::mat4& lightRotation,
                            glm::vec3 center,
                            float radius,
                            float depthNear,
                            float depthFar,
                            uint32_t resolution);

// 인스턴스 변환을 반영한 세계 공간 경계 구. 비균등 스케일은 최대 성분으로 보수적으로 잡는다.
glm::vec4 transformBoundingSphere(const glm::mat4& model, const glm::vec4& sphere);

} // namespace gfx
