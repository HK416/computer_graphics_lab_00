#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "scene/scene.h"

namespace core {
class JobSystem;
} // namespace core

namespace physics {

// GPU 경로의 shaders/fluid_common.glsl 과 같은 값이어야 한다. 두 백엔드가 같은 격자 규칙을 쓴다.
inline constexpr uint32_t FLUID_MAX_COLLIDERS = 8;
inline constexpr uint32_t FLUID_CELL_CAPACITY = 32;

// 유체가 부딪히는 강체 하나. 일방향이라 입자는 밀려나지만 강체는 밀리지 않는다.
struct FluidCollider {
    scene::ColliderShape shape = scene::ColliderShape::SPHERE;
    // 구.
    glm::vec3 center{0.0F};
    float radius = 0.0F;
    // 상자. 지역 공간에서 판정하고 세계 공간으로 되돌린다.
    glm::vec3 halfExtents{0.0F};
    glm::mat4 world{1.0F};
    glm::mat4 inverseWorld{1.0F};
    // 평면. dot(normal, x) = offset.
    glm::vec3 normal{0.0F, 1.0F, 0.0F};
    float offset = 0.0F;
};

// 부품 설정과 장면에서 끌어낸 시뮬레이션 상수. CPU 솔버와 GPU 셰이더가 «같은 함수»로 만든 같은
// 값을 본다. 두 벌로 두면 백엔드를 바꿀 때마다 물이 달리 흐른다.
struct FluidParams {
    glm::mat4 emitterWorld{1.0F};
    glm::vec3 emitterHalfExtents{0.5F};
    // 방출 격자의 간격이자 입자 지름.
    float spacing = 0.05F;
    glm::vec3 containerMin{-1.0F, 0.0F, -1.0F};
    float particleRadius = 0.025F;
    glm::vec3 containerMax{1.0F, 2.0F, 1.0F};
    // 용기 벽에서 튀는 정도.
    float wallRestitution = 0.3F;
    glm::vec3 gravity{0.0F, -9.81F, 0.0F};
    // 입자 하나가 대신하는 물의 질량. 간격 세제곱만큼 잡아야 밀도가 기준 밀도 언저리에서 시작한다.
    float particleMass = 1.0F;
    float smoothingRadius = 0.1F;
    float restDensity = 1000.0F;
    float stiffness = 50.0F;
    float viscosity = 0.5F;
    // 방출 격자의 축별 개수. 상자 안에 x, y 로 채우고 남는 입자는 z 로 이어 쌓는다.
    glm::uvec3 lattice{1U};
    // 해시 격자의 셀 수. 마스크로 접으므로 2 의 거듭제곱이어야 한다.
    uint32_t cellCount = 1024;
    uint32_t colliderCount = 0;
    std::array<FluidCollider, FLUID_MAX_COLLIDERS> colliders{};
};

FluidParams deriveFluidParams(const scene::Fluid& settings,
                              const glm::mat4& emitterWorld,
                              uint32_t particleCount,
                              uint32_t cellCount,
                              const scene::Scene& scene);

// 프레임 하나가 한 번에 진행할 최대 시간. 프레임이 길어도 이만큼만 따라잡아 나선형으로 느려지지 않는다.
inline constexpr float FLUID_MAX_FRAME_STEP = 1.0F / 30.0F;

// 프레임 하나를 몇 번에 나눠 풀지. 강성·중력과 커널 반지름에서 어림한 안정 간격으로 정한다.
// GPU 경로도 이 함수를 쓴다. 두 벌로 두면 백엔드를 바꿀 때 물이 달리 흐른다.
uint32_t fluidSubsteps(const FluidParams& params, float deltaSeconds);

// CPU SPH. 입자 상태를 여기 들고 밀도·힘·적분을 JobSystem 으로 나눠 푼다. Vulkan 을 타지 않아 테스트할
// 수 있고, 같은 시작 상태에서 워커 수와 무관하게 같은 결과가 나온다(GPU 백엔드는 그렇지 않다).
class FluidSolver {
public:
    // 입자를 방출 상자 안의 격자에 다시 놓는다.
    void emit(const FluidParams& params, uint32_t count);
    // deltaSeconds 만큼 진행한다. 안쪽에서 안정 간격으로 나눠 돈다.
    void step(const FluidParams& params, float deltaSeconds, core::JobSystem* jobs);

    uint32_t particleCount() const { return static_cast<uint32_t>(positions.size()); }
    // xyz 가 위치, w 가 밀도다.
    const std::vector<glm::vec4>& particles() const { return positions; }
    // 지난 프레임에 그린 위치. 모션 벡터에 쓴다.
    const std::vector<glm::vec4>& previousParticles() const { return previousRendered; }
    // 지난 위치를 지금 위치로 맞춘다. 인스턴스를 쓴 뒤와 다시 뿌린 프레임에 부른다.
    void keepRendered();

private:
    void substep(const FluidParams& params, float dt, core::JobSystem* jobs);
    void buildGrid(const FluidParams& params, core::JobSystem* jobs);

    // xyz 위치, w 밀도.
    std::vector<glm::vec4> positions;
    // xyz 속도, w 압력.
    std::vector<glm::vec4> velocities;
    std::vector<glm::vec4> nextPositions;
    std::vector<glm::vec4> nextVelocities;
    std::vector<glm::vec4> previousRendered;
    // 해시 격자. 셀 번호는 나눠 계산하고 버킷에 넣는 것은 직렬이라 결과가 워커 수와 무관하다.
    std::vector<uint32_t> cellOf;
    std::vector<uint32_t> cellCounts;
    std::vector<uint32_t> cellParticles;
};

} // namespace physics
