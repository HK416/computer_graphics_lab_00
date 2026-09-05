#include "physics/fluid_sph.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <numbers>

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/quaternion.hpp>

#include "core/job_system.h"
#include "physics/collider_shapes.h"

namespace physics {

namespace {

constexpr float PI = std::numbers::pi_v<float>;
// 입자가 이보다 적으면 나누는 값이 더 든다.
constexpr uint32_t GRANULARITY = 256;

// shaders/fluid_common.glsl 의 같은 이름 함수와 결과가 같아야 한다.
glm::ivec3 fluidCell(const glm::vec3& position, float h) {
    return glm::ivec3{glm::floor(position / h)};
}

uint32_t fluidHash(const glm::ivec3& cell, uint32_t cellCount) {
    uint32_t hashed = static_cast<uint32_t>(cell.x) * 73856093U ^ static_cast<uint32_t>(cell.y) * 19349663U ^
                      static_cast<uint32_t>(cell.z) * 83492791U;
    return hashed & (cellCount - 1U);
}

float poly6(float squaredDistance, float h) {
    float h2 = h * h;
    if (squaredDistance >= h2) {
        return 0.0F;
    }
    float d = h2 - squaredDistance;
    return 315.0F / (64.0F * PI * std::pow(h, 9.0F)) * d * d * d;
}

glm::vec3 spikyGradient(const glm::vec3& delta, float length, float h) {
    if (length >= h || length <= 1e-6F) {
        return glm::vec3{0.0F};
    }
    float d = h - length;
    return -45.0F / (PI * std::pow(h, 6.0F)) * d * d * (delta / length);
}

float viscosityLaplacian(float length, float h) {
    if (length >= h) {
        return 0.0F;
    }
    return 45.0F / (PI * std::pow(h, 6.0F)) * (h - length);
}

void bounce(glm::vec3& velocity, const glm::vec3& normal, float restitution) {
    float along = glm::dot(velocity, normal);
    if (along < 0.0F) {
        velocity -= (1.0F + restitution) * along * normal;
    }
}

void forRange(core::JobSystem* jobs, uint32_t count, const std::function<void(uint32_t, uint32_t)>& body) {
    if (jobs != nullptr && count > GRANULARITY) {
        jobs->parallelFor(count, GRANULARITY, body);
    } else {
        body(0, count);
    }
}

} // namespace

FluidParams deriveFluidParams(const scene::Fluid& settings,
                              const glm::mat4& emitterWorld,
                              uint32_t particleCount,
                              uint32_t cellCount,
                              const scene::Scene& scene) {
    FluidParams params;
    float spacing = settings.particleRadius * 2.0F;
    params.emitterWorld = emitterWorld;
    params.emitterHalfExtents = settings.emitterHalfExtents;
    params.spacing = spacing;
    params.containerMin = settings.containerMin;
    params.containerMax = settings.containerMax;
    params.particleRadius = settings.particleRadius;
    params.wallRestitution = 0.3F;
    params.gravity = settings.gravity;
    // 입자 하나가 간격 세제곱의 물을 대신하도록 잡는다. 그래야 밀도가 기준 밀도 언저리에서 시작한다.
    params.particleMass = settings.restDensity * spacing * spacing * spacing;
    // 커널이 간격보다 작으면 이웃을 하나도 못 찾아 밀도가 무너진다.
    params.smoothingRadius = std::max(settings.smoothingRadius, spacing);
    params.restDensity = settings.restDensity;
    params.stiffness = settings.stiffness;
    params.viscosity = settings.viscosity;
    params.cellCount = cellCount;

    auto along = [spacing](float half) {
        return std::max(1U, static_cast<uint32_t>(std::floor(half * 2.0F / spacing)));
    };
    uint32_t nx = along(settings.emitterHalfExtents.x);
    uint32_t ny = along(settings.emitterHalfExtents.y);
    uint32_t nz = std::max(1U, (particleCount + nx * ny - 1) / (nx * ny));
    params.lattice = glm::uvec3{nx, ny, nz};

    params.colliderCount = 0;
    for (uint32_t index = 0; index < scene.objects.size() && params.colliderCount < FLUID_MAX_COLLIDERS; ++index) {
        int32_t slot = scene.objects[index].rigidBody;
        if (slot < 0 || static_cast<size_t>(slot) >= scene.rigidBodies.size() || !scene.visibleCached(index)) {
            continue;
        }
        const scene::RigidBody& body = scene.rigidBodies[static_cast<size_t>(slot)];
        if (body.shape == scene::ColliderShape::MESH) {
            // ponytail: 입자 대 삼각형 판정은 아직 없다. 메쉬 콜라이더는 유체가 통과한다.
            continue;
        }
        // 크기 규칙은 강체 솔버·콜라이더 표시와 같은 함수로 낸다.
        scene::ColliderPose pose = scene::colliderPose(body, scene.world(index));
        FluidCollider& collider = params.colliders[params.colliderCount++];
        collider.shape = body.shape;
        collider.radius = pose.radius;
        collider.halfExtents = pose.halfExtents;
        glm::mat4 rigid = glm::translate(glm::mat4{1.0F}, pose.position) * glm::mat4_cast(pose.rotation);
        collider.world = rigid;
        collider.inverseWorld = glm::inverse(rigid);
    }
    return params;
}

uint32_t fluidSubsteps(const FluidParams& params, float deltaSeconds) {
    // 압력파 속도(√강성)와 낙하 속도로 안정 간격을 어림한다. 둘 중 짧은 쪽을 쓴다.
    float h = params.smoothingRadius;
    float gravity = std::max(glm::length(params.gravity), 0.1F);
    float stable = std::min(0.4F * h / std::sqrt(std::max(params.stiffness, 1.0F)), 0.25F * std::sqrt(h / gravity));
    float frameStep = std::min(std::max(deltaSeconds, 0.0F), FLUID_MAX_FRAME_STEP);
    auto steps = static_cast<uint32_t>(std::ceil(frameStep / std::max(stable, 1e-5F)));
    return std::clamp(steps, 1U, 8U);
}

void FluidSolver::emit(const FluidParams& params, uint32_t count) {
    positions.assign(count, glm::vec4{0.0F});
    velocities.assign(count, glm::vec4{0.0F});
    nextPositions.assign(count, glm::vec4{0.0F});
    nextVelocities.assign(count, glm::vec4{0.0F});
    previousRendered.assign(count, glm::vec4{0.0F});

    glm::uvec3 counts = glm::max(params.lattice, glm::uvec3{1U});
    glm::vec3 extent = glm::vec3{counts - 1U} * params.spacing * 0.5F;
    for (uint32_t i = 0; i < count; ++i) {
        glm::uvec3 index{i % counts.x, (i / counts.x) % counts.y, i / (counts.x * counts.y)};
        glm::vec3 local = glm::vec3{index} * params.spacing - extent;
        glm::vec3 world = glm::vec3{params.emitterWorld * glm::vec4{local, 1.0F}};
        positions[i] = glm::vec4{world, params.restDensity};
        previousRendered[i] = glm::vec4{world, 0.0F};
    }
}

void FluidSolver::keepRendered() {
    previousRendered = positions;
}

void FluidSolver::buildGrid(const FluidParams& params, core::JobSystem* jobs) {
    cellCounts.assign(params.cellCount, 0U);
    cellParticles.resize(static_cast<size_t>(params.cellCount) * FLUID_CELL_CAPACITY);
    auto count = static_cast<uint32_t>(positions.size());
    cellOf.resize(count);

    // 셀 번호는 서로 독립이라 나눠 계산한다.
    forRange(jobs, count, [&](uint32_t begin, uint32_t end) {
        for (uint32_t i = begin; i < end; ++i) {
            cellOf[i] = fluidHash(fluidCell(glm::vec3{positions[i]}, params.smoothingRadius), params.cellCount);
        }
    });

    // 버킷에 넣는 것은 직렬이다. GPU 경로는 원자 카운터로 자리를 나눠 주므로 한 버킷 안의 순서가
    // 스레드 인터리빙마다 달라지고, 그 순서가 곧 밀도·힘의 부동소수 누적 순서라 결과가 갈린다.
    // SPH 는 혼돈계라 마지막 비트 차이가 몇십 프레임이면 눈에 보이는 차이로 자란다. 입자 번호
    // 오름차순으로 넣으면 워커 수와 무관하게 같은 물이 나온다. O(n) 이라 이웃 순회에 견주면 없는
    // 비용이다(입자 8192 에 몇 마이크로초, 이웃 순회는 밀리초 단위).
    //
    // ponytail: 버킷이 넘치면 번호가 앞선 32 개만 남고 나머지는 이웃 검색에서 빠진다. 정확히
    // 하려면 계수 정렬로 버킷 용량을 없애야 한다. GPU 경로도 같은 한계를 갖는다.
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t cell = cellOf[i];
        uint32_t slot = cellCounts[cell]++;
        if (slot < FLUID_CELL_CAPACITY) {
            cellParticles[static_cast<size_t>(cell) * FLUID_CELL_CAPACITY + slot] = i;
        }
    }
}

void FluidSolver::substep(const FluidParams& params, float dt, core::JobSystem* jobs) {
    buildGrid(params, jobs);

    auto count = static_cast<uint32_t>(positions.size());
    float h = params.smoothingRadius;
    float mass = params.particleMass;

    // 이웃 27 셀을 훑어 밀도와 압력을 구한다. 자기 원소의 w 에만 쓰고 이웃의 xyz 만 읽는다. w 와 xyz 는
    // 같은 vec4 의 다른 스칼라 멤버라 표준상 서로 다른 객체이고, 컴파일러가 저장을 넓히지 않는 한
    // 겹치지 않는다.
    forRange(jobs, count, [&](uint32_t begin, uint32_t end) {
        for (uint32_t i = begin; i < end; ++i) {
            glm::vec3 position{positions[i]};
            glm::ivec3 cell = fluidCell(position, h);
            float density = 0.0F;
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        uint32_t bucket = fluidHash(cell + glm::ivec3{dx, dy, dz}, params.cellCount);
                        uint32_t bucketCount = std::min(cellCounts[bucket], FLUID_CELL_CAPACITY);
                        for (uint32_t k = 0; k < bucketCount; ++k) {
                            uint32_t j = cellParticles[static_cast<size_t>(bucket) * FLUID_CELL_CAPACITY + k];
                            glm::vec3 delta = position - glm::vec3{positions[j]};
                            density += mass * poly6(glm::dot(delta, delta), h);
                        }
                    }
                }
            }
            density = std::max(density, 1e-6F);
            positions[i].w = density;
            velocities[i].w = std::max(params.stiffness * (density - params.restDensity), 0.0F);
        }
    });

    // 압력·점성·중력으로 가속을 구해 반암시적 오일러로 적분하고 용기와 강체에 부딪힌다.
    forRange(jobs, count, [&](uint32_t begin, uint32_t end) {
        for (uint32_t i = begin; i < end; ++i) {
            glm::vec3 position{positions[i]};
            float density = std::max(positions[i].w, 1e-6F);
            glm::vec3 velocity{velocities[i]};
            float pressure = velocities[i].w;
            glm::ivec3 cell = fluidCell(position, h);

            glm::vec3 force{0.0F};
            for (int dz = -1; dz <= 1; ++dz) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        uint32_t bucket = fluidHash(cell + glm::ivec3{dx, dy, dz}, params.cellCount);
                        uint32_t bucketCount = std::min(cellCounts[bucket], FLUID_CELL_CAPACITY);
                        for (uint32_t k = 0; k < bucketCount; ++k) {
                            uint32_t j = cellParticles[static_cast<size_t>(bucket) * FLUID_CELL_CAPACITY + k];
                            if (j == i) {
                                continue;
                            }
                            glm::vec3 delta = position - glm::vec3{positions[j]};
                            float distance = glm::length(delta);
                            if (distance >= h || distance <= 1e-6F) {
                                continue;
                            }
                            float otherDensity = std::max(positions[j].w, 1e-6F);
                            // 압력은 대칭으로 평균 내야 작용·반작용이 맞는다.
                            force += -mass * (pressure + velocities[j].w) / (2.0F * otherDensity) *
                                     spikyGradient(delta, distance, h);
                            force += params.viscosity * mass * (glm::vec3{velocities[j]} - velocity) / otherDensity *
                                     viscosityLaplacian(distance, h);
                        }
                    }
                }
            }

            glm::vec3 acceleration = force / density + params.gravity;
            velocity += acceleration * dt;
            // 한 스텝에 커널 반지름의 일부 이상 움직이지 못하게 잘라 폭주를 막는다.
            float maxSpeed = 0.4F * h / dt;
            float speed = glm::length(velocity);
            if (speed > maxSpeed) {
                velocity *= maxSpeed / speed;
            }
            position += velocity * dt;

            glm::vec3 low = params.containerMin + params.particleRadius;
            glm::vec3 high = params.containerMax - params.particleRadius;
            for (int axis = 0; axis < 3; ++axis) {
                if (position[axis] < low[axis]) {
                    position[axis] = low[axis];
                    if (velocity[axis] < 0.0F) {
                        velocity[axis] = -velocity[axis] * params.wallRestitution;
                    }
                } else if (position[axis] > high[axis]) {
                    position[axis] = high[axis];
                    if (velocity[axis] > 0.0F) {
                        velocity[axis] = -velocity[axis] * params.wallRestitution;
                    }
                }
            }

            // 강체 콜라이더. 일방향이라 입자는 밀려나지만 강체는 밀리지 않는다. 모양별 기하는 강체
            // 솔버와 같은 collider_shapes.h 다.
            for (uint32_t c = 0; c < params.colliderCount; ++c) {
                const FluidCollider& collider = params.colliders[c];
                glm::vec3 local = glm::vec3{collider.inverseWorld * glm::vec4{position, 1.0F}};
                SurfacePoint surface =
                    closestOnColliderLocal(ColliderLocal{collider.shape, collider.radius, collider.halfExtents}, local);
                if (surface.distance < params.particleRadius) {
                    glm::vec3 normal = glm::normalize(glm::mat3{collider.world} * surface.normal);
                    position =
                        glm::vec3{collider.world * glm::vec4{surface.point, 1.0F}} + normal * params.particleRadius;
                    bounce(velocity, normal, params.wallRestitution);
                }
            }

            nextPositions[i] = glm::vec4{position, density};
            nextVelocities[i] = glm::vec4{velocity, pressure};
        }
    });

    positions.swap(nextPositions);
    velocities.swap(nextVelocities);
}

void FluidSolver::step(const FluidParams& params, float deltaSeconds, core::JobSystem* jobs) {
    if (positions.empty() || deltaSeconds <= 0.0F) {
        return;
    }
    uint32_t substeps = fluidSubsteps(params, deltaSeconds);
    float frameStep = std::min(deltaSeconds, FLUID_MAX_FRAME_STEP);
    float dt = frameStep / static_cast<float>(substeps);
    for (uint32_t i = 0; i < substeps; ++i) {
        substep(params, dt, jobs);
    }
}

} // namespace physics
