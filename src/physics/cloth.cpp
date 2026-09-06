#include "physics/cloth.h"

#include <algorithm>
#include <cmath>
#include <functional>

#include <glm/geometric.hpp>

#include "core/job_system.h"
#include "physics/collider_shapes.h"

namespace physics {
namespace {

constexpr uint32_t GRANULARITY = 256;

void forRange(core::JobSystem* jobs, uint32_t count, const std::function<void(uint32_t, uint32_t)>& body) {
    if (jobs != nullptr && count > GRANULARITY) {
        jobs->parallelFor(count, GRANULARITY, body);
    } else {
        body(0, count);
    }
}

} // namespace

bool buildClothTopology(const std::vector<glm::vec3>& localPositions,
                        const glm::mat4& world,
                        const scene::Cloth& settings,
                        ClothTopology& out) {
    uint32_t n = settings.resolution;
    uint32_t side = n + 1;
    out = ClothTopology{};
    if (n < 2 || localPositions.size() != static_cast<size_t>(side) * side) {
        return false;
    }
    out.resolution = n;
    out.gridToVertex.assign(static_cast<size_t>(side) * side, UINT32_MAX);
    out.vertices.resize(localPositions.size());
    out.rest.resize(localPositions.size());

    // 내장 격자는 XZ 평면 ±1 이다. 좌표를 양자화해 격자 자리를 되찾는다. meshopt 가 정점을 재배열해도 위치는
    // 그대로라 순서에 기대지 않는다.
    for (uint32_t vertex = 0; vertex < localPositions.size(); ++vertex) {
        const glm::vec3& local = localPositions[vertex];
        float fx = (local.x + 1.0F) * 0.5F * static_cast<float>(n);
        float fz = (local.z + 1.0F) * 0.5F * static_cast<float>(n);
        auto gx = static_cast<int32_t>(std::lround(fx));
        auto gz = static_cast<int32_t>(std::lround(fz));
        if (gx < 0 || gz < 0 || gx > static_cast<int32_t>(n) || gz > static_cast<int32_t>(n) ||
            std::abs(fx - static_cast<float>(gx)) > 0.25F || std::abs(fz - static_cast<float>(gz)) > 0.25F) {
            return false;
        }
        uint32_t cell = static_cast<uint32_t>(gx) + static_cast<uint32_t>(gz) * side;
        if (out.gridToVertex[cell] != UINT32_MAX) {
            return false;
        }
        out.gridToVertex[cell] = vertex;
        out.vertices[vertex].gridX = static_cast<uint32_t>(gx);
        out.vertices[vertex].gridZ = static_cast<uint32_t>(gz);
        out.rest[vertex] = glm::vec4{glm::vec3{world * glm::vec4{local, 1.0F}}, 0.0F};
    }

    // 고정: 네 변 가운데 월드 Y 평균이 가장 높은 변. 오브젝트를 어떻게 돌려 놓아도 «위» 가 고정된다.
    auto at = [&](uint32_t x, uint32_t z) { return out.gridToVertex[x + z * side]; };
    std::array<float, 4> edgeHeight{};
    for (uint32_t i = 0; i < side; ++i) {
        edgeHeight[0] += out.rest[at(i, 0)].y;
        edgeHeight[1] += out.rest[at(i, n)].y;
        edgeHeight[2] += out.rest[at(0, i)].y;
        edgeHeight[3] += out.rest[at(n, i)].y;
    }
    size_t topEdge = static_cast<size_t>(std::max_element(edgeHeight.begin(), edgeHeight.end()) - edgeHeight.begin());
    auto edgeVertex = [&](size_t edge, uint32_t i) {
        switch (edge) {
        case 0:
            return at(i, 0);
        case 1:
            return at(i, n);
        case 2:
            return at(0, i);
        default:
            return at(n, i);
        }
    };
    std::vector<uint8_t> pinned(localPositions.size(), 0);
    if (settings.pin == scene::ClothPin::TOP_EDGE) {
        for (uint32_t i = 0; i < side; ++i) {
            pinned[edgeVertex(topEdge, i)] = 1;
        }
    } else if (settings.pin == scene::ClothPin::TWO_CORNERS) {
        pinned[edgeVertex(topEdge, 0)] = 1;
        pinned[edgeVertex(topEdge, n)] = 1;
    }
    float inverseMass = static_cast<float>(side * side) / std::max(settings.mass, 1.0e-3F);
    for (uint32_t vertex = 0; vertex < localPositions.size(); ++vertex) {
        out.rest[vertex].w = pinned[vertex] != 0 ? 0.0F : inverseMass;
    }

    // 제약을 색별로 모은다. 한 색 안의 제약들은 정점을 나눠 갖지 않는다:
    //   0·1  가로 신장 (x,z)-(x+1,z)      색 = x 홀짝
    //   2·3  세로 신장 (x,z)-(x,z+1)      색 = z 홀짝
    //   4·5  전단      (x,z)-(x+1,z+1)    색 = x 홀짝
    //   6·7  전단      (x+1,z)-(x,z+1)    색 = x 홀짝
    //   8·9  가로 굽힘 (x,z)-(x+2,z)      색 = (x/2) 홀짝
    //  10·11 세로 굽힘 (x,z)-(x,z+2)      색 = (z/2) 홀짝
    std::array<std::vector<ClothConstraint>, CLOTH_COLORS> buckets;
    auto link = [&](uint32_t color, uint32_t a, uint32_t b, float compliance) {
        ClothConstraint constraint;
        constraint.a = a;
        constraint.b = b;
        constraint.restLength = glm::distance(glm::vec3{out.rest[a]}, glm::vec3{out.rest[b]});
        constraint.compliance = compliance;
        buckets[color].push_back(constraint);
    };
    for (uint32_t z = 0; z < side; ++z) {
        for (uint32_t x = 0; x < side; ++x) {
            if (x + 1 < side) {
                link(x % 2, at(x, z), at(x + 1, z), settings.stretchCompliance);
            }
            if (z + 1 < side) {
                link(2 + z % 2, at(x, z), at(x, z + 1), settings.stretchCompliance);
            }
            if (x + 1 < side && z + 1 < side) {
                link(4 + x % 2, at(x, z), at(x + 1, z + 1), settings.shearCompliance);
                link(6 + x % 2, at(x + 1, z), at(x, z + 1), settings.shearCompliance);
            }
            if (x + 2 < side) {
                link(8 + (x / 2) % 2, at(x, z), at(x + 2, z), settings.bendCompliance);
            }
            if (z + 2 < side) {
                link(10 + (z / 2) % 2, at(x, z), at(x, z + 2), settings.bendCompliance);
            }
        }
    }
    for (uint32_t color = 0; color < CLOTH_COLORS; ++color) {
        out.colorBegin[color] = static_cast<uint32_t>(out.constraints.size());
        out.constraints.insert(out.constraints.end(), buckets[color].begin(), buckets[color].end());
    }
    out.colorBegin[CLOTH_COLORS] = static_cast<uint32_t>(out.constraints.size());
    out.stretchCount = out.colorBegin[4];
    out.shearCount = out.colorBegin[8] - out.colorBegin[4];
    out.bendCount = out.colorBegin[CLOTH_COLORS] - out.colorBegin[8];

    glm::vec3 minimum{out.rest[0]};
    glm::vec3 maximum = minimum;
    for (const glm::vec4& rest : out.rest) {
        minimum = glm::min(minimum, glm::vec3{rest});
        maximum = glm::max(maximum, glm::vec3{rest});
    }
    out.restCenter = (minimum + maximum) * 0.5F;
    out.restRadius = glm::length(maximum - minimum) * 0.5F;
    return true;
}

void collectMeshTriangles(const scene::Scene& scene, uint32_t skipObject, std::vector<ClothTriangle>& out) {
    out.clear();
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        int32_t slot = scene.objects[index].rigidBody;
        if (index == skipObject || slot < 0 || static_cast<size_t>(slot) >= scene.rigidBodies.size() ||
            !scene.visibleCached(index) ||
            scene.rigidBodies[static_cast<size_t>(slot)].shape != scene::ColliderShape::MESH) {
            continue;
        }
        const scene::ColliderMesh* mesh = scene.colliderMesh(index);
        if (mesh == nullptr) {
            continue;
        }
        const glm::mat4& world = scene.world(index);
        auto toWorld = [&](uint32_t vertex) { return glm::vec3{world * glm::vec4{mesh->positions[vertex], 1.0F}}; };
        for (size_t t = 0; t + 2 < mesh->indices.size(); t += 3) {
            out.push_back(
                ClothTriangle{toWorld(mesh->indices[t]), toWorld(mesh->indices[t + 1]), toWorld(mesh->indices[t + 2])});
        }
    }
}

ClothParams deriveClothParams(const scene::Cloth& settings, const scene::Scene& scene, float deltaSeconds) {
    ClothParams params;
    params.gravity = settings.gravity;
    params.wind = settings.wind;
    params.damping = std::max(settings.damping, 0.0F);
    params.thickness = std::max(settings.thickness, 0.0F);
    params.friction = std::clamp(settings.friction, 0.0F, 1.0F);
    params.substeps = std::clamp(settings.substeps, 1U, 16U);
    params.iterations = std::clamp(settings.iterations, 1U, 32U);
    params.frameStep = std::min(std::max(deltaSeconds, 0.0F), CLOTH_MAX_FRAME_STEP);
    params.colliderCount = collectShapeColliders(scene, params.colliders);
    return params;
}

void ClothSolver::reset(const ClothTopology& topology) {
    size_t count = topology.rest.size();
    current.resize(count);
    for (size_t i = 0; i < count; ++i) {
        current[i] = glm::vec3{topology.rest[i]};
    }
    predicted = current;
    velocity.assign(count, glm::vec3{0.0F});
    normal.assign(count, glm::vec3{0.0F, 1.0F, 0.0F});
    tangent.assign(count, glm::vec3{1.0F, 0.0F, 0.0F});
    lambda.assign(topology.constraints.size(), 0.0F);
    computeNormals(topology);
}

void ClothSolver::step(const ClothTopology& topology,
                       const ClothParams& params,
                       const std::vector<ClothTriangle>& triangles,
                       core::JobSystem* jobs) {
    auto vertexCount = static_cast<uint32_t>(current.size());
    auto constraintCount = static_cast<uint32_t>(topology.constraints.size());
    if (vertexCount == 0 || params.frameStep <= 0.0F) {
        return;
    }
    float h = params.frameStep / static_cast<float>(params.substeps);
    float inverseH2 = 1.0F / (h * h);
    glm::vec3 acceleration = params.gravity + params.wind;

    // 메쉬 삼각형은 천의 현재 경계 구와 겹치는 것만 본다. O(V·T) 협역 검사의 T 를 줄인다.
    //
    // ponytail: 삼각형마다 훑는다. 큰 메쉬 콜라이더 위의 촘촘한 천은 느리다. BVH 가 답이다.
    std::vector<const ClothTriangle*> nearby;
    if (!triangles.empty()) {
        glm::vec4 sphere = bounds();
        float reach = sphere.w + glm::length(acceleration) * params.frameStep * params.frameStep + params.thickness +
                      params.frameStep * 10.0F;
        for (const ClothTriangle& triangle : triangles) {
            glm::vec3 closest = closestOnTriangle(triangle.a, triangle.b, triangle.c, glm::vec3{sphere});
            if (glm::distance(closest, glm::vec3{sphere}) <= reach) {
                nearby.push_back(&triangle);
            }
        }
    }

    for (uint32_t substep = 0; substep < params.substeps; ++substep) {
        // 예측. 고정점(질량 역수 0)은 제자리다.
        forRange(jobs, vertexCount, [&](uint32_t begin, uint32_t end) {
            for (uint32_t i = begin; i < end; ++i) {
                if (topology.rest[i].w <= 0.0F) {
                    predicted[i] = current[i];
                    velocity[i] = glm::vec3{0.0F};
                    continue;
                }
                velocity[i] += acceleration * h;
                velocity[i] *= std::max(1.0F - params.damping * h, 0.0F);
                predicted[i] = current[i] + velocity[i] * h;
            }
        });
        std::fill(lambda.begin(), lambda.end(), 0.0F);

        for (uint32_t iteration = 0; iteration < params.iterations; ++iteration) {
            // 색마다 병렬. 한 색 안에서는 정점 하나를 제약 하나만 만지므로 바로 위치를 고쳐도 경쟁이 없다.
            for (uint32_t color = 0; color < CLOTH_COLORS; ++color) {
                uint32_t first = topology.colorBegin[color];
                uint32_t count = topology.colorBegin[color + 1] - first;
                forRange(jobs, count, [&](uint32_t begin, uint32_t end) {
                    for (uint32_t c = first + begin; c < first + end; ++c) {
                        const ClothConstraint& constraint = topology.constraints[c];
                        float wa = topology.rest[constraint.a].w;
                        float wb = topology.rest[constraint.b].w;
                        glm::vec3 direction = predicted[constraint.a] - predicted[constraint.b];
                        float length = glm::length(direction);
                        if (length <= 1.0e-6F || wa + wb <= 0.0F) {
                            continue;
                        }
                        direction /= length;
                        float violation = length - constraint.restLength;
                        float alpha = constraint.compliance * inverseH2;
                        float deltaLambda = (-violation - alpha * lambda[c]) / (wa + wb + alpha);
                        lambda[c] += deltaLambda;
                        predicted[constraint.a] += direction * (wa * deltaLambda);
                        predicted[constraint.b] -= direction * (wb * deltaLambda);
                    }
                });
            }
        }

        // 마무리: 충돌, 속도, 위치.
        forRange(jobs, vertexCount, [&](uint32_t begin, uint32_t end) {
            for (uint32_t i = begin; i < end; ++i) {
                if (topology.rest[i].w <= 0.0F) {
                    continue;
                }
                glm::vec3 position = predicted[i];
                glm::vec3 contactNormal{0.0F};
                // 도형 콜라이더. 유체 입자와 같은 식이다.
                for (uint32_t c = 0; c < params.colliderCount; ++c) {
                    const FluidCollider& collider = params.colliders[c];
                    glm::vec3 local = glm::vec3{collider.inverseWorld * glm::vec4{position, 1.0F}};
                    SurfacePoint surface = closestOnColliderLocal(
                        ColliderLocal{collider.shape, collider.radius, collider.halfExtents}, local);
                    if (surface.distance < params.thickness) {
                        contactNormal = glm::normalize(glm::mat3{collider.world} * surface.normal);
                        position = glm::vec3{collider.world * glm::vec4{surface.point, 1.0F}} +
                                   contactNormal * params.thickness;
                    }
                }
                // 메쉬 콜라이더. 삼각형마다 구 대 삼각형이다. 뒷면으로 두께만큼 들어간 점은 앞면으로 올린다.
                for (const ClothTriangle* triangle : nearby) {
                    SurfacePoint surface;
                    if (closestOnTriangleSurface(triangle->a,
                                                 triangle->b,
                                                 triangle->c,
                                                 position,
                                                 params.thickness,
                                                 params.thickness,
                                                 surface)) {
                        contactNormal = surface.normal;
                        position = surface.point + surface.normal * params.thickness;
                    }
                }
                // 마찰: 닿았으면 이번 서브스텝의 접선 이동을 friction 만큼 죽인다. GPU cloth_finish.comp 와 같은 식.
                if (contactNormal != glm::vec3{0.0F}) {
                    glm::vec3 motion = position - current[i];
                    glm::vec3 tangential = motion - contactNormal * glm::dot(motion, contactNormal);
                    position -= tangential * params.friction;
                }
                velocity[i] = (position - current[i]) / h;
                current[i] = position;
            }
        });
    }
    computeNormals(topology);
}

void ClothSolver::computeNormals(const ClothTopology& topology) {
    uint32_t n = topology.resolution;
    uint32_t side = n + 1;
    auto at = [&](uint32_t x, uint32_t z) { return topology.gridToVertex[x + z * side]; };
    for (uint32_t i = 0; i < current.size(); ++i) {
        const ClothVertexInfo& info = topology.vertices[i];
        uint32_t x0 = info.gridX > 0 ? info.gridX - 1 : 0;
        uint32_t x1 = std::min(info.gridX + 1, n);
        uint32_t z0 = info.gridZ > 0 ? info.gridZ - 1 : 0;
        uint32_t z1 = std::min(info.gridZ + 1, n);
        glm::vec3 dx = current[at(x1, info.gridZ)] - current[at(x0, info.gridZ)];
        glm::vec3 dz = current[at(info.gridX, z1)] - current[at(info.gridX, z0)];
        // 격자 v 는 +Z 로 늘어나고 정지 법선은 +Y 다. cross(dz, dx) 가 +Y 를 낸다.
        glm::vec3 cross = glm::cross(dz, dx);
        float length = glm::length(cross);
        normal[i] = length > 1.0e-8F ? cross / length : glm::vec3{0.0F, 1.0F, 0.0F};
        float dxLength = glm::length(dx);
        tangent[i] = dxLength > 1.0e-8F ? dx / dxLength : glm::vec3{1.0F, 0.0F, 0.0F};
    }
}

glm::vec4 ClothSolver::bounds() const {
    if (current.empty()) {
        return glm::vec4{0.0F};
    }
    glm::vec3 minimum = current[0];
    glm::vec3 maximum = minimum;
    for (const glm::vec3& position : current) {
        minimum = glm::min(minimum, position);
        maximum = glm::max(maximum, position);
    }
    glm::vec3 center = (minimum + maximum) * 0.5F;
    return glm::vec4{center, glm::length(maximum - minimum) * 0.5F};
}

} // namespace physics
