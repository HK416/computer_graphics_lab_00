#include <algorithm>
#include <atomic>
#include <cstdint>
#include <limits>
#include <vector>

#include <glm/geometric.hpp>
#include <meshoptimizer.h>
#include <spdlog/spdlog.h>

#include "asset/load_progress.h"
#include "asset/model.h"
#include "core/job_system.h"

namespace asset {
namespace {

// NVIDIA 와 AMD 의 mesh shader 권장 한도에 맞춘 값이다.
constexpr size_t MAX_MESHLET_VERTICES = 64;
constexpr size_t MAX_MESHLET_TRIANGLES = 124;
constexpr float CONE_WEIGHT = 0.5F;

// 한 그룹으로 묶을 meshlet 수와 단순화 목표. Nanite 와 같은 절반 감축을 노린다.
constexpr size_t TARGET_GROUP_SIZE = 8;
constexpr float SIMPLIFY_RATIO = 0.5F;
constexpr float SIMPLIFY_TARGET_ERROR = 0.5F;
// 그룹이 이 비율보다 덜 줄어들면 단순화에 실패한 것으로 본다.
constexpr float MIN_SIMPLIFY_PROGRESS = 0.9F;
// 단순화에 실패한 그룹도 다음 단계로 그대로 올려 각 단계가 완전한 메쉬가 되게 한다. 이때 오차를
// 아주 조금 올려 부모 오차가 자식보다 항상 크다는 성질을 지킨다.
constexpr float CARRY_ERROR_EPSILON = 1e-4F;
constexpr uint32_t MAX_LOD_LEVELS = 12;

// 빌드 중에만 쓰는 표현. 정점은 원본 배열을 가리키고, 삼각형은 meshlet 지역 인덱스다.
struct BuildMeshlet {
    std::vector<uint32_t> vertices;
    std::vector<uint8_t> triangles;
    glm::vec4 boundingSphere{0.0F};
    glm::vec4 cone{0.0F};
    glm::vec4 errorSphere{0.0F};
    glm::vec4 parentSphere{0.0F};
    float error = 0.0F;
    float parentError = std::numeric_limits<float>::max();
    uint32_t level = 0;
};

std::vector<uint32_t> expandIndices(const BuildMeshlet& meshlet) {
    std::vector<uint32_t> indices;
    indices.reserve(meshlet.triangles.size());
    for (uint8_t local : meshlet.triangles) {
        indices.push_back(meshlet.vertices[local]);
    }
    return indices;
}

// 여러 경계 구를 모두 감싸는 구. 중심을 평균으로 잡는 보수적 근사다.
glm::vec4 mergeSpheres(const std::vector<glm::vec4>& spheres) {
    if (spheres.empty()) {
        return glm::vec4{0.0F};
    }
    glm::vec3 center{0.0F};
    for (const glm::vec4& sphere : spheres) {
        center += glm::vec3(sphere);
    }
    center /= static_cast<float>(spheres.size());

    float radius = 0.0F;
    for (const glm::vec4& sphere : spheres) {
        radius = std::max(radius, glm::distance(center, glm::vec3(sphere)) + sphere.w);
    }
    return glm::vec4{center, radius};
}

std::vector<BuildMeshlet>
splitIntoMeshlets(const std::vector<uint32_t>& indices, const std::vector<Vertex>& vertices, uint32_t level) {
    std::vector<BuildMeshlet> result;
    if (indices.empty()) {
        return result;
    }

    const float* positions = &vertices.front().position.x;
    size_t maxMeshlets = meshopt_buildMeshletsBound(indices.size(), MAX_MESHLET_VERTICES, MAX_MESHLET_TRIANGLES);
    std::vector<meshopt_Meshlet> raw(maxMeshlets);
    std::vector<uint32_t> meshletVertices(maxMeshlets * MAX_MESHLET_VERTICES);
    std::vector<uint8_t> meshletTriangles(maxMeshlets * MAX_MESHLET_TRIANGLES * 3);

    size_t count = meshopt_buildMeshlets(raw.data(),
                                         meshletVertices.data(),
                                         meshletTriangles.data(),
                                         indices.data(),
                                         indices.size(),
                                         positions,
                                         vertices.size(),
                                         sizeof(Vertex),
                                         MAX_MESHLET_VERTICES,
                                         MAX_MESHLET_TRIANGLES,
                                         CONE_WEIGHT);
    result.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        const meshopt_Meshlet& source = raw[i];
        meshopt_Bounds bounds = meshopt_computeMeshletBounds(&meshletVertices[source.vertex_offset],
                                                             &meshletTriangles[source.triangle_offset],
                                                             source.triangle_count,
                                                             positions,
                                                             vertices.size(),
                                                             sizeof(Vertex));

        BuildMeshlet meshlet;
        meshlet.level = level;
        meshlet.boundingSphere = glm::vec4{bounds.center[0], bounds.center[1], bounds.center[2], bounds.radius};
        meshlet.cone = glm::vec4{bounds.cone_axis[0], bounds.cone_axis[1], bounds.cone_axis[2], bounds.cone_cutoff};
        meshlet.vertices.assign(&meshletVertices[source.vertex_offset],
                                &meshletVertices[source.vertex_offset] + source.vertex_count);
        meshlet.triangles.assign(&meshletTriangles[source.triangle_offset],
                                 &meshletTriangles[source.triangle_offset] + source.triangle_count * 3);
        result.push_back(std::move(meshlet));
    }
    return result;
}

// 인접한 meshlet 을 묶어 함께 단순화한다. 그룹 경계 정점을 잠가야 이웃 그룹과 틈이 생기지 않는다.
std::vector<std::vector<size_t>> partitionMeshlets(const std::vector<BuildMeshlet>& meshlets,
                                                   const std::vector<Vertex>& vertices) {
    std::vector<uint32_t> clusterIndices;
    std::vector<uint32_t> clusterCounts(meshlets.size());
    for (size_t i = 0; i < meshlets.size(); ++i) {
        std::vector<uint32_t> expanded = expandIndices(meshlets[i]);
        clusterCounts[i] = static_cast<uint32_t>(expanded.size());
        clusterIndices.insert(clusterIndices.end(), expanded.begin(), expanded.end());
    }

    std::vector<uint32_t> partitionOf(meshlets.size());
    size_t partitionCount = meshopt_partitionClusters(partitionOf.data(),
                                                      clusterIndices.data(),
                                                      clusterIndices.size(),
                                                      clusterCounts.data(),
                                                      meshlets.size(),
                                                      &vertices.front().position.x,
                                                      vertices.size(),
                                                      sizeof(Vertex),
                                                      TARGET_GROUP_SIZE);

    std::vector<std::vector<size_t>> groups(partitionCount);
    for (size_t i = 0; i < meshlets.size(); ++i) {
        groups[partitionOf[i]].push_back(i);
    }
    return groups;
}

} // namespace

uint64_t lodWorkEstimate(const Mesh& mesh) {
    // 0 단계 분할이 인덱스 수만큼, 그 뒤 단계가 절반씩 줄어드는 인덱스 수만큼 든다고 보면 합은 세 배쯤이다.
    return static_cast<uint64_t>(mesh.indices.size()) * 3;
}

void buildLodHierarchy(Mesh& mesh, core::JobSystem* jobs, LoadProgress* progress) {
    // 이 메쉬가 진행률에 더하는 몫은 어림값과 정확히 같아야 한다. 덜 쓰면 끝에서 메우고, 더 쓰면
    // (단순화에 실패한 그룹이 많아 단계마다 절반으로 줄지 않을 때) 어림값에서 자른다. 그래야 메쉬
    // 여럿이 총량 하나를 나눠 쓸 때 한 메쉬가 남의 몫까지 채우지 않는다.
    uint64_t workBudget = lodWorkEstimate(mesh);
    std::atomic<uint64_t> workConsumed{0};
    auto reportWork = [&](uint64_t amount) {
        uint64_t before = workConsumed.fetch_add(amount, std::memory_order_relaxed);
        uint64_t remaining = before < workBudget ? workBudget - before : 0;
        if (progress != nullptr && remaining > 0) {
            progress->advance(std::min(amount, remaining));
        }
    };
    if (mesh.indices.empty() || mesh.vertices.empty()) {
        reportWork(workBudget);
        return;
    }

    std::vector<Vertex> canonical = std::move(mesh.vertices);
    std::vector<uint32_t> indices = std::move(mesh.indices);
    const float* positions = &canonical.front().position.x;

    meshopt_optimizeVertexCache(indices.data(), indices.data(), indices.size(), canonical.size());
    meshopt_optimizeOverdraw(
        indices.data(), indices.data(), indices.size(), positions, canonical.size(), sizeof(Vertex), 1.05F);

    // 단순화 오차는 정규화된 값이라 월드 단위로 되돌려야 화면 오차로 환산할 수 있다.
    float simplifyScale = meshopt_simplifyScale(positions, canonical.size(), sizeof(Vertex));

    std::vector<std::vector<BuildMeshlet>> levels;
    levels.push_back(splitIntoMeshlets(indices, canonical, 0));
    reportWork(indices.size());

    for (uint32_t level = 1; level < MAX_LOD_LEVELS; ++level) {
        std::vector<BuildMeshlet>& previous = levels.back();
        if (previous.size() <= 1) {
            break;
        }

        std::vector<std::vector<size_t>> groups = partitionMeshlets(previous, canonical);
        std::vector<BuildMeshlet> next;
        bool anySimplified = false;

        // 그룹끼리는 서로 겹치지 않으므로 단순화를 병렬로 돌릴 수 있다. 자식에 쓰는 부모 정보도
        // 그룹마다 서로 다른 meshlet 을 건드리므로 경합이 없다.
        struct GroupResult {
            std::vector<BuildMeshlet> produced;
            bool reduced = false;
            bool valid = false;
        };
        std::vector<GroupResult> results(groups.size());

        auto processGroups = [&](uint32_t begin, uint32_t end) {
            for (uint32_t groupIndex = begin; groupIndex < end; ++groupIndex) {
                const std::vector<size_t>& group = groups[groupIndex];
                if (group.empty()) {
                    continue;
                }

                std::vector<uint32_t> merged;
                std::vector<glm::vec4> childSpheres;
                float childError = 0.0F;
                for (size_t child : group) {
                    std::vector<uint32_t> expanded = expandIndices(previous[child]);
                    merged.insert(merged.end(), expanded.begin(), expanded.end());
                    childSpheres.push_back(previous[child].boundingSphere);
                    childError = std::max(childError, previous[child].error);
                }

                auto targetIndexCount = static_cast<size_t>(static_cast<float>(merged.size()) * SIMPLIFY_RATIO);
                targetIndexCount = (targetIndexCount / 3) * 3;
                if (targetIndexCount < 3) {
                    continue;
                }

                // 그룹이 쓰는 정점만 모아 지역 번호로 바꾼다. meshopt_simplify 는 넘긴 정점 수에 비례해
                // 배열을 만들고 훑으므로, 전체 메쉬를 넘기면 그룹마다 메쉬 전체를 다시 읽는 셈이 된다.
                // 정점 2천만 개 메쉬에서 그룹 하나가 0.6 초씩 걸리던 것이 이 압축으로 밀리초 아래로 내려간다.
                std::vector<uint32_t> groupVertices = merged;
                std::sort(groupVertices.begin(), groupVertices.end());
                groupVertices.erase(std::unique(groupVertices.begin(), groupVertices.end()), groupVertices.end());
                std::vector<glm::vec3> groupPositions(groupVertices.size());
                for (size_t i = 0; i < groupVertices.size(); ++i) {
                    groupPositions[i] = canonical[groupVertices[i]].position;
                }
                std::vector<uint32_t> localIndices(merged.size());
                for (size_t i = 0; i < merged.size(); ++i) {
                    localIndices[i] =
                        static_cast<uint32_t>(std::lower_bound(groupVertices.begin(), groupVertices.end(), merged[i]) -
                                              groupVertices.begin());
                }

                // 오차는 절대 단위로 주고받는다. 상대 오차는 넘긴 정점의 범위 기준이라 그룹마다 달라진다.
                std::vector<uint32_t> simplified(merged.size());
                float absoluteError = 0.0F;
                size_t simplifiedCount = meshopt_simplify(simplified.data(),
                                                          localIndices.data(),
                                                          localIndices.size(),
                                                          &groupPositions.front().x,
                                                          groupPositions.size(),
                                                          sizeof(glm::vec3),
                                                          targetIndexCount,
                                                          SIMPLIFY_TARGET_ERROR * simplifyScale,
                                                          meshopt_SimplifyLockBorder | meshopt_SimplifyErrorAbsolute,
                                                          &absoluteError);
                for (size_t i = 0; i < simplifiedCount; ++i) {
                    simplified[i] = groupVertices[simplified[i]];
                }

                bool reduced =
                    static_cast<float>(simplifiedCount) <= static_cast<float>(merged.size()) * MIN_SIMPLIFY_PROGRESS;
                if (reduced) {
                    simplified.resize(simplifiedCount);
                } else {
                    // 줄지 않은 그룹도 그대로 올려 두어야 이 단계만 그렸을 때 빈 곳이 생기지 않는다.
                    simplified = merged;
                }

                glm::vec4 groupSphere = mergeSpheres(childSpheres);
                float groupError = childError + (reduced ? absoluteError : simplifyScale * CARRY_ERROR_EPSILON);

                for (size_t child : group) {
                    previous[child].parentSphere = groupSphere;
                    previous[child].parentError = groupError;
                }

                results[groupIndex].produced = splitIntoMeshlets(simplified, canonical, level);
                for (BuildMeshlet& meshlet : results[groupIndex].produced) {
                    meshlet.errorSphere = groupSphere;
                    meshlet.error = groupError;
                }
                results[groupIndex].reduced = reduced;
                results[groupIndex].valid = true;
                reportWork(merged.size());
            }
        };

        if (jobs != nullptr) {
            jobs->parallelFor(static_cast<uint32_t>(groups.size()), 1, processGroups);
        } else {
            processGroups(0, static_cast<uint32_t>(groups.size()));
        }

        for (GroupResult& result : results) {
            if (!result.valid) {
                continue;
            }
            anySimplified = anySimplified || result.reduced;
            for (BuildMeshlet& meshlet : result.produced) {
                next.push_back(std::move(meshlet));
            }
        }

        if (next.empty() || !anySimplified) {
            // 어느 그룹도 줄지 않았다면 이 단계는 복사본일 뿐이므로 버리고 부모 연결도 되돌린다.
            for (BuildMeshlet& meshlet : previous) {
                meshlet.parentSphere = glm::vec4{0.0F};
                meshlet.parentError = std::numeric_limits<float>::max();
            }
            break;
        }
        levels.push_back(std::move(next));
    }

    // 0단계 meshlet 은 원본 그대로이므로 오차가 없다.
    for (BuildMeshlet& meshlet : levels.front()) {
        meshlet.errorSphere = meshlet.boundingSphere;
        meshlet.error = 0.0F;
    }

    if (uint64_t consumed = workConsumed.load(std::memory_order_relaxed); consumed < workBudget) {
        reportWork(workBudget - consumed);
    }

    // GPU 배치로 펼친다. 정점은 원본 그대로 두고 모든 단계가 공유한다. meshlet 은 쓰는 정점의
    // 번호 목록을 갖고, 인덱스는 그 번호를 풀어 LOD 단계별로 이어 둔다.
    mesh.vertices = std::move(canonical);
    mesh.indices.clear();
    mesh.meshlets.clear();
    mesh.meshletTriangles.clear();
    mesh.meshletVertices.clear();
    mesh.lods.clear();
    mesh.lods.reserve(levels.size());

    for (const std::vector<BuildMeshlet>& level : levels) {
        MeshLod lod;
        lod.indexOffset = static_cast<uint32_t>(mesh.indices.size());
        lod.meshletOffset = static_cast<uint32_t>(mesh.meshlets.size());

        for (const BuildMeshlet& source : level) {
            Meshlet meshlet;
            meshlet.boundingSphere = source.boundingSphere;
            meshlet.cone = source.cone;
            meshlet.errorSphere = source.errorSphere;
            meshlet.parentSphere = source.parentSphere;
            meshlet.error = source.error;
            meshlet.parentError = source.parentError;
            meshlet.level = source.level;
            meshlet.indexOffset = static_cast<uint32_t>(mesh.indices.size());
            meshlet.vertexOffset = static_cast<uint32_t>(mesh.meshletVertices.size());
            meshlet.triangleOffset = static_cast<uint32_t>(mesh.meshletTriangles.size());
            meshlet.vertexCount = static_cast<uint32_t>(source.vertices.size());
            meshlet.triangleCount = static_cast<uint32_t>(source.triangles.size() / 3);

            mesh.meshletVertices.insert(mesh.meshletVertices.end(), source.vertices.begin(), source.vertices.end());
            for (uint8_t local : source.triangles) {
                mesh.meshletTriangles.push_back(local);
                mesh.indices.push_back(source.vertices[local]);
            }
            mesh.meshlets.push_back(meshlet);
        }

        lod.indexCount = static_cast<uint32_t>(mesh.indices.size()) - lod.indexOffset;
        lod.meshletCount = static_cast<uint32_t>(mesh.meshlets.size()) - lod.meshletOffset;
        mesh.lods.push_back(lod);
    }
}

} // namespace asset
