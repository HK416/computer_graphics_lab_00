#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <glm/geometric.hpp>
#include <meshoptimizer.h>
#include <spdlog/spdlog.h>

#include "asset/model.h"

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

void buildLodHierarchy(Mesh& mesh) {
    if (mesh.indices.empty() || mesh.vertices.empty()) {
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

    for (uint32_t level = 1; level < MAX_LOD_LEVELS; ++level) {
        std::vector<BuildMeshlet>& previous = levels.back();
        if (previous.size() <= 1) {
            break;
        }

        std::vector<std::vector<size_t>> groups = partitionMeshlets(previous, canonical);
        std::vector<BuildMeshlet> next;
        bool anySimplified = false;

        for (const std::vector<size_t>& group : groups) {
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

            std::vector<uint32_t> simplified(merged.size());
            float relativeError = 0.0F;
            size_t simplifiedCount = meshopt_simplify(simplified.data(),
                                                      merged.data(),
                                                      merged.size(),
                                                      positions,
                                                      canonical.size(),
                                                      sizeof(Vertex),
                                                      targetIndexCount,
                                                      SIMPLIFY_TARGET_ERROR,
                                                      meshopt_SimplifyLockBorder,
                                                      &relativeError);
            bool reduced =
                static_cast<float>(simplifiedCount) <= static_cast<float>(merged.size()) * MIN_SIMPLIFY_PROGRESS;
            anySimplified = anySimplified || reduced;
            if (reduced) {
                simplified.resize(simplifiedCount);
            } else {
                // 줄지 않은 그룹도 그대로 올려 두어야 이 단계만 그렸을 때 빈 곳이 생기지 않는다.
                simplified = merged;
            }

            glm::vec4 groupSphere = mergeSpheres(childSpheres);
            float groupError =
                childError + (reduced ? relativeError * simplifyScale : simplifyScale * CARRY_ERROR_EPSILON);

            for (size_t child : group) {
                previous[child].parentSphere = groupSphere;
                previous[child].parentError = groupError;
            }

            std::vector<BuildMeshlet> produced = splitIntoMeshlets(simplified, canonical, level);
            for (BuildMeshlet& meshlet : produced) {
                meshlet.errorSphere = groupSphere;
                meshlet.error = groupError;
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

    // GPU 배치로 펼친다. meshlet 마다 정점을 소유하고, 인덱스는 LOD 단계별로 이어 둔다.
    mesh.vertices.clear();
    mesh.indices.clear();
    mesh.meshlets.clear();
    mesh.meshletTriangles.clear();
    mesh.vertexMeshlets.clear();
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
            meshlet.vertexOffset = static_cast<uint32_t>(mesh.vertices.size());
            meshlet.triangleOffset = static_cast<uint32_t>(mesh.meshletTriangles.size());
            meshlet.vertexCount = static_cast<uint32_t>(source.vertices.size());
            meshlet.triangleCount = static_cast<uint32_t>(source.triangles.size() / 3);

            auto meshletIndex = static_cast<uint32_t>(mesh.meshlets.size());
            for (uint32_t vertex : source.vertices) {
                mesh.vertices.push_back(canonical[vertex]);
                mesh.vertexMeshlets.push_back(meshletIndex);
            }
            for (uint8_t local : source.triangles) {
                mesh.meshletTriangles.push_back(local);
                mesh.indices.push_back(meshlet.vertexOffset + local);
            }
            mesh.meshlets.push_back(meshlet);
        }

        lod.indexCount = static_cast<uint32_t>(mesh.indices.size()) - lod.indexOffset;
        lod.meshletCount = static_cast<uint32_t>(mesh.meshlets.size()) - lod.meshletOffset;
        mesh.lods.push_back(lod);
    }
}

} // namespace asset
