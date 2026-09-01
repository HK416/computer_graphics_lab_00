#include <vector>

#include <meshoptimizer.h>
#include <spdlog/spdlog.h>

#include "asset/model.h"

namespace asset {
namespace {

// NVIDIA 와 AMD 의 mesh shader 권장 한도에 맞춘 값이다.
constexpr size_t MAX_MESHLET_VERTICES = 64;
constexpr size_t MAX_MESHLET_TRIANGLES = 124;
constexpr float CONE_WEIGHT = 0.5F;

} // namespace

void buildMeshlets(Mesh& mesh) {
    if (mesh.indices.empty() || mesh.vertices.empty()) {
        return;
    }

    const float* positions = &mesh.vertices.front().position.x;
    size_t vertexCount = mesh.vertices.size();
    size_t indexCount = mesh.indices.size();

    meshopt_optimizeVertexCache(mesh.indices.data(), mesh.indices.data(), indexCount, vertexCount);
    meshopt_optimizeOverdraw(
        mesh.indices.data(), mesh.indices.data(), indexCount, positions, vertexCount, sizeof(Vertex), 1.05F);
    meshopt_optimizeVertexFetch(
        mesh.vertices.data(), mesh.indices.data(), indexCount, mesh.vertices.data(), vertexCount, sizeof(Vertex));

    size_t maxMeshlets = meshopt_buildMeshletsBound(indexCount, MAX_MESHLET_VERTICES, MAX_MESHLET_TRIANGLES);
    std::vector<meshopt_Meshlet> rawMeshlets(maxMeshlets);
    mesh.meshletVertices.resize(maxMeshlets * MAX_MESHLET_VERTICES);
    mesh.meshletTriangles.resize(maxMeshlets * MAX_MESHLET_TRIANGLES * 3);

    size_t meshletCount = meshopt_buildMeshlets(rawMeshlets.data(),
                                                mesh.meshletVertices.data(),
                                                mesh.meshletTriangles.data(),
                                                mesh.indices.data(),
                                                indexCount,
                                                positions,
                                                vertexCount,
                                                sizeof(Vertex),
                                                MAX_MESHLET_VERTICES,
                                                MAX_MESHLET_TRIANGLES,
                                                CONE_WEIGHT);
    if (meshletCount == 0) {
        return;
    }

    const meshopt_Meshlet& last = rawMeshlets[meshletCount - 1];
    mesh.meshletVertices.resize(last.vertex_offset + last.vertex_count);
    mesh.meshletTriangles.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3U));
    rawMeshlets.resize(meshletCount);

    mesh.meshlets.reserve(meshletCount);
    // 고전 경로도 meshlet 순서로 그리도록 인덱스 버퍼를 다시 만든다. gl_PrimitiveID 가 곧 meshlet 조회 키가 된다.
    std::vector<uint32_t> reordered;
    reordered.reserve(indexCount);
    mesh.triangleMeshlets.clear();
    mesh.triangleMeshlets.reserve(indexCount / 3);

    for (size_t i = 0; i < meshletCount; ++i) {
        const meshopt_Meshlet& raw = rawMeshlets[i];
        meshopt_Bounds bounds = meshopt_computeMeshletBounds(&mesh.meshletVertices[raw.vertex_offset],
                                                             &mesh.meshletTriangles[raw.triangle_offset],
                                                             raw.triangle_count,
                                                             positions,
                                                             vertexCount,
                                                             sizeof(Vertex));

        Meshlet meshlet;
        meshlet.boundingSphere = glm::vec4{bounds.center[0], bounds.center[1], bounds.center[2], bounds.radius};
        meshlet.cone = glm::vec4{bounds.cone_axis[0], bounds.cone_axis[1], bounds.cone_axis[2], bounds.cone_cutoff};
        meshlet.vertexOffset = raw.vertex_offset;
        meshlet.triangleOffset = raw.triangle_offset;
        meshlet.vertexCount = raw.vertex_count;
        meshlet.triangleCount = raw.triangle_count;
        mesh.meshlets.push_back(meshlet);

        for (uint32_t triangle = 0; triangle < raw.triangle_count; ++triangle) {
            for (uint32_t corner = 0; corner < 3; ++corner) {
                uint32_t local = mesh.meshletTriangles[raw.triangle_offset + triangle * 3 + corner];
                reordered.push_back(mesh.meshletVertices[raw.vertex_offset + local]);
            }
            mesh.triangleMeshlets.push_back(static_cast<uint32_t>(i));
        }
    }

    mesh.indices = std::move(reordered);
}

} // namespace asset
