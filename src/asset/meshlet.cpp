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

    size_t maxMeshlets = meshopt_buildMeshletsBound(indexCount, MAX_MESHLET_VERTICES, MAX_MESHLET_TRIANGLES);
    std::vector<meshopt_Meshlet> rawMeshlets(maxMeshlets);
    std::vector<uint32_t> meshletVertices(maxMeshlets * MAX_MESHLET_VERTICES);
    std::vector<uint8_t> meshletTriangles(maxMeshlets * MAX_MESHLET_TRIANGLES * 3);

    size_t meshletCount = meshopt_buildMeshlets(rawMeshlets.data(),
                                                meshletVertices.data(),
                                                meshletTriangles.data(),
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
    rawMeshlets.resize(meshletCount);

    // meshlet 마다 정점을 따로 소유하도록 정점 버퍼를 다시 만든다. 경계 정점이 조금 중복되지만,
    // mesh shader 경로가 지역 인덱스를 그대로 쓸 수 있고 고전 경로에서도 flat 보간으로 meshlet 을
    // 프래그먼트까지 내릴 수 있다. gl_PrimitiveID 는 Geometry capability 를 요구해 쓰지 않는다.
    std::vector<Vertex> rebuiltVertices;
    std::vector<uint32_t> rebuiltIndices;
    rebuiltVertices.reserve(vertexCount);
    rebuiltIndices.reserve(indexCount);

    mesh.meshlets.clear();
    mesh.meshlets.reserve(meshletCount);
    mesh.meshletTriangles.clear();
    mesh.vertexMeshlets.clear();
    mesh.vertexMeshlets.reserve(vertexCount);

    for (size_t i = 0; i < meshletCount; ++i) {
        const meshopt_Meshlet& raw = rawMeshlets[i];
        meshopt_Bounds bounds = meshopt_computeMeshletBounds(&meshletVertices[raw.vertex_offset],
                                                             &meshletTriangles[raw.triangle_offset],
                                                             raw.triangle_count,
                                                             positions,
                                                             vertexCount,
                                                             sizeof(Vertex));

        Meshlet meshlet;
        meshlet.boundingSphere = glm::vec4{bounds.center[0], bounds.center[1], bounds.center[2], bounds.radius};
        meshlet.cone = glm::vec4{bounds.cone_axis[0], bounds.cone_axis[1], bounds.cone_axis[2], bounds.cone_cutoff};
        meshlet.vertexOffset = static_cast<uint32_t>(rebuiltVertices.size());
        meshlet.triangleOffset = static_cast<uint32_t>(mesh.meshletTriangles.size());
        meshlet.vertexCount = raw.vertex_count;
        meshlet.triangleCount = raw.triangle_count;

        for (uint32_t v = 0; v < raw.vertex_count; ++v) {
            rebuiltVertices.push_back(mesh.vertices[meshletVertices[raw.vertex_offset + v]]);
            mesh.vertexMeshlets.push_back(static_cast<uint32_t>(i));
        }
        for (uint32_t index = 0; index < raw.triangle_count * 3; ++index) {
            uint8_t local = meshletTriangles[raw.triangle_offset + index];
            mesh.meshletTriangles.push_back(local);
            rebuiltIndices.push_back(meshlet.vertexOffset + local);
        }

        mesh.meshlets.push_back(meshlet);
    }

    mesh.vertices = std::move(rebuiltVertices);
    mesh.indices = std::move(rebuiltIndices);
}

} // namespace asset
