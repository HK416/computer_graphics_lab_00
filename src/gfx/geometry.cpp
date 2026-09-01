#include "gfx/geometry.h"

#include <spdlog/spdlog.h>

#include "core/error.h"
#include "gfx/context.h"
#include "gfx/uploader.h"

namespace gfx {

GeometryStore::GeometryStore(Context& context) : context(context) {}

GeometryStore::~GeometryStore() {
    destroyBuffer(context, vertexMeshletBuffer);
    destroyBuffer(context, meshletTriangleBuffer);
    destroyBuffer(context, meshletBuffer);
    destroyBuffer(context, materialBuffer);
    destroyBuffer(context, meshBuffer);
    destroyBuffer(context, indexBuffer);
    destroyBuffer(context, vertexBuffer);
}

namespace {
uint32_t resolveTexture(uint32_t textureIndex, const std::vector<uint32_t>& textureSlots) {
    if (textureIndex == asset::INVALID_TEXTURE || textureIndex >= textureSlots.size()) {
        return asset::INVALID_TEXTURE;
    }
    return textureSlots[textureIndex];
}
} // namespace

uint32_t GeometryStore::addModel(const asset::Model& model, const std::vector<uint32_t>& textureSlots) {
    auto firstMesh = static_cast<uint32_t>(meshes.size());
    auto materialBase = static_cast<uint32_t>(materials.size());

    for (const asset::Material& source : model.materials) {
        GpuMaterial material{};
        material.baseColorFactor = source.baseColorFactor;
        material.emissiveAndCutoff = glm::vec4{source.emissiveFactor, source.alphaCutoff};
        material.metallicFactor = source.metallicFactor;
        material.roughnessFactor = source.roughnessFactor;
        material.normalScale = source.normalScale;
        material.occlusionStrength = source.occlusionStrength;
        material.baseColorTexture = resolveTexture(source.baseColorTexture, textureSlots);
        material.metallicRoughnessTexture = resolveTexture(source.metallicRoughnessTexture, textureSlots);
        material.normalTexture = resolveTexture(source.normalTexture, textureSlots);
        material.occlusionTexture = resolveTexture(source.occlusionTexture, textureSlots);
        material.emissiveTexture = resolveTexture(source.emissiveTexture, textureSlots);
        material.alphaMode = static_cast<uint32_t>(source.alphaMode);
        material.flags = source.doubleSided ? 1U : 0U;
        materials.push_back(material);
        sourceMaterials.push_back(source);
    }

    for (const asset::Mesh& source : model.meshes) {
        auto vertexBase = static_cast<uint32_t>(vertices.size());
        auto triangleBase = static_cast<uint32_t>(meshletTriangles.size());

        GpuMesh mesh{};
        mesh.boundingSphere = glm::vec4{source.boundsCenter, source.boundsRadius};
        mesh.indexOffset = static_cast<uint32_t>(indices.size());
        mesh.indexCount = static_cast<uint32_t>(source.indices.size());
        mesh.vertexOffset = static_cast<int32_t>(vertexBase);
        mesh.materialIndex = materialBase + source.materialIndex;
        mesh.meshletOffset = static_cast<uint32_t>(meshlets.size());
        mesh.meshletCount = static_cast<uint32_t>(source.meshlets.size());
        meshes.push_back(mesh);
        meshNames.push_back(source.name);

        for (const asset::Meshlet& sourceMeshlet : source.meshlets) {
            GpuMeshlet meshlet{};
            meshlet.boundingSphere = sourceMeshlet.boundingSphere;
            meshlet.cone = sourceMeshlet.cone;
            meshlet.vertexOffset = vertexBase + sourceMeshlet.vertexOffset;
            meshlet.triangleOffset = triangleBase + sourceMeshlet.triangleOffset;
            meshlet.vertexCount = sourceMeshlet.vertexCount;
            meshlet.triangleCount = sourceMeshlet.triangleCount;
            meshlets.push_back(meshlet);
        }
        for (uint8_t local : source.meshletTriangles) {
            meshletTriangles.push_back(local);
        }
        for (uint32_t meshletIndex : source.vertexMeshlets) {
            vertexMeshlets.push_back(mesh.meshletOffset + meshletIndex);
        }

        vertices.insert(vertices.end(), source.vertices.begin(), source.vertices.end());
        indices.insert(indices.end(), source.indices.begin(), source.indices.end());
    }
    return firstMesh;
}

void GeometryStore::build() {
    if (meshes.empty()) {
        core::fatal("장면에 그릴 메쉬가 없습니다");
    }

    vertexBuffer = createBuffer(context,
                                vertices.size() * sizeof(asset::Vertex),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                MemoryLocation::DEVICE,
                                "정점");
    indexBuffer = createBuffer(context,
                               indices.size() * sizeof(uint32_t),
                               VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               MemoryLocation::DEVICE,
                               "인덱스");
    meshBuffer = createBuffer(context,
                              meshes.size() * sizeof(GpuMesh),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              MemoryLocation::DEVICE,
                              "메쉬");
    materialBuffer = createBuffer(context,
                                  materials.size() * sizeof(GpuMaterial),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  MemoryLocation::DEVICE,
                                  "재질");

    meshletBuffer = createBuffer(context,
                                 meshlets.size() * sizeof(GpuMeshlet),
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 MemoryLocation::DEVICE,
                                 "meshlet");
    meshletTriangleBuffer = createBuffer(context,
                                         meshletTriangles.size() * sizeof(uint32_t),
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                         MemoryLocation::DEVICE,
                                         "meshlet 삼각형");
    vertexMeshletBuffer = createBuffer(context,
                                       vertexMeshlets.size() * sizeof(uint32_t),
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       MemoryLocation::DEVICE,
                                       "정점별 meshlet");

    Uploader uploader(context);
    uploader.uploadBuffer(vertexBuffer, 0, vertices.data(), vertexBuffer.size);
    uploader.uploadBuffer(indexBuffer, 0, indices.data(), indexBuffer.size);
    uploader.uploadBuffer(meshBuffer, 0, meshes.data(), meshBuffer.size);
    uploader.uploadBuffer(materialBuffer, 0, materials.data(), materialBuffer.size);
    uploader.uploadBuffer(meshletBuffer, 0, meshlets.data(), meshletBuffer.size);
    uploader.uploadBuffer(meshletTriangleBuffer, 0, meshletTriangles.data(), meshletTriangleBuffer.size);
    uploader.uploadBuffer(vertexMeshletBuffer, 0, vertexMeshlets.data(), vertexMeshletBuffer.size);
    uploader.flush();

    spdlog::info("지오메트리 업로드: 정점 {}, 인덱스 {}, 메쉬 {}, 재질 {}, meshlet {}",
                 vertices.size(),
                 indices.size(),
                 meshes.size(),
                 materials.size(),
                 meshlets.size());

    // CPU 사본은 이후 meshlet 분할 단계에서 다시 필요하지만, 지금은 GPU 버퍼만 유지한다.
    vertices.clear();
    vertices.shrink_to_fit();
    indices.clear();
    indices.shrink_to_fit();
    meshletTriangles.clear();
    meshletTriangles.shrink_to_fit();
    vertexMeshlets.clear();
    vertexMeshlets.shrink_to_fit();
}

} // namespace gfx
