#include "gfx/geometry.h"

#include <algorithm>

#include <spdlog/spdlog.h>

#include "core/error.h"
#include "gfx/context.h"
#include "gfx/uploader.h"

namespace gfx {

GeometryStore::GeometryStore(Context& context) : context(context) {}

GeometryStore::~GeometryStore() {
    destroyBuffer(context, meshletVertexBuffer);
    destroyBuffer(context, meshletTriangleBuffer);
    destroyBuffer(context, lodBuffer);
    destroyBuffer(context, meshletBuffer);
    destroyBuffer(context, materialBuffer);
    destroyBuffer(context, meshBuffer);
    destroyBuffer(context, indexBuffer);
    destroyBuffer(context, skinWeightBuffer);
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
        material.flags = source.doubleSided ? MATERIAL_FLAG_DOUBLE_SIDED : 0U;
        if (source.normalTexture != asset::INVALID_TEXTURE && source.normalTexture < model.textures.size()) {
            asset::TextureFormat normalFormat = model.textures[source.normalTexture].format;
            if (normalFormat == asset::TextureFormat::BC4 || normalFormat == asset::TextureFormat::BC5) {
                material.flags |= MATERIAL_FLAG_TWO_CHANNEL_NORMAL;
            }
        }
        materials.push_back(material);
        sourceMaterials.push_back(source);
    }

    for (const asset::Mesh& source : model.meshes) {
        auto vertexBase = static_cast<uint32_t>(vertices.next());
        auto triangleBase = static_cast<uint32_t>(meshletTriangles.next());
        auto meshletVertexBase = static_cast<uint32_t>(meshletVertices.next());

        GpuMesh mesh{};
        mesh.boundingSphere = glm::vec4{source.boundsCenter, source.boundsRadius};
        mesh.indexOffset = static_cast<uint32_t>(indices.next());
        mesh.indexCount = static_cast<uint32_t>(source.indices.size());
        mesh.vertexOffset = static_cast<int32_t>(vertexBase);
        mesh.materialIndex = materialBase + source.materialIndex;
        mesh.meshletOffset = static_cast<uint32_t>(meshlets.size());
        mesh.meshletCount = static_cast<uint32_t>(source.meshlets.size());
        mesh.lodOffset = static_cast<uint32_t>(lods.size());
        mesh.lodCount = static_cast<uint32_t>(source.lods.size());
        maxLods = std::max(maxLods, mesh.lodCount);
        meshes.push_back(mesh);
        meshNames.push_back(source.name);
        meshVertexCounts.push_back(static_cast<uint32_t>(source.vertices.size()));
        if (source.skinWeights.empty()) {
            meshSkinOffsets.push_back(NO_SKIN_WEIGHTS);
        } else {
            meshSkinOffsets.push_back(static_cast<uint32_t>(skinWeights.next()));
            skinWeights.append(source.skinWeights.data(), source.skinWeights.size());
        }

        for (const asset::Meshlet& sourceMeshlet : source.meshlets) {
            GpuMeshlet meshlet{};
            meshlet.boundingSphere = sourceMeshlet.boundingSphere;
            meshlet.cone = sourceMeshlet.cone;
            meshlet.errorSphere = sourceMeshlet.errorSphere;
            meshlet.parentSphere = sourceMeshlet.parentSphere;
            meshlet.error = sourceMeshlet.error;
            meshlet.parentError = sourceMeshlet.parentError;
            meshlet.indexOffset = mesh.indexOffset + sourceMeshlet.indexOffset;
            meshlet.vertexOffset = meshletVertexBase + sourceMeshlet.vertexOffset;
            meshlet.triangleOffset = triangleBase + sourceMeshlet.triangleOffset;
            meshlet.vertexCount = sourceMeshlet.vertexCount;
            meshlet.triangleCount = sourceMeshlet.triangleCount;
            meshlet.level = sourceMeshlet.level;
            meshlets.push_back(meshlet);
        }
        for (const asset::MeshLod& sourceLod : source.lods) {
            GpuMeshLod lod{};
            lod.indexOffset = mesh.indexOffset + sourceLod.indexOffset;
            lod.indexCount = sourceLod.indexCount;
            lod.meshletOffset = mesh.meshletOffset + sourceLod.meshletOffset;
            lod.meshletCount = sourceLod.meshletCount;
            lods.push_back(lod);
        }
        // 지역 인덱스는 바이트 그대로 잇고, 목록의 값은 전역 정점 번호로 바꿔 둔다. 셰이더가 메쉬 오프셋을
        // 더하지 않아도 된다.
        meshletTriangles.append(source.meshletTriangles.data(), source.meshletTriangles.size());
        std::vector<uint32_t> globalVertices(source.meshletVertices.size());
        for (size_t i = 0; i < globalVertices.size(); ++i) {
            globalVertices[i] = vertexBase + source.meshletVertices[i];
        }
        meshletVertices.append(globalVertices.data(), globalVertices.size());

        vertices.append(source.vertices.data(), source.vertices.size());
        indices.append(source.indices.data(), source.indices.size());
    }
    return firstMesh;
}

VkDeviceSize GeometryStore::estimateModelBytes(const asset::Model& model) {
    VkDeviceSize bytes = model.materials.size() * sizeof(GpuMaterial);
    for (const asset::Mesh& mesh : model.meshes) {
        bytes += sizeof(GpuMesh) + mesh.lods.size() * sizeof(GpuMeshLod);
        bytes += mesh.vertices.size() * sizeof(asset::Vertex);
        bytes += mesh.skinWeights.size() * sizeof(asset::SkinWeight);
        bytes += mesh.indices.size() * sizeof(uint32_t);
        bytes += mesh.meshlets.size() * sizeof(GpuMeshlet);
        bytes += mesh.meshletTriangles.size();
        bytes += mesh.meshletVertices.size() * sizeof(uint32_t);
    }
    return bytes;
}

VkDeviceSize GeometryStore::residentBytes() const {
    return vertexBuffer.size + skinWeightBuffer.size + indexBuffer.size + meshletTriangleBuffer.size +
           meshletVertexBuffer.size;
}

VkDeviceSize GeometryStore::pendingBytes() const {
    return vertices.pending.size() * sizeof(asset::Vertex) + skinWeights.pending.size() * sizeof(asset::SkinWeight) +
           indices.pending.size() * sizeof(uint32_t) + meshletTriangles.pending.size() +
           meshletVertices.pending.size() * sizeof(uint32_t);
}

template <typename T>
void GeometryStore::growAndUpload(Uploader& uploader,
                                  Buffer& buffer,
                                  GrowingArray<T>& array,
                                  VkBufferUsageFlags usage,
                                  const char* name,
                                  std::vector<Buffer>& retired) {
    // 주소는 늘 유효해야 하므로 비어 있어도 한 칸은 둔다. 바이트 배열을 uint 로 읽는 셰이더가 끝을
    // 넘지 않도록 4 바이트 배수로 잡는다.
    VkDeviceSize newSize = static_cast<VkDeviceSize>(std::max<size_t>(array.total, 1)) * sizeof(T);
    newSize = (newSize + 3) / 4 * 4;
    VkDeviceSize uploadedBytes = static_cast<VkDeviceSize>(array.uploaded) * sizeof(T);
    if (buffer.handle == VK_NULL_HANDLE || buffer.size < newSize) {
        // ponytail: 꼭 필요한 크기로만 잡아 여유를 두지 않는다. 모델을 더할 때마다 전체를 한 번 옮기고
        // 그 순간 GPU 에 옛 버퍼와 새 버퍼가 함께 있다. 자주 더한다면 배수로 키우는 편이 낫다.
        Buffer replacement = createBuffer(context,
                                          newSize,
                                          usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                          MemoryLocation::DEVICE,
                                          name);
        if (buffer.handle != VK_NULL_HANDLE) {
            uploader.copyBuffer(buffer, replacement, std::min(uploadedBytes, buffer.size));
            retired.push_back(buffer);
        }
        buffer = replacement;
    }
    if (!array.pending.empty()) {
        uploader.uploadBuffer(buffer, uploadedBytes, array.pending.data(), array.pending.size() * sizeof(T));
    }
}

void GeometryStore::build() {
    if (meshes.empty()) {
        core::fatal("장면에 그릴 메쉬가 없습니다");
    }
    bool nothingNew = vertices.pending.empty() && skinWeights.pending.empty() && indices.pending.empty() &&
                      meshletTriangles.pending.empty() && meshletVertices.pending.empty() &&
                      meshes.size() == uploadedMeshCount && materials.size() == uploadedMaterialCount;
    if (nothingNew && vertexBuffer.handle != VK_NULL_HANDLE) {
        return;
    }

    // 하위 가속 구조는 정점과 인덱스 버퍼를 그대로 입력으로 읽는다. 용도 비트가 없으면 규격 위반이라
    // 드라이버가 어떤 구조를 세울지 정해져 있지 않다. 광선 확장이 없는 장치에는 붙일 수 없다.
    VkBufferUsageFlags accelerationInput = 0;
    if (context.caps.rayTracingPipeline || context.caps.rayQuery) {
        accelerationInput = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    }

    // 큰 배열은 GPU 안에서 옮겨 키우고 꼬리만 올린다. 옛 버퍼는 복사가 끝난 뒤에 지운다.
    Uploader uploader(context);
    std::vector<Buffer> retired;
    growAndUpload(
        uploader, vertexBuffer, vertices, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | accelerationInput, "정점", retired);
    growAndUpload(uploader, skinWeightBuffer, skinWeights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, "스킨 가중치", retired);
    growAndUpload(
        uploader, indexBuffer, indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | accelerationInput, "인덱스", retired);
    growAndUpload(uploader,
                  meshletTriangleBuffer,
                  meshletTriangles,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  "meshlet 삼각형",
                  retired);
    growAndUpload(uploader,
                  meshletVertexBuffer,
                  meshletVertices,
                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                  "meshlet 정점 목록",
                  retired);

    // 표는 작고 렌더러가 CPU 에서도 읽으므로 통째로 다시 올린다. 호출 전에 장치가 놀고 있어야 한다.
    auto rebuildTable = [&](Buffer& buffer, const void* data, size_t bytes, const char* name) {
        destroyBuffer(context, buffer);
        buffer = createBuffer(context,
                              bytes,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                              MemoryLocation::DEVICE,
                              name);
        uploader.uploadBuffer(buffer, 0, data, bytes);
    };
    rebuildTable(meshBuffer, meshes.data(), meshes.size() * sizeof(GpuMesh), "메쉬");
    rebuildTable(materialBuffer, materials.data(), materials.size() * sizeof(GpuMaterial), "재질");
    rebuildTable(meshletBuffer, meshlets.data(), meshlets.size() * sizeof(GpuMeshlet), "meshlet");
    rebuildTable(lodBuffer, lods.data(), lods.size() * sizeof(GpuMeshLod), "LOD");
    uploader.flush();

    for (Buffer& buffer : retired) {
        destroyBuffer(context, buffer);
    }
    // 올린 꼬리는 버린다. 다음 build 는 GPU 의 옛 버퍼에서 옮기므로 CPU 사본이 필요 없다.
    auto settle = [](auto& array) {
        array.uploaded = array.total;
        array.pending.clear();
        array.pending.shrink_to_fit();
    };
    settle(vertices);
    settle(skinWeights);
    settle(indices);
    settle(meshletTriangles);
    settle(meshletVertices);
    uploadedMeshCount = meshes.size();
    uploadedMaterialCount = materials.size();

    spdlog::info("지오메트리 업로드: 정점 {}, 인덱스 {}, 메쉬 {}, 재질 {}, meshlet {}",
                 vertices.total,
                 indices.total,
                 meshes.size(),
                 materials.size(),
                 meshlets.size());
    spdlog::info("LOD 단계 최대 {}, 총 LOD 항목 {}", maxLods, lods.size());
}

} // namespace gfx
