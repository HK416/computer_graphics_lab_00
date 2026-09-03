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

size_t GeometryStore::RangeAllocator::allocate(size_t count) {
    if (count == 0) {
        return total;
    }
    // ponytail: first-fit. 모델이 수십 개 수준이라 빈 구간도 몇 개뿐이다.
    for (size_t i = 0; i < freeRanges.size(); ++i) {
        Range& range = freeRanges[i];
        if (range.count < count) {
            continue;
        }
        size_t offset = range.offset;
        range.offset += count;
        range.count -= count;
        if (range.count == 0) {
            freeRanges.erase(freeRanges.begin() + static_cast<std::ptrdiff_t>(i));
        }
        return offset;
    }
    size_t offset = total;
    total += count;
    return offset;
}

void GeometryStore::RangeAllocator::release(size_t offset, size_t count) {
    if (count == 0) {
        return;
    }
    // 오프셋 순서를 지키며 넣고 양옆과 붙으면 합친다.
    auto position =
        std::lower_bound(freeRanges.begin(), freeRanges.end(), offset, [](const Range& range, size_t value) {
            return range.offset < value;
        });
    position = freeRanges.insert(position, Range{offset, count});
    if (position + 1 != freeRanges.end() && position->offset + position->count == (position + 1)->offset) {
        position->count += (position + 1)->count;
        freeRanges.erase(position + 1);
    }
    if (position != freeRanges.begin()) {
        auto previous = position - 1;
        if (previous->offset + previous->count == position->offset) {
            previous->count += position->count;
            freeRanges.erase(position);
        }
    }
    // 꼬리가 비었으면 그만큼 줄인다. 버퍼가 실제로 작아지는 유일한 길이다.
    if (!freeRanges.empty() && freeRanges.back().offset + freeRanges.back().count == total) {
        total = freeRanges.back().offset;
        freeRanges.pop_back();
    }
}

template <typename T> void GeometryStore::GrowingArray<T>::release(size_t offset, size_t count) {
    // 올리기도 전에 해제된 조각은 그냥 버린다.
    std::erase_if(pending, [offset, count](const Chunk& chunk) {
        return chunk.offset >= offset && chunk.offset + chunk.data.size() <= offset + count;
    });
    ranges.release(offset, count);
}

template <typename T> size_t GeometryStore::GrowingArray<T>::pendingCount() const {
    size_t count = 0;
    for (const Chunk& chunk : pending) {
        count += chunk.data.size();
    }
    return count;
}

template <typename T>
uint32_t GeometryStore::allocateTable(std::vector<T>& table, RangeAllocator& ranges, size_t count) {
    size_t offset = ranges.allocate(count);
    if (table.size() < offset + count) {
        table.resize(offset + count);
    }
    return static_cast<uint32_t>(offset);
}

template <typename T>
void GeometryStore::releaseTable(std::vector<T>& table, RangeAllocator& ranges, size_t offset, size_t count) {
    for (size_t i = offset; i < offset + count && i < table.size(); ++i) {
        table[i] = T{};
    }
    ranges.release(offset, count);
    if (table.size() > ranges.total) {
        table.resize(ranges.total);
    }
}

GeometryStore::ModelRange GeometryStore::addModel(const asset::Model& model,
                                                  const std::vector<uint32_t>& textureSlots) {
    ModelRange range;
    range.materialCount = static_cast<uint32_t>(model.materials.size());
    range.materialBase = allocateTable(materials, materialRanges, range.materialCount);
    sourceMaterials.resize(materials.size());
    range.meshCount = static_cast<uint32_t>(model.meshes.size());
    range.meshBase = allocateTable(meshes, meshRanges, range.meshCount);
    meshNames.resize(meshes.size());
    meshVertexCounts.resize(meshes.size());
    meshSkinOffsets.resize(meshes.size(), NO_SKIN_WEIGHTS);
    meshTriangleRanges.resize(meshes.size());
    meshMeshletVertexRanges.resize(meshes.size());

    for (uint32_t i = 0; i < range.materialCount; ++i) {
        const asset::Material& source = model.materials[i];
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
        materials[range.materialBase + i] = material;
        sourceMaterials[range.materialBase + i] = source;
    }

    for (uint32_t i = 0; i < range.meshCount; ++i) {
        const asset::Mesh& source = model.meshes[i];
        uint32_t meshIndex = range.meshBase + i;
        auto vertexBase = static_cast<uint32_t>(vertices.allocate(source.vertices.size()));
        auto triangleBase = static_cast<uint32_t>(meshletTriangles.allocate(source.meshletTriangles.size()));
        auto meshletVertexBase = static_cast<uint32_t>(meshletVertices.allocate(source.meshletVertices.size()));

        GpuMesh mesh{};
        mesh.boundingSphere = glm::vec4{source.boundsCenter, source.boundsRadius};
        mesh.indexOffset = static_cast<uint32_t>(indices.allocate(source.indices.size()));
        mesh.indexCount = static_cast<uint32_t>(source.indices.size());
        mesh.vertexOffset = static_cast<int32_t>(vertexBase);
        mesh.materialIndex = range.materialBase + source.materialIndex;
        mesh.meshletCount = static_cast<uint32_t>(source.meshlets.size());
        mesh.meshletOffset = allocateTable(meshlets, meshletRanges, mesh.meshletCount);
        mesh.lodCount = static_cast<uint32_t>(source.lods.size());
        mesh.lodOffset = allocateTable(lods, lodRanges, mesh.lodCount);
        maxLods = std::max(maxLods, mesh.lodCount);
        meshes[meshIndex] = mesh;
        meshNames[meshIndex] = source.name;
        meshVertexCounts[meshIndex] = static_cast<uint32_t>(source.vertices.size());
        meshTriangleRanges[meshIndex] = Range{triangleBase, source.meshletTriangles.size()};
        meshMeshletVertexRanges[meshIndex] = Range{meshletVertexBase, source.meshletVertices.size()};
        if (source.skinWeights.empty()) {
            meshSkinOffsets[meshIndex] = NO_SKIN_WEIGHTS;
        } else {
            size_t skinBase = skinWeights.allocate(source.skinWeights.size());
            meshSkinOffsets[meshIndex] = static_cast<uint32_t>(skinBase);
            skinWeights.write(skinBase, source.skinWeights.data(), source.skinWeights.size());
        }

        for (uint32_t m = 0; m < mesh.meshletCount; ++m) {
            const asset::Meshlet& sourceMeshlet = source.meshlets[m];
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
            meshlets[mesh.meshletOffset + m] = meshlet;
        }
        for (uint32_t l = 0; l < mesh.lodCount; ++l) {
            const asset::MeshLod& sourceLod = source.lods[l];
            GpuMeshLod lod{};
            lod.indexOffset = mesh.indexOffset + sourceLod.indexOffset;
            lod.indexCount = sourceLod.indexCount;
            lod.meshletOffset = mesh.meshletOffset + sourceLod.meshletOffset;
            lod.meshletCount = sourceLod.meshletCount;
            lods[mesh.lodOffset + l] = lod;
        }
        // 지역 인덱스는 바이트 그대로 잇고, 목록의 값은 전역 정점 번호로 바꿔 둔다. 셰이더가 메쉬 오프셋을
        // 더하지 않아도 된다. 구간을 옮기는 일이 없으므로 이 번호는 해제 뒤에도 그대로다.
        meshletTriangles.write(triangleBase, source.meshletTriangles.data(), source.meshletTriangles.size());
        std::vector<uint32_t> globalVertices(source.meshletVertices.size());
        for (size_t v = 0; v < globalVertices.size(); ++v) {
            globalVertices[v] = vertexBase + source.meshletVertices[v];
        }
        meshletVertices.write(meshletVertexBase, globalVertices.data(), globalVertices.size());

        vertices.write(vertexBase, source.vertices.data(), source.vertices.size());
        indices.write(mesh.indexOffset, source.indices.data(), source.indices.size());
    }
    tablesDirty = true;
    return range;
}

void GeometryStore::removeModel(const ModelRange& range) {
    for (uint32_t i = 0; i < range.meshCount; ++i) {
        uint32_t meshIndex = range.meshBase + i;
        if (meshIndex >= meshes.size()) {
            break;
        }
        const GpuMesh& mesh = meshes[meshIndex];
        vertices.release(static_cast<size_t>(mesh.vertexOffset), meshVertexCounts[meshIndex]);
        indices.release(mesh.indexOffset, mesh.indexCount);
        if (meshSkinOffsets[meshIndex] != NO_SKIN_WEIGHTS) {
            skinWeights.release(meshSkinOffsets[meshIndex], meshVertexCounts[meshIndex]);
        }
        meshletTriangles.release(meshTriangleRanges[meshIndex].offset, meshTriangleRanges[meshIndex].count);
        meshletVertices.release(meshMeshletVertexRanges[meshIndex].offset, meshMeshletVertexRanges[meshIndex].count);
        releaseTable(meshlets, meshletRanges, mesh.meshletOffset, mesh.meshletCount);
        releaseTable(lods, lodRanges, mesh.lodOffset, mesh.lodCount);

        meshes[meshIndex] = GpuMesh{};
        meshNames[meshIndex] = "(해제됨)";
        meshVertexCounts[meshIndex] = 0;
        meshSkinOffsets[meshIndex] = NO_SKIN_WEIGHTS;
        meshTriangleRanges[meshIndex] = Range{};
        meshMeshletVertexRanges[meshIndex] = Range{};
    }
    releaseTable(meshes, meshRanges, range.meshBase, range.meshCount);
    meshNames.resize(meshes.size());
    meshVertexCounts.resize(meshes.size());
    meshSkinOffsets.resize(meshes.size());
    meshTriangleRanges.resize(meshes.size());
    meshMeshletVertexRanges.resize(meshes.size());
    releaseTable(materials, materialRanges, range.materialBase, range.materialCount);
    sourceMaterials.resize(materials.size());

    maxLods = 1;
    for (const GpuMesh& mesh : meshes) {
        maxLods = std::max(maxLods, mesh.lodCount);
    }
    tablesDirty = true;
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
    return vertices.pendingCount() * sizeof(asset::Vertex) + skinWeights.pendingCount() * sizeof(asset::SkinWeight) +
           indices.pendingCount() * sizeof(uint32_t) + meshletTriangles.pendingCount() +
           meshletVertices.pendingCount() * sizeof(uint32_t);
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
    VkDeviceSize newSize = static_cast<VkDeviceSize>(std::max<size_t>(array.total(), 1)) * sizeof(T);
    newSize = (newSize + 3) / 4 * 4;
    // 해제로 꼬리가 줄었으면 유효한 것도 거기까지다.
    array.uploaded = std::min(array.uploaded, array.total());
    VkDeviceSize uploadedBytes = static_cast<VkDeviceSize>(array.uploaded) * sizeof(T);
    // 모자라면 키우고, 절반 이하로 줄었으면 작은 버퍼로 옮겨 메모리를 실제로 돌려준다.
    bool shrink = buffer.handle != VK_NULL_HANDLE && newSize * 2 < buffer.size;
    if (buffer.handle == VK_NULL_HANDLE || buffer.size < newSize || shrink) {
        // ponytail: 꼭 필요한 크기로만 잡아 여유를 두지 않는다. 모델을 더할 때마다 전체를 한 번 옮기고
        // 그 순간 GPU 에 옛 버퍼와 새 버퍼가 함께 있다. 자주 더한다면 배수로 키우는 편이 낫다.
        Buffer replacement = createBuffer(context,
                                          newSize,
                                          usage | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                          MemoryLocation::DEVICE,
                                          name);
        if (buffer.handle != VK_NULL_HANDLE) {
            uploader.copyBuffer(buffer, replacement, std::min({uploadedBytes, buffer.size, newSize}));
            retired.push_back(buffer);
        }
        buffer = replacement;
    }
    for (const auto& chunk : array.pending) {
        uploader.uploadBuffer(buffer,
                              static_cast<VkDeviceSize>(chunk.offset) * sizeof(T),
                              chunk.data.data(),
                              chunk.data.size() * sizeof(T));
    }
}

void GeometryStore::build() {
    // 메쉬가 하나도 없어도 돌아간다. 기본 장면은 비어 있고, 모델은 편집기에서 올린다.
    bool nothingNew = vertices.pending.empty() && skinWeights.pending.empty() && indices.pending.empty() &&
                      meshletTriangles.pending.empty() && meshletVertices.pending.empty() && !tablesDirty;
    if (nothingNew && vertexBuffer.handle != VK_NULL_HANDLE) {
        return;
    }

    // 하위 가속 구조는 정점과 인덱스 버퍼를 그대로 입력으로 읽는다. 용도 비트가 없으면 규격 위반이라
    // 드라이버가 어떤 구조를 세울지 정해져 있지 않다. 광선 확장이 없는 장치에는 붙일 수 없다.
    VkBufferUsageFlags accelerationInput = 0;
    if (context.caps.rayTracingPipeline || context.caps.rayQuery) {
        accelerationInput = VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    }

    // 큰 배열은 GPU 안에서 옮겨 키우고 조각만 제자리에 올린다. 옛 버퍼는 복사가 끝난 뒤에 지운다.
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
        // 표가 비어도 주소는 유효해야 하므로 한 칸은 잡는다.
        buffer = createBuffer(context,
                              std::max<size_t>(bytes, 16),
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
    // 올린 조각은 버린다. 다음 build 는 GPU 의 옛 버퍼에서 옮기므로 CPU 사본이 필요 없다.
    auto settle = [](auto& array) {
        array.uploaded = array.total();
        array.pending.clear();
        array.pending.shrink_to_fit();
    };
    settle(vertices);
    settle(skinWeights);
    settle(indices);
    settle(meshletTriangles);
    settle(meshletVertices);
    tablesDirty = false;

    constexpr double MB = 1024.0 * 1024.0;
    spdlog::info("지오메트리 업로드: 정점 {}, 인덱스 {}, 메쉬 {}, 재질 {}, meshlet {}, GPU {:.1f} MB",
                 vertices.total(),
                 indices.total(),
                 meshes.size(),
                 materials.size(),
                 meshlets.size(),
                 static_cast<double>(residentBytes()) / MB);
    spdlog::info("LOD 단계 최대 {}, 총 LOD 항목 {}", maxLods, lods.size());
}

} // namespace gfx
