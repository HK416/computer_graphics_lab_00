#include "gfx/raytracing.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <string>

#include <glm/mat4x4.hpp>
#include <spdlog/spdlog.h>

#include "core/error.h"
#include "gfx/bindless.h"
#include "gfx/context.h"
#include "gfx/geometry.h"
#include "gfx/uploader.h"
#include "gfx/vk_check.h"
#include "scene/scene.h"

namespace gfx {
namespace {

struct PathTracePushConstants {
    VkDeviceAddress vertices;
    VkDeviceAddress skinnedVertices;
    VkDeviceAddress indices;
    VkDeviceAddress meshes;
    VkDeviceAddress instances;
    VkDeviceAddress materials;
    VkDeviceAddress lods;
    VkDeviceAddress camera;
    VkDeviceAddress lights;
    uint32_t accumulationImage;
    uint32_t velocityImage;
    uint32_t frameIndex;
    uint32_t sampleCount;
    uint32_t maxBounces;
    uint32_t samplesPerFrame;
    uint32_t flags;
    float radianceClamp;
    float skyIntensity;
    uint32_t debugMode;
    // 안내 버퍼 슬롯을 둘씩 16비트로 묶는다. 이유는 pathtrace_common.glsl 쪽 주석 참조.
    uint32_t guideAlbedoSlots;
    uint32_t guideNormalRoughnessSlots;
    uint32_t guideDepthSlot;
};

constexpr uint32_t PATH_FLAG_NEXT_EVENT = 1;
constexpr uint32_t PATH_FLAG_RUSSIAN_ROULETTE = 2;
constexpr uint32_t PATH_FLAG_WRITE_GUIDES = 4;

uint64_t alignUp(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

// 알파를 보는 재질은 OPAQUE 로 올리면 안 된다. 그래야 적중 셰이더가 불려 컷오프와 반투명을
// 가려낼 수 있다. 불투명 재질은 표시해 두어야 교차 판정이 빨라진다.
VkGeometryFlagsKHR geometryFlagsFor(const asset::Material& material) {
    return material.alphaMode == asset::AlphaMode::SOLID ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
}

} // namespace

RayTracer::RayTracer(Context& context, GeometryStore& geometry, BindlessTextures& bindless)
    : context(context), geometry(geometry), bindless(bindless) {
    loadFunctions();
    createPipeline();
}

RayTracer::~RayTracer() {
    destroyStructure(topLevel);
    for (AccelerationStructure& structure : bottomLevels) {
        destroyStructure(structure);
    }
    for (AccelerationStructure& structure : skinnedBottomLevels) {
        destroyStructure(structure);
    }
    destroyBuffer(context, shaderBindingTable);
    destroyBuffer(context, scratchBuffer);
    destroyBuffer(context, skinnedScratchBuffer);
    for (Buffer& buffer : instanceBuffers) {
        destroyBuffer(context, buffer);
    }
    vkDestroyPipeline(context.device, pipeline, nullptr);
    vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
    vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(context.device, descriptorSetLayout, nullptr);
}

void RayTracer::loadFunctions() {
    auto load = [this](const char* name) { return vkGetDeviceProcAddr(context.device, name); };
    createAccelerationStructure =
        reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(load("vkCreateAccelerationStructureKHR"));
    destroyAccelerationStructure =
        reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(load("vkDestroyAccelerationStructureKHR"));
    getBuildSizes =
        reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(load("vkGetAccelerationStructureBuildSizesKHR"));
    cmdBuildAccelerationStructures =
        reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(load("vkCmdBuildAccelerationStructuresKHR"));
    getStructureAddress = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
        load("vkGetAccelerationStructureDeviceAddressKHR"));
    createRayTracingPipelines =
        reinterpret_cast<PFN_vkCreateRayTracingPipelinesKHR>(load("vkCreateRayTracingPipelinesKHR"));
    getShaderGroupHandles =
        reinterpret_cast<PFN_vkGetRayTracingShaderGroupHandlesKHR>(load("vkGetRayTracingShaderGroupHandlesKHR"));
    cmdTraceRays = reinterpret_cast<PFN_vkCmdTraceRaysKHR>(load("vkCmdTraceRaysKHR"));

    if (createAccelerationStructure == nullptr || cmdTraceRays == nullptr) {
        core::fatal("레이트레이싱 진입점을 찾을 수 없습니다");
    }
}

RayTracer::AccelerationStructure RayTracer::createStructure(VkAccelerationStructureTypeKHR type, VkDeviceSize size) {
    AccelerationStructure structure;
    structure.storage =
        createBuffer(context,
                     size,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     MemoryLocation::DEVICE,
                     "가속 구조");

    VkAccelerationStructureCreateInfoKHR info{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    info.buffer = structure.storage.handle;
    info.size = size;
    info.type = type;
    VK_CHECK(createAccelerationStructure(context.device, &info, nullptr, &structure.handle));

    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    addressInfo.accelerationStructure = structure.handle;
    structure.address = getStructureAddress(context.device, &addressInfo);
    return structure;
}

void RayTracer::destroyStructure(AccelerationStructure& structure) {
    if (structure.handle != VK_NULL_HANDLE) {
        destroyAccelerationStructure(context.device, structure.handle, nullptr);
        structure.handle = VK_NULL_HANDLE;
    }
    destroyBuffer(context, structure.storage);
    structure.address = 0;
}

void RayTracer::reserveScratch(Buffer& buffer, VkDeviceSize size, const char* debugName) {
    if (buffer.size >= size) {
        return;
    }
    destroyBuffer(context, buffer);
    // 구축 스크래치는 장치가 요구하는 정렬을 지켜야 한다.
    buffer = createBuffer(context,
                          size,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                          MemoryLocation::DEVICE,
                          debugName,
                          context.accelerationStructureProperties.minAccelerationStructureScratchOffsetAlignment);
}

void RayTracer::invalidateBottomLevel() {
    for (AccelerationStructure& structure : bottomLevels) {
        destroyStructure(structure);
    }
    bottomLevels.clear();
    // 상위 구조는 하위 구조 주소를 담고 있어 함께 버린다. ready() 가 거짓이 되어 다음 프레임에 다시 세운다.
    destroyStructure(topLevel);
}

bool RayTracer::buildBottomLevel(std::string& reason) {
    invalidateBottomLevel();
    auto buildStart = std::chrono::steady_clock::now();

    // 먼저 크기만 재서 예산에 들어가는지 본다. 넘기면 할당이 시스템 메모리로 넘어가 통과한 뒤 빌드
    // 중에 장치를 잃으므로, 만들기 전에 거른다.
    uint32_t meshCount = geometry.meshCount();
    std::vector<VkAccelerationStructureGeometryKHR> geometries(meshCount);
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos(meshCount);
    std::vector<VkAccelerationStructureBuildSizesInfoKHR> sizes(meshCount);
    std::vector<uint32_t> primitiveCounts(meshCount);
    VkDeviceSize structureBytes = 0;
    VkDeviceSize largestScratch = 0;
    VkDeviceSize totalScratch = 0;
    for (uint32_t index = 0; index < meshCount; ++index) {
        const GpuMesh& mesh = geometry.mesh(index);
        const GpuMeshLod& lod = geometry.lod(mesh.lodOffset);

        VkAccelerationStructureGeometryTrianglesDataKHR triangles{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress =
            geometry.vertexBuffer.address + static_cast<VkDeviceSize>(mesh.vertexOffset) * sizeof(asset::Vertex);
        triangles.vertexStride = sizeof(asset::Vertex);
        // 이 메쉬가 가진 정점 중 가장 큰 번호다. 인덱스 개수와는 무관하다.
        triangles.maxVertex = geometry.meshVertexCount(index) - 1;
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress =
            geometry.indexBuffer.address + static_cast<VkDeviceSize>(lod.indexOffset) * sizeof(uint32_t);

        geometries[index] = VkAccelerationStructureGeometryKHR{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geometries[index].geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometries[index].geometry.triangles = triangles;
        geometries[index].flags = geometryFlagsFor(geometry.material(mesh.materialIndex));

        buildInfos[index] = VkAccelerationStructureBuildGeometryInfoKHR{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        buildInfos[index].type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfos[index].flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        buildInfos[index].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        buildInfos[index].geometryCount = 1;
        buildInfos[index].pGeometries = &geometries[index];

        primitiveCounts[index] = lod.indexCount / 3;
        sizes[index] =
            VkAccelerationStructureBuildSizesInfoKHR{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        getBuildSizes(context.device,
                      VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                      &buildInfos[index],
                      &primitiveCounts[index],
                      &sizes[index]);
        structureBytes += sizes[index].accelerationStructureSize;
        VkDeviceSize scratch = alignUp(sizes[index].buildScratchSize, 256);
        largestScratch = std::max(largestScratch, scratch);
        totalScratch += scratch;
    }

    // 한 제출에 넣을 스크래치 합의 한도. 전부 합쳐 이 아래면 한 번에, 아니면 이만큼씩 끊어 제출한다.
    // 메쉬 하나가 이보다 크면 그것만 따로 제출한다.
    constexpr VkDeviceSize SCRATCH_BATCH_BYTES = 256ULL * 1024 * 1024;
    VkDeviceSize scratchBytes = std::max(std::min(totalScratch, SCRATCH_BATCH_BYTES), largestScratch);
    Context::MemoryBudget budget = context.deviceMemoryBudget();
    VkDeviceSize available = budget.budget > budget.usage ? budget.budget - budget.usage : 0;
    constexpr double MB = 1024.0 * 1024.0;
    if (structureBytes + scratchBytes > available) {
        reason = std::format("하위 가속 구조 {:.0f} MB 와 스크래치 {:.0f} MB 가 남은 GPU 예산 {:.0f} MB 를 넘습니다",
                             static_cast<double>(structureBytes) / MB,
                             static_cast<double>(scratchBytes) / MB,
                             static_cast<double>(available) / MB);
        return false;
    }

    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = context.queueFamilies.graphics;
    VkCommandPool pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateCommandPool(context.device, &poolInfo, nullptr, &pool));
    VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocateInfo.commandPool = pool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(context.device, &allocateInfo, &commandBuffer));

    bottomLevels.resize(meshCount);
    reserveScratch(scratchBuffer, std::max<VkDeviceSize>(scratchBytes, 256), "가속 구조 스크래치");

    // 스크래치 한도만큼 묶어 제출하고 기다린다. 묶음마다 스크래치를 처음부터 다시 쓴다.
    uint32_t submissions = 0;
    uint32_t first = 0;
    while (first < meshCount) {
        VkDeviceSize used = 0;
        uint32_t last = first;
        std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
        std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> rangePointers;
        while (last < meshCount) {
            VkDeviceSize scratch = alignUp(sizes[last].buildScratchSize, 256);
            if (last > first && used + scratch > scratchBuffer.size) {
                break;
            }
            bottomLevels[last] =
                createStructure(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, sizes[last].accelerationStructureSize);
            buildInfos[last].dstAccelerationStructure = bottomLevels[last].handle;
            buildInfos[last].scratchData.deviceAddress = scratchBuffer.address + used;
            used += scratch;
            ++last;
        }
        ranges.resize(last - first);
        rangePointers.resize(last - first);
        for (uint32_t index = first; index < last; ++index) {
            ranges[index - first] = {};
            ranges[index - first].primitiveCount = primitiveCounts[index];
            rangePointers[index - first] = &ranges[index - first];
        }

        VK_CHECK(vkResetCommandPool(context.device, pool, 0));
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
        cmdBuildAccelerationStructures(commandBuffer, last - first, &buildInfos[first], rangePointers.data());
        VK_CHECK(vkEndCommandBuffer(commandBuffer));

        VkCommandBufferSubmitInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        commandInfo.commandBuffer = commandBuffer;
        VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &commandInfo;
        VK_CHECK(vkQueueSubmit2(context.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(context.graphicsQueue));
        ++submissions;
        first = last;
    }
    vkDestroyCommandPool(context.device, pool, nullptr);

    double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - buildStart).count();
    spdlog::info("하위 가속 구조 {}개 생성: {:.0f} MB, 스크래치 {:.0f} MB, 제출 {}회, {:.1f} ms",
                 bottomLevels.size(),
                 static_cast<double>(structureBytes) / MB,
                 static_cast<double>(scratchBuffer.size) / MB,
                 submissions,
                 elapsed);
    return true;
}

void RayTracer::barrierBeforeBuild(VkCommandBuffer commandBuffer) {
    // 구조와 스크래치 버퍼는 하나씩만 두고 프레임마다 다시 쓴다. 진행 중인 프레임이 아직 이전
    // 구조를 추적하거나 광선 질의로 읽고 있을 수 있으므로, 덮어쓰기 전에 그것들이 끝나야 한다.
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                           VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    // 스크래치 버퍼 접근도 구축 단계에서는 가속 구조 읽기/쓰기로 친다.
    barrier.srcAccessMask =
        VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.dstAccessMask =
        VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

void RayTracer::updateSkinnedBottomLevel(VkCommandBuffer commandBuffer,
                                         const Buffer& skinnedVertices,
                                         const std::vector<SkinnedInstance>& skinned) {
    if (skinned.empty()) {
        return;
    }

    std::vector<VkAccelerationStructureGeometryKHR> geometries(skinned.size());
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos(skinned.size());
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(skinned.size());
    std::vector<VkDeviceSize> scratchOffsets(skinned.size());
    VkDeviceSize scratchNeeded = 0;

    if (skinnedBottomLevels.size() < skinned.size()) {
        skinnedBottomLevels.resize(skinned.size());
    }

    for (size_t index = 0; index < skinned.size(); ++index) {
        const GpuMesh& mesh = geometry.mesh(skinned[index].meshIndex);
        const GpuMeshLod& lod = geometry.lod(mesh.lodOffset);
        uint32_t vertexCount = geometry.meshVertexCount(skinned[index].meshIndex);

        VkAccelerationStructureGeometryTrianglesDataKHR triangles{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        // 인덱스는 메쉬 지역 번호이므로, 변형 정점 구간의 시작을 정점 기준점으로 삼으면 그대로 맞는다.
        triangles.vertexData.deviceAddress =
            skinnedVertices.address + static_cast<VkDeviceSize>(skinned[index].vertexOffset) * sizeof(asset::Vertex);
        triangles.vertexStride = sizeof(asset::Vertex);
        triangles.maxVertex = vertexCount - 1;
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress =
            geometry.indexBuffer.address + static_cast<VkDeviceSize>(lod.indexOffset) * sizeof(uint32_t);

        geometries[index] = VkAccelerationStructureGeometryKHR{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geometries[index].geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometries[index].geometry.triangles = triangles;
        geometries[index].flags = geometryFlagsFor(geometry.material(mesh.materialIndex));

        buildInfos[index] = VkAccelerationStructureBuildGeometryInfoKHR{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        buildInfos[index].type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        // 매 프레임 다시 세우므로 추적 속도보다 구축 속도를 고른다.
        buildInfos[index].flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
        buildInfos[index].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        buildInfos[index].geometryCount = 1;
        buildInfos[index].pGeometries = &geometries[index];

        uint32_t primitiveCount = lod.indexCount / 3;
        VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        getBuildSizes(context.device,
                      VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                      &buildInfos[index],
                      &primitiveCount,
                      &sizes);

        // 같은 메쉬가 계속 오면 자리를 그대로 다시 쓴다. 크기가 모자랄 때만 새로 잡는다.
        if (skinnedBottomLevels[index].storage.size < sizes.accelerationStructureSize) {
            destroyStructure(skinnedBottomLevels[index]);
            skinnedBottomLevels[index] =
                createStructure(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, sizes.accelerationStructureSize);
        }
        buildInfos[index].dstAccelerationStructure = skinnedBottomLevels[index].handle;

        scratchOffsets[index] = scratchNeeded;
        scratchNeeded += alignUp(sizes.buildScratchSize, 256);

        ranges[index] = {};
        ranges[index].primitiveCount = primitiveCount;
    }

    reserveScratch(skinnedScratchBuffer, std::max<VkDeviceSize>(scratchNeeded, 256), "스킨 가속 구조 스크래치");

    std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> rangePointers(skinned.size());
    for (size_t index = 0; index < skinned.size(); ++index) {
        buildInfos[index].scratchData.deviceAddress = skinnedScratchBuffer.address + scratchOffsets[index];
        rangePointers[index] = &ranges[index];
    }

    barrierBeforeBuild(commandBuffer);
    cmdBuildAccelerationStructures(
        commandBuffer, static_cast<uint32_t>(buildInfos.size()), buildInfos.data(), rangePointers.data());

    // 상위 구조가 이 결과를 읽고, 스크래치도 다시 쓴다.
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.dstAccessMask =
        VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

void RayTracer::updateTopLevel(VkCommandBuffer commandBuffer,
                               const scene::Scene& sceneToTrace,
                               const std::vector<uint32_t>& instanceSlots,
                               const std::vector<uint32_t>& skinnedBlasSlots,
                               uint32_t frameSlot) {
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    instances.reserve(sceneToTrace.objects.size());

    for (uint32_t index = 0; index < sceneToTrace.objects.size(); ++index) {
        uint32_t mesh = sceneToTrace.meshOf(index);
        if (index >= instanceSlots.size() || instanceSlots[index] == INVALID_INSTANCE_SLOT ||
            mesh >= bottomLevels.size()) {
            continue;
        }
        // 스킨 오브젝트는 이번 프레임의 포즈로 다시 세운 구조를 가리킨다. 바인드 포즈 구조를
        // 그대로 두면 화면에서만 움직이고 광선은 서 있는 몸을 맞힌다.
        uint32_t skinnedSlot = index < skinnedBlasSlots.size() ? skinnedBlasSlots[index] : NO_SKINNED_BLAS;
        VkDeviceAddress blasAddress = bottomLevels[mesh].address;
        if (skinnedSlot != NO_SKINNED_BLAS && skinnedSlot < skinnedBottomLevels.size() &&
            skinnedBottomLevels[skinnedSlot].handle != VK_NULL_HANDLE) {
            blasAddress = skinnedBottomLevels[skinnedSlot].address;
        }

        glm::mat4 model = glm::transpose(sceneToTrace.world(index));

        VkAccelerationStructureInstanceKHR instance{};
        std::memcpy(&instance.transform, &model, sizeof(VkTransformMatrixKHR));
        // 적중 셰이더가 인스턴스 배열을 찾는 번호. 그리기 인스턴스는 버킷 순서로 채워지므로
        // 장면 순서로 매기면 어긋난다. buildDrawCommands 가 만든 슬롯을 그대로 쓴다.
        instance.instanceCustomIndex = instanceSlots[index];
        instance.mask = 0xFF;
        // 래스터가 vkCmdSetCullMode 로 하는 것과 같은 판단이다. 양면 재질만 컬링을 끈다.
        instance.flags = 0;
        if (geometry.material(geometry.mesh(mesh).materialIndex).doubleSided) {
            instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        }
        instance.accelerationStructureReference = blasAddress;
        instances.push_back(instance);
    }
    // 인스턴스가 비어도 그냥 돌아가지 않는다. 건너뛰면 지운 모델이 담긴 지난 구조가 그대로 남아
    // 경로 추적과 광선 질의 그림자에 계속 보인다.
    if (instanceBuffers.size() <= frameSlot) {
        instanceBuffers.resize(frameSlot + 1);
        instanceCapacities.resize(frameSlot + 1, 0);
    }
    Buffer& instanceBuffer = instanceBuffers[frameSlot];
    // 비어 있어도 구축 입력 주소는 유효해야 하므로 최소 한 칸은 잡아 둔다.
    const auto neededInstances = std::max<uint32_t>(static_cast<uint32_t>(instances.size()), 1);
    if (neededInstances > instanceCapacities[frameSlot]) {
        destroyBuffer(context, instanceBuffer);
        instanceCapacities[frameSlot] = neededInstances * 2;
        instanceBuffer = createBuffer(context,
                                      static_cast<VkDeviceSize>(instanceCapacities[frameSlot]) *
                                          sizeof(VkAccelerationStructureInstanceKHR),
                                      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                      MemoryLocation::HOST_WRITE,
                                      "가속 구조 인스턴스");
    }
    if (!instances.empty()) {
        std::memcpy(
            instanceBuffer.mapped, instances.data(), instances.size() * sizeof(VkAccelerationStructureInstanceKHR));
    }

    VkAccelerationStructureGeometryKHR geometryInfo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometryInfo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometryInfo.geometry.instances = VkAccelerationStructureGeometryInstancesDataKHR{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR};
    geometryInfo.geometry.instances.data.deviceAddress = instanceBuffer.address;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometryInfo;

    auto instanceCount = static_cast<uint32_t>(instances.size());
    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    getBuildSizes(context.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &instanceCount, &sizes);

    if (topLevel.storage.size < sizes.accelerationStructureSize) {
        destroyStructure(topLevel);
        topLevel = createStructure(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, sizes.accelerationStructureSize);

        VkWriteDescriptorSetAccelerationStructureKHR accelerationWrite{
            VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
        accelerationWrite.accelerationStructureCount = 1;
        accelerationWrite.pAccelerationStructures = &topLevel.handle;

        VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        write.pNext = &accelerationWrite;
        write.dstSet = descriptorSet;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
        vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
    }
    // 빈 장면이면 세울 구조도 추적할 것도 없다.
    if (topLevel.handle == VK_NULL_HANDLE) {
        return;
    }
    reserveScratch(scratchBuffer, std::max<VkDeviceSize>(sizes.buildScratchSize, 256), "가속 구조 스크래치");

    buildInfo.dstAccelerationStructure = topLevel.handle;
    buildInfo.scratchData.deviceAddress = scratchBuffer.address;

    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = instanceCount;
    const VkAccelerationStructureBuildRangeInfoKHR* rangePointer = &range;
    barrierBeforeBuild(commandBuffer);
    cmdBuildAccelerationStructures(commandBuffer, 1, &buildInfo, &rangePointer);

    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    // 경로 추적기뿐 아니라 래스터 경로의 광선 질의 그림자와 반사 컴퓨트도 이 구조를 읽는다.
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

void RayTracer::createPipeline() {
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    binding.descriptorCount = 1;
    // 하이브리드 그림자의 광선 질의는 프래그먼트 셰이더에서, 광선 반사는 컴퓨트에서 같은 TLAS 를 읽는다.
    binding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                         VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.bindingCount = 1;
    layoutInfo.pBindings = &binding;
    VK_CHECK(vkCreateDescriptorSetLayout(context.device, &layoutInfo, nullptr, &descriptorSetLayout));

    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    VK_CHECK(vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &descriptorPool));

    VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocateInfo.descriptorPool = descriptorPool;
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &descriptorSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(context.device, &allocateInfo, &descriptorSet));

    std::array<VkDescriptorSetLayout, 2> setLayouts{bindless.layout(), descriptorSetLayout};
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                   VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    pushConstantRange.size = sizeof(PathTracePushConstants);
    // 규격 보장 최소치가 128 바이트라 여유가 넉넉하지 않다.
    if (pushConstantRange.size > context.properties.limits.maxPushConstantsSize) {
        core::fatal("경로 추적 푸시 상수가 장치 한도를 넘습니다: {} > {} 바이트",
                    pushConstantRange.size,
                    context.properties.limits.maxPushConstantsSize);
    }

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipelineLayoutInfo.setLayoutCount = static_cast<uint32_t>(setLayouts.size());
    pipelineLayoutInfo.pSetLayouts = setLayouts.data();
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &pipelineLayoutInfo, nullptr, &pipelineLayout));

    VkShaderModule raygenModule = createShaderModule(context.device, "pathtrace.rgen.spv");
    VkShaderModule missModule = createShaderModule(context.device, "pathtrace.rmiss.spv");
    VkShaderModule shadowMissModule = createShaderModule(context.device, "pathtrace_shadow.rmiss.spv");
    VkShaderModule hitModule = createShaderModule(context.device, "pathtrace.rchit.spv");
    VkShaderModule anyHitModule = createShaderModule(context.device, "pathtrace.rahit.spv");

    std::array<VkPipelineShaderStageCreateInfo, 5> stages{};
    stages[0] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[0].module = raygenModule;
    stages[0].pName = "main";
    stages[1] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[1].module = missModule;
    stages[1].pName = "main";
    // 그림자 광선은 페이로드가 달라 미스 셰이더도 따로 둔다.
    stages[2] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[2].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[2].module = shadowMissModule;
    stages[2].pName = "main";
    stages[3] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[3].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[3].module = hitModule;
    stages[3].pName = "main";
    // 알파 컷오프와 반투명을 가려낸다. 불투명 재질은 가속 구조가 OPAQUE 라 불리지 않는다.
    stages[4] = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stages[4].stage = VK_SHADER_STAGE_ANY_HIT_BIT_KHR;
    stages[4].module = anyHitModule;
    stages[4].pName = "main";

    std::array<VkRayTracingShaderGroupCreateInfoKHR, 4> groups{};
    for (size_t i = 0; i < groups.size(); ++i) {
        groups[i] = {VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR};
        groups[i].generalShader = VK_SHADER_UNUSED_KHR;
        groups[i].closestHitShader = VK_SHADER_UNUSED_KHR;
        groups[i].anyHitShader = VK_SHADER_UNUSED_KHR;
        groups[i].intersectionShader = VK_SHADER_UNUSED_KHR;
    }
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0;
    groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader = 1;
    groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[2].generalShader = 2;
    groups[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[3].closestHitShader = 3;
    groups[3].anyHitShader = 4;

    VkRayTracingPipelineCreateInfoKHR pipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
    pipelineInfo.pStages = stages.data();
    pipelineInfo.groupCount = static_cast<uint32_t>(groups.size());
    pipelineInfo.pGroups = groups.data();
    pipelineInfo.maxPipelineRayRecursionDepth = 1;
    pipelineInfo.layout = pipelineLayout;
    VK_CHECK(createRayTracingPipelines(
        context.device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));

    vkDestroyShaderModule(context.device, anyHitModule, nullptr);
    vkDestroyShaderModule(context.device, hitModule, nullptr);
    vkDestroyShaderModule(context.device, shadowMissModule, nullptr);
    vkDestroyShaderModule(context.device, missModule, nullptr);
    vkDestroyShaderModule(context.device, raygenModule, nullptr);

    // 셰이더 바인딩 테이블. 그룹마다 핸들 하나씩만 두므로 스트라이드가 곧 정렬된 핸들 크기다.
    uint32_t handleSize = context.rayTracingPipelineProperties.shaderGroupHandleSize;
    uint32_t handleAlignment = context.rayTracingPipelineProperties.shaderGroupHandleAlignment;
    uint32_t baseAlignment = context.rayTracingPipelineProperties.shaderGroupBaseAlignment;
    auto handleStride = static_cast<uint32_t>(alignUp(handleSize, handleAlignment));
    auto regionStride = static_cast<uint32_t>(alignUp(handleStride, baseAlignment));

    std::vector<uint8_t> handles(static_cast<size_t>(handleSize) * groups.size());
    VK_CHECK(getShaderGroupHandles(
        context.device, pipeline, 0, static_cast<uint32_t>(groups.size()), handles.size(), handles.data()));

    VkDeviceSize tableSize = static_cast<VkDeviceSize>(regionStride) * groups.size();
    shaderBindingTable = createBuffer(context,
                                      tableSize,
                                      VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR,
                                      MemoryLocation::HOST_WRITE,
                                      "셰이더 바인딩 테이블");
    auto* table = static_cast<uint8_t*>(shaderBindingTable.mapped);
    std::memset(table, 0, tableSize);
    for (size_t i = 0; i < groups.size(); ++i) {
        std::memcpy(table + i * regionStride, handles.data() + i * handleSize, handleSize);
    }

    raygenRegion = {shaderBindingTable.address, regionStride, regionStride};
    // 미스 그룹이 둘이라 이 구간만 두 배다.
    missRegion = {shaderBindingTable.address + regionStride, regionStride, 2ULL * regionStride};
    hitRegion = {shaderBindingTable.address + 3ULL * regionStride, regionStride, regionStride};
    callableRegion = {};

    spdlog::info("경로 추적 파이프라인 준비 완료");
}

void RayTracer::trace(VkCommandBuffer commandBuffer,
                      VkExtent2D extent,
                      VkDeviceAddress cameraAddress,
                      VkDeviceAddress instanceAddress,
                      VkDeviceAddress lightAddress,
                      VkDeviceAddress skinnedVertexAddress,
                      uint32_t accumulationImage,
                      uint32_t velocityImage,
                      uint32_t frameIndex,
                      uint32_t sampleCount,
                      const PathTraceOptions& options,
                      const PathGuideTargets& guides) {
    PathTracePushConstants pushConstants{};
    pushConstants.vertices = geometry.vertexBuffer.address;
    pushConstants.skinnedVertices = skinnedVertexAddress;
    pushConstants.indices = geometry.indexBuffer.address;
    pushConstants.meshes = geometry.meshBuffer.address;
    pushConstants.instances = instanceAddress;
    pushConstants.materials = geometry.materialBuffer.address;
    pushConstants.lods = geometry.lodBuffer.address;
    pushConstants.camera = cameraAddress;
    pushConstants.lights = lightAddress;
    pushConstants.accumulationImage = accumulationImage;
    pushConstants.velocityImage = velocityImage;
    pushConstants.frameIndex = frameIndex;
    pushConstants.sampleCount = sampleCount;
    pushConstants.maxBounces = options.maxBounces;
    pushConstants.samplesPerFrame = options.samplesPerFrame;
    pushConstants.flags = (options.nextEventEstimation ? PATH_FLAG_NEXT_EVENT : 0U) |
                          (options.russianRoulette ? PATH_FLAG_RUSSIAN_ROULETTE : 0U) |
                          (guides.write ? PATH_FLAG_WRITE_GUIDES : 0U);
    pushConstants.radianceClamp = options.radianceClamp;
    pushConstants.skyIntensity = options.skyIntensity;
    pushConstants.debugMode = options.debugMode;
    pushConstants.guideAlbedoSlots = (guides.diffuseAlbedo & 0xFFFFU) | (guides.specularAlbedo << 16);
    pushConstants.guideNormalRoughnessSlots = (guides.normal & 0xFFFFU) | (guides.roughness << 16);
    pushConstants.guideDepthSlot = guides.depth;

    std::array<VkDescriptorSet, 2> sets{bindless.set(), descriptorSet};
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, pipeline);
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR,
                            pipelineLayout,
                            0,
                            static_cast<uint32_t>(sets.size()),
                            sets.data(),
                            0,
                            nullptr);
    vkCmdPushConstants(commandBuffer,
                       pipelineLayout,
                       VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                           VK_SHADER_STAGE_MISS_BIT_KHR | VK_SHADER_STAGE_ANY_HIT_BIT_KHR,
                       0,
                       sizeof(pushConstants),
                       &pushConstants);
    cmdTraceRays(
        commandBuffer, &raygenRegion, &missRegion, &hitRegion, &callableRegion, extent.width, extent.height, 1);
}

} // namespace gfx
