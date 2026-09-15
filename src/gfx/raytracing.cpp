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
#include "gfx/cloth.h"
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
    VkDeviceAddress fluidSurfaces;
    uint32_t accumulationImage;
    uint32_t velocityImage;
    uint32_t frameIndex;
    uint32_t sampleCount;
    // 128 바이트 한도 때문에 작은 값은 둘씩 16비트로 묶는다. 이유는 pathtrace_common.glsl 쪽 주석 참조.
    uint32_t bouncesAndSamples; // 하위 최대 바운스, 상위 프레임당 표본
    uint32_t flags;
    float radianceClamp;
    float skyIntensity;
    uint32_t debugModeAndDepthSlot; // 하위 디버그 모드, 상위 안내 깊이 슬롯
    uint32_t guideAlbedoSlots;
    uint32_t guideNormalRoughnessSlots;
};
static_assert(sizeof(PathTracePushConstants) <= 128, "경로 추적 푸시 상수는 규격이 보장하는 128 바이트 안이어야 한다");

constexpr uint32_t PATH_FLAG_NEXT_EVENT = 1;
constexpr uint32_t PATH_FLAG_RUSSIAN_ROULETTE = 2;
constexpr uint32_t PATH_FLAG_WRITE_GUIDES = 4;
constexpr uint32_t PATH_FLAG_FIXED_JITTER = 8;

uint64_t alignUp(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
}

// 알파를 보는 재질은 OPAQUE 로 올리면 안 된다. 그래야 적중 셰이더가 불려 컷오프와 반투명을
// 가려낼 수 있다. 불투명 재질은 표시해 두어야 교차 판정이 빨라진다.
VkGeometryFlagsKHR geometryFlagsFor(const asset::Material& material) {
    return material.alphaMode == asset::AlphaMode::SOLID ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;
}

// shaders/cluster_gather.comp 의 푸시 상수와 배치가 같아야 한다.
struct ClusterGatherPushConstants {
    VkDeviceAddress vertices;
    VkDeviceAddress meshlets;
    VkDeviceAddress meshletVertices;
    VkDeviceAddress positions;
    uint32_t meshletOffset;
    uint32_t meshletCount;
    uint32_t meshVertexOffset;
    uint32_t sourceVertexOffset;
    uint32_t meshletVertexBase;
    uint32_t positionBase;
};

VkPipeline createComputePipeline(Context& context, VkPipelineLayout layout, const char* shaderName) {
    VkShaderModule module = createShaderModule(context.device, shaderName);
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
    vkDestroyShaderModule(context.device, module, nullptr);
    return pipeline;
}

} // namespace

struct RayTracer::ClusterSet {
    Buffer positions;
    Buffer clusterInfos;     // VkClusterAccelerationStructureBuildTriangleClusterInfoNV[], CPU 가 쓴다
    Buffer clusterStorage;   // 암시적 목적지
    Buffer clusterAddresses; // 클러스터 주소. 클러스터 빌드가 쓰고 하위 구조 빌드가 읽는다
    Buffer bottomInfos;      // VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV[], CPU 가 쓴다
    Buffer bottomAddresses;  // 명시적 목적지 주소, CPU 가 쓴다
    Buffer scratch;
    VkClusterAccelerationStructureTriangleClusterInputNV triangleInput{
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV};
    VkClusterAccelerationStructureClustersBottomLevelInputNV bottomInput{
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV};
    VkClusterAccelerationStructureInputInfoNV clusterBuild{
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    VkClusterAccelerationStructureInputInfoNV bottomBuild{
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    VkDeviceSize clusterScratch = 0;
    VkDeviceSize bottomScratch = 0;
    struct Gather {
        ClusterGatherPushConstants push;
        uint32_t groups;
    };
    std::vector<Gather> gathers;
};

RayTracer::RayTracer(Context& context, GeometryStore& geometry, BindlessTextures& bindless)
    : context(context), geometry(geometry), bindless(bindless) {
    loadFunctions();
    createPipeline();
    if (context.caps.clusterAccelerationStructure) {
        VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ClusterGatherPushConstants)};
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &range;
        VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &gatherLayout));
        gatherPipeline = createComputePipeline(context, gatherLayout, "cluster_gather.comp.spv");
        // 하드웨어 가속 경로가 있으면 그것이 기본이다.
        useClusters = true;
    }
}

RayTracer::~RayTracer() {
    destroyStructure(topLevel);
    for (AccelerationStructure& structure : bottomLevels) {
        destroyStructure(structure);
    }
    for (AccelerationStructure& structure : skinnedBottomLevels) {
        destroyStructure(structure);
    }
    for (DynamicBottomLevel& dynamic : dynamicBottomLevels) {
        destroyStructure(dynamic.structure);
        destroyBuffer(context, dynamic.scratch);
    }
    auto destroySet = [this](std::unique_ptr<ClusterSet>& set) {
        if (set == nullptr) {
            return;
        }
        for (Buffer* buffer : {&set->positions,
                               &set->clusterInfos,
                               &set->clusterStorage,
                               &set->clusterAddresses,
                               &set->bottomInfos,
                               &set->bottomAddresses,
                               &set->scratch}) {
            destroyBuffer(context, *buffer);
        }
    };
    destroySet(staticClusters);
    for (std::unique_ptr<ClusterSet>& set : skinnedClusters) {
        destroySet(set);
    }
    vkDestroyPipeline(context.device, gatherPipeline, nullptr);
    vkDestroyPipelineLayout(context.device, gatherLayout, nullptr);
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
    // 간접 구축은 기능 게이트다. 장치가 지원하지 않으면 진입점이 없고, 그러면 물 표면을 올리지 않는다.
    if (context.caps.accelerationStructureIndirectBuild) {
        cmdBuildAccelerationStructuresIndirect = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresIndirectKHR>(
            load("vkCmdBuildAccelerationStructuresIndirectKHR"));
    }
    if (context.caps.clusterAccelerationStructure) {
        getClusterBuildSizes = reinterpret_cast<PFN_vkGetClusterAccelerationStructureBuildSizesNV>(
            load("vkGetClusterAccelerationStructureBuildSizesNV"));
        cmdBuildClusters = reinterpret_cast<PFN_vkCmdBuildClusterAccelerationStructureIndirectNV>(
            load("vkCmdBuildClusterAccelerationStructureIndirectNV"));
        if (getClusterBuildSizes == nullptr || cmdBuildClusters == nullptr) {
            core::fatal("클러스터 가속 구조 진입점을 찾을 수 없습니다");
        }
    }
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

void RayTracer::retireStructure(AccelerationStructure& structure) {
    if (structure.handle != VK_NULL_HANDLE) {
        VkDevice device = context.device;
        PFN_vkDestroyAccelerationStructureKHR destroy = destroyAccelerationStructure;
        VkAccelerationStructureKHR handle = structure.handle;
        context.retireDeferred([device, destroy, handle]() { destroy(device, handle, nullptr); });
        structure.handle = VK_NULL_HANDLE;
    }
    context.retireBuffer(structure.storage);
    structure.address = 0;
}

void RayTracer::reserveScratch(Buffer& buffer, VkDeviceSize size, const char* debugName) {
    if (buffer.size >= size) {
        return;
    }
    // 스크래치도 기록 중에 커진다. 지난 프레임의 구축이 아직 옛 버퍼를 쓰고 있을 수 있다.
    context.retireBuffer(buffer);
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
    bottomLevelBuilt = false;
    // 상위 구조는 하위 구조 주소를 담고 있어 함께 버린다. ready() 가 거짓이 되어 다음 프레임에 다시 세운다.
    destroyStructure(topLevel);
}

bool RayTracer::buildBottomLevel(std::string& reason) {
    invalidateBottomLevel();
    if (useClusters) {
        return buildClusterBottomLevel(reason);
    }
    auto buildStart = std::chrono::steady_clock::now();

    // 먼저 크기만 재서 예산에 들어가는지 본다. 넘기면 할당이 시스템 메모리로 넘어가 통과한 뒤 빌드
    // 중에 장치를 잃으므로, 만들기 전에 거른다.
    uint32_t meshCount = geometry.meshCount();
    std::vector<VkAccelerationStructureGeometryKHR> geometries(meshCount);
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos(meshCount);
    std::vector<VkAccelerationStructureBuildSizesInfoKHR> sizes(meshCount);
    std::vector<uint32_t> primitiveCounts(meshCount);
    // 해제된 무덤 메쉬는 세우지 않고 핸들을 null 로 둔다. 정점 수 0 에서 maxVertex 가 아래로 넘친다.
    auto live = [this](uint32_t index) { return geometry.meshLive(index) && geometry.meshVertexCount(index) > 0; };
    VkDeviceSize structureBytes = 0;
    VkDeviceSize largestScratch = 0;
    VkDeviceSize totalScratch = 0;
    for (uint32_t index = 0; index < meshCount; ++index) {
        if (!live(index)) {
            continue;
        }
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
        // 무덤 메쉬는 묶음에 넣지 않는다. 구조 생성도 빌드 명령도 살아 있는 메쉬만 모은다.
        std::vector<VkAccelerationStructureBuildGeometryInfoKHR> batchInfos;
        while (last < meshCount) {
            if (!live(last)) {
                ++last;
                continue;
            }
            VkDeviceSize scratch = alignUp(sizes[last].buildScratchSize, 256);
            if (!batchInfos.empty() && used + scratch > scratchBuffer.size) {
                break;
            }
            bottomLevels[last] =
                createStructure(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, sizes[last].accelerationStructureSize);
            buildInfos[last].dstAccelerationStructure = bottomLevels[last].handle;
            buildInfos[last].scratchData.deviceAddress = scratchBuffer.address + used;
            used += scratch;
            batchInfos.push_back(buildInfos[last]);
            ranges.push_back(VkAccelerationStructureBuildRangeInfoKHR{primitiveCounts[last], 0, 0, 0});
            ++last;
        }
        if (batchInfos.empty()) {
            first = last;
            continue;
        }
        rangePointers.resize(ranges.size());
        for (size_t i = 0; i < ranges.size(); ++i) {
            rangePointers[i] = &ranges[i];
        }

        VK_CHECK(vkResetCommandPool(context.device, pool, 0));
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
        cmdBuildAccelerationStructures(
            commandBuffer, static_cast<uint32_t>(batchInfos.size()), batchInfos.data(), rangePointers.data());
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
    bottomLevelBuilt = true;

    size_t built = 0;
    for (const AccelerationStructure& structure : bottomLevels) {
        built += structure.address != 0 ? 1 : 0;
    }
    double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - buildStart).count();
    spdlog::info("하위 가속 구조 {}개 생성: {:.0f} MB, 스크래치 {:.0f} MB, 제출 {}회, {:.1f} ms",
                 built,
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
    // 반사·DDGI·입자는 컴퓨트 광선 질의로 읽으므로 그 단계도 기다린다.
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR |
                           VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
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

    if (skinnedBottomLevels.size() < skinned.size()) {
        skinnedBottomLevels.resize(skinned.size());
    }
    // 포즈가 바뀐 것과 아직 구조가 없는 것만 세운다. 번호는 skinnedBottomLevels 와 같다.
    std::vector<size_t> toBuild;
    for (size_t index = 0; index < skinned.size(); ++index) {
        if (skinned[index].rebuild || skinnedBottomLevels[index].address == 0) {
            toBuild.push_back(index);
        }
    }
    if (toBuild.empty()) {
        return;
    }
    if (useClusters) {
        updateSkinnedClusterBottomLevel(commandBuffer, skinnedVertices, skinned, toBuild);
        return;
    }

    std::vector<VkAccelerationStructureGeometryKHR> geometries(toBuild.size());
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos(toBuild.size());
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(toBuild.size());
    std::vector<VkDeviceSize> scratchOffsets(toBuild.size());
    VkDeviceSize scratchNeeded = 0;

    for (size_t slot = 0; slot < toBuild.size(); ++slot) {
        size_t index = toBuild[slot];
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

        geometries[slot] = VkAccelerationStructureGeometryKHR{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geometries[slot].geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometries[slot].geometry.triangles = triangles;
        geometries[slot].flags = geometryFlagsFor(geometry.material(mesh.materialIndex));

        buildInfos[slot] = VkAccelerationStructureBuildGeometryInfoKHR{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        buildInfos[slot].type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        // 포즈가 바뀌는 프레임마다 다시 세우므로 추적 속도보다 구축 속도를 고른다.
        buildInfos[slot].flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
        buildInfos[slot].mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        buildInfos[slot].geometryCount = 1;
        buildInfos[slot].pGeometries = &geometries[slot];

        uint32_t primitiveCount = lod.indexCount / 3;
        VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        getBuildSizes(context.device,
                      VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
                      &buildInfos[slot],
                      &primitiveCount,
                      &sizes);

        // 같은 메쉬가 계속 오면 자리를 그대로 다시 쓴다. 크기가 모자라거나 클러스터 구조(핸들이 없다)가 들어
        // 있던 자리면 새로 잡는다.
        if (skinnedBottomLevels[index].handle == VK_NULL_HANDLE ||
            skinnedBottomLevels[index].storage.size < sizes.accelerationStructureSize) {
            retireStructure(skinnedBottomLevels[index]);
            skinnedBottomLevels[index] =
                createStructure(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, sizes.accelerationStructureSize);
        }
        buildInfos[slot].dstAccelerationStructure = skinnedBottomLevels[index].handle;

        scratchOffsets[slot] = scratchNeeded;
        scratchNeeded += alignUp(sizes.buildScratchSize, 256);

        ranges[slot] = {};
        ranges[slot].primitiveCount = primitiveCount;
    }

    reserveScratch(skinnedScratchBuffer, std::max<VkDeviceSize>(scratchNeeded, 256), "스킨 가속 구조 스크래치");

    std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> rangePointers(toBuild.size());
    for (size_t slot = 0; slot < toBuild.size(); ++slot) {
        buildInfos[slot].scratchData.deviceAddress = skinnedScratchBuffer.address + scratchOffsets[slot];
        rangePointers[slot] = &ranges[slot];
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

namespace {

// 동적 구조의 지오메트리 서술. 준비(크기 계산)와 구축이 같은 서술을 써야 크기가 맞는다.
VkAccelerationStructureGeometryKHR
dynamicTriangles(VkDeviceAddress vertices, uint32_t vertexStride, uint32_t maxTriangles) {
    VkAccelerationStructureGeometryTrianglesDataKHR triangles{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
    triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    triangles.vertexData.deviceAddress = vertices;
    triangles.vertexStride = vertexStride;
    triangles.maxVertex = maxTriangles * 3 - 1;
    triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
    VkAccelerationStructureGeometryKHR geometry{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geometry.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    geometry.geometry.triangles = triangles;
    // 물은 알파가 없다. 불투명으로 올려 임의 적중 셰이더가 돌지 않게 한다.
    geometry.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    return geometry;
}

} // namespace

VkDeviceAddress RayTracer::ensureDynamicBottomLevel(uint32_t key, uint32_t maxTriangles) {
    if (maxTriangles == 0) {
        return 0;
    }
    if (dynamicBottomLevels.size() <= key) {
        dynamicBottomLevels.resize(key + 1);
    }
    DynamicBottomLevel& dynamic = dynamicBottomLevels[key];
    if (dynamic.structure.handle != VK_NULL_HANDLE && dynamic.maxTriangles == maxTriangles) {
        return dynamic.structure.address;
    }
    // 크기는 정점 주소와 무관하다. 상한 삼각형 수로 재어 두고 프레임마다 그 안에서 세운다.
    VkAccelerationStructureGeometryKHR geometry = dynamicTriangles(0, sizeof(float) * 4, maxTriangles);
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;
    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    getBuildSizes(context.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &maxTriangles, &sizes);

    retireStructure(dynamic.structure);
    dynamic.structure =
        createStructure(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, sizes.accelerationStructureSize);
    reserveScratch(dynamic.scratch, std::max<VkDeviceSize>(sizes.buildScratchSize, 256), "물 표면 가속 구조 스크래치");
    dynamic.maxTriangles = maxTriangles;
    return dynamic.structure.address;
}

void RayTracer::buildDynamicBottomLevel(VkCommandBuffer commandBuffer,
                                        uint32_t key,
                                        VkDeviceAddress vertices,
                                        uint32_t vertexStride,
                                        uint32_t maxTriangles,
                                        VkDeviceAddress rangeAddress) {
    if (key >= dynamicBottomLevels.size() || dynamicBottomLevels[key].structure.handle == VK_NULL_HANDLE) {
        return;
    }
    DynamicBottomLevel& dynamic = dynamicBottomLevels[key];
    VkAccelerationStructureGeometryKHR geometry = dynamicTriangles(vertices, vertexStride, maxTriangles);
    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometry;
    buildInfo.dstAccelerationStructure = dynamic.structure.handle;
    buildInfo.scratchData.deviceAddress = dynamic.scratch.address;

    barrierBeforeBuild(commandBuffer);
    if (indirectBuildAvailable()) {
        // 삼각형 수는 rangeAddress 의 VkAccelerationStructureBuildRangeInfoKHR 에서 읽는다. 마칭 컴퓨트가 채운 값이다.
        uint32_t stride = sizeof(VkAccelerationStructureBuildRangeInfoKHR);
        const uint32_t* maxPrimitiveCounts = &maxTriangles;
        cmdBuildAccelerationStructuresIndirect(
            commandBuffer, 1, &buildInfo, &rangeAddress, &stride, &maxPrimitiveCounts);
    } else {
        // 상한 개수로 세운다. 쓰지 않은 꼬리는 0 으로 지워진 퇴화 삼각형이라 광선이 맞히지 않는다.
        // ponytail: 실제 삼각형 수와 무관하게 늘 상한만큼 구축한다. 프레임당 65536 삼각형 빠른 구축이다.
        VkAccelerationStructureBuildRangeInfoKHR range{};
        range.primitiveCount = maxTriangles;
        const VkAccelerationStructureBuildRangeInfoKHR* rangePointer = &range;
        cmdBuildAccelerationStructures(commandBuffer, 1, &buildInfo, &rangePointer);
    }

    // 상위 구조가 이 결과를 읽는다.
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

void RayTracer::reserveInstances(uint32_t frameSlot, uint32_t count) {
    if (instanceBuffers.size() <= frameSlot) {
        instanceBuffers.resize(frameSlot + 1);
        instanceCapacities.resize(frameSlot + 1, 0);
    }
    // 비어 있어도 구축 입력 주소는 유효해야 하므로 최소 한 칸은 잡아 둔다.
    uint32_t needed = std::max<uint32_t>(count, 1);
    if (needed <= instanceCapacities[frameSlot]) {
        return;
    }
    destroyBuffer(context, instanceBuffers[frameSlot]);
    instanceCapacities[frameSlot] = needed * 2;
    // 유체 컴퓨트가 앞쪽을 직접 쓰므로 스토리지 용도도 붙인다.
    instanceBuffers[frameSlot] = createBuffer(
        context,
        static_cast<VkDeviceSize>(instanceCapacities[frameSlot]) * sizeof(VkAccelerationStructureInstanceKHR),
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        MemoryLocation::HOST_WRITE,
        "가속 구조 인스턴스");
}

VkDeviceAddress RayTracer::instanceBufferAddress(uint32_t frameSlot) const {
    return frameSlot < instanceBuffers.size() ? instanceBuffers[frameSlot].address : 0;
}

void* RayTracer::instanceBufferMapped(uint32_t frameSlot) const {
    return frameSlot < instanceBuffers.size() ? instanceBuffers[frameSlot].mapped : nullptr;
}

VkDeviceAddress RayTracer::bottomLevelAddress(uint32_t mesh) const {
    if (mesh >= bottomLevels.size() || bottomLevels[mesh].address == 0) {
        return 0;
    }
    return bottomLevels[mesh].address;
}

void RayTracer::updateTopLevel(VkCommandBuffer commandBuffer,
                               const scene::Scene& sceneToTrace,
                               const std::vector<uint32_t>& instanceSlots,
                               const std::vector<uint32_t>& skinnedBlasSlots,
                               uint32_t frameSlot,
                               uint32_t prependedInstances) {
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    instances.reserve(sceneToTrace.objects.size());

    for (uint32_t index = 0; index < sceneToTrace.objects.size(); ++index) {
        uint32_t mesh = sceneToTrace.meshOf(index);
        if (index >= instanceSlots.size() || instanceSlots[index] == INVALID_INSTANCE_SLOT ||
            mesh >= bottomLevels.size() || bottomLevels[mesh].address == 0) {
            continue;
        }
        // 스킨 오브젝트는 이번 프레임의 포즈로 다시 세운 구조를 가리킨다. 바인드 포즈 구조를
        // 그대로 두면 화면에서만 움직이고 광선은 서 있는 몸을 맞힌다.
        uint32_t skinnedSlot = index < skinnedBlasSlots.size() ? skinnedBlasSlots[index] : NO_SKINNED_BLAS;
        VkDeviceAddress blasAddress = bottomLevels[mesh].address;
        if (skinnedSlot != NO_SKINNED_BLAS && skinnedSlot < skinnedBottomLevels.size() &&
            skinnedBottomLevels[skinnedSlot].address != 0) {
            blasAddress = skinnedBottomLevels[skinnedSlot].address;
        }

        // 천은 월드 공간에서 풀어 변형 정점에 이미 월드 위치가 들어 있다. 그리기 인스턴스처럼 변환은 항등이다.
        bool worldSpaceVertices = sceneToTrace.objects[index].cloth >= 0 && skinnedSlot != NO_SKINNED_BLAS;
        glm::mat4 model = worldSpaceVertices ? glm::mat4{1.0F} : glm::transpose(sceneToTrace.world(index));

        VkAccelerationStructureInstanceKHR instance{};
        std::memcpy(&instance.transform, &model, sizeof(VkTransformMatrixKHR));
        // 적중 셰이더가 인스턴스 배열을 찾는 번호. 그리기 인스턴스는 버킷 순서로 채워지므로
        // 장면 순서로 매기면 어긋난다. buildDrawCommands 가 만든 슬롯을 그대로 쓴다.
        instance.instanceCustomIndex = instanceSlots[index];
        // 천은 자기 충돌 광선(CLOTH_SELF_RAY_MASK 만 켠 마스크)이 지나가도록 그 비트를 내린다.
        instance.mask = sceneToTrace.objects[index].cloth >= 0 ? (0xFFU & ~CLOTH_SELF_RAY_MASK) : 0xFFU;
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
    // 앞쪽 prependedInstances 칸은 GPU 가 이미 채웠다. 그때는 reserveInstances 가 먼저 불렸으므로 여기서
    // 버퍼가 바뀌지 않는다.
    reserveInstances(frameSlot, prependedInstances + static_cast<uint32_t>(instances.size()));
    Buffer& instanceBuffer = instanceBuffers[frameSlot];
    if (!instances.empty()) {
        std::memcpy(static_cast<VkAccelerationStructureInstanceKHR*>(instanceBuffer.mapped) + prependedInstances,
                    instances.data(),
                    instances.size() * sizeof(VkAccelerationStructureInstanceKHR));
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

    auto instanceCount = prependedInstances + static_cast<uint32_t>(instances.size());
    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    getBuildSizes(context.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &instanceCount, &sizes);

    if (topLevel.storage.size < sizes.accelerationStructureSize) {
        retireStructure(topLevel);
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
    // 클러스터 장치는 gl_ClusterIDNV 를 읽는 변종을 쓴다. 클러스터가 아닌 구조를 맞히면 -1 이라 두 방식을 다 푼다.
    bool clusters = context.caps.clusterAccelerationStructure;
    VkShaderModule hitModule =
        createShaderModule(context.device, clusters ? "pathtrace_cluster.rchit.spv" : "pathtrace.rchit.spv");
    VkShaderModule anyHitModule =
        createShaderModule(context.device, clusters ? "pathtrace_cluster.rahit.spv" : "pathtrace.rahit.spv");

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
    // 클러스터 하위 구조를 추적하려면 파이프라인에 허용을 걸어야 한다.
    VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV clusterInfo{
        VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CLUSTER_ACCELERATION_STRUCTURE_CREATE_INFO_NV};
    clusterInfo.allowClusterAccelerationStructure = VK_TRUE;
    if (clusters) {
        pipelineInfo.pNext = &clusterInfo;
    }
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
                      VkDeviceAddress fluidSurfaceAddress,
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
    pushConstants.fluidSurfaces = fluidSurfaceAddress;
    pushConstants.accumulationImage = accumulationImage;
    pushConstants.velocityImage = velocityImage;
    pushConstants.frameIndex = frameIndex;
    pushConstants.sampleCount = sampleCount;
    pushConstants.bouncesAndSamples = (options.maxBounces & 0xFFFFU) | (options.samplesPerFrame << 16);
    // 비트 8~15 는 광원 후보 수(pathtrace_common.glsl 의 pathLightCandidates).
    pushConstants.flags = (options.nextEventEstimation ? PATH_FLAG_NEXT_EVENT : 0U) |
                          (options.russianRoulette ? PATH_FLAG_RUSSIAN_ROULETTE : 0U) |
                          (guides.write ? PATH_FLAG_WRITE_GUIDES : 0U) |
                          (guides.fixedJitter ? PATH_FLAG_FIXED_JITTER : 0U) |
                          (std::min(std::max(options.lightCandidates, 1U), 255U) << 8U);
    pushConstants.radianceClamp = options.radianceClamp;
    pushConstants.skyIntensity = options.skyIntensity;
    pushConstants.debugModeAndDepthSlot = (options.debugMode & 0xFFFFU) | (guides.depth << 16);
    pushConstants.guideAlbedoSlots = (guides.diffuseAlbedo & 0xFFFFU) | (guides.specularAlbedo << 16);
    pushConstants.guideNormalRoughnessSlots = (guides.normal & 0xFFFFU) | (guides.roughness << 16);

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

// ---- 클러스터 하위 구조(VK_NV_cluster_acceleration_structure) ----
//
// meshlet 하나가 클러스터 가속 구조(CLAS) 하나다. 클러스터의 입력은 «클러스터 안 지역 인덱스 + 빽빽한 위치 배열»
// 인데, meshlet 삼각형 버퍼가 이미 지역 인덱스(8비트)이므로 위치만 meshlet 정점 목록 순서로 모으면(cluster_gather.comp)
// 그 둘이 짝이 된다. 클러스터는 암시적 목적지로 한 버퍼에 몰아 세우고(주소는 장치가 배열에 써 준다), 메쉬의 하위
// 구조는 명시적 목적지로 세운다 — CPU 가 상위 구조 인스턴스에 그 주소를 써야 하므로 미리 알아야 한다.
// 클러스터 번호는 meshlet 의 첫 삼각형 번호(indexOffset / 3)라, 히트 셰이더가 표 없이 인덱스 버퍼로 바로 간다.
// ponytail: LOD 0 만 올린다. 래스터가 고른 LOD 의 meshlet 로 세우면 mesh shader 경로와 기하가 정확히 같아진다.

bool RayTracer::clusterAvailable() const {
    return context.caps.clusterAccelerationStructure && gatherPipeline != VK_NULL_HANDLE;
}

void RayTracer::setClusterMode(bool enabled) {
    enabled = enabled && clusterAvailable();
    if (enabled == useClusters) {
        return;
    }
    useClusters = enabled;
    // 프레임 기록 중에 불리므로 바로 지우지 않고 맡겨 둔다. 주소가 0 이면 다음에 다시 세운다.
    for (AccelerationStructure& structure : bottomLevels) {
        retireStructure(structure);
    }
    bottomLevels.clear();
    bottomLevelBuilt = false;
    retireStructure(topLevel);
    for (AccelerationStructure& structure : skinnedBottomLevels) {
        retireStructure(structure);
    }
}

namespace {

// 크기가 모자랄 때만 새로 잡는다. 기록 중일 수 있어 옛 버퍼는 맡겨 둔다.
void ensureBuffer(Context& context,
                  Buffer& buffer,
                  VkDeviceSize size,
                  VkBufferUsageFlags usage,
                  MemoryLocation location,
                  const char* debugName,
                  VkDeviceSize alignment = 0) {
    if (buffer.size >= size) {
        return;
    }
    context.retireBuffer(buffer);
    buffer = createBuffer(context, std::max<VkDeviceSize>(size, 16), usage, location, debugName, alignment);
}

constexpr VkBufferUsageFlags CLUSTER_INPUT_USAGE =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
constexpr VkBufferUsageFlags CLUSTER_STORAGE_USAGE =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR;

} // namespace

bool RayTracer::prepareClusterBuild(ClusterSet& set,
                                    const std::vector<uint32_t>& meshes,
                                    const std::vector<AccelerationStructure*>& destinations,
                                    const Buffer* skinnedVertices,
                                    const std::vector<uint32_t>& skinnedVertexOffsets,
                                    VkBuildAccelerationStructureFlagsKHR flags,
                                    bool budgetCheck,
                                    std::string& reason) {
    const VkPhysicalDeviceClusterAccelerationStructurePropertiesNV& props = context.clusterProperties;
    std::vector<VkClusterAccelerationStructureBuildTriangleClusterInfoNV> clusterInfos;
    std::vector<VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV> bottomInfos(meshes.size());
    set.gathers.clear();
    set.triangleInput = VkClusterAccelerationStructureTriangleClusterInputNV{
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV};
    set.triangleInput.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    set.triangleInput.maxClusterUniqueGeometryCount = 1;
    uint32_t positionCount = 0;
    uint32_t maxClustersPerMesh = 0;

    for (size_t i = 0; i < meshes.size(); ++i) {
        const GpuMesh& mesh = geometry.mesh(meshes[i]);
        const GpuMeshLod& lod = geometry.lod(mesh.lodOffset);
        bool opaque = geometryFlagsFor(geometry.material(mesh.materialIndex)) == VK_GEOMETRY_OPAQUE_BIT_KHR;
        uint32_t firstEntry = geometry.meshlet(lod.meshletOffset).vertexOffset;

        ClusterSet::Gather gather{};
        gather.push.vertices = skinnedVertices != nullptr ? skinnedVertices->address : geometry.vertexBuffer.address;
        gather.push.meshlets = geometry.meshletBuffer.address;
        gather.push.meshletVertices = geometry.meshletVertexBuffer.address;
        gather.push.meshletOffset = lod.meshletOffset;
        gather.push.meshletCount = lod.meshletCount;
        gather.push.meshVertexOffset = skinnedVertices != nullptr ? static_cast<uint32_t>(mesh.vertexOffset) : 0;
        gather.push.sourceVertexOffset = skinnedVertices != nullptr ? skinnedVertexOffsets[i] : 0;
        gather.push.meshletVertexBase = firstEntry;
        gather.push.positionBase = positionCount;
        gather.groups = lod.meshletCount;
        set.gathers.push_back(gather);

        bottomInfos[i].clusterReferencesCount = lod.meshletCount;
        bottomInfos[i].clusterReferencesStride = sizeof(VkDeviceAddress);
        // 주소 배열은 아직 없을 수 있어 아래서 기준 주소를 더한다.
        bottomInfos[i].clusterReferences = clusterInfos.size() * sizeof(VkDeviceAddress);
        maxClustersPerMesh = std::max(maxClustersPerMesh, lod.meshletCount);

        for (uint32_t m = 0; m < lod.meshletCount; ++m) {
            const GpuMeshlet& meshlet = geometry.meshlet(lod.meshletOffset + m);
            VkClusterAccelerationStructureBuildTriangleClusterInfoNV info{};
            info.clusterID = meshlet.indexOffset / 3;
            info.triangleCount = meshlet.triangleCount;
            info.vertexCount = meshlet.vertexCount;
            info.indexType = VK_CLUSTER_ACCELERATION_STRUCTURE_INDEX_FORMAT_8BIT_NV;
            info.baseGeometryIndexAndGeometryFlags.geometryFlags =
                opaque ? VK_CLUSTER_ACCELERATION_STRUCTURE_GEOMETRY_OPAQUE_BIT_NV : 0;
            info.indexBufferStride = 1;
            info.vertexBufferStride = sizeof(float) * 3;
            info.indexBuffer = geometry.meshletTriangleBuffer.address + meshlet.triangleOffset;
            info.vertexBuffer =
                static_cast<VkDeviceSize>(positionCount + meshlet.vertexOffset - firstEntry) * sizeof(float) * 3;
            clusterInfos.push_back(info);
            set.triangleInput.maxClusterTriangleCount =
                std::max(set.triangleInput.maxClusterTriangleCount, meshlet.triangleCount);
            set.triangleInput.maxClusterVertexCount =
                std::max(set.triangleInput.maxClusterVertexCount, meshlet.vertexCount);
            set.triangleInput.maxTotalTriangleCount += meshlet.triangleCount;
            set.triangleInput.maxTotalVertexCount += meshlet.vertexCount;
        }
        const GpuMeshlet& last = geometry.meshlet(lod.meshletOffset + lod.meshletCount - 1);
        positionCount += last.vertexOffset + last.vertexCount - firstEntry;
    }
    if (set.triangleInput.maxClusterTriangleCount > props.maxTrianglesPerCluster ||
        set.triangleInput.maxClusterVertexCount > props.maxVerticesPerCluster) {
        core::fatal("meshlet 이 장치의 클러스터 한도를 넘습니다: 삼각형 {} / {}, 정점 {} / {}",
                    set.triangleInput.maxClusterTriangleCount,
                    props.maxTrianglesPerCluster,
                    set.triangleInput.maxClusterVertexCount,
                    props.maxVerticesPerCluster);
    }
    auto clusterCount = static_cast<uint32_t>(clusterInfos.size());

    // 크기. 클러스터는 암시적 목적지라 합이 나오고, 하위 구조는 명시적이라 메쉬마다 따로 묻는다.
    set.clusterBuild =
        VkClusterAccelerationStructureInputInfoNV{VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    set.clusterBuild.maxAccelerationStructureCount = clusterCount;
    set.clusterBuild.flags = flags;
    set.clusterBuild.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_TRIANGLE_CLUSTER_NV;
    set.clusterBuild.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_IMPLICIT_DESTINATIONS_NV;
    set.clusterBuild.opInput.pTriangleClusters = &set.triangleInput;
    VkAccelerationStructureBuildSizesInfoKHR clusterSizes{
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    getClusterBuildSizes(context.device, &set.clusterBuild, &clusterSizes);
    set.clusterScratch = clusterSizes.buildScratchSize;

    set.bottomInput = VkClusterAccelerationStructureClustersBottomLevelInputNV{
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV};
    set.bottomInput.maxTotalClusterCount = clusterCount;
    set.bottomInput.maxClusterCountPerAccelerationStructure = maxClustersPerMesh;
    set.bottomBuild =
        VkClusterAccelerationStructureInputInfoNV{VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    set.bottomBuild.maxAccelerationStructureCount = static_cast<uint32_t>(meshes.size());
    set.bottomBuild.flags = flags;
    set.bottomBuild.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
    set.bottomBuild.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    set.bottomBuild.opInput.pClustersBottomLevel = &set.bottomInput;
    VkAccelerationStructureBuildSizesInfoKHR bottomSizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    getClusterBuildSizes(context.device, &set.bottomBuild, &bottomSizes);
    set.bottomScratch = bottomSizes.buildScratchSize;

    std::vector<VkDeviceSize> bottomBytes(meshes.size());
    VkDeviceSize structureBytes = clusterSizes.accelerationStructureSize;
    for (size_t i = 0; i < meshes.size(); ++i) {
        VkClusterAccelerationStructureClustersBottomLevelInputNV one = set.bottomInput;
        one.maxTotalClusterCount = bottomInfos[i].clusterReferencesCount;
        one.maxClusterCountPerAccelerationStructure = bottomInfos[i].clusterReferencesCount;
        VkClusterAccelerationStructureInputInfoNV query = set.bottomBuild;
        query.maxAccelerationStructureCount = 1;
        query.opInput.pClustersBottomLevel = &one;
        VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        getClusterBuildSizes(context.device, &query, &sizes);
        bottomBytes[i] = sizes.accelerationStructureSize;
        structureBytes += bottomBytes[i];
    }
    VkDeviceSize scratchBytes = std::max(set.clusterScratch, set.bottomScratch);
    // 일반 경로와 달리 입력(모은 위치, 주소 배열, 서술)도 새로 잡는 버퍼라 예산에 넣는다.
    VkDeviceSize positionBytes = VkDeviceSize{positionCount} * sizeof(float) * 3;
    VkDeviceSize inputBytes = positionBytes +
                              VkDeviceSize{clusterCount} * (sizeof(VkDeviceAddress) + sizeof(clusterInfos[0])) +
                              meshes.size() * (sizeof(bottomInfos[0]) + sizeof(VkDeviceAddress));
    constexpr double MB = 1024.0 * 1024.0;
    if (budgetCheck) {
        Context::MemoryBudget budget = context.deviceMemoryBudget();
        VkDeviceSize available = budget.budget > budget.usage ? budget.budget - budget.usage : 0;
        if (structureBytes + inputBytes + scratchBytes > available) {
            reason = std::format("클러스터 하위 가속 구조 {:.0f} MB, 입력 {:.0f} MB, 스크래치 {:.0f} MB 가 남은 GPU "
                                 "예산 {:.0f} MB 를 넘습니다",
                                 static_cast<double>(structureBytes) / MB,
                                 static_cast<double>(inputBytes) / MB,
                                 static_cast<double>(scratchBytes) / MB,
                                 static_cast<double>(available) / MB);
            return false;
        }
    }

    // 버퍼. 목적지는 명시적이라 여기서 잡은 주소가 곧 하위 구조 주소다.
    ensureBuffer(
        context, set.positions, positionBytes, CLUSTER_INPUT_USAGE, MemoryLocation::DEVICE, "클러스터 정점 위치");
    ensureBuffer(context,
                 set.clusterInfos,
                 clusterInfos.size() * sizeof(clusterInfos[0]),
                 CLUSTER_INPUT_USAGE,
                 MemoryLocation::HOST_WRITE,
                 "클러스터 빌드 서술");
    ensureBuffer(context,
                 set.clusterStorage,
                 clusterSizes.accelerationStructureSize,
                 CLUSTER_STORAGE_USAGE,
                 MemoryLocation::DEVICE,
                 "클러스터 가속 구조",
                 props.clusterByteAlignment);
    ensureBuffer(context,
                 set.clusterAddresses,
                 VkDeviceSize{clusterCount} * sizeof(VkDeviceAddress),
                 CLUSTER_INPUT_USAGE,
                 MemoryLocation::DEVICE,
                 "클러스터 주소");
    ensureBuffer(context,
                 set.bottomInfos,
                 bottomInfos.size() * sizeof(bottomInfos[0]),
                 CLUSTER_INPUT_USAGE,
                 MemoryLocation::HOST_WRITE,
                 "클러스터 하위 구조 서술");
    ensureBuffer(context,
                 set.bottomAddresses,
                 meshes.size() * sizeof(VkDeviceAddress),
                 CLUSTER_INPUT_USAGE,
                 MemoryLocation::HOST_WRITE,
                 "클러스터 하위 구조 주소");
    ensureBuffer(context,
                 set.scratch,
                 scratchBytes,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 MemoryLocation::DEVICE,
                 "클러스터 빌드 스크래치",
                 props.clusterScratchByteAlignment);

    std::vector<VkDeviceAddress> bottomAddresses(meshes.size());
    for (size_t i = 0; i < meshes.size(); ++i) {
        AccelerationStructure& destination = *destinations[i];
        // 일반 구조가 들어 있던 자리(핸들이 있다)거나 모자라면 새로 잡는다. 같은 메쉬가 계속 오면 그대로 쓴다.
        if (destination.handle != VK_NULL_HANDLE || destination.storage.size < bottomBytes[i]) {
            retireStructure(destination);
            destination.storage = createBuffer(context,
                                               bottomBytes[i],
                                               CLUSTER_STORAGE_USAGE,
                                               MemoryLocation::DEVICE,
                                               "클러스터 하위 가속 구조",
                                               props.clusterBottomLevelByteAlignment);
        }
        destination.address = destination.storage.address;
        bottomAddresses[i] = destination.address;
        bottomInfos[i].clusterReferences += set.clusterAddresses.address;
    }
    for (VkClusterAccelerationStructureBuildTriangleClusterInfoNV& info : clusterInfos) {
        info.vertexBuffer += set.positions.address;
    }
    for (ClusterSet::Gather& gather : set.gathers) {
        gather.push.positions = set.positions.address;
    }
    std::memcpy(set.clusterInfos.mapped, clusterInfos.data(), clusterInfos.size() * sizeof(clusterInfos[0]));
    std::memcpy(set.bottomInfos.mapped, bottomInfos.data(), bottomInfos.size() * sizeof(bottomInfos[0]));
    std::memcpy(set.bottomAddresses.mapped, bottomAddresses.data(), bottomAddresses.size() * sizeof(VkDeviceAddress));
    return true;
}

void RayTracer::recordClusterBuild(VkCommandBuffer commandBuffer, ClusterSet& set) {
    auto memoryBarrier = [commandBuffer](VkPipelineStageFlags2 srcStage,
                                         VkAccessFlags2 srcAccess,
                                         VkPipelineStageFlags2 dstStage,
                                         VkAccessFlags2 dstAccess) {
        VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        barrier.srcStageMask = srcStage;
        barrier.srcAccessMask = srcAccess;
        barrier.dstStageMask = dstStage;
        barrier.dstAccessMask = dstAccess;
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependency);
    };
    constexpr VkAccessFlags2 BUILD_ACCESS =
        VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

    // 지난 프레임의 빌드가 아직 위치를 읽고 있을 수 있다. 스킨 컴퓨트의 변형 정점 쓰기는 호출자가 막는다.
    memoryBarrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                  VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_WRITE_BIT);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, gatherPipeline);
    for (const ClusterSet::Gather& gather : set.gathers) {
        vkCmdPushConstants(
            commandBuffer, gatherLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gather.push), &gather.push);
        vkCmdDispatch(commandBuffer, gather.groups, 1, 1);
    }
    memoryBarrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                  VK_ACCESS_2_SHADER_READ_BIT | BUILD_ACCESS);
    // 지난 프레임의 추적이 덮어쓸 구조를 아직 읽고 있을 수 있다.
    barrierBeforeBuild(commandBuffer);

    VkClusterAccelerationStructureCommandsInfoNV commands{
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
    commands.input = set.clusterBuild;
    commands.dstImplicitData = set.clusterStorage.address;
    commands.scratchData = set.scratch.address;
    commands.dstAddressesArray = {set.clusterAddresses.address, sizeof(VkDeviceAddress), set.clusterAddresses.size};
    commands.srcInfosArray = {set.clusterInfos.address,
                              sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV),
                              set.clusterInfos.size};
    cmdBuildClusters(commandBuffer, &commands);

    // 하위 구조 빌드가 클러스터와 그 주소 배열을 읽고, 같은 스크래치를 다시 쓴다.
    memoryBarrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                  BUILD_ACCESS,
                  VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                  VK_ACCESS_2_SHADER_READ_BIT | BUILD_ACCESS);

    commands =
        VkClusterAccelerationStructureCommandsInfoNV{VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
    commands.input = set.bottomBuild;
    commands.scratchData = set.scratch.address;
    commands.dstAddressesArray = {set.bottomAddresses.address, sizeof(VkDeviceAddress), set.bottomAddresses.size};
    commands.srcInfosArray = {set.bottomInfos.address,
                              sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV),
                              set.bottomInfos.size};
    cmdBuildClusters(commandBuffer, &commands);

    // 상위 구조가 이 결과를 읽고, 스크래치도 다시 쓴다.
    memoryBarrier(VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                  BUILD_ACCESS,
                  VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                  BUILD_ACCESS);
}

bool RayTracer::buildClusterBottomLevel(std::string& reason) {
    auto buildStart = std::chrono::steady_clock::now();
    uint32_t meshCount = geometry.meshCount();
    bottomLevels.resize(meshCount);
    std::vector<uint32_t> meshes;
    std::vector<AccelerationStructure*> destinations;
    for (uint32_t index = 0; index < meshCount; ++index) {
        // 무덤 메쉬는 세우지 않고 주소를 0 으로 둔다.
        if (geometry.meshLive(index) && geometry.meshVertexCount(index) > 0) {
            meshes.push_back(index);
            destinations.push_back(&bottomLevels[index]);
        }
    }
    if (staticClusters == nullptr) {
        staticClusters = std::make_unique<ClusterSet>();
    }
    if (!meshes.empty()) {
        if (!prepareClusterBuild(*staticClusters,
                                 meshes,
                                 destinations,
                                 nullptr,
                                 {},
                                 VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
                                 true,
                                 reason)) {
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
        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
        recordClusterBuild(commandBuffer, *staticClusters);
        VK_CHECK(vkEndCommandBuffer(commandBuffer));
        VkCommandBufferSubmitInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        commandInfo.commandBuffer = commandBuffer;
        VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &commandInfo;
        VK_CHECK(vkQueueSubmit2(context.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE));
        VK_CHECK(vkQueueWaitIdle(context.graphicsQueue));
        vkDestroyCommandPool(context.device, pool, nullptr);
    }
    bottomLevelBuilt = true;
    double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - buildStart).count();
    constexpr double MB = 1024.0 * 1024.0;
    spdlog::info("클러스터 하위 가속 구조 {}개 생성: 클러스터 {}개 {:.0f} MB, {:.1f} ms",
                 meshes.size(),
                 meshes.empty() ? 0 : staticClusters->clusterBuild.maxAccelerationStructureCount,
                 static_cast<double>(staticClusters->clusterStorage.size) / MB,
                 elapsed);
    return true;
}

void RayTracer::updateSkinnedClusterBottomLevel(VkCommandBuffer commandBuffer,
                                                const Buffer& skinnedVertices,
                                                const std::vector<SkinnedInstance>& skinned,
                                                const std::vector<size_t>& toBuild) {
    // 포즈가 바뀐 슬롯 전부를 한 벌로 세운다. 벌은 진행 중인 프레임 수만큼 돌려 쓴다 — CPU 가 채우는 서술과
    // 컴퓨트가 채우는 위치를 지난 프레임의 빌드가 아직 읽고 있을 수 있다(인스턴스 버퍼와 같은 이유).
    std::unique_ptr<ClusterSet>& set = skinnedClusters[skinnedClusterCursor++ % skinnedClusters.size()];
    if (set == nullptr) {
        set = std::make_unique<ClusterSet>();
    }
    std::vector<uint32_t> meshes;
    std::vector<AccelerationStructure*> destinations;
    std::vector<uint32_t> vertexOffsets;
    for (size_t index : toBuild) {
        meshes.push_back(skinned[index].meshIndex);
        destinations.push_back(&skinnedBottomLevels[index]);
        vertexOffsets.push_back(skinned[index].vertexOffset);
    }
    // 포즈가 바뀌는 프레임마다 다시 세우므로 추적 속도보다 구축 속도를 고른다. 예산은 보지 않는다(일반 스킨 경로와
    // 같다).
    std::string reason;
    prepareClusterBuild(*set,
                        meshes,
                        destinations,
                        &skinnedVertices,
                        vertexOffsets,
                        VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
                        false,
                        reason);
    recordClusterBuild(commandBuffer, *set);
}

} // namespace gfx
