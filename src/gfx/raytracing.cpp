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

// shaders/cluster_select.comp 의 ClusterSelectObject·푸시 상수와 배치가 같아야 한다.
struct ClusterSelectObject {
    VkDeviceAddress clusterAddresses;
    VkDeviceAddress references;
    VkDeviceAddress info;
    uint32_t instanceSlot;
    uint32_t clusterBase;
};
struct ClusterSelectPushConstants {
    VkDeviceAddress instances;
    VkDeviceAddress meshes;
    VkDeviceAddress meshlets;
    VkDeviceAddress camera;
    VkDeviceAddress network;
    VkDeviceAddress objects;
    uint32_t objectCount;
    uint32_t useNetwork;
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
    Buffer clusterStorage;   // 암시적 목적지
    Buffer clusterAddresses; // 클러스터 주소. 클러스터 빌드가 쓰고, 선택 컴퓨트와 하위 구조 빌드가 읽는다
    // VkClusterAccelerationStructureBuildTriangleClusterInfoNV[]. CPU 가 쓰므로 지난 프레임의 빌드가 아직 읽고 있을
    // 수 있어 프레임 수만큼 돌려 쓴다.
    std::array<Buffer, 3> clusterInfos;
    uint32_t activeInfos = 0;
    Buffer scratch;
    VkClusterAccelerationStructureTriangleClusterInputNV triangleInput{
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV};
    VkClusterAccelerationStructureInputInfoNV clusterBuild{
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    struct Gather {
        ClusterGatherPushConstants push;
        uint32_t groups;
    };
    std::vector<Gather> gathers;
    // 벌에 든 메쉬와 clusterAddresses 안에서 그 메쉬의 첫 클러스터 자리. 비어 있으면 아직 세우지 않은 것이다.
    struct MeshEntry {
        uint32_t mesh;
        uint32_t clusterBase;
    };
    std::vector<MeshEntry> meshes;
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
        VkPushConstantRange selectRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ClusterSelectPushConstants)};
        layoutInfo.pPushConstantRanges = &selectRange;
        VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &selectLayout));
        selectPipeline = createComputePipeline(context, selectLayout, "cluster_select.comp.spv");
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
        for (Buffer* buffer : {&set->positions, &set->clusterStorage, &set->clusterAddresses, &set->scratch}) {
            destroyBuffer(context, *buffer);
        }
        for (Buffer& buffer : set->clusterInfos) {
            destroyBuffer(context, buffer);
        }
    };
    for (AccelerationStructure& structure : objectBottomLevels) {
        destroyStructure(structure);
    }
    for (ClusterSelectBuffers& buffers : selectBuffers) {
        for (Buffer* buffer : {&buffers.objects,
                               &buffers.bottomAddresses,
                               &buffers.references,
                               &buffers.bottomInfos,
                               &buffers.scratch}) {
            destroyBuffer(context, *buffer);
        }
    }
    vkDestroyPipeline(context.device, selectPipeline, nullptr);
    vkDestroyPipelineLayout(context.device, selectLayout, nullptr);
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
    // 오브젝트 하위 구조는 메쉬 크기에 맞춰 잡은 것이라 지오메트리가 바뀌면 함께 버린다.
    for (AccelerationStructure& structure : objectBottomLevels) {
        destroyStructure(structure);
    }
    objectBottomLevels.clear();
    staticClusterBase.clear();
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
    skinnedRebuilt.assign(skinned.size(), 0);
    if (skinned.empty()) {
        return;
    }

    if (skinnedBottomLevels.size() < skinned.size()) {
        skinnedBottomLevels.resize(skinned.size());
    }
    // 포즈가 바뀐 것과 아직 구조가 없는 것만 세운다. 번호는 skinnedBottomLevels 와 같다.
    std::vector<size_t> toBuild;
    for (size_t index = 0; index < skinned.size(); ++index) {
        bool built = useClusters ? skinnedClusterBuilt(index) : skinnedBottomLevels[index].address != 0;
        if (skinned[index].rebuild || !built) {
            toBuild.push_back(index);
            skinnedRebuilt[index] = 1;
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
                               uint32_t prependedInstances,
                               bool refitAllowed) {
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    instances.reserve(sceneToTrace.objects.size());

    for (uint32_t index = 0; index < sceneToTrace.objects.size(); ++index) {
        uint32_t mesh = sceneToTrace.meshOf(index);
        if (index >= instanceSlots.size() || instanceSlots[index] == INVALID_INSTANCE_SLOT) {
            continue;
        }
        uint32_t skinnedSlot = index < skinnedBlasSlots.size() ? skinnedBlasSlots[index] : NO_SKINNED_BLAS;
        VkDeviceAddress blasAddress = 0;
        if (useClusters) {
            // 클러스터 모드는 오브젝트마다 이번 프레임의 LOD 컷으로 세운 구조를 가리킨다(selectClusters).
            if (index >= objectHasClusters.size() || objectHasClusters[index] == 0) {
                continue;
            }
            blasAddress = objectBottomLevels[index].address;
        } else {
            if (mesh >= bottomLevels.size() || bottomLevels[mesh].address == 0) {
                continue;
            }
            // 스킨 오브젝트는 이번 프레임의 포즈로 다시 세운 구조를 가리킨다. 바인드 포즈 구조를
            // 그대로 두면 화면에서만 움직이고 광선은 서 있는 몸을 맞힌다.
            blasAddress = bottomLevels[mesh].address;
            if (skinnedSlot != NO_SKINNED_BLAS && skinnedSlot < skinnedBottomLevels.size() &&
                skinnedBottomLevels[skinnedSlot].address != 0) {
                blasAddress = skinnedBottomLevels[skinnedSlot].address;
            }
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
    // 클러스터 모드는 부분 재구축 프레임에 제자리 갱신을 쓰므로 그 허용을 걸고 세운다. 일반 모드는 그대로 둔다.
    if (useClusters) {
        buildInfo.flags |= VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    }
    buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &geometryInfo;

    auto instanceCount = prependedInstances + static_cast<uint32_t>(instances.size());
    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    getBuildSizes(context.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &instanceCount, &sizes);

    // 갱신은 같은 구조(ALLOW_UPDATE 로 세운 것)에 같은 수의 인스턴스일 때만 된다. 변환과 하위 구조 참조는 바뀌어도
    // 된다.
    bool refit = refitAllowed && useClusters && topLevel.handle != VK_NULL_HANDLE &&
                 topLevelInstanceCount == instanceCount && topLevel.storage.size >= sizes.accelerationStructureSize;
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
    reserveScratch(scratchBuffer,
                   std::max<VkDeviceSize>(refit ? sizes.updateScratchSize : sizes.buildScratchSize, 256),
                   "가속 구조 스크래치");

    if (refit) {
        buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        buildInfo.srcAccelerationStructure = topLevel.handle;
    }
    buildInfo.dstAccelerationStructure = topLevel.handle;
    buildInfo.scratchData.deviceAddress = scratchBuffer.address;
    topLevelInstanceCount = instanceCount;

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
// 그 둘이 짝이 된다. 메쉬의 **모든 LOD 단계** meshlet 을 클러스터로 세워 두고(정적 메쉬는 한 번, 스킨 슬롯은 포즈가
// 바뀔 때), 오브젝트의 하위 구조는 프레임마다 cluster_select.comp 가 래스터와 같은 LOD 컷으로 고른 클러스터들로
// 짓는다 — 그래서 광선 경로가 mesh shader 경로와 같은 meshlet 을 본다. 클러스터는 암시적 목적지로 한 버퍼에 몰아
// 세우고(주소는 장치가 배열에 써 준다), 오브젝트의 하위 구조는 명시적 목적지로 세운다 — CPU 가 상위 구조 인스턴스에
// 그 주소를 써야 하므로 미리 알아야 한다. 클러스터 번호는 meshlet 의 첫 삼각형 번호(indexOffset / 3)라, 히트 셰이더가
// 표 없이 인덱스 버퍼로 바로 간다.

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
constexpr VkAccessFlags2 BUILD_ACCESS =
    VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;

void memoryBarrier2(VkCommandBuffer commandBuffer,
                    VkPipelineStageFlags2 srcStage,
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
}

} // namespace

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
    for (AccelerationStructure& structure : objectBottomLevels) {
        retireStructure(structure);
    }
    for (std::unique_ptr<ClusterSet>& set : skinnedClusters) {
        if (set != nullptr) {
            set->meshes.clear();
        }
    }
}

bool RayTracer::skinnedClusterBuilt(size_t slot) const {
    return slot < skinnedClusters.size() && skinnedClusters[slot] != nullptr && !skinnedClusters[slot]->meshes.empty();
}

bool RayTracer::prepareClusterBuild(ClusterSet& set,
                                    const std::vector<uint32_t>& meshes,
                                    const Buffer* skinnedVertices,
                                    const std::vector<uint32_t>& skinnedVertexOffsets,
                                    VkBuildAccelerationStructureFlagsKHR flags,
                                    bool budgetCheck,
                                    std::string& reason) {
    const VkPhysicalDeviceClusterAccelerationStructurePropertiesNV& props = context.clusterProperties;
    std::vector<VkClusterAccelerationStructureBuildTriangleClusterInfoNV> clusterInfos;
    set.gathers.clear();
    set.meshes.clear();
    set.triangleInput = VkClusterAccelerationStructureTriangleClusterInputNV{
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_TRIANGLE_CLUSTER_INPUT_NV};
    set.triangleInput.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    set.triangleInput.maxClusterUniqueGeometryCount = 1;
    uint32_t positionCount = 0;

    for (size_t i = 0; i < meshes.size(); ++i) {
        const GpuMesh& mesh = geometry.mesh(meshes[i]);
        bool opaque = geometryFlagsFor(geometry.material(mesh.materialIndex)) == VK_GEOMETRY_OPAQUE_BIT_KHR;
        // 모든 LOD 단계. 메쉬의 meshlet 과 그 정점 목록은 단계 순서로 이어져 있다.
        uint32_t firstEntry = geometry.meshlet(mesh.meshletOffset).vertexOffset;
        set.meshes.push_back({meshes[i], static_cast<uint32_t>(clusterInfos.size())});

        ClusterSet::Gather gather{};
        gather.push.vertices = skinnedVertices != nullptr ? skinnedVertices->address : geometry.vertexBuffer.address;
        gather.push.meshlets = geometry.meshletBuffer.address;
        gather.push.meshletVertices = geometry.meshletVertexBuffer.address;
        gather.push.meshletOffset = mesh.meshletOffset;
        gather.push.meshletCount = mesh.meshletCount;
        gather.push.meshVertexOffset = skinnedVertices != nullptr ? static_cast<uint32_t>(mesh.vertexOffset) : 0;
        gather.push.sourceVertexOffset = skinnedVertices != nullptr ? skinnedVertexOffsets[i] : 0;
        gather.push.meshletVertexBase = firstEntry;
        gather.push.positionBase = positionCount;
        gather.groups = mesh.meshletCount;
        set.gathers.push_back(gather);

        for (uint32_t m = 0; m < mesh.meshletCount; ++m) {
            const GpuMeshlet& meshlet = geometry.meshlet(mesh.meshletOffset + m);
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
            // 위치 버퍼는 아직 없을 수 있어 아래서 기준 주소를 더한다.
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
        const GpuMeshlet& last = geometry.meshlet(mesh.meshletOffset + mesh.meshletCount - 1);
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

    // 암시적 목적지라 크기가 합으로 나온다.
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

    // 일반 경로와 달리 입력(모은 위치, 주소 배열, 서술)도 새로 잡는 버퍼라 예산에 넣는다.
    VkDeviceSize positionBytes = VkDeviceSize{positionCount} * sizeof(float) * 3;
    VkDeviceSize infoBytes = clusterInfos.size() * sizeof(clusterInfos[0]);
    VkDeviceSize addressBytes = VkDeviceSize{clusterCount} * sizeof(VkDeviceAddress);
    constexpr double MB = 1024.0 * 1024.0;
    if (budgetCheck) {
        Context::MemoryBudget budget = context.deviceMemoryBudget();
        VkDeviceSize available = budget.budget > budget.usage ? budget.budget - budget.usage : 0;
        VkDeviceSize needed = clusterSizes.accelerationStructureSize + clusterSizes.buildScratchSize + positionBytes +
                              infoBytes * set.clusterInfos.size() + addressBytes;
        if (needed > available) {
            reason =
                std::format("클러스터 가속 구조 {:.0f} MB(입력·스크래치 포함)가 남은 GPU 예산 {:.0f} MB 를 넘습니다",
                            static_cast<double>(needed) / MB,
                            static_cast<double>(available) / MB);
            set.meshes.clear();
            return false;
        }
    }

    ensureBuffer(
        context, set.positions, positionBytes, CLUSTER_INPUT_USAGE, MemoryLocation::DEVICE, "클러스터 정점 위치");
    ensureBuffer(context,
                 set.clusterStorage,
                 clusterSizes.accelerationStructureSize,
                 CLUSTER_STORAGE_USAGE,
                 MemoryLocation::DEVICE,
                 "클러스터 가속 구조",
                 props.clusterByteAlignment);
    ensureBuffer(
        context, set.clusterAddresses, addressBytes, CLUSTER_INPUT_USAGE, MemoryLocation::DEVICE, "클러스터 주소");
    ensureBuffer(context,
                 set.scratch,
                 clusterSizes.buildScratchSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 MemoryLocation::DEVICE,
                 "클러스터 빌드 스크래치",
                 props.clusterScratchByteAlignment);
    // CPU 가 쓰는 서술은 지난 프레임의 빌드가 아직 읽고 있을 수 있어 프레임 수만큼 돌려 쓴다.
    set.activeInfos = (set.activeInfos + 1) % static_cast<uint32_t>(set.clusterInfos.size());
    Buffer& infos = set.clusterInfos[set.activeInfos];
    ensureBuffer(context, infos, infoBytes, CLUSTER_INPUT_USAGE, MemoryLocation::HOST_WRITE, "클러스터 빌드 서술");

    for (VkClusterAccelerationStructureBuildTriangleClusterInfoNV& info : clusterInfos) {
        info.vertexBuffer += set.positions.address;
    }
    for (ClusterSet::Gather& gather : set.gathers) {
        gather.push.positions = set.positions.address;
    }
    std::memcpy(infos.mapped, clusterInfos.data(), infoBytes);
    return true;
}

void RayTracer::recordClusterBuild(VkCommandBuffer commandBuffer, ClusterSet& set) {
    // 지난 프레임의 빌드가 아직 위치를 읽고 있을 수 있다. 스킨 컴퓨트의 변형 정점 쓰기는 호출자가 막는다.
    memoryBarrier2(commandBuffer,
                   VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                   VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_WRITE_BIT);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, gatherPipeline);
    for (const ClusterSet::Gather& gather : set.gathers) {
        vkCmdPushConstants(
            commandBuffer, gatherLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gather.push), &gather.push);
        vkCmdDispatch(commandBuffer, gather.groups, 1, 1);
    }
    memoryBarrier2(commandBuffer,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                   VK_ACCESS_2_SHADER_READ_BIT | BUILD_ACCESS);
    // 지난 프레임의 추적이 덮어쓸 클러스터를 아직 읽고 있을 수 있다.
    barrierBeforeBuild(commandBuffer);

    const Buffer& infos = set.clusterInfos[set.activeInfos];
    VkClusterAccelerationStructureCommandsInfoNV commands{
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
    commands.input = set.clusterBuild;
    commands.dstImplicitData = set.clusterStorage.address;
    commands.scratchData = set.scratch.address;
    commands.dstAddressesArray = {set.clusterAddresses.address, sizeof(VkDeviceAddress), set.clusterAddresses.size};
    commands.srcInfosArray = {
        infos.address, sizeof(VkClusterAccelerationStructureBuildTriangleClusterInfoNV), infos.size};
    cmdBuildClusters(commandBuffer, &commands);

    // 선택 컴퓨트가 주소 배열을 읽고, 하위 구조 빌드가 클러스터를 읽는다.
    memoryBarrier2(commandBuffer,
                   VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                   BUILD_ACCESS,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                   VK_ACCESS_2_SHADER_READ_BIT | BUILD_ACCESS);
}

bool RayTracer::buildClusterBottomLevel(std::string& reason) {
    auto buildStart = std::chrono::steady_clock::now();
    uint32_t meshCount = geometry.meshCount();
    std::vector<uint32_t> meshes;
    for (uint32_t index = 0; index < meshCount; ++index) {
        // 무덤 메쉬는 세우지 않는다.
        if (geometry.meshLive(index) && geometry.meshVertexCount(index) > 0) {
            meshes.push_back(index);
        }
    }
    if (staticClusters == nullptr) {
        staticClusters = std::make_unique<ClusterSet>();
    }
    staticClusters->meshes.clear();
    if (!meshes.empty()) {
        if (!prepareClusterBuild(*staticClusters,
                                 meshes,
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
        // 클러스터는 자기 완결이라 입력(모은 위치, 서술, 스크래치)은 빌드가 끝나면 아무도 읽지 않는다. 정점 버퍼와
        // 맞먹는 크기라 바로 놓는다. 다음 빌드가 다시 잡는다.
        destroyBuffer(context, staticClusters->positions);
        destroyBuffer(context, staticClusters->scratch);
        for (Buffer& buffer : staticClusters->clusterInfos) {
            destroyBuffer(context, buffer);
        }
    }
    // 메쉬 번호 -> 정적 벌 안의 첫 클러스터 자리.
    staticClusterBase.assign(meshCount, NO_CLUSTERS);
    for (const ClusterSet::MeshEntry& entry : staticClusters->meshes) {
        staticClusterBase[entry.mesh] = entry.clusterBase;
    }
    clusterBottomBytes.assign(meshCount, 0);
    bottomLevelBuilt = true;
    double elapsed = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - buildStart).count();
    constexpr double MB = 1024.0 * 1024.0;
    spdlog::info("클러스터 {}개 생성(메쉬 {}개, 모든 LOD): {:.0f} MB, {:.1f} ms",
                 meshes.empty() ? 0 : staticClusters->clusterBuild.maxAccelerationStructureCount,
                 meshes.size(),
                 static_cast<double>(staticClusters->clusterStorage.size) / MB,
                 elapsed);
    return true;
}

void RayTracer::updateSkinnedClusterBottomLevel(VkCommandBuffer commandBuffer,
                                                const Buffer& skinnedVertices,
                                                const std::vector<SkinnedInstance>& skinned,
                                                const std::vector<size_t>& toBuild) {
    if (skinnedClusters.size() < skinned.size()) {
        skinnedClusters.resize(skinned.size());
    }
    std::string reason;
    for (size_t index : toBuild) {
        if (skinnedClusters[index] == nullptr) {
            skinnedClusters[index] = std::make_unique<ClusterSet>();
        }
        // 슬롯마다 클러스터 저장을 따로 둔다 — 오브젝트 하위 구조가 그 클러스터를 가리키므로 포즈가 그대로인 슬롯의
        // 클러스터는 살아 있어야 한다. 포즈가 바뀌는 프레임마다 다시 세우므로 추적 속도보다 구축 속도를 고른다.
        prepareClusterBuild(*skinnedClusters[index],
                            {skinned[index].meshIndex},
                            &skinnedVertices,
                            {skinned[index].vertexOffset},
                            VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR,
                            false,
                            reason);
        recordClusterBuild(commandBuffer, *skinnedClusters[index]);
    }
}

bool RayTracer::selectClusters(VkCommandBuffer commandBuffer,
                               const scene::Scene& sceneToTrace,
                               const std::vector<uint32_t>& instanceSlots,
                               const std::vector<uint32_t>& skinnedBlasSlots,
                               uint32_t frameSlot,
                               const ClusterSelection& selection,
                               const std::vector<uint8_t>* transformChanged,
                               std::string& reason) {
    const VkPhysicalDeviceClusterAccelerationStructurePropertiesNV& props = context.clusterProperties;
    size_t objectCount = sceneToTrace.objects.size();
    if (objectBottomLevels.size() < objectCount) {
        objectBottomLevels.resize(objectCount);
    }
    // 부분 재구축은 지난 프레임의 «세웠다» 표시를 남기고, 다시 세우는 것만 새로 표시한다.
    bool partial = transformChanged != nullptr && objectHasClusters.size() == objectCount;
    if (!partial) {
        objectHasClusters.assign(objectCount, 0);
    }

    // 오브젝트마다 클러스터 주소 배열(정적 벌 또는 스킨 슬롯 벌)과 참조 목록 자리를 정한다.
    std::vector<ClusterSelectObject> objects;
    std::vector<VkDeviceAddress> bottomAddresses;
    std::vector<uint32_t> objectIndices;
    VkDeviceSize referenceBytes = 0;
    uint32_t maxClusters = 0;
    uint32_t totalClusters = 0;
    for (uint32_t index = 0; index < objectCount; ++index) {
        uint32_t mesh = sceneToTrace.meshOf(index);
        uint32_t skinnedSlot = index < skinnedBlasSlots.size() ? skinnedBlasSlots[index] : NO_SKINNED_BLAS;
        if (partial && objectHasClusters[index] != 0 && (*transformChanged)[index] == 0 &&
            (skinnedSlot == NO_SKINNED_BLAS || skinnedSlot >= skinnedRebuilt.size() ||
             skinnedRebuilt[skinnedSlot] == 0)) {
            // 변환도 포즈도 그대로라 지난 구조가 아직 맞는다.
            continue;
        }
        objectHasClusters[index] = 0;
        if (index >= instanceSlots.size() || instanceSlots[index] == INVALID_INSTANCE_SLOT ||
            mesh >= staticClusterBase.size() || !geometry.meshLive(mesh)) {
            continue;
        }
        ClusterSelectObject object{};
        if (skinnedSlot != NO_SKINNED_BLAS) {
            if (!skinnedClusterBuilt(skinnedSlot)) {
                continue;
            }
            object.clusterAddresses = skinnedClusters[skinnedSlot]->clusterAddresses.address;
            object.clusterBase = skinnedClusters[skinnedSlot]->meshes[0].clusterBase;
        } else {
            if (staticClusterBase[mesh] == NO_CLUSTERS) {
                continue;
            }
            object.clusterAddresses = staticClusters->clusterAddresses.address;
            object.clusterBase = staticClusterBase[mesh];
        }
        object.instanceSlot = instanceSlots[index];
        uint32_t clusterCount = geometry.mesh(mesh).meshletCount;
        // 참조 목록과 서술의 자리는 아래서 기준 주소를 더한다.
        object.references = referenceBytes;
        object.info = objects.size() * sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV);
        referenceBytes += VkDeviceSize{clusterCount} * sizeof(VkDeviceAddress);
        maxClusters = std::max(maxClusters, clusterCount);
        totalClusters += clusterCount;

        // 오브젝트의 하위 구조 저장. 메쉬의 모든 meshlet 이 뽑히는 경우로 잡아 어떤 컷도 들어간다. 크기는 메쉬의
        // meshlet 수로 묻는다 — 구조 크기는 그 안의 클러스터 수가 정하므로, 한 빌드에 더 큰 메쉬가 섞여 있어도 이
        // 메쉬의 구조는 이 크기를 넘지 않는다.
        if (clusterBottomBytes[mesh] == 0) {
            VkClusterAccelerationStructureClustersBottomLevelInputNV one{
                VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV};
            one.maxTotalClusterCount = clusterCount;
            one.maxClusterCountPerAccelerationStructure = clusterCount;
            VkClusterAccelerationStructureInputInfoNV query{
                VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
            query.maxAccelerationStructureCount = 1;
            query.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
            query.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
            query.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
            query.opInput.pClustersBottomLevel = &one;
            VkAccelerationStructureBuildSizesInfoKHR sizes{
                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
            getClusterBuildSizes(context.device, &query, &sizes);
            clusterBottomBytes[mesh] = sizes.accelerationStructureSize;
        }
        AccelerationStructure& destination = objectBottomLevels[index];
        if (destination.handle != VK_NULL_HANDLE || destination.storage.size < clusterBottomBytes[mesh]) {
            // 인스턴스마다 잡는 것이라 정적 빌드의 예산 검사가 세지 못했다. 넘기면 할당이 시스템 메모리로 넘어가
            // 빌드 중 장치를 잃으므로 여기서 거른다. 호출자가 사유를 보이고 광선 기능을 끈다.
            Context::MemoryBudget budget = context.deviceMemoryBudget();
            VkDeviceSize available = budget.budget > budget.usage ? budget.budget - budget.usage : 0;
            if (clusterBottomBytes[mesh] > available) {
                constexpr double MB = 1024.0 * 1024.0;
                reason =
                    std::format("오브젝트 클러스터 하위 가속 구조 {:.0f} MB 가 남은 GPU 예산 {:.0f} MB 를 넘습니다",
                                static_cast<double>(clusterBottomBytes[mesh]) / MB,
                                static_cast<double>(available) / MB);
                return false;
            }
            retireStructure(destination);
            destination.storage = createBuffer(context,
                                               clusterBottomBytes[mesh],
                                               CLUSTER_STORAGE_USAGE,
                                               MemoryLocation::DEVICE,
                                               "클러스터 하위 가속 구조",
                                               props.clusterBottomLevelByteAlignment);
        }
        destination.address = destination.storage.address;
        bottomAddresses.push_back(destination.address);
        objects.push_back(object);
        objectIndices.push_back(index);
        objectHasClusters[index] = 1;
    }
    if (objects.empty()) {
        return true;
    }

    VkClusterAccelerationStructureInputInfoNV build{VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_INPUT_INFO_NV};
    VkClusterAccelerationStructureClustersBottomLevelInputNV bottomInput{
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_CLUSTERS_BOTTOM_LEVEL_INPUT_NV};
    bottomInput.maxTotalClusterCount = totalClusters;
    bottomInput.maxClusterCountPerAccelerationStructure = maxClusters;
    build.maxAccelerationStructureCount = static_cast<uint32_t>(objects.size());
    build.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    build.opType = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_TYPE_BUILD_CLUSTERS_BOTTOM_LEVEL_NV;
    build.opMode = VK_CLUSTER_ACCELERATION_STRUCTURE_OP_MODE_EXPLICIT_DESTINATIONS_NV;
    build.opInput.pClustersBottomLevel = &bottomInput;
    VkAccelerationStructureBuildSizesInfoKHR sizes{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    getClusterBuildSizes(context.device, &build, &sizes);

    ClusterSelectBuffers& buffers = selectBuffers[frameSlot];
    VkDeviceSize infoBytes = objects.size() * sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV);
    ensureBuffer(context,
                 buffers.objects,
                 objects.size() * sizeof(ClusterSelectObject),
                 CLUSTER_INPUT_USAGE,
                 MemoryLocation::HOST_WRITE,
                 "클러스터 선택 오브젝트");
    ensureBuffer(context,
                 buffers.bottomAddresses,
                 bottomAddresses.size() * sizeof(VkDeviceAddress),
                 CLUSTER_INPUT_USAGE,
                 MemoryLocation::HOST_WRITE,
                 "클러스터 하위 구조 주소");
    ensureBuffer(
        context, buffers.references, referenceBytes, CLUSTER_INPUT_USAGE, MemoryLocation::DEVICE, "클러스터 참조 목록");
    ensureBuffer(context,
                 buffers.bottomInfos,
                 infoBytes,
                 CLUSTER_INPUT_USAGE,
                 MemoryLocation::DEVICE,
                 "클러스터 하위 구조 서술");
    ensureBuffer(context,
                 buffers.scratch,
                 sizes.buildScratchSize,
                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                 MemoryLocation::DEVICE,
                 "클러스터 하위 구조 스크래치",
                 props.clusterScratchByteAlignment);
    for (ClusterSelectObject& object : objects) {
        object.references += buffers.references.address;
        object.info += buffers.bottomInfos.address;
    }
    std::memcpy(buffers.objects.mapped, objects.data(), objects.size() * sizeof(ClusterSelectObject));
    std::memcpy(
        buffers.bottomAddresses.mapped, bottomAddresses.data(), bottomAddresses.size() * sizeof(VkDeviceAddress));

    // 지난 프레임의 하위 구조 빌드가 아직 참조 목록·서술을 읽고 있을 수 있고, 클러스터 빌드가 주소 배열을 썼다.
    memoryBarrier2(commandBuffer,
                   VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                   BUILD_ACCESS,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT);
    ClusterSelectPushConstants push{};
    push.instances = selection.instances;
    push.meshes = geometry.meshBuffer.address;
    push.meshlets = geometry.meshletBuffer.address;
    push.camera = selection.camera;
    push.network = selection.network;
    push.objects = buffers.objects.address;
    push.objectCount = static_cast<uint32_t>(objects.size());
    push.useNetwork = selection.useNetwork ? 1U : 0U;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, selectPipeline);
    vkCmdPushConstants(commandBuffer, selectLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(commandBuffer, push.objectCount, 1, 1);
    memoryBarrier2(commandBuffer,
                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                   VK_ACCESS_2_SHADER_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                   VK_ACCESS_2_SHADER_READ_BIT | BUILD_ACCESS);
    // 지난 프레임의 추적이 제자리에서 덮어쓸 하위 구조를 아직 읽고 있을 수 있다.
    barrierBeforeBuild(commandBuffer);

    VkClusterAccelerationStructureCommandsInfoNV commands{
        VK_STRUCTURE_TYPE_CLUSTER_ACCELERATION_STRUCTURE_COMMANDS_INFO_NV};
    commands.input = build;
    commands.scratchData = buffers.scratch.address;
    commands.dstAddressesArray = {
        buffers.bottomAddresses.address, sizeof(VkDeviceAddress), buffers.bottomAddresses.size};
    commands.srcInfosArray = {buffers.bottomInfos.address,
                              sizeof(VkClusterAccelerationStructureBuildClustersBottomLevelInfoNV),
                              buffers.bottomInfos.size};
    cmdBuildClusters(commandBuffer, &commands);

    // 상위 구조가 이 결과를 읽는다.
    memoryBarrier2(commandBuffer,
                   VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                   BUILD_ACCESS,
                   VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                   BUILD_ACCESS);
    return true;
}

} // namespace gfx
