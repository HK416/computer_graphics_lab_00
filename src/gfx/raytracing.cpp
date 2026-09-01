#include "gfx/raytracing.h"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
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
    VkDeviceAddress indices;
    VkDeviceAddress meshes;
    VkDeviceAddress instances;
    VkDeviceAddress materials;
    VkDeviceAddress lods;
    VkDeviceAddress camera;
    VkDeviceAddress lights;
    uint32_t accumulationImage;
    uint32_t outputImage;
    uint32_t frameIndex;
    uint32_t sampleCount;
    uint32_t maxBounces;
    uint32_t samplesPerFrame;
    uint32_t flags;
    float radianceClamp;
    float skyIntensity;
};

constexpr uint32_t PATH_FLAG_NEXT_EVENT = 1;
constexpr uint32_t PATH_FLAG_RUSSIAN_ROULETTE = 2;

uint64_t alignUp(uint64_t value, uint64_t alignment) {
    return (value + alignment - 1) & ~(alignment - 1);
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
    destroyBuffer(context, shaderBindingTable);
    destroyBuffer(context, scratchBuffer);
    destroyBuffer(context, instanceBuffer);
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

void RayTracer::reserveScratch(VkDeviceSize size) {
    if (scratchBuffer.size >= size) {
        return;
    }
    destroyBuffer(context, scratchBuffer);
    scratchBuffer =
        createBuffer(context, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryLocation::DEVICE, "가속 구조 스크래치");
}

void RayTracer::buildBottomLevel() {
    for (AccelerationStructure& structure : bottomLevels) {
        destroyStructure(structure);
    }
    bottomLevels.clear();
    bottomLevels.resize(geometry.meshCount());

    // 빌드는 한 번에 제출한다. 메쉬가 많아도 명령 버퍼 하나로 끝난다.
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

    VkDeviceSize scratchNeeded = 0;
    std::vector<VkAccelerationStructureGeometryKHR> geometries(geometry.meshCount());
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos(geometry.meshCount());
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(geometry.meshCount());
    std::vector<VkDeviceSize> scratchOffsets(geometry.meshCount());

    for (uint32_t index = 0; index < geometry.meshCount(); ++index) {
        const GpuMesh& mesh = geometry.mesh(index);
        const GpuMeshLod& lod = geometry.lod(mesh.lodOffset);

        VkAccelerationStructureGeometryTrianglesDataKHR triangles{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR};
        triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        triangles.vertexData.deviceAddress =
            geometry.vertexBuffer.address + static_cast<VkDeviceSize>(mesh.vertexOffset) * sizeof(asset::Vertex);
        triangles.vertexStride = sizeof(asset::Vertex);
        triangles.maxVertex = lod.indexCount;
        triangles.indexType = VK_INDEX_TYPE_UINT32;
        triangles.indexData.deviceAddress =
            geometry.indexBuffer.address + static_cast<VkDeviceSize>(lod.indexOffset) * sizeof(uint32_t);

        geometries[index] = VkAccelerationStructureGeometryKHR{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geometries[index].geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        geometries[index].geometry.triangles = triangles;
        geometries[index].flags = VK_GEOMETRY_OPAQUE_BIT_KHR;

        buildInfos[index] = VkAccelerationStructureBuildGeometryInfoKHR{
            VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        buildInfos[index].type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfos[index].flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
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

        bottomLevels[index] =
            createStructure(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, sizes.accelerationStructureSize);
        buildInfos[index].dstAccelerationStructure = bottomLevels[index].handle;

        scratchOffsets[index] = scratchNeeded;
        scratchNeeded += alignUp(sizes.buildScratchSize, 256);

        ranges[index] = {};
        ranges[index].primitiveCount = primitiveCount;
    }

    reserveScratch(std::max<VkDeviceSize>(scratchNeeded, 256));

    std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> rangePointers(geometry.meshCount());
    for (uint32_t index = 0; index < geometry.meshCount(); ++index) {
        buildInfos[index].scratchData.deviceAddress = scratchBuffer.address + scratchOffsets[index];
        rangePointers[index] = &ranges[index];
    }
    cmdBuildAccelerationStructures(
        commandBuffer, static_cast<uint32_t>(buildInfos.size()), buildInfos.data(), rangePointers.data());

    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VkCommandBufferSubmitInfo commandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    commandInfo.commandBuffer = commandBuffer;
    VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandInfo;
    VK_CHECK(vkQueueSubmit2(context.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(context.graphicsQueue));
    vkDestroyCommandPool(context.device, pool, nullptr);

    spdlog::info("하위 가속 구조 {}개 생성", bottomLevels.size());
}

void RayTracer::updateTopLevel(VkCommandBuffer commandBuffer,
                               const scene::Scene& sceneToTrace,
                               const std::vector<uint32_t>& instanceSlots) {
    std::vector<VkAccelerationStructureInstanceKHR> instances;
    instances.reserve(sceneToTrace.objects.size());

    for (uint32_t index = 0; index < sceneToTrace.objects.size(); ++index) {
        const scene::Object& object = sceneToTrace.objects[index];
        if (index >= instanceSlots.size() || instanceSlots[index] == INVALID_INSTANCE_SLOT ||
            object.meshIndex >= bottomLevels.size()) {
            continue;
        }
        glm::mat4 model = glm::transpose(sceneToTrace.world(index));

        VkAccelerationStructureInstanceKHR instance{};
        std::memcpy(&instance.transform, &model, sizeof(VkTransformMatrixKHR));
        // 적중 셰이더가 인스턴스 배열을 찾는 번호. 그리기 인스턴스는 버킷 순서로 채워지므로
        // 장면 순서로 매기면 어긋난다. buildDrawCommands 가 만든 슬롯을 그대로 쓴다.
        instance.instanceCustomIndex = instanceSlots[index];
        instance.mask = 0xFF;
        instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        instance.accelerationStructureReference = bottomLevels[object.meshIndex].address;
        instances.push_back(instance);
    }
    if (instances.empty()) {
        return;
    }

    if (instances.size() > instanceCapacity) {
        destroyBuffer(context, instanceBuffer);
        instanceCapacity = static_cast<uint32_t>(instances.size()) * 2;
        instanceBuffer =
            createBuffer(context,
                         static_cast<VkDeviceSize>(instanceCapacity) * sizeof(VkAccelerationStructureInstanceKHR),
                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                         MemoryLocation::HOST_WRITE,
                         "가속 구조 인스턴스");
    }
    std::memcpy(instanceBuffer.mapped, instances.data(), instances.size() * sizeof(VkAccelerationStructureInstanceKHR));

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
    reserveScratch(std::max<VkDeviceSize>(sizes.buildScratchSize, 256));

    buildInfo.dstAccelerationStructure = topLevel.handle;
    buildInfo.scratchData.deviceAddress = scratchBuffer.address;

    VkAccelerationStructureBuildRangeInfoKHR range{};
    range.primitiveCount = instanceCount;
    const VkAccelerationStructureBuildRangeInfoKHR* rangePointer = &range;
    cmdBuildAccelerationStructures(commandBuffer, 1, &buildInfo, &rangePointer);

    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
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
    binding.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;

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
    pushConstantRange.stageFlags =
        VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
    pushConstantRange.size = sizeof(PathTracePushConstants);

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

    std::array<VkPipelineShaderStageCreateInfo, 4> stages{};
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

    VkRayTracingPipelineCreateInfoKHR pipelineInfo{VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR};
    pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
    pipelineInfo.pStages = stages.data();
    pipelineInfo.groupCount = static_cast<uint32_t>(groups.size());
    pipelineInfo.pGroups = groups.data();
    pipelineInfo.maxPipelineRayRecursionDepth = 1;
    pipelineInfo.layout = pipelineLayout;
    VK_CHECK(createRayTracingPipelines(
        context.device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));

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
                      uint32_t accumulationImage,
                      uint32_t outputImage,
                      uint32_t frameIndex,
                      uint32_t sampleCount,
                      const PathTraceOptions& options) {
    PathTracePushConstants pushConstants{};
    pushConstants.vertices = geometry.vertexBuffer.address;
    pushConstants.indices = geometry.indexBuffer.address;
    pushConstants.meshes = geometry.meshBuffer.address;
    pushConstants.instances = instanceAddress;
    pushConstants.materials = geometry.materialBuffer.address;
    pushConstants.lods = geometry.lodBuffer.address;
    pushConstants.camera = cameraAddress;
    pushConstants.lights = lightAddress;
    pushConstants.accumulationImage = accumulationImage;
    pushConstants.outputImage = outputImage;
    pushConstants.frameIndex = frameIndex;
    pushConstants.sampleCount = sampleCount;
    pushConstants.maxBounces = options.maxBounces;
    pushConstants.samplesPerFrame = options.samplesPerFrame;
    pushConstants.flags = (options.nextEventEstimation ? PATH_FLAG_NEXT_EVENT : 0U) |
                          (options.russianRoulette ? PATH_FLAG_RUSSIAN_ROULETTE : 0U);
    pushConstants.radianceClamp = options.radianceClamp;
    pushConstants.skyIntensity = options.skyIntensity;

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
                           VK_SHADER_STAGE_MISS_BIT_KHR,
                       0,
                       sizeof(pushConstants),
                       &pushConstants);
    cmdTraceRays(
        commandBuffer, &raygenRegion, &missRegion, &hitRegion, &callableRegion, extent.width, extent.height, 1);
}

} // namespace gfx
