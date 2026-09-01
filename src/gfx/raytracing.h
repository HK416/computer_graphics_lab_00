#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "gfx/resources.h"

namespace scene {
struct Scene;
}

namespace gfx {

struct Context;
class BindlessTextures;
class GeometryStore;

// 레이트레이싱 가속 구조와 경로 추적 파이프라인.
// 하드웨어가 VK_KHR_ray_tracing_pipeline 을 지원할 때만 만들어지며, 폴백 경로는 두지 않는다.
class RayTracer {
    struct AccelerationStructure {
        VkAccelerationStructureKHR handle = VK_NULL_HANDLE;
        Buffer storage;
        VkDeviceAddress address = 0;
    };

public:
    RayTracer(Context& context, GeometryStore& geometry, BindlessTextures& bindless);
    ~RayTracer();
    RayTracer(const RayTracer&) = delete;
    RayTracer& operator=(const RayTracer&) = delete;

    // 메쉬마다 0단계 LOD 로 하위 가속 구조를 만든다. 적재가 끝난 뒤 한 번 호출한다.
    void buildBottomLevel();
    // 장면 인스턴스로 상위 가속 구조를 다시 만든다. 매 프레임 호출해도 된다.
    void updateTopLevel(VkCommandBuffer commandBuffer, const scene::Scene& sceneToTrace);
    void trace(VkCommandBuffer commandBuffer,
               VkExtent2D extent,
               VkDeviceAddress cameraAddress,
               VkDeviceAddress instanceAddress,
               uint32_t accumulationImage,
               uint32_t outputImage,
               uint32_t frameIndex,
               uint32_t sampleCount,
               uint32_t maxBounces);

    bool ready() const { return topLevel.handle != VK_NULL_HANDLE; }

private:
    void loadFunctions();
    void createPipeline();
    AccelerationStructure createStructure(VkAccelerationStructureTypeKHR type, VkDeviceSize size);
    void destroyStructure(AccelerationStructure& structure);
    void reserveScratch(VkDeviceSize size);

    Context& context;
    GeometryStore& geometry;
    BindlessTextures& bindless;

    std::vector<AccelerationStructure> bottomLevels;
    AccelerationStructure topLevel;
    Buffer instanceBuffer;
    Buffer scratchBuffer;
    uint32_t instanceCapacity = 0;

    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    Buffer shaderBindingTable;
    VkStridedDeviceAddressRegionKHR raygenRegion{};
    VkStridedDeviceAddressRegionKHR missRegion{};
    VkStridedDeviceAddressRegionKHR hitRegion{};
    VkStridedDeviceAddressRegionKHR callableRegion{};

    PFN_vkCreateAccelerationStructureKHR createAccelerationStructure = nullptr;
    PFN_vkDestroyAccelerationStructureKHR destroyAccelerationStructure = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR getBuildSizes = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR cmdBuildAccelerationStructures = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR getStructureAddress = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR createRayTracingPipelines = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR getShaderGroupHandles = nullptr;
    PFN_vkCmdTraceRaysKHR cmdTraceRays = nullptr;
};

} // namespace gfx
