#pragma once

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "gfx/geometry.h"
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
// 편집기에서 조절하는 경로 추적 설정.
struct PathTraceOptions {
    uint32_t maxBounces = 3;
    // 한 프레임에 픽셀마다 쏘는 경로 수. 늘리면 빨리 수렴하지만 프레임이 길어진다.
    uint32_t samplesPerFrame = 1;
    // 0 이면 계속 누적한다. 그 외에는 이 수만큼 쌓고 멈춘다.
    uint32_t maxSamples = 0;
    // 다음 사건 추정. 끄면 조명을 우연히 맞출 때만 밝아져 훨씬 느리게 수렴한다.
    bool nextEventEstimation = true;
    bool russianRoulette = true;
    // 반딧불이 표본을 자르는 상한.
    float radianceClamp = 8.0F;
    float skyIntensity = 1.0F;

    bool operator==(const PathTraceOptions&) const = default;
};

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
    // instanceSlots 는 오브젝트 인덱스 -> 그리기 인스턴스 슬롯. 적중 셰이더가 인스턴스 배열을
    // 이 번호로 찾으므로 buildDrawCommands 가 만든 것과 반드시 같아야 한다.
    void updateTopLevel(VkCommandBuffer commandBuffer,
                        const scene::Scene& sceneToTrace,
                        const std::vector<uint32_t>& instanceSlots);
    void trace(VkCommandBuffer commandBuffer,
               VkExtent2D extent,
               VkDeviceAddress cameraAddress,
               VkDeviceAddress instanceAddress,
               VkDeviceAddress lightAddress,
               uint32_t accumulationImage,
               uint32_t outputImage,
               uint32_t frameIndex,
               uint32_t sampleCount,
               const PathTraceOptions& options);

    bool ready() const { return topLevel.handle != VK_NULL_HANDLE; }

    // 래스터 경로의 광선 질의 그림자가 같은 TLAS 를 집합 1 로 묶어 쓴다.
    VkDescriptorSetLayout accelerationLayout() const { return descriptorSetLayout; }
    VkDescriptorSet accelerationSet() const { return descriptorSet; }

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
