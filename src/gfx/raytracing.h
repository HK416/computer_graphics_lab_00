#pragma once

#include <cstdint>
#include <string>
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
    // shaders/scene_types.glsl 의 DEBUG_MODE_*. 렌더러가 채운다. 여기 있으면 모드를 바꿀 때
    // operator== 비교가 달라져 누적이 자동으로 초기화된다.
    uint32_t debugMode = 0;

    bool operator==(const PathTraceOptions&) const = default;
};

// DLSS Ray Reconstruction 이 요구하는 안내 버퍼의 bindless 스토리지 슬롯.
// write 가 거짓이면 광선 생성 셰이더가 아무것도 쓰지 않고 나머지 값도 보지 않는다.
struct PathGuideTargets {
    bool write = false;
    uint32_t diffuseAlbedo = 0;
    uint32_t specularAlbedo = 0;
    uint32_t normal = 0;
    uint32_t roughness = 0;
    uint32_t depth = 0;
};

// 스킨 인스턴스 하나가 쓸 하위 가속 구조의 재료. 변형 정점 버퍼 안의 구간과 그 메쉬 번호다.
struct SkinnedInstance {
    uint32_t meshIndex;
    uint32_t vertexOffset;
    // 이번 프레임에 포즈가 바뀌어 하위 가속 구조를 다시 세워야 하는지. 구조가 아직 없으면 어차피 세운다.
    bool rebuild = true;

    bool operator==(const SkinnedInstance&) const = default;
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

    // 메쉬마다 0단계 LOD 로 하위 가속 구조를 만든다. 광선 기능이 처음 필요할 때 부른다. 구조와
    // 스크래치가 남은 GPU 예산을 넘으면 아무것도 만들지 않고 사유를 reason 에 적은 뒤 false 를 돌려준다.
    // 스크래치 합이 한도를 넘을 때마다 제출을 끊어, 한 번의 제출이 길어져 GPU 가 재설정되는 일을 줄인다.
    bool buildBottomLevel(std::string& reason);
    // 지오메트리 버퍼가 바뀌면 옛 주소를 가리키는 하위·상위 구조를 버린다. 다음 요청 때 다시 만든다.
    void invalidateBottomLevel();
    // 메쉬가 하나도 없어도 «세웠다»가 참이어야 한다. 비어 있음으로 판단하면 빈 장면에서 매 프레임 다시 세운다.
    bool bottomLevelReady() const { return bottomLevelBuilt; }
    // 스킨 인스턴스마다 변형 정점으로 하위 가속 구조를 다시 세운다. 포즈가 바뀌면 매 프레임
    // 불러야 한다. 스킨 컴퓨트가 끝난 뒤, updateTopLevel 앞에 온다.
    void updateSkinnedBottomLevel(VkCommandBuffer commandBuffer,
                                  const Buffer& skinnedVertices,
                                  const std::vector<SkinnedInstance>& skinned);
    // GPU 가 만든 비인덱스 삼각형 목록(물 표면)의 하위 가속 구조. 삼각형 수는 CPU 가 모른다. 장치가 간접 구축을
    // 지원하면 rangeAddress 의 VkAccelerationStructureBuildRangeInfoKHR 에서 읽고, 아니면(NVIDIA) 상한 개수로
    // 세운다 — 그때는 정점 버퍼의 쓰지 않은 꼬리가 0(퇴화 삼각형)이어야 하므로 마칭이 버퍼를 먼저 지운다.
    bool indirectBuildAvailable() const { return cmdBuildAccelerationStructuresIndirect != nullptr; }
    // key 번째 동적 구조를 maxTriangles 크기로 준비하고 주소를 돌려준다. 인스턴스가 이 주소를 가리킬 수 있게
    // 구축보다 먼저 부른다. 같은 크기면 자리를 그대로 쓴다.
    VkDeviceAddress ensureDynamicBottomLevel(uint32_t key, uint32_t maxTriangles);
    // key 번째 동적 구조를 세운다. vertices 는 위치가 앞에 오는 stride 바이트 정점, rangeAddress 는 삼각형 수가
    // 든 범위 구조체. 정점을 만든 컴퓨트 뒤, updateTopLevel 앞에 온다.
    void buildDynamicBottomLevel(VkCommandBuffer commandBuffer,
                                 uint32_t key,
                                 VkDeviceAddress vertices,
                                 uint32_t vertexStride,
                                 uint32_t maxTriangles,
                                 VkDeviceAddress rangeAddress);
    // 장면 인스턴스로 상위 가속 구조를 다시 만든다. 매 프레임 호출해도 된다.
    // instanceSlots 는 오브젝트 인덱스 -> 그리기 인스턴스 슬롯. 적중 셰이더가 인스턴스 배열을
    // 이 번호로 찾으므로 buildDrawCommands 가 만든 것과 반드시 같아야 한다.
    // skinnedBlasSlots 는 오브젝트 인덱스 -> updateSkinnedBottomLevel 에 넘긴 배열의 번호이며,
    // 스킨이 아닌 오브젝트는 NO_SKINNED_BLAS 다.
    // frameSlot 은 진행 중인 프레임 번호. 인스턴스 버퍼를 프레임마다 나눠 쓰는 데 쓴다.
    // prependedInstances 는 GPU(유체 컴퓨트)가 인스턴스 버퍼 앞쪽에 이미 써 둔 개수. 오브젝트 인스턴스는
    // 그 뒤에 이어 붙고 구축 개수에 합쳐진다.
    void updateTopLevel(VkCommandBuffer commandBuffer,
                        const scene::Scene& sceneToTrace,
                        const std::vector<uint32_t>& instanceSlots,
                        const std::vector<uint32_t>& skinnedBlasSlots,
                        uint32_t frameSlot,
                        uint32_t prependedInstances = 0);
    // 인스턴스 버퍼를 미리 잡는다. GPU 가 앞쪽을 쓰는 패스는 updateTopLevel 이 버퍼를 다시 잡기 전에
    // 주소를 기록하므로 먼저 불러야 한다.
    void reserveInstances(uint32_t frameSlot, uint32_t count);
    VkDeviceAddress instanceBufferAddress(uint32_t frameSlot) const;
    // 호스트에서 보이는 자리라 CPU 유체 백엔드가 앞쪽 인스턴스를 직접 쓴다.
    void* instanceBufferMapped(uint32_t frameSlot) const;
    // 메쉬의 하위 가속 구조 주소. 없으면(무덤, 미구축) 0.
    VkDeviceAddress bottomLevelAddress(uint32_t mesh) const;
    void trace(VkCommandBuffer commandBuffer,
               VkExtent2D extent,
               VkDeviceAddress cameraAddress,
               VkDeviceAddress instanceAddress,
               VkDeviceAddress lightAddress,
               VkDeviceAddress skinnedVertexAddress,
               // 유체마다의 물 표면 정보(FluidSurfaceInfo). 0 이면 물 표면이 없다.
               VkDeviceAddress fluidSurfaceAddress,
               uint32_t accumulationImage,
               // 화면 UV 모션 벡터를 쓸 rg16f 스토리지 슬롯.
               uint32_t velocityImage,
               uint32_t frameIndex,
               uint32_t sampleCount,
               const PathTraceOptions& options,
               const PathGuideTargets& guides);

    // 스킨이 아닌 오브젝트의 하위 가속 구조 번호.
    static constexpr uint32_t NO_SKINNED_BLAS = 0xFFFFFFFFU;

    bool ready() const { return topLevel.handle != VK_NULL_HANDLE; }

    // 래스터 경로의 광선 질의 그림자가 같은 TLAS 를 집합 1 로 묶어 쓴다.
    VkDescriptorSetLayout accelerationLayout() const { return descriptorSetLayout; }
    VkDescriptorSet accelerationSet() const { return descriptorSet; }

private:
    void loadFunctions();
    void createPipeline();
    // 이번 프레임의 구축이 지난 프레임의 추적/질의와 겹치지 않게 막는다. 구조와 스크래치 버퍼를
    // 하나씩만 두고 프레임마다 다시 쓰기 때문에 필요하다.
    void barrierBeforeBuild(VkCommandBuffer commandBuffer);
    AccelerationStructure createStructure(VkAccelerationStructureTypeKHR type, VkDeviceSize size);
    // 장치가 놀고 있을 때만 부른다(소멸자, 지오메트리 교체). 기록 중이면 retireStructure 를 쓴다.
    void destroyStructure(AccelerationStructure& structure);
    // 명령 기록 도중 구조를 갈아 끼울 때 쓴다. 지난 프레임의 추적·질의가 아직 옛 구조를 읽고 있을 수
    // 있어 바로 지우면 안 된다. 구조를 먼저 맡겨야 그 뒤에 맡기는 저장 버퍼보다 먼저 지워진다.
    void retireStructure(AccelerationStructure& structure);
    void reserveScratch(Buffer& buffer, VkDeviceSize size, const char* debugName);

    Context& context;
    GeometryStore& geometry;
    BindlessTextures& bindless;

    // 메쉬 번호마다 하나. 해제된 무덤 메쉬는 핸들이 null 이다.
    std::vector<AccelerationStructure> bottomLevels;
    bool bottomLevelBuilt = false;
    // 스킨 인스턴스마다 하나. 포즈가 바뀔 때마다 같은 자리에 다시 세운다.
    std::vector<AccelerationStructure> skinnedBottomLevels;
    // 동적(물 표면) 구조. 키마다 구조·스크래치·크기 근거를 둔다.
    struct DynamicBottomLevel {
        AccelerationStructure structure;
        Buffer scratch;
        uint32_t maxTriangles = 0;
    };
    std::vector<DynamicBottomLevel> dynamicBottomLevels;
    AccelerationStructure topLevel;
    // 구축 입력은 CPU 가 기록 시점에 채우므로 진행 중인 프레임 수만큼 나눠 둬야 한다. 하나만
    // 두면 지난 프레임의 구축이 아직 읽는 중에 덮어쓰게 된다.
    std::vector<Buffer> instanceBuffers;
    std::vector<uint32_t> instanceCapacities;
    Buffer scratchBuffer;
    // 스킨 하위 구조는 상위 구조와 같은 명령 버퍼 안에서 세워진다. 스크래치를 하나로 묶으면
    // 상위 구조 차례에 버퍼가 커지면서 이미 기록해 둔 주소가 날아간다.
    Buffer skinnedScratchBuffer;

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
    PFN_vkCmdBuildAccelerationStructuresIndirectKHR cmdBuildAccelerationStructuresIndirect = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR getStructureAddress = nullptr;
    PFN_vkCreateRayTracingPipelinesKHR createRayTracingPipelines = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR getShaderGroupHandles = nullptr;
    PFN_vkCmdTraceRaysKHR cmdTraceRays = nullptr;
};

} // namespace gfx
