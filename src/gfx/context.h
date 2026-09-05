#pragma once

#include <cstdint>
#include <functional>
#include <vector>

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include "gfx/resources.h"

struct SDL_Window;

namespace gfx {

// 렌더 경로 선택에 쓰이는 선택적 하드웨어 기능. 미지원 기능에 대한 소프트웨어 폴백은 만들지 않고,
// 해당 경로 자체를 비활성화한다.
struct Capabilities {
    bool meshShader = false;
    bool taskShader = false;
    bool accelerationStructure = false;
    bool rayTracingPipeline = false;
    bool rayQuery = false;
    // 하위 가속 구조의 삼각형 수를 장치 버퍼에서 읽어 세울 수 있는지(간접 구축). GPU 가 만든 물 표면을
    // 광선 경로에 올리는 데 쓴다. 없으면 물 표면은 Path Tracing 에 보이지 않는다.
    bool accelerationStructureIndirectBuild = false;
    // 간접 그리기 개수를 GPU 버퍼에서 읽어 압축 드로우가 가능한지. 없으면 고정 개수로 디스패치하고
    // 컬링된 드로우는 instanceCount 를 0 으로 기록한다.
    bool drawIndirectCount = false;
    bool pipelineStatistics = false;
    bool shaderFloat16 = false;
    // FSR 셰이더가 SPIR-V Int16 능력을 선언한다.
    bool shaderInt16 = false;
    bool shaderInt8 = false;
    bool subgroupSizeControl = false;
    bool textureCompressionBc = false;
    bool textureCompressionAstc = false;
    bool memoryBudget = false;
    bool memoryPriority = false;
    // 타임스탬프 쿼리로 GPU 구간 시간을 잴 수 있는지. 주기와 유효 비트가 모두 있어야 한다.
    bool timestamps = false;
    // 그림자 절두체 근평면 앞의 캐스터를 잘라 내지 않고 눌러 담을 수 있는지.
    bool depthClamp = false;
    // 1 화소보다 굵은 선을 그릴 수 있는지. 콜라이더 표시가 굵게 그리는 데 쓴다.
    bool wideLines = false;
    // 정점 셰이더가 gl_DrawID 를 읽을 수 있는지. MoltenVK 는 shaderDrawParameters 를 지원한다고
    // 보고하지만 MSL 에 DrawIndex 가 없어 SPIR-V 변환에서 죽는다. 없으면 meshlet 디버그 뷰가 메쉬
    // 단위로 뭉개진다.
    bool shaderDrawIndex = false;
    uint32_t subgroupSize = 0;
};

// 용도별 큐 패밀리. 전용 패밀리가 없으면 그래픽스 패밀리로 접힌다.
struct QueueFamilies {
    uint32_t graphics = VK_QUEUE_FAMILY_IGNORED;
    uint32_t compute = VK_QUEUE_FAMILY_IGNORED;
    uint32_t transfer = VK_QUEUE_FAMILY_IGNORED;
    // 그래픽스 큐가 돌려주는 타임스탬프의 유효 비트 수. 0 이면 그 큐에서 타임스탬프를 쓸 수 없다.
    uint32_t graphicsTimestampBits = 0;

    bool hasAsyncCompute() const { return compute != graphics; }
    bool hasSeparateTransfer() const { return transfer != graphics && transfer != compute; }
};

struct Context {
    explicit Context(SDL_Window* window);
    ~Context();
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VmaAllocator allocator = VK_NULL_HANDLE;

    VkPhysicalDeviceProperties properties{};
    VkPhysicalDeviceMeshShaderPropertiesEXT meshShaderProperties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
    VkPhysicalDeviceAccelerationStructurePropertiesKHR accelerationStructureProperties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingPipelineProperties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR};

    QueueFamilies queueFamilies;
    VkQueue graphicsQueue = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    VkQueue transferQueue = VK_NULL_HANDLE;

    Capabilities caps;

    // 장치 전용 힙의 예산과 사용량(바이트). VK_EXT_memory_budget 이 있으면 드라이버가 알려 주는 예산이고,
    // 없으면 VMA 가 힙 크기의 80% 를 예산으로, 자기가 할당한 만큼만 사용량으로 본다. 큰 모델을 올리기
    // 전에 들어갈지 미리 재는 데 쓴다.
    struct MemoryBudget {
        VkDeviceSize budget = 0;
        VkDeviceSize usage = 0;
    };
    MemoryBudget deviceMemoryBudget() const;
    // 장치 전용 힙의 «크기» 합. 예산과 달리 다른 프로세스나 이미 만든 자원에 영향받지 않아 기기의
    // 성격을 재는 데 쓴다. 자동 튜닝의 등급 판정이 이 값을 본다.
    VkDeviceSize deviceLocalMemoryBytes() const;
    // 0 이 아니면 드라이버 예산 대신 이 값을 예산으로 본다. --gpu-budget 이 채우며, 모델과 가속 구조의
    // 예산 검사가 모두 같은 값을 보게 여기 둔다.
    VkDeviceSize memoryBudgetOverride = 0;

    // 진행 중인 프레임이 아직 읽고 있을 수 있는 자원을 바로 지우지 않고 모아 둔다. 편집기가 유체
    // 부품을 떼는 것처럼 자원이 사라지는 일은 프레임 한가운데 일어나는데, 그때마다 장치를 세우면
    // 편집할 때마다 화면이 끊긴다. 대신 다음 프레임 첫머리에서 지운다.
    void retireBuffer(Buffer& buffer);
    // 파괴 함수가 여기 없는 자원(가속 구조 등)을 같은 규칙으로 맡긴다. 맡긴 순서대로 지우므로,
    // 다른 자원을 붙들고 있는 것을 먼저 맡기면 수명 순서도 지켜진다.
    void retireDeferred(std::function<void()> destroy);
    bool hasRetired() const { return !retiredResources.empty(); }
    // 제출이 하나도 남아 있지 않을 때 부른다. 맡긴 것을 전부 지운다. «FRAMES_IN_FLIGHT 뒤» 같은
    // 프레임 번호 판정으로는 모자란다. MoltenVK 는 장치 주소 버퍼를 제출마다 전부 상주시키므로,
    // 살아 있던 동안 제출된 명령이 하나라도 돌고 있으면 지우는 순간 장치가 죽는다.
    void collectRetired();

private:
    std::vector<std::function<void()>> retiredResources;
};

} // namespace gfx
