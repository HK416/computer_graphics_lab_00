#pragma once

#include <cstdint>

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.h>

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
    // 간접 그리기 개수를 GPU 버퍼에서 읽어 압축 드로우가 가능한지. 없으면 고정 개수로 디스패치하고
    // 컬링된 드로우는 instanceCount 를 0 으로 기록한다.
    bool drawIndirectCount = false;
    // HZB 밈맵 생성 시 min/max 리덕션 샘플러 사용 가능 여부.
    bool samplerFilterMinmax = false;
    bool pipelineStatistics = false;
    bool shaderFloat16 = false;
    bool shaderInt8 = false;
    bool subgroupSizeControl = false;
    bool textureCompressionBc = false;
    bool textureCompressionAstc = false;
    bool memoryBudget = false;
    bool memoryPriority = false;
    uint32_t subgroupSize = 0;
};

// 용도별 큐 패밀리. 전용 패밀리가 없으면 그래픽스 패밀리로 접힌다.
struct QueueFamilies {
    uint32_t graphics = VK_QUEUE_FAMILY_IGNORED;
    uint32_t compute = VK_QUEUE_FAMILY_IGNORED;
    uint32_t transfer = VK_QUEUE_FAMILY_IGNORED;

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
};

} // namespace gfx
