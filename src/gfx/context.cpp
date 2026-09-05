#include "gfx/context.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <format>
#include <string>
#include <vector>

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <spdlog/spdlog.h>

#include "core/error.h"
#include "gfx/upscaler.h"
#include "gfx/vk_check.h"

namespace gfx {
namespace {

constexpr const char* VALIDATION_LAYER = "VK_LAYER_KHRONOS_validation";

// NGX 가 요구하는 확장. 인스턴스는 생성자에서, 장치는 후보를 고르는 곳에서 쓰므로 파일 범위에 둔다.
std::vector<const char*> ngxInstanceExtensions;
std::vector<const char*> ngxDeviceExtensions;

#ifdef NDEBUG
constexpr bool WANT_VALIDATION = false;
#else
constexpr bool WANT_VALIDATION = true;
#endif

// Vulkan SDK 를 시스템에 설치하지 않고 압축만 풀어둔 개발 환경에서도 로더가 ICD 와 레이어를 찾도록 한다.
void configureLoaderPaths() {
    const char* sdk = SDL_getenv("VULKAN_SDK");
    if (sdk == nullptr) {
        return;
    }
    std::string root = sdk;
    // 레이어 매니페스트 위치는 배포판마다 다르다. 실제로 있는 디렉터리일 때만 지정한다. 없는 경로를
    // 넣으면 로더가 시스템에 설치된 레이어까지 못 찾아 오히려 망가진다.
    if (SDL_getenv("VK_LAYER_PATH") == nullptr) {
        for (const char* candidate : {"/share/vulkan/explicit_layer.d", "/Bin", "/etc/vulkan/explicit_layer.d"}) {
            std::error_code error;
            if (std::filesystem::is_directory(root + candidate, error)) {
                SDL_setenv_unsafe("VK_LAYER_PATH", (root + candidate).c_str(), 1);
                break;
            }
        }
    }
#if defined(__APPLE__)
    if (SDL_getenv("VK_ICD_FILENAMES") == nullptr && SDL_getenv("VK_DRIVER_FILES") == nullptr) {
        SDL_setenv_unsafe("VK_DRIVER_FILES", (root + "/share/vulkan/icd.d/MoltenVK_icd.json").c_str(), 1);
    }
#endif
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT /*types*/,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data,
                                             void* /*userData*/) {
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0) {
        spdlog::error("[vulkan] {}", data->pMessage);
    } else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0) {
        spdlog::warn("[vulkan] {}", data->pMessage);
    } else {
        spdlog::debug("[vulkan] {}", data->pMessage);
    }
    return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT debugMessengerInfo() {
    VkDebugUtilsMessengerCreateInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    info.messageSeverity =
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = debugCallback;
    return info;
}

bool contains(const std::vector<VkExtensionProperties>& list, const char* name) {
    return std::ranges::any_of(
        list, [name](const VkExtensionProperties& item) { return std::strcmp(item.extensionName, name) == 0; });
}

std::vector<VkExtensionProperties> enumerateInstanceExtensions() {
    uint32_t count = 0;
    VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr));
    std::vector<VkExtensionProperties> extensions(count);
    VK_CHECK(vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data()));
    return extensions;
}

std::vector<VkExtensionProperties> enumerateDeviceExtensions(VkPhysicalDevice device) {
    uint32_t count = 0;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr));
    std::vector<VkExtensionProperties> extensions(count);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data()));
    return extensions;
}

bool validationLayerAvailable() {
    uint32_t count = 0;
    VK_CHECK(vkEnumerateInstanceLayerProperties(&count, nullptr));
    std::vector<VkLayerProperties> layers(count);
    VK_CHECK(vkEnumerateInstanceLayerProperties(&count, layers.data()));
    return std::ranges::any_of(
        layers, [](const VkLayerProperties& layer) { return std::strcmp(layer.layerName, VALIDATION_LAYER) == 0; });
}

// 기능 조회와 활성화에 함께 쓰이는 pNext 체인.
struct FeatureChain {
    VkPhysicalDeviceFeatures2 features2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    VkPhysicalDeviceVulkan11Features v11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
    VkPhysicalDeviceVulkan12Features v12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceVulkan13Features v13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rayTracing{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR rayQuery{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};

    // 드라이버가 광고하지 않은 확장의 구조체를 체인에 넣는 것은 규정 밖이므로 지원 여부로 걸러 연결한다.
    void link(bool withMesh, bool withAccel, bool withRayTracing, bool withRayQuery) {
        features2.pNext = nullptr;
        void** next = &features2.pNext;
        auto append = [&next](auto& node) {
            *next = &node;
            next = &node.pNext;
            *next = nullptr;
        };
        append(v11);
        append(v12);
        append(v13);
        if (withMesh) {
            append(mesh);
        }
        if (withAccel) {
            append(accel);
        }
        if (withRayTracing) {
            append(rayTracing);
        }
        if (withRayQuery) {
            append(rayQuery);
        }
    }
};

// GPU-Driven 렌더링과 bindless 를 전제하므로 아래 기능은 필수로 요구한다.
std::vector<const char*> missingRequiredFeatures(const FeatureChain& f) {
    std::vector<const char*> missing;
    auto require = [&missing](VkBool32 supported, const char* name) {
        if (supported == VK_FALSE) {
            missing.push_back(name);
        }
    };
    require(f.features2.features.multiDrawIndirect, "multiDrawIndirect");
    require(f.features2.features.drawIndirectFirstInstance, "drawIndirectFirstInstance");
    require(f.features2.features.fillModeNonSolid, "fillModeNonSolid");
    require(f.features2.features.independentBlend, "independentBlend");
    require(f.features2.features.fragmentStoresAndAtomics, "fragmentStoresAndAtomics");
    require(f.features2.features.samplerAnisotropy, "samplerAnisotropy");
    require(f.features2.features.shaderInt64, "shaderInt64");
    require(f.v11.shaderDrawParameters, "shaderDrawParameters");
    require(f.v12.bufferDeviceAddress, "bufferDeviceAddress");
    require(f.v12.descriptorIndexing, "descriptorIndexing");
    require(f.v12.runtimeDescriptorArray, "runtimeDescriptorArray");
    require(f.v12.shaderSampledImageArrayNonUniformIndexing, "shaderSampledImageArrayNonUniformIndexing");
    require(f.v12.shaderStorageBufferArrayNonUniformIndexing, "shaderStorageBufferArrayNonUniformIndexing");
    require(f.v12.shaderStorageImageArrayNonUniformIndexing, "shaderStorageImageArrayNonUniformIndexing");
    require(f.v12.descriptorBindingPartiallyBound, "descriptorBindingPartiallyBound");
    require(f.v12.descriptorBindingUpdateUnusedWhilePending, "descriptorBindingUpdateUnusedWhilePending");
    require(f.v12.descriptorBindingVariableDescriptorCount, "descriptorBindingVariableDescriptorCount");
    require(f.v12.descriptorBindingSampledImageUpdateAfterBind, "descriptorBindingSampledImageUpdateAfterBind");
    require(f.v12.descriptorBindingStorageBufferUpdateAfterBind, "descriptorBindingStorageBufferUpdateAfterBind");
    require(f.v12.descriptorBindingStorageImageUpdateAfterBind, "descriptorBindingStorageImageUpdateAfterBind");
    require(f.v12.timelineSemaphore, "timelineSemaphore");
    require(f.v12.scalarBlockLayout, "scalarBlockLayout");
    require(f.v12.hostQueryReset, "hostQueryReset");
    require(f.v13.dynamicRendering, "dynamicRendering");
    require(f.v13.synchronization2, "synchronization2");
    require(f.v13.maintenance4, "maintenance4");
    require(f.v13.shaderDemoteToHelperInvocation, "shaderDemoteToHelperInvocation");
    return missing;
}

struct DeviceCandidate {
    VkPhysicalDevice device = VK_NULL_HANDLE;
    // 적격 여부는 점수와 분리한다. 요구 조건을 만족해도 장치 종류에 따라 점수는 0 일 수 있다.
    bool suitable = false;
    std::string rejectionReason;
    VkPhysicalDeviceProperties properties{};
    QueueFamilies queueFamilies;
    Capabilities caps;
    FeatureChain features;
    std::vector<const char*> enabledExtensions;
    int score = 0;
};

bool selectQueueFamilies(VkPhysicalDevice device, VkSurfaceKHR surface, QueueFamilies& out) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

    for (uint32_t i = 0; i < count; ++i) {
        VkBool32 presentSupported = VK_FALSE;
        VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupported));
        if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0 && presentSupported == VK_TRUE) {
            out.graphics = i;
            out.graphicsTimestampBits = families[i].timestampValidBits;
            break;
        }
    }
    if (out.graphics == VK_QUEUE_FAMILY_IGNORED) {
        return false;
    }

    // 비동기 컴퓨트: 그래픽스 비트가 없는 전용 패밀리를 우선하고, 없으면 그래픽스와 다른 패밀리를 쓴다.
    out.compute = out.graphics;
    for (uint32_t i = 0; i < count; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0) {
            continue;
        }
        if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
            out.compute = i;
            break;
        }
        if (i != out.graphics && out.compute == out.graphics) {
            out.compute = i;
        }
    }

    // 전송: DMA 전용 패밀리를 우선하고, 없으면 그래픽스/컴퓨트와 겹치지 않는 패밀리를 쓴다.
    out.transfer = out.graphics;
    for (uint32_t i = 0; i < count; ++i) {
        if ((families[i].queueFlags & VK_QUEUE_TRANSFER_BIT) == 0) {
            continue;
        }
        if ((families[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) == 0) {
            out.transfer = i;
            break;
        }
        if (i != out.graphics && i != out.compute && out.transfer == out.graphics) {
            out.transfer = i;
        }
    }
    return true;
}

Capabilities queryCapabilities(const FeatureChain& f,
                               const std::vector<VkExtensionProperties>& extensions,
                               const VkPhysicalDeviceSubgroupProperties& subgroup) {
    Capabilities caps;
    caps.meshShader = f.mesh.meshShader == VK_TRUE;
    caps.taskShader = f.mesh.taskShader == VK_TRUE;
    caps.accelerationStructure = f.accel.accelerationStructure == VK_TRUE;
    caps.rayTracingPipeline = caps.accelerationStructure && f.rayTracing.rayTracingPipeline == VK_TRUE;
    caps.rayQuery = caps.accelerationStructure && f.rayQuery.rayQuery == VK_TRUE;
    caps.drawIndirectCount = f.v12.drawIndirectCount == VK_TRUE;
    caps.pipelineStatistics = f.features2.features.pipelineStatisticsQuery == VK_TRUE;
    caps.depthClamp = f.features2.features.depthClamp == VK_TRUE;
    caps.wideLines = f.features2.features.wideLines == VK_TRUE;
    // portability_subset 을 내거는 구현(MoltenVK)은 gl_DrawID 를 MSL 로 옮기지 못한다.
    caps.shaderDrawIndex = !contains(extensions, "VK_KHR_portability_subset");
    caps.shaderFloat16 = f.v12.shaderFloat16 == VK_TRUE;
    // FSR 셰이더가 SPIR-V Int16 능력을 선언한다. 켜 두지 않으면 셰이더 모듈 생성이 거부된다.
    caps.shaderInt16 = f.features2.features.shaderInt16 == VK_TRUE && f.v11.storageBuffer16BitAccess == VK_TRUE;
    caps.shaderInt8 = f.v12.shaderInt8 == VK_TRUE;
    caps.subgroupSizeControl = f.v13.subgroupSizeControl == VK_TRUE;
    caps.textureCompressionBc = f.features2.features.textureCompressionBC == VK_TRUE;
    caps.textureCompressionAstc = f.features2.features.textureCompressionASTC_LDR == VK_TRUE;
    caps.memoryBudget = contains(extensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    caps.memoryPriority = contains(extensions, VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME);
    caps.subgroupSize = subgroup.subgroupSize;
    return caps;
}

int scoreDevice(const VkPhysicalDeviceProperties& properties, const Capabilities& caps) {
    int score = 0;
    switch (properties.deviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        score += 10000;
        break;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        score += 5000;
        break;
    default:
        break;
    }
    // 최적화 경로를 더 많이 여는 장치를 선호한다.
    score += caps.meshShader ? 2000 : 0;
    score += caps.rayTracingPipeline ? 2000 : 0;
    score += caps.rayQuery ? 500 : 0;
    score += caps.drawIndirectCount ? 500 : 0;
    return score;
}

DeviceCandidate evaluateDevice(VkPhysicalDevice device, VkSurfaceKHR surface) {
    DeviceCandidate candidate;
    candidate.device = device;

    std::vector<VkExtensionProperties> extensions = enumerateDeviceExtensions(device);
    if (!contains(extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
        candidate.rejectionReason = "VK_KHR_swapchain 미지원";
        return candidate;
    }

    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties2.pNext = &subgroup;
    vkGetPhysicalDeviceProperties2(device, &properties2);
    candidate.properties = properties2.properties;
    if (candidate.properties.apiVersion < VK_API_VERSION_1_3) {
        candidate.rejectionReason = "Vulkan 1.3 미만";
        return candidate;
    }

    bool meshExt = contains(extensions, VK_EXT_MESH_SHADER_EXTENSION_NAME);
    bool accelExt = contains(extensions, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) &&
                    contains(extensions, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    bool rayTracingExt = accelExt && contains(extensions, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    bool rayQueryExt = accelExt && contains(extensions, VK_KHR_RAY_QUERY_EXTENSION_NAME);

    candidate.features.link(meshExt, accelExt, rayTracingExt, rayQueryExt);
    vkGetPhysicalDeviceFeatures2(device, &candidate.features.features2);

    std::vector<const char*> missing = missingRequiredFeatures(candidate.features);
    if (!missing.empty()) {
        candidate.rejectionReason = "필수 기능 누락:";
        for (const char* name : missing) {
            candidate.rejectionReason += std::string(" ") + name;
        }
        return candidate;
    }
    if (!selectQueueFamilies(device, surface, candidate.queueFamilies)) {
        candidate.rejectionReason = "표시 가능한 그래픽스 큐 패밀리 없음";
        return candidate;
    }

    candidate.caps = queryCapabilities(candidate.features, extensions, subgroup);
    // 타임스탬프는 주기와 큐의 유효 비트가 모두 있어야 쓸 수 있다. 어느 하나라도 0 이면 GPU 구간을
    // 잴 수 없고 프로파일러는 CPU 만 잰다.
    candidate.caps.timestamps =
        candidate.properties.limits.timestampPeriod > 0.0F && candidate.queueFamilies.graphicsTimestampBits > 0;

    candidate.enabledExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    // 코어 1.3 로도 쓸 수 있지만 ImGui Vulkan 백엔드가 확장 활성화를 요구한다.
    if (contains(extensions, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME)) {
        candidate.enabledExtensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    }
    if (contains(extensions, "VK_KHR_portability_subset")) {
        candidate.enabledExtensions.push_back("VK_KHR_portability_subset");
    }
    // 1.1 에서 코어로 올라간 확장이지만, vkGetDeviceProcAddr 은 확장을 명시적으로 켜야 KHR 접미사
    // 별칭을 돌려준다. FidelityFX 백엔드가 vkGetBufferMemoryRequirements2KHR 을 그 이름으로 찾아
    // 가드 없이 부르므로, 켜 두지 않으면 널 포인터를 호출한다.
    if (contains(extensions, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME)) {
        candidate.enabledExtensions.push_back(VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME);
    }
    for (const char* name : ngxDeviceExtensions) {
        // bufferDeviceAddress 는 코어 1.2 기능으로 이미 켜 두었다. 승격 전 확장을 함께 켜는 것은
        // 규격 위반이라 장치 생성이 거부된다.
        if (std::string_view{name} == VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) {
            continue;
        }
        if (contains(extensions, name)) {
            candidate.enabledExtensions.push_back(name);
        }
    }
    if (candidate.caps.meshShader) {
        candidate.enabledExtensions.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
    }
    if (candidate.caps.accelerationStructure) {
        candidate.enabledExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        candidate.enabledExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    }
    if (candidate.caps.rayTracingPipeline) {
        candidate.enabledExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
    }
    if (candidate.caps.rayQuery) {
        candidate.enabledExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
    }
    if (candidate.caps.memoryBudget) {
        candidate.enabledExtensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    }
    if (candidate.caps.memoryPriority) {
        candidate.enabledExtensions.push_back(VK_EXT_MEMORY_PRIORITY_EXTENSION_NAME);
    }

    candidate.suitable = true;
    candidate.score = scoreDevice(candidate.properties, candidate.caps);
    return candidate;
}

const char* deviceTypeName(VkPhysicalDeviceType type) {
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        return "외장";
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        return "내장";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        return "가상";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        return "CPU";
    default:
        return "기타";
    }
}

void logCapabilities(const VkPhysicalDeviceProperties& properties,
                     const QueueFamilies& families,
                     const Capabilities& caps) {
    spdlog::info("GPU: {} ({}, Vulkan {}.{}.{})",
                 properties.deviceName,
                 deviceTypeName(properties.deviceType),
                 VK_API_VERSION_MAJOR(properties.apiVersion),
                 VK_API_VERSION_MINOR(properties.apiVersion),
                 VK_API_VERSION_PATCH(properties.apiVersion));
    spdlog::info("큐 패밀리: 그래픽스 {}, 컴퓨트 {}{}, 전송 {}{}",
                 families.graphics,
                 families.compute,
                 families.hasAsyncCompute() ? " (별도)" : " (공유)",
                 families.transfer,
                 families.hasSeparateTransfer() ? " (별도)" : " (공유)");
    spdlog::info("mesh shader: {}, task shader: {}", caps.meshShader, caps.taskShader);
    spdlog::info("ray tracing pipeline: {}, ray query: {}", caps.rayTracingPipeline, caps.rayQuery);
    spdlog::info("drawIndirectCount: {}, subgroup {}", caps.drawIndirectCount, caps.subgroupSize);
    spdlog::info("타임스탬프 쿼리: {} (주기 {:.2f} ns, 유효 비트 {})",
                 caps.timestamps,
                 properties.limits.timestampPeriod,
                 families.graphicsTimestampBits);
}

} // namespace

Context::Context(SDL_Window* window) {
    configureLoaderPaths();

    uint32_t sdlExtensionCount = 0;
    const char* const* sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&sdlExtensionCount);
    if (sdlExtensions == nullptr) {
        core::fatal("Vulkan 인스턴스 확장 목록을 가져오지 못했습니다: {}", SDL_GetError());
    }

    std::vector<VkExtensionProperties> availableInstanceExtensions = enumerateInstanceExtensions();
    std::vector<const char*> instanceExtensions(sdlExtensions, sdlExtensions + sdlExtensionCount);

    VkInstanceCreateFlags instanceFlags = 0;
    if (contains(availableInstanceExtensions, VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        instanceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        instanceFlags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }

    // NGX 는 자기 셰이더를 직접 올리느라 확장 몇 개를 요구하고, 그건 인스턴스/장치를 만들 때만
    // 켤 수 있다. 없는 확장은 조용히 건너뛰고, 그러면 DLSS 만 쓸 수 없게 된다.
    ngxInstanceExtensions.clear();
    ngxDeviceExtensions.clear();
    dlssRequiredExtensions(ngxInstanceExtensions, ngxDeviceExtensions);
    for (const char* name : ngxInstanceExtensions) {
        if (contains(availableInstanceExtensions, name)) {
            instanceExtensions.push_back(name);
        }
    }

    bool useValidation = WANT_VALIDATION && validationLayerAvailable() &&
                         contains(availableInstanceExtensions, VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (useValidation) {
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    } else if (WANT_VALIDATION) {
        spdlog::warn("검증 레이어를 찾지 못해 비활성화합니다");
    }

    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "Computer Graphics Lab";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.pEngineName = "cg_lab";
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkDebugUtilsMessengerCreateInfoEXT messengerInfo = debugMessengerInfo();
    VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    instanceInfo.pNext = useValidation ? &messengerInfo : nullptr;
    instanceInfo.flags = instanceFlags;
    instanceInfo.pApplicationInfo = &appInfo;
    instanceInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
    instanceInfo.ppEnabledExtensionNames = instanceExtensions.data();
    if (useValidation) {
        instanceInfo.enabledLayerCount = 1;
        instanceInfo.ppEnabledLayerNames = &VALIDATION_LAYER;
    }
    VK_CHECK(vkCreateInstance(&instanceInfo, nullptr, &instance));

    if (useValidation) {
        auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        VK_CHECK(createMessenger(instance, &messengerInfo, nullptr, &debugMessenger));
    }

    if (!SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface)) {
        core::fatal("Vulkan 서피스 생성에 실패했습니다: {}", SDL_GetError());
    }

    uint32_t deviceCount = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr));
    if (deviceCount == 0) {
        core::fatal("Vulkan 을 지원하는 그래픽 장치를 찾지 못했습니다");
    }
    std::vector<VkPhysicalDevice> devices(deviceCount);
    VK_CHECK(vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data()));

    DeviceCandidate best;
    std::string rejectionDetail;
    for (VkPhysicalDevice device : devices) {
        DeviceCandidate candidate = evaluateDevice(device, surface);
        if (!candidate.suitable) {
            VkPhysicalDeviceProperties rejected{};
            vkGetPhysicalDeviceProperties(device, &rejected);
            rejectionDetail += std::format("\n  - {}: {}", rejected.deviceName, candidate.rejectionReason);
            continue;
        }
        if (!best.suitable || candidate.score > best.score) {
            best = candidate;
        }
    }
    if (!best.suitable) {
        core::fatal("요구 조건(Vulkan 1.3, bindless, GPU-Driven 간접 그리기)을 만족하는 장치가 없습니다:{}",
                    rejectionDetail);
    }

    physicalDevice = best.device;
    properties = best.properties;
    queueFamilies = best.queueFamilies;
    caps = best.caps;

    std::vector<uint32_t> uniqueFamilies{queueFamilies.graphics};
    if (queueFamilies.compute != queueFamilies.graphics) {
        uniqueFamilies.push_back(queueFamilies.compute);
    }
    if (queueFamilies.transfer != queueFamilies.graphics && queueFamilies.transfer != queueFamilies.compute) {
        uniqueFamilies.push_back(queueFamilies.transfer);
    }

    float queuePriority = 1.0F;
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    queueInfos.reserve(uniqueFamilies.size());
    for (uint32_t family : uniqueFamilies) {
        VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        queueInfo.queueFamilyIndex = family;
        queueInfo.queueCount = 1;
        queueInfo.pQueuePriorities = &queuePriority;
        queueInfos.push_back(queueInfo);
    }

    // 조회 결과를 그대로 넘기면 robustBufferAccess 같은 비용 있는 기능까지 켜지므로 필요한 것만 다시 세운다.
    FeatureChain enabled;
    enabled.link(caps.meshShader, caps.accelerationStructure, caps.rayTracingPipeline, caps.rayQuery);
    enabled.features2.features.multiDrawIndirect = VK_TRUE;
    enabled.features2.features.drawIndirectFirstInstance = VK_TRUE;
    enabled.features2.features.fillModeNonSolid = VK_TRUE;
    // 콜라이더 표시가 굵은 선을 쓴다. 없으면 1 화소로 그린다.
    enabled.features2.features.wideLines = caps.wideLines ? VK_TRUE : VK_FALSE;
    enabled.features2.features.independentBlend = VK_TRUE;
    enabled.features2.features.fragmentStoresAndAtomics = VK_TRUE;
    enabled.features2.features.samplerAnisotropy = VK_TRUE;
    enabled.features2.features.shaderInt64 = VK_TRUE;
    enabled.features2.features.shaderInt16 = caps.shaderInt16 ? VK_TRUE : VK_FALSE;
    enabled.features2.features.pipelineStatisticsQuery = caps.pipelineStatistics ? VK_TRUE : VK_FALSE;
    enabled.features2.features.depthClamp = caps.depthClamp ? VK_TRUE : VK_FALSE;
    enabled.features2.features.textureCompressionBC = caps.textureCompressionBc ? VK_TRUE : VK_FALSE;
    enabled.features2.features.textureCompressionASTC_LDR = caps.textureCompressionAstc ? VK_TRUE : VK_FALSE;
    enabled.v11.shaderDrawParameters = VK_TRUE;
    enabled.v11.storageBuffer16BitAccess = caps.shaderInt16 ? VK_TRUE : VK_FALSE;
    enabled.v12.bufferDeviceAddress = VK_TRUE;
    enabled.v12.descriptorIndexing = VK_TRUE;
    enabled.v12.runtimeDescriptorArray = VK_TRUE;
    enabled.v12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    enabled.v12.shaderStorageBufferArrayNonUniformIndexing = VK_TRUE;
    enabled.v12.shaderStorageImageArrayNonUniformIndexing = VK_TRUE;
    enabled.v12.descriptorBindingPartiallyBound = VK_TRUE;
    enabled.v12.descriptorBindingUpdateUnusedWhilePending = VK_TRUE;
    enabled.v12.descriptorBindingVariableDescriptorCount = VK_TRUE;
    enabled.v12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    enabled.v12.descriptorBindingStorageBufferUpdateAfterBind = VK_TRUE;
    enabled.v12.descriptorBindingStorageImageUpdateAfterBind = VK_TRUE;
    enabled.v12.timelineSemaphore = VK_TRUE;
    enabled.v12.scalarBlockLayout = VK_TRUE;
    enabled.v12.hostQueryReset = VK_TRUE;
    enabled.v12.drawIndirectCount = caps.drawIndirectCount ? VK_TRUE : VK_FALSE;
    enabled.v12.shaderFloat16 = caps.shaderFloat16 ? VK_TRUE : VK_FALSE;
    enabled.v12.shaderInt8 = caps.shaderInt8 ? VK_TRUE : VK_FALSE;
    enabled.v13.dynamicRendering = VK_TRUE;
    enabled.v13.synchronization2 = VK_TRUE;
    enabled.v13.maintenance4 = VK_TRUE;
    enabled.v13.shaderDemoteToHelperInvocation = VK_TRUE;
    enabled.v13.subgroupSizeControl = caps.subgroupSizeControl ? VK_TRUE : VK_FALSE;
    enabled.v13.computeFullSubgroups = caps.subgroupSizeControl ? VK_TRUE : VK_FALSE;
    enabled.mesh.meshShader = caps.meshShader ? VK_TRUE : VK_FALSE;
    enabled.mesh.taskShader = caps.taskShader ? VK_TRUE : VK_FALSE;
    enabled.accel.accelerationStructure = caps.accelerationStructure ? VK_TRUE : VK_FALSE;
    enabled.rayTracing.rayTracingPipeline = caps.rayTracingPipeline ? VK_TRUE : VK_FALSE;
    enabled.rayQuery.rayQuery = caps.rayQuery ? VK_TRUE : VK_FALSE;

    VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    deviceInfo.pNext = &enabled.features2;
    deviceInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    deviceInfo.pQueueCreateInfos = queueInfos.data();
    deviceInfo.enabledExtensionCount = static_cast<uint32_t>(best.enabledExtensions.size());
    deviceInfo.ppEnabledExtensionNames = best.enabledExtensions.data();
    VK_CHECK(vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device));

    vkGetDeviceQueue(device, queueFamilies.graphics, 0, &graphicsQueue);
    vkGetDeviceQueue(device, queueFamilies.compute, 0, &computeQueue);
    vkGetDeviceQueue(device, queueFamilies.transfer, 0, &transferQueue);

    VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    void** propertyNext = &properties2.pNext;
    if (caps.meshShader) {
        *propertyNext = &meshShaderProperties;
        propertyNext = &meshShaderProperties.pNext;
    }
    if (caps.accelerationStructure) {
        *propertyNext = &accelerationStructureProperties;
        propertyNext = &accelerationStructureProperties.pNext;
    }
    if (caps.rayTracingPipeline) {
        *propertyNext = &rayTracingPipelineProperties;
        propertyNext = &rayTracingPipelineProperties.pNext;
    }
    *propertyNext = nullptr;
    vkGetPhysicalDeviceProperties2(physicalDevice, &properties2);

    VmaAllocatorCreateInfo allocatorInfo{};
    allocatorInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    if (caps.memoryBudget) {
        allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;
    }
    if (caps.memoryPriority) {
        allocatorInfo.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_PRIORITY_BIT;
    }
    allocatorInfo.physicalDevice = physicalDevice;
    allocatorInfo.device = device;
    allocatorInfo.instance = instance;
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_3;
    VK_CHECK(vmaCreateAllocator(&allocatorInfo, &allocator));

    logCapabilities(properties, queueFamilies, caps);
}

Context::MemoryBudget Context::deviceMemoryBudget() const {
    const VkPhysicalDeviceMemoryProperties* memory = nullptr;
    vmaGetMemoryProperties(allocator, &memory);
    std::vector<VmaBudget> budgets(memory->memoryHeapCount);
    vmaGetHeapBudgets(allocator, budgets.data());
    MemoryBudget total;
    for (uint32_t heap = 0; heap < memory->memoryHeapCount; ++heap) {
        if ((memory->memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) == 0) {
            continue;
        }
        total.budget += budgets[heap].budget;
        total.usage += budgets[heap].usage;
    }
    if (memoryBudgetOverride != 0) {
        total.budget = memoryBudgetOverride;
    }
    return total;
}

VkDeviceSize Context::deviceLocalMemoryBytes() const {
    const VkPhysicalDeviceMemoryProperties* memory = nullptr;
    vmaGetMemoryProperties(allocator, &memory);
    VkDeviceSize total = 0;
    for (uint32_t heap = 0; heap < memory->memoryHeapCount; ++heap) {
        if ((memory->memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0) {
            total += memory->memoryHeaps[heap].size;
        }
    }
    return total;
}

Context::~Context() {
    // 할당기를 지우기 전에 맡아 둔 자원을 모두 비운다.
    collectRetired();
    vmaDestroyAllocator(allocator);
    vkDestroyDevice(device, nullptr);
    if (debugMessenger != VK_NULL_HANDLE) {
        auto destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        destroyMessenger(instance, debugMessenger, nullptr);
    }
    vkDestroySurfaceKHR(instance, surface, nullptr);
    vkDestroyInstance(instance, nullptr);
}

} // namespace gfx
