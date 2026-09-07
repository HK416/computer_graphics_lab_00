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
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coop{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};

    // 드라이버가 광고하지 않은 확장의 구조체를 체인에 넣는 것은 규정 밖이므로 지원 여부로 걸러 연결한다.
    void link(bool withMesh, bool withAccel, bool withRayTracing, bool withRayQuery, bool withCoop) {
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
        if (withCoop) {
            append(coop);
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
        // 서피스가 없으면(헤드리스) 표시 지원을 묻지 않는다. 그릴 곳이 없으니 볼 것도 없다.
        VkBool32 presentSupported = surface == VK_NULL_HANDLE ? VK_TRUE : VK_FALSE;
        if (surface != VK_NULL_HANDLE) {
            VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupported));
        }
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

// 미리 컴파일해 둔 협력 행렬 모양. **셰이더 변종과 하나씩 짝이 맞아야 한다** — 여기에 줄을 더하면
// CMakeLists 의 shader_variants 에도 같은 이름의 변종을 더한다. 앞에 있는 것을 먼저 고른다.
//
// fp32 A/B 를 먼저 두는 이유: 캐스팅이 없어 FMA 경로와 오차가 거의 없고, 그래서 «가속을 켜도 학습이
// 같은 곳으로 간다» 를 확인하기 쉽다. fp16 은 A/B 만 반정밀도이고 누산기는 fp32 다.
struct CoopCandidate {
    uint32_t m;
    uint32_t n;
    uint32_t k;
    bool float32;
    // 이 모양으로 컴파일해 둔 변종의 이름. 여기 함께 두어야 모양과 셰이더가 갈라지지 않는다.
    const char* shader;
};
constexpr CoopCandidate COOP_CANDIDATES[] = {
    // ponytail: fp32 A/B 는 개발 기기(RTX 3060)가 광고하지 않아 **한 번도 돌려 보지 못했다.** 컴파일만
    // 확인했다. 자기 검사가 지금 증명하는 것은 fp16 갈래뿐이다.
    {16, 16, 16, true, "neural_linear_coop_f32_16.comp.spv"},
    {16, 16, 16, false, "neural_linear_coop_f16_16.comp.spv"},
};

// 장치가 광고하는 모양과 후보의 교집합에서 첫 번째를 고른다. 없으면 caps 를 건드리지 않는다(가속만 꺼진다).
//
// **아래 게이트들은 이 기기에서 검사할 방법이 없다.** 개발 기기가 모든 조건을 만족하므로, 조건을 하나씩
// 빼는 돌연변이를 넣어도 자기 검사가 그대로 통과한다(실측). 조건을 어기는 기기에서만 갈린다 — 하드웨어
// 게이트가 원래 그런 것이고, 그래서 여기서는 «돌려 보고 맞추는» 대신 규격이 요구하는 것을 하나씩 적는다.
void selectCooperativeMatrix(VkInstance instance,
                             VkPhysicalDevice device,
                             const VkPhysicalDeviceVulkan13Properties& v13,
                             uint32_t subgroupSize,
                             Capabilities& caps) {
    // 서브그룹 크기를 못 박을 수 없으면 시작하지도 않는다. AMD 처럼 wave32/wave64 가 섞이는 기기에서
    // 작업 그룹 32 개가 64 폭 서브그룹의 절반만 채우면 협력 행렬 연산 자체가 규정 밖이 된다.
    if (!caps.subgroupSizeControl || (v13.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0) {
        return;
    }
    // 협력 행렬 셰이더는 Vulkan 메모리 모델을 선언한다. 기능을 못 켜면 모듈을 만들 수 없다.
    if (!caps.vulkanMemoryModel) {
        return;
    }
    // **컴퓨트 단계를 광고해야 한다.** 모양이 맞아도 이 단계가 빠지면 OpTypeCooperativeMatrixKHR 을 담은
    // 모듈 자체가 규정 밖이다(VUID-RuntimeSpirv-cooperativeMatrixSupportedStages-08985).
    VkPhysicalDeviceCooperativeMatrixPropertiesKHR coopProperties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties2.pNext = &coopProperties;
    vkGetPhysicalDeviceProperties2(device, &properties2);
    if ((coopProperties.cooperativeMatrixSupportedStages & VK_SHADER_STAGE_COMPUTE_BIT) == 0) {
        return;
    }
    if (subgroupSize < v13.minSubgroupSize || subgroupSize > v13.maxSubgroupSize) {
        return;
    }
    // **로더가 이 함수를 내보내지 않는다.** vkGetInstanceProcAddr 로 받아야 한다.
    auto enumerate = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
        vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    if (enumerate == nullptr) {
        return;
    }
    uint32_t count = 0;
    if (enumerate(device, &count, nullptr) != VK_SUCCESS || count == 0) {
        return;
    }
    std::vector<VkCooperativeMatrixPropertiesKHR> properties(
        count, VkCooperativeMatrixPropertiesKHR{VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
    if (enumerate(device, &count, properties.data()) != VK_SUCCESS) {
        return;
    }
    properties.resize(count);

    for (const CoopCandidate& candidate : COOP_CANDIDATES) {
        VkComponentTypeKHR ab = candidate.float32 ? VK_COMPONENT_TYPE_FLOAT32_KHR : VK_COMPONENT_TYPE_FLOAT16_KHR;
        // fp16 변종은 float16 산술만 있으면 된다. A·B 를 공유 메모리에 담아 싣기 때문에 16비트 «저장
        // 버퍼» 접근은 필요 없다 — 버퍼 안의 값은 끝까지 fp32 다.
        if (!candidate.float32 && !caps.shaderFloat16) {
            continue;
        }
        bool found = std::ranges::any_of(properties, [&](const VkCooperativeMatrixPropertiesKHR& item) {
            // 누산기(C·Result)는 언제나 fp32 로 받는다. 반정밀도로 접으면 열 걸음도 못 간다.
            return item.scope == VK_SCOPE_SUBGROUP_KHR && item.MSize == candidate.m && item.NSize == candidate.n &&
                   item.KSize == candidate.k && item.AType == ab && item.BType == ab &&
                   item.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && item.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR;
        });
        if (found) {
            caps.cooperativeMatrix = true;
            caps.coopSubgroupSize = subgroupSize;
            caps.coopM = candidate.m;
            caps.coopN = candidate.n;
            caps.coopK = candidate.k;
            caps.coopFloat32 = candidate.float32;
            caps.coopShader = candidate.shader;
            return;
        }
    }
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
    caps.accelerationStructureIndirectBuild =
        caps.accelerationStructure && f.accel.accelerationStructureIndirectBuild == VK_TRUE;
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
    caps.vulkanMemoryModel = f.v12.vulkanMemoryModel == VK_TRUE;
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

DeviceCandidate evaluateDevice(VkInstance instance, VkPhysicalDevice device, VkSurfaceKHR surface) {
    DeviceCandidate candidate;
    candidate.device = device;

    std::vector<VkExtensionProperties> extensions = enumerateDeviceExtensions(device);
    if (surface != VK_NULL_HANDLE && !contains(extensions, VK_KHR_SWAPCHAIN_EXTENSION_NAME)) {
        candidate.rejectionReason = "VK_KHR_swapchain 미지원";
        return candidate;
    }

    VkPhysicalDeviceVulkan13Properties v13Properties{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    VkPhysicalDeviceSubgroupProperties subgroup{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    subgroup.pNext = &v13Properties;
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
    bool coopExt = contains(extensions, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);

    candidate.features.link(meshExt, accelExt, rayTracingExt, rayQueryExt, coopExt);
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
        candidate.rejectionReason =
            surface != VK_NULL_HANDLE ? "표시 가능한 그래픽스 큐 패밀리 없음" : "그래픽스 큐 패밀리 없음";
        return candidate;
    }

    candidate.caps = queryCapabilities(candidate.features, extensions, subgroup);
    // 모양을 여기서 고른다. 장치를 만들기 **전**이라야 «쓸 모양이 없으면 확장을 아예 안 켠다» 가 된다.
    if (coopExt && candidate.features.coop.cooperativeMatrix == VK_TRUE) {
        selectCooperativeMatrix(instance, device, v13Properties, subgroup.subgroupSize, candidate.caps);
    }
    // 타임스탬프는 주기와 큐의 유효 비트가 모두 있어야 쓸 수 있다. 어느 하나라도 0 이면 GPU 구간을
    // 잴 수 없고 프로파일러는 CPU 만 잰다.
    candidate.caps.timestamps =
        candidate.properties.limits.timestampPeriod > 0.0F && candidate.queueFamilies.graphicsTimestampBits > 0;

    if (surface != VK_NULL_HANDLE) {
        candidate.enabledExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }
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
    if (candidate.caps.cooperativeMatrix) {
        candidate.enabledExtensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
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
    if (caps.cooperativeMatrix) {
        spdlog::info("협력 행렬: {}x{}x{} ({} A/B, fp32 누산기)",
                     caps.coopM,
                     caps.coopN,
                     caps.coopK,
                     caps.coopFloat32 ? "fp32" : "fp16");
    } else {
        spdlog::info("협력 행렬: 없음 (신경망 선형 층은 FMA 경로로만 돈다)");
    }
    spdlog::info("타임스탬프 쿼리: {} (주기 {:.2f} ns, 유효 비트 {})",
                 caps.timestamps,
                 properties.limits.timestampPeriod,
                 families.graphicsTimestampBits);
}

} // namespace

Context::Context(SDL_Window* window) {
    configureLoaderPaths();

    // 창이 없으면(--headless) 서피스를 만들지 않는다. SDL 의 인스턴스 확장은 서피스를 위한 것이고
    // SDL 비디오가 올라와 있어야 물을 수 있으므로 그때는 아예 묻지 않는다.
    std::vector<const char*> instanceExtensions;
    if (window != nullptr) {
        uint32_t sdlExtensionCount = 0;
        const char* const* sdlExtensions = SDL_Vulkan_GetInstanceExtensions(&sdlExtensionCount);
        if (sdlExtensions == nullptr) {
            core::fatal("Vulkan 인스턴스 확장 목록을 가져오지 못했습니다: {}", SDL_GetError());
        }
        instanceExtensions.assign(sdlExtensions, sdlExtensions + sdlExtensionCount);
    }

    std::vector<VkExtensionProperties> availableInstanceExtensions = enumerateInstanceExtensions();

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

    if (window != nullptr && !SDL_Vulkan_CreateSurface(window, instance, nullptr, &surface)) {
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
        DeviceCandidate candidate = evaluateDevice(instance, device, surface);
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
    enabled.link(
        caps.meshShader, caps.accelerationStructure, caps.rayTracingPipeline, caps.rayQuery, caps.cooperativeMatrix);
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
    // 협력 행렬 셰이더만 이 모델을 쓴다. 메모리 모델은 모듈마다 선언하므로 나머지 셰이더는 영향받지 않는다.
    enabled.v12.vulkanMemoryModel = caps.vulkanMemoryModel ? VK_TRUE : VK_FALSE;
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
    enabled.accel.accelerationStructureIndirectBuild = caps.accelerationStructureIndirectBuild ? VK_TRUE : VK_FALSE;
    enabled.rayTracing.rayTracingPipeline = caps.rayTracingPipeline ? VK_TRUE : VK_FALSE;
    enabled.rayQuery.rayQuery = caps.rayQuery ? VK_TRUE : VK_FALSE;
    enabled.coop.cooperativeMatrix = caps.cooperativeMatrix ? VK_TRUE : VK_FALSE;

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
    if (surface != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(instance, surface, nullptr);
    }
    vkDestroyInstance(instance, nullptr);
}

} // namespace gfx
