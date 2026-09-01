#include "gfx/upscaler.h"

#ifdef CG_LAB_DLSS

#include <array>
#include <filesystem>
#include <string>

#include <nvsdk_ngx_defs.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_vk.h>

#include <spdlog/spdlog.h>

#include "gfx/context.h"
#include "gfx/resources.h"

namespace gfx {
namespace {

// NGX 는 등록된 애플리케이션 ID 나 프로젝트 ID 를 요구한다. 등록하지 않은 저장소라 고정 UUID 를
// 쓴다. 개발용으로는 이걸로 충분하다.
constexpr const char* NGX_PROJECT_ID = "a1b2c3d4-5e6f-4a7b-8c9d-0e1f2a3b4c5d";

// NGX 초기화는 프로세스마다 한 번이다. 여러 업스케일러가 생겼다 사라져도 다시 하지 않는다.
struct NgxRuntime {
    bool initialized = false;
    bool superSamplingAvailable = false;
    NVSDK_NGX_Parameter* capabilities = nullptr;
    VkDevice device = VK_NULL_HANDLE;
    std::string reason = "초기화하지 않음";

    void start(Context& context) {
        if (initialized) {
            return;
        }
        // 가중치 파일과 로그를 둘 곳. 실행 파일 옆에 nvngx_dlss.dll 이 있어야 한다.
        std::filesystem::path dataPath = std::filesystem::current_path();
        NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_Init_with_ProjectID(NGX_PROJECT_ID,
                                                                       NVSDK_NGX_ENGINE_TYPE_CUSTOM,
                                                                       "0.1.0",
                                                                       dataPath.wstring().c_str(),
                                                                       context.instance,
                                                                       context.physicalDevice,
                                                                       context.device);
        if (NVSDK_NGX_FAILED(result)) {
            reason = "NGX 초기화 실패";
            spdlog::warn("DLSS: NGX 초기화 실패 (0x{:08X})", static_cast<uint32_t>(result));
            return;
        }
        device = context.device;
        initialized = true;

        result = NVSDK_NGX_VULKAN_GetCapabilityParameters(&capabilities);
        if (NVSDK_NGX_FAILED(result)) {
            reason = "NGX 성능 조회 실패";
            return;
        }

        int available = 0;
        capabilities->Get(NVSDK_NGX_Parameter_SuperSampling_Available, &available);
        if (available == 0) {
            int reasonCode = 0;
            capabilities->Get(NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &reasonCode);
            reason = "이 장치/드라이버에서 DLSS 미지원";
            spdlog::warn("DLSS 사용 불가 (초기화 결과 0x{:08X})", static_cast<uint32_t>(reasonCode));
            return;
        }
        superSamplingAvailable = true;
        reason.clear();
    }

    // 장치가 살아 있을 때 명시적으로 부른다. 정적 소멸자는 장치가 없어진 뒤에 돌아 쓸 수 없다.
    void stop() {
        if (capabilities != nullptr) {
            NVSDK_NGX_VULKAN_DestroyParameters(capabilities);
            capabilities = nullptr;
        }
        if (initialized) {
            NVSDK_NGX_VULKAN_Shutdown1(device);
            initialized = false;
        }
        superSamplingAvailable = false;
        reason = "종료됨";
    }
};

NgxRuntime& ngxRuntime() {
    static NgxRuntime runtime;
    return runtime;
}

NVSDK_NGX_Resource_VK makeResource(const Image& image, VkExtent2D extent, bool readWrite) {
    VkImageSubresourceRange range{};
    range.aspectMask =
        image.format == VK_FORMAT_D32_SFLOAT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = 1;
    range.layerCount = 1;
    return NVSDK_NGX_Create_ImageView_Resource_VK(
        image.view, image.handle, range, image.format, extent.width, extent.height, readWrite);
}

class DlssUpscaler final : public TemporalUpscaler {
public:
    DlssUpscaler(Context& context) : context(context) {}

    ~DlssUpscaler() override { releaseFeature(); }

    void resize(VkExtent2D render, VkExtent2D display) override {
        releaseFeature();
        renderExtent = render;
        displayExtent = display;
        // 기능 생성은 명령 버퍼를 요구하므로 다음 평가 때 만든다.
        needsCreate = true;
    }

    void evaluate(VkCommandBuffer commandBuffer, const UpscaleInputs& inputs) override {
        if (needsCreate && !createFeature(commandBuffer)) {
            return;
        }
        if (handle == nullptr) {
            return;
        }

        NVSDK_NGX_Resource_VK color = makeResource(*inputs.color, renderExtent, false);
        NVSDK_NGX_Resource_VK depth = makeResource(*inputs.depth, renderExtent, false);
        NVSDK_NGX_Resource_VK velocity = makeResource(*inputs.velocity, renderExtent, false);
        NVSDK_NGX_Resource_VK output = makeResource(*inputs.output, displayExtent, true);

        NVSDK_NGX_VK_DLSS_Eval_Params params{};
        params.Feature.pInColor = &color;
        params.Feature.pInOutput = &output;
        params.Feature.InSharpness = 0.0F;
        params.pInDepth = &depth;
        params.pInMotionVectors = &velocity;
        // 지터는 렌더 픽셀 단위다. 화면에서 내용을 미는 방향과 부호를 맞춘다.
        params.InJitterOffsetX = inputs.jitter.x;
        params.InJitterOffsetY = inputs.jitter.y;
        params.InRenderSubrectDimensions = {renderExtent.width, renderExtent.height};
        params.InReset = inputs.reset ? 1 : 0;
        // 모션 벡터를 화면 UV 로 담았으므로 렌더 픽셀 단위로 되돌린다.
        params.InMVScaleX = static_cast<float>(renderExtent.width);
        params.InMVScaleY = static_cast<float>(renderExtent.height);
        params.InPreExposure = 1.0F;
        params.InFrameTimeDeltaInMsec = inputs.deltaSeconds * 1000.0F;

        NVSDK_NGX_Result result = NGX_VULKAN_EVALUATE_DLSS_EXT(commandBuffer, handle, ngxRuntime().capabilities, &params);
        if (NVSDK_NGX_FAILED(result) && !evaluateFailed) {
            // 매 프레임 같은 오류를 쏟지 않도록 한 번만 남긴다.
            spdlog::error("DLSS 평가 실패 (0x{:08X})", static_cast<uint32_t>(result));
            evaluateFailed = true;
        }
    }

    bool ready() const override { return handle != nullptr || needsCreate; }

private:
    bool createFeature(VkCommandBuffer commandBuffer) {
        needsCreate = false;
        NVSDK_NGX_DLSS_Create_Params create{};
        create.Feature.InWidth = renderExtent.width;
        create.Feature.InHeight = renderExtent.height;
        create.Feature.InTargetWidth = displayExtent.width;
        create.Feature.InTargetHeight = displayExtent.height;
        // 배율은 렌더 배율이 이미 정한다. NGX 에는 그 배율에 가장 가까운 사전 설정을 알려 준다.
        create.Feature.InPerfQualityValue = qualityForRatio();
        // 톤 매핑 앞의 선형 HDR 을 넘기고, 깊이는 reverse-Z 이며, 변위는 렌더 해상도에 지터가
        // 빠진 값이다.
        create.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                                      NVSDK_NGX_DLSS_Feature_Flags_DepthInverted |
                                      NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;

        NVSDK_NGX_Result result =
            NGX_VULKAN_CREATE_DLSS_EXT(commandBuffer, 1, 1, &handle, ngxRuntime().capabilities, &create);
        if (NVSDK_NGX_FAILED(result)) {
            spdlog::error("DLSS 기능 생성 실패 (0x{:08X})", static_cast<uint32_t>(result));
            handle = nullptr;
            return false;
        }
        spdlog::info("DLSS 준비 완료: 렌더 {}x{} -> 표시 {}x{}",
                     renderExtent.width,
                     renderExtent.height,
                     displayExtent.width,
                     displayExtent.height);
        evaluateFailed = false;
        return true;
    }

    NVSDK_NGX_PerfQuality_Value qualityForRatio() const {
        float ratio = displayExtent.width > 0 && renderExtent.width > 0
                          ? static_cast<float>(displayExtent.width) / static_cast<float>(renderExtent.width)
                          : 1.0F;
        if (ratio <= 1.05F) {
            return NVSDK_NGX_PerfQuality_Value_DLAA;
        }
        if (ratio <= 1.4F) {
            return NVSDK_NGX_PerfQuality_Value_MaxQuality;
        }
        if (ratio <= 1.8F) {
            return NVSDK_NGX_PerfQuality_Value_Balanced;
        }
        if (ratio <= 2.5F) {
            return NVSDK_NGX_PerfQuality_Value_MaxPerf;
        }
        return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
    }

    void releaseFeature() {
        if (handle != nullptr) {
            NVSDK_NGX_VULKAN_ReleaseFeature(handle);
            handle = nullptr;
        }
        needsCreate = false;
        evaluateFailed = false;
    }

    Context& context;
    NVSDK_NGX_Handle* handle = nullptr;
    VkExtent2D renderExtent{};
    VkExtent2D displayExtent{};
    bool needsCreate = false;
    bool evaluateFailed = false;
};

} // namespace

void dlssRequiredExtensions(std::vector<const char*>& instanceExtensions,
                            std::vector<const char*>& deviceExtensions) {
    unsigned int instanceCount = 0;
    unsigned int deviceCount = 0;
    const char** instanceNames = nullptr;
    const char** deviceNames = nullptr;
    if (NVSDK_NGX_FAILED(
            NVSDK_NGX_VULKAN_RequiredExtensions(&instanceCount, &instanceNames, &deviceCount, &deviceNames))) {
        return;
    }
    instanceExtensions.assign(instanceNames, instanceNames + instanceCount);
    deviceExtensions.assign(deviceNames, deviceNames + deviceCount);
}

const char* dlssUnavailableReason(const Context& context) {
    if (context.properties.vendorID != 0x10DE) {
        return "NVIDIA 장치 아님";
    }
    // 상수 문자열을 돌려줘야 해서 사유를 정적 버퍼에 옮긴다. 편집기가 매 프레임 읽는다.
    static std::string reason;
    NgxRuntime& runtime = ngxRuntime();
    reason = runtime.reason;
    return runtime.superSamplingAvailable ? nullptr : reason.c_str();
}

std::unique_ptr<TemporalUpscaler> createDlssUpscaler(Context& context, BindlessTextures&) {
    ngxRuntime().start(context);
    if (!ngxRuntime().superSamplingAvailable) {
        return nullptr;
    }
    return std::make_unique<DlssUpscaler>(context);
}

void startDlssRuntime(Context& context) {
    if (context.properties.vendorID == 0x10DE) {
        ngxRuntime().start(context);
    }
}

void shutdownDlssRuntime() {
    ngxRuntime().stop();
}

} // namespace gfx

#else

#include "gfx/context.h"

namespace gfx {

void dlssRequiredExtensions(std::vector<const char*>&, std::vector<const char*>&) {}

const char* dlssUnavailableReason(const Context& context) {
    return context.properties.vendorID == 0x10DE ? "NGX SDK 미포함" : "NVIDIA 장치 아님";
}

std::unique_ptr<TemporalUpscaler> createDlssUpscaler(Context&, BindlessTextures&) {
    return nullptr;
}

void startDlssRuntime(Context&) {}

void shutdownDlssRuntime() {}

} // namespace gfx

#endif
