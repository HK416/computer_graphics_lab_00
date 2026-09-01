#include "gfx/upscaler.h"

#ifdef CG_LAB_FSR

#include <cmath>
#include <string>

#include <windows.h>

#include <ffx_api/ffx_api.h>
#include <ffx_api/ffx_api_loader.h>
#include <ffx_api/ffx_upscale.h>
#include <ffx_api/vk/ffx_api_vk.h>

#include <spdlog/spdlog.h>

#include "gfx/context.h"
#include "gfx/resources.h"

namespace gfx {
namespace {

// 서명된 배포 DLL 이름. 링크하지 않고 실행 시점에 찾으므로, 없으면 FSR 만 비활성이 된다.
constexpr const char* FFX_LIBRARY = "amd_fidelityfx_vk.dll";

// DLL 은 프로세스마다 한 번만 찾는다. 편집기가 매 프레임 가용성을 묻는다.
struct FfxLibrary {
    HMODULE module = nullptr;
    ffxFunctions functions{};
    const char* reason = nullptr;

    FfxLibrary() {
        module = LoadLibraryA(FFX_LIBRARY);
        if (module == nullptr) {
            reason = "amd_fidelityfx_vk.dll 없음";
            return;
        }
        ffxLoadFunctions(&functions, module);
        if (functions.CreateContext == nullptr || functions.DestroyContext == nullptr ||
            functions.Dispatch == nullptr || functions.Query == nullptr) {
            reason = "amd_fidelityfx_vk.dll 진입점 없음";
        }
    }
};

const FfxLibrary& ffxLibrary() {
    static FfxLibrary library;
    return library;
}

// 런타임이 내보내는 메시지. 여기로 받지 않으면 컨텍스트 생성 실패 이유를 알 길이 없다.
void ffxMessage(uint32_t type, const wchar_t* message) {
    std::wstring wide = message != nullptr ? message : L"";
    std::string narrow(wide.size(), '?');
    for (size_t i = 0; i < wide.size(); ++i) {
        narrow[i] = wide[i] < 128 ? static_cast<char>(wide[i]) : '?';
    }
    if (type == FFX_API_MESSAGE_TYPE_ERROR) {
        spdlog::error("FSR: {}", narrow);
    } else {
        spdlog::warn("FSR: {}", narrow);
    }
}

FfxApiResource makeResource(const Image& image, VkExtent2D extent, uint32_t state, uint32_t usage) {
    FfxApiResourceDescription description{};
    description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    description.format = ffxApiGetSurfaceFormatVK(image.format);
    description.width = extent.width;
    description.height = extent.height;
    description.depth = 1;
    description.mipCount = 1;
    description.flags = FFX_API_RESOURCE_FLAGS_NONE;
    description.usage = usage;
    return ffxApiGetResourceVK(image.handle, description, state);
}

// FSR 3.1 업스케일러. FidelityFX API 는 같은 진입점으로 하드웨어에 맞는 구현을 고르므로, RDNA4
// 에서는 드라이버가 이 자리에 FSR 4 를 끼워 넣는다. 우리가 붙이는 코드는 그대로다.
class FsrUpscaler final : public TemporalUpscaler {
public:
    FsrUpscaler(Context& context) : context(context) {}

    ~FsrUpscaler() override { destroyContext(); }

    void resize(VkExtent2D render, VkExtent2D display) override {
        // 컨텍스트는 최대 크기로 잡히므로 해상도가 바뀌면 다시 만든다. 히스토리도 함께 버려진다.
        destroyContext();
        renderExtent = render;
        displayExtent = display;

        // 생성 서술자에 엮은 포인터는 컨텍스트를 지울 때까지 살아 있어야 한다. 지역 변수로 두면
        // 런타임이 나중에 되읽을 때 이미 없어진 스택을 본다.
        backend = ffxCreateBackendVKDesc{};
        backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
        backend.vkDevice = context.device;
        backend.vkPhysicalDevice = context.physicalDevice;
        backend.vkDeviceProcAddr = vkGetDeviceProcAddr;

        create = ffxCreateContextDescUpscale{};
        create.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
        create.header.pNext = &backend.header;
        // 톤 매핑 앞의 선형 HDR 을 넘긴다. 깊이는 reverse-Z 무한 원거리다.
        create.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE | FFX_UPSCALE_ENABLE_DEPTH_INVERTED |
                       FFX_UPSCALE_ENABLE_DEPTH_INFINITE | FFX_UPSCALE_ENABLE_AUTO_EXPOSURE;
        create.maxRenderSize = {render.width, render.height};
        create.maxUpscaleSize = {display.width, display.height};
        create.fpMessage = ffxMessage;

        spdlog::info("FSR 컨텍스트 생성: 렌더 {}x{} -> 표시 {}x{}",
                     render.width,
                     render.height,
                     display.width,
                     display.height);
        ffxReturnCode_t result = ffxLibrary().functions.CreateContext(&handle, &create.header, nullptr);
        if (result != FFX_API_RETURN_OK) {
            spdlog::error("FSR 컨텍스트 생성 실패 ({})", result);
            handle = nullptr;
            return;
        }
        spdlog::info("FSR 컨텍스트 준비 완료");
    }

    void evaluate(VkCommandBuffer commandBuffer, const UpscaleInputs& inputs) override {
        if (handle == nullptr) {
            return;
        }

        ffxDispatchDescUpscale dispatch{};
        dispatch.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
        dispatch.commandList = commandBuffer;
        dispatch.color = makeResource(
            *inputs.color, renderExtent, FFX_API_RESOURCE_STATE_COMPUTE_READ, FFX_API_RESOURCE_USAGE_READ_ONLY);
        dispatch.depth = makeResource(
            *inputs.depth, renderExtent, FFX_API_RESOURCE_STATE_COMPUTE_READ, FFX_API_RESOURCE_USAGE_DEPTHTARGET);
        dispatch.motionVectors = makeResource(
            *inputs.velocity, renderExtent, FFX_API_RESOURCE_STATE_COMPUTE_READ, FFX_API_RESOURCE_USAGE_READ_ONLY);
        dispatch.output = makeResource(
            *inputs.output, displayExtent, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS, FFX_API_RESOURCE_USAGE_UAV);

        dispatch.jitterOffset = {inputs.jitter.x, inputs.jitter.y};
        // 모션 벡터를 화면 UV 로 담았으므로 렌더 픽셀 단위로 되돌린다. 방향은 현재에서 이전으로,
        // FSR 이 기대하는 것과 같다.
        dispatch.motionVectorScale = {static_cast<float>(renderExtent.width),
                                      static_cast<float>(renderExtent.height)};
        dispatch.renderSize = {renderExtent.width, renderExtent.height};
        dispatch.upscaleSize = {displayExtent.width, displayExtent.height};
        // 선명화는 편집기의 공간 업스케일 쪽 값과 뜻이 달라 붙이지 않는다.
        dispatch.enableSharpening = false;
        dispatch.sharpness = 0.0F;
        dispatch.frameTimeDelta = inputs.deltaSeconds * 1000.0F;
        dispatch.preExposure = 1.0F;
        dispatch.reset = inputs.reset;
        dispatch.cameraNear = inputs.nearPlane;
        // 무한 원거리 투영이라 원평면이 없다. DEPTH_INFINITE 를 켰으므로 이 값은 쓰이지 않는다.
        dispatch.cameraFar = INFINITY;
        dispatch.cameraFovAngleVertical = inputs.verticalFovRadians;
        dispatch.viewSpaceToMetersFactor = 1.0F;

        ffxReturnCode_t result = ffxLibrary().functions.Dispatch(&handle, &dispatch.header);
        if (result != FFX_API_RETURN_OK && !dispatchFailed) {
            // 매 프레임 같은 오류를 쏟지 않도록 한 번만 남긴다.
            spdlog::error("FSR 디스패치 실패 ({})", result);
            dispatchFailed = true;
        }
    }

    bool valid() const { return handle != nullptr; }

private:
    void destroyContext() {
        if (handle != nullptr) {
            ffxLibrary().functions.DestroyContext(&handle, nullptr);
            handle = nullptr;
        }
        dispatchFailed = false;
    }

    Context& context;
    ffxContext handle = nullptr;
    ffxCreateContextDescUpscale create{};
    ffxCreateBackendVKDesc backend{};
    VkExtent2D renderExtent{};
    VkExtent2D displayExtent{};
    bool dispatchFailed = false;
};

} // namespace

const char* fsrUnavailableReason() {
    return ffxLibrary().reason;
}

std::unique_ptr<TemporalUpscaler> createFsrUpscaler(Context& context, BindlessTextures&) {
    if (ffxLibrary().reason != nullptr) {
        return nullptr;
    }
    return std::make_unique<FsrUpscaler>(context);
}

} // namespace gfx

#else

namespace gfx {

const char* fsrUnavailableReason() {
    return "FidelityFX SDK 미포함";
}

std::unique_ptr<TemporalUpscaler> createFsrUpscaler(Context&, BindlessTextures&) {
    return nullptr;
}

} // namespace gfx

#endif
