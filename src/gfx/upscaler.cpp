#include "gfx/upscaler.h"

#include "gfx/context.h"

namespace gfx {
namespace {
constexpr uint32_t NVIDIA_VENDOR_ID = 0x10DE;
} // namespace

UpscalerInfo upscalerInfo(Upscaler kind, const Context& context) {
    switch (kind) {
    case Upscaler::NONE:
        return {kind, "없음 (통과)", true, ""};
    case Upscaler::SPATIAL:
        return {kind, "내장 공간 업스케일", true, ""};
    case Upscaler::TAAU:
        return {kind, "내장 시간축 업스케일", true, ""};
    case Upscaler::FSR:
        return {kind, "FSR 3.1", false, "FidelityFX SDK 미포함"};
    case Upscaler::DLSS:
        return {kind,
                "DLSS",
                false,
                context.properties.vendorID == NVIDIA_VENDOR_ID ? "NGX SDK 미포함" : "NVIDIA 장치 아님"};
    }
    return {kind, "알 수 없음", false, "지원하지 않는 방식"};
}

std::unique_ptr<TemporalUpscaler> createUpscaler(Upscaler kind, Context& context, BindlessTextures& bindless) {
    if (!upscalerInfo(kind, context).available || !isTemporal(kind)) {
        return nullptr;
    }
    if (kind == Upscaler::TAAU) {
        return createTaauUpscaler(context, bindless);
    }
    return nullptr;
}

} // namespace gfx
