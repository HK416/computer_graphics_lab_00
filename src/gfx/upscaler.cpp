#include "gfx/upscaler.h"

#include "gfx/context.h"

namespace gfx {
UpscalerInfo upscalerInfo(Upscaler kind, const Context& context) {
    switch (kind) {
    case Upscaler::NONE:
        return {kind, "없음 (통과)", true, ""};
    case Upscaler::SPATIAL:
        return {kind, "내장 공간 업스케일", true, ""};
    case Upscaler::TAAU:
        return {kind, "내장 시간축 업스케일", true, ""};
    case Upscaler::FSR: {
        const char* reason = fsrUnavailableReason();
        return {kind, "FSR 3.1", reason == nullptr, reason == nullptr ? "" : reason};
    }
    case Upscaler::DLSS: {
        const char* reason = dlssUnavailableReason(context);
        return {kind, "DLSS", reason == nullptr, reason == nullptr ? "" : reason};
    }
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
    if (kind == Upscaler::FSR) {
        return createFsrUpscaler(context, bindless);
    }
    if (kind == Upscaler::DLSS) {
        return createDlssUpscaler(context, bindless);
    }
    return nullptr;
}

} // namespace gfx
