#include "gfx/upscaler.h"

#include "gfx/context.h"

namespace gfx {
namespace {
// NVIDIA 가 정한 사전 설정별 렌더 배율.
constexpr float QUALITY_SCALES[DLSS_QUALITY_COUNT] = {1.0F, 1.0F / 1.5F, 1.0F / 1.72F, 0.5F, 1.0F / 3.0F};
constexpr const char* QUALITY_NAMES[DLSS_QUALITY_COUNT] = {
    "DLAA (1.00)", "품질 (0.67)", "균형 (0.58)", "성능 (0.50)", "울트라 성능 (0.33)"};
} // namespace

float dlssQualityScale(DlssQuality quality) {
    return QUALITY_SCALES[static_cast<uint32_t>(quality) % DLSS_QUALITY_COUNT];
}

DlssQuality dlssQualityForScale(float scale) {
    // 경계는 이웃한 두 사전 설정 배율의 중간값이다.
    if (scale >= 0.833F) {
        return DlssQuality::DLAA;
    }
    if (scale >= 0.624F) {
        return DlssQuality::QUALITY;
    }
    if (scale >= 0.541F) {
        return DlssQuality::BALANCED;
    }
    if (scale >= 0.417F) {
        return DlssQuality::PERFORMANCE;
    }
    return DlssQuality::ULTRA_PERFORMANCE;
}

const char* dlssQualityName(DlssQuality quality) {
    return QUALITY_NAMES[static_cast<uint32_t>(quality) % DLSS_QUALITY_COUNT];
}

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
    case Upscaler::DLSS_RR: {
        const char* reason = dlssRayReconstructionUnavailableReason(context);
        return {kind, "DLSS Ray Reconstruction", reason == nullptr, reason == nullptr ? "" : reason};
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
    if (kind == Upscaler::DLSS_RR) {
        return createDlssRayReconstruction(context, bindless);
    }
    return nullptr;
}

} // namespace gfx
