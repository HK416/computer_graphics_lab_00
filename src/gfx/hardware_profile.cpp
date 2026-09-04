#include "gfx/hardware_profile.h"

#include <format>

namespace gfx {

namespace {

constexpr uint64_t GIGABYTE = 1024ULL * 1024ULL * 1024ULL;

// 등급을 가르는 장치 메모리 «크기». 외장 그래픽과 가상 장치에 쓴다.
constexpr uint64_t HIGH_TIER_MEMORY = 6 * GIGABYTE;
constexpr uint64_t MEDIUM_TIER_MEMORY = 3 * GIGABYTE;

// 이 화소 수를 넘으면 한 등급 보수적으로 고른다. 1440p 를 넘는 크기다.
constexpr uint64_t HIGH_RESOLUTION_PIXELS = 2560ULL * 1440ULL;

HardwareTier tierOf(const ProfileInputs& inputs, std::vector<std::string>& reasons) {
    double memoryGb = static_cast<double>(inputs.deviceMemoryBytes) / static_cast<double>(GIGABYTE);
    auto byMemory = [&] {
        if (inputs.deviceMemoryBytes >= HIGH_TIER_MEMORY) {
            return HardwareTier::HIGH;
        }
        return inputs.deviceMemoryBytes >= MEDIUM_TIER_MEMORY ? HardwareTier::MEDIUM : HardwareTier::LOW;
    };

    HardwareTier tier = HardwareTier::MEDIUM;
    switch (inputs.deviceType) {
    case ProfileInputs::DeviceType::DISCRETE:
        tier = byMemory();
        reasons.push_back(std::format("외장 그래픽, 장치 메모리 {:.1f} GB", memoryGb));
        break;
    case ProfileInputs::DeviceType::VIRTUAL:
        // 데이터센터 GPU 가 가상 장치로 보이기도 한다. 메모리로 재는 편이 낫다.
        tier = byMemory();
        reasons.push_back(std::format("가상 장치, 장치 메모리 {:.1f} GB", memoryGb));
        break;
    case ProfileInputs::DeviceType::INTEGRATED:
        // 내장 그래픽의 «장치 메모리» 는 시스템 메모리를 그대로 보고하는 드라이버가 많아 외장과
        // 견줄 수 없다. 대역폭이 CPU 와 나뉘는 것도 감안해 늘 낮은 등급으로 둔다.
        tier = HardwareTier::LOW;
        reasons.push_back("내장 그래픽은 대역폭을 CPU 와 나눠 쓴다");
        break;
    case ProfileInputs::DeviceType::CPU:
        tier = HardwareTier::LOW;
        reasons.push_back("소프트웨어 장치");
        break;
    case ProfileInputs::DeviceType::OTHER:
        reasons.push_back("알 수 없는 장치라 중간 등급으로 본다");
        break;
    }

    auto pixels = static_cast<uint64_t>(inputs.displayWidth) * static_cast<uint64_t>(inputs.displayHeight);
    if (pixels > HIGH_RESOLUTION_PIXELS && tier != HardwareTier::LOW) {
        tier = tier == HardwareTier::HIGH ? HardwareTier::MEDIUM : HardwareTier::LOW;
        reasons.push_back(std::format("{}x{} 는 넓어 한 등급 낮춘다", inputs.displayWidth, inputs.displayHeight));
    }
    return tier;
}

} // namespace

HardwareProfile chooseProfile(const ProfileInputs& inputs, AutoTune level) {
    HardwareProfile profile;
    if (level == AutoTune::OFF) {
        profile.reasons.push_back("자동 튜닝이 꺼져 있어 기본값을 그대로 쓴다");
        return profile;
    }

    bool aggressive = level == AutoTune::AGGRESSIVE;
    profile.tier = tierOf(inputs, profile.reasons);
    // 광선 반사와 광선 그림자는 하위 가속 구조를 요구하고, 이 저장소는 그것을 RayTracer 가 세운다.
    // RayTracer 는 광선 추적 파이프라인이 있어야 만들어지므로 광선 질의만으로는 켤 수 없다.
    bool rayReady = inputs.rayQuery && inputs.rayTracingPipeline;

    switch (profile.tier) {
    case HardwareTier::LOW:
        // 낮은 배율로 그리고 키운다. 시간축 업스케일은 공간 업스케일보다 잘 버텨 더 낮출 수 있다.
        //
        // ponytail: 여기서 aggressive 는 «여유를 재서» 올리는 것이 아니라 사용자가 프레임 대신
        // 화질을 택했다는 뜻이다. 프레임 시간을 보고 정하려면 첫 몇 초를 재서 다시 고르는 고리가
        // 있어야 한다.
        profile.upscaler = inputs.bestTemporalUpscaler;
        profile.renderScale = aggressive ? 0.67F : 0.5F;
        profile.ssaoSamples = aggressive ? 12 : 8;
        profile.shadowCascades = 2;
        profile.fluidParticleLimit = 8192;
        break;
    case HardwareTier::MEDIUM:
        profile.renderScale = 1.0F;
        if (!aggressive) {
            // 그대로 그려도 되지만 조금 낮춰 여유를 남긴다. 시간축 업스케일이라 손실이 적다.
            profile.upscaler = inputs.bestTemporalUpscaler;
            profile.renderScale = 0.85F;
        }
        profile.ssaoSamples = 16;
        profile.shadowCascades = 3;
        profile.reflections = aggressive && rayReady;
        profile.fluidParticleLimit = 16384;
        break;
    case HardwareTier::HIGH:
        profile.renderScale = 1.0F;
        profile.ssaoSamples = aggressive ? 32 : 24;
        profile.shadowCascades = 4;
        profile.reflections = rayReady;
        profile.rayQueryShadows = aggressive && rayReady;
        profile.fluidParticleLimit = 32768;
        break;
    }

    if (profile.renderScale < 1.0F) {
        profile.reasons.push_back(
            std::format("렌더 배율 {:.2f}, 업스케일러 {}", profile.renderScale, profile.upscaler));
    }
    if (profile.reflections) {
        profile.reasons.push_back("광선 기능이 있어 광선 반사를 켠다");
    } else if (rayReady) {
        profile.reasons.push_back("광선 기능은 있지만 이 등급에서는 반사를 끈다");
    } else {
        profile.reasons.push_back("광선 기능이 없어 반사와 광선 그림자를 쓸 수 없다");
    }
    if (profile.rayQueryShadows) {
        profile.reasons.push_back("가까운 그림자를 광선으로 판정한다");
    }
    if (!inputs.meshShader) {
        profile.reasons.push_back("mesh shader 가 없어 고전 경로로 돈다");
    }
    profile.reasons.push_back(std::format("그림자 캐스케이드 {}, SSAO 표본 {}, 유체 입자 상한 {}",
                                          profile.shadowCascades,
                                          profile.ssaoSamples,
                                          profile.fluidParticleLimit));
    return profile;
}

const char* tierName(HardwareTier tier) {
    switch (tier) {
    case HardwareTier::LOW:
        return "낮음";
    case HardwareTier::MEDIUM:
        return "중간";
    case HardwareTier::HIGH:
        return "높음";
    }
    return "알 수 없음";
}

const char* autoTuneName(AutoTune level) {
    switch (level) {
    case AutoTune::OFF:
        return "끔";
    case AutoTune::SAFE:
        return "안전";
    case AutoTune::AGGRESSIVE:
        return "공격적";
    }
    return "알 수 없음";
}

bool parseAutoTune(std::string_view text, AutoTune& out) {
    if (text == "off") {
        out = AutoTune::OFF;
        return true;
    }
    if (text == "safe") {
        out = AutoTune::SAFE;
        return true;
    }
    if (text == "aggressive") {
        out = AutoTune::AGGRESSIVE;
        return true;
    }
    return false;
}

} // namespace gfx
