#include <cassert>
#include <cstdio>
#include <string>

#include "gfx/hardware_profile.h"

namespace {

constexpr uint64_t GIGABYTE = 1024ULL * 1024ULL * 1024ULL;

gfx::ProfileInputs discrete(uint64_t gigabytes) {
    gfx::ProfileInputs inputs;
    inputs.deviceType = gfx::ProfileInputs::DeviceType::DISCRETE;
    inputs.deviceMemoryBytes = gigabytes * GIGABYTE;
    inputs.meshShader = true;
    inputs.rayQuery = true;
    inputs.rayTracingPipeline = true;
    return inputs;
}

gfx::ProfileInputs integrated(uint64_t gigabytes) {
    gfx::ProfileInputs inputs;
    inputs.deviceType = gfx::ProfileInputs::DeviceType::INTEGRATED;
    inputs.deviceMemoryBytes = gigabytes * GIGABYTE;
    return inputs;
}

bool mentions(const gfx::HardwareProfile& profile, const std::string& fragment) {
    for (const std::string& reason : profile.reasons) {
        if (reason.find(fragment) != std::string::npos) {
            return true;
        }
    }
    return false;
}

} // namespace

int main() {
    // ---- 등급 판정 ----
    assert(gfx::chooseProfile(discrete(8), gfx::AutoTune::SAFE).tier == gfx::HardwareTier::HIGH);
    assert(gfx::chooseProfile(discrete(4), gfx::AutoTune::SAFE).tier == gfx::HardwareTier::MEDIUM);
    assert(gfx::chooseProfile(discrete(2), gfx::AutoTune::SAFE).tier == gfx::HardwareTier::LOW);
    // 경계값 그 자체. 6GB 와 3GB 는 그 등급에 들어간다.
    assert(gfx::chooseProfile(discrete(6), gfx::AutoTune::SAFE).tier == gfx::HardwareTier::HIGH);
    assert(gfx::chooseProfile(discrete(3), gfx::AutoTune::SAFE).tier == gfx::HardwareTier::MEDIUM);

    // 내장 그래픽은 보고하는 메모리를 외장과 견줄 수 없어 늘 낮은 등급이다.
    assert(gfx::chooseProfile(integrated(16), gfx::AutoTune::SAFE).tier == gfx::HardwareTier::LOW);
    assert(gfx::chooseProfile(integrated(1), gfx::AutoTune::SAFE).tier == gfx::HardwareTier::LOW);

    // 알 수 없는 장치는 중간으로 본다.
    {
        gfx::ProfileInputs unknown;
        unknown.deviceMemoryBytes = 16 * GIGABYTE;
        gfx::HardwareProfile profile = gfx::chooseProfile(unknown, gfx::AutoTune::SAFE);
        assert(profile.tier == gfx::HardwareTier::MEDIUM);
        assert(mentions(profile, "알 수 없는"));
    }

    // 가상 장치는 메모리로 잰다. 데이터센터 GPU 가 이렇게 보이기도 한다.
    {
        gfx::ProfileInputs virtualGpu;
        virtualGpu.deviceType = gfx::ProfileInputs::DeviceType::VIRTUAL;
        virtualGpu.deviceMemoryBytes = 24 * GIGABYTE;
        assert(gfx::chooseProfile(virtualGpu, gfx::AutoTune::SAFE).tier == gfx::HardwareTier::HIGH);
    }

    {
        gfx::ProfileInputs software;
        software.deviceType = gfx::ProfileInputs::DeviceType::CPU;
        software.deviceMemoryBytes = 32 * GIGABYTE;
        gfx::HardwareProfile profile = gfx::chooseProfile(software, gfx::AutoTune::AGGRESSIVE);
        assert(profile.tier == gfx::HardwareTier::LOW && "메모리가 많아도 소프트웨어 장치는 낮은 등급이다");
        assert(mentions(profile, "소프트웨어"));
    }

    // 넓은 화면은 한 등급 낮춘다.
    {
        gfx::ProfileInputs wide = discrete(8);
        wide.displayWidth = 3840;
        wide.displayHeight = 2160;
        gfx::HardwareProfile profile = gfx::chooseProfile(wide, gfx::AutoTune::SAFE);
        assert(profile.tier == gfx::HardwareTier::MEDIUM && "4K 는 높은 등급을 중간으로 내린다");
        assert(mentions(profile, "3840x2160"));
    }

    // ---- OFF 는 아무것도 바꾸지 않는다 ----
    {
        gfx::HardwareProfile off = gfx::chooseProfile(discrete(2), gfx::AutoTune::OFF);
        gfx::HardwareProfile defaults;
        assert(off.tier == defaults.tier);
        assert(off.renderScale == defaults.renderScale);
        assert(off.upscaler == defaults.upscaler);
        assert(off.ssaoSamples == defaults.ssaoSamples);
        assert(off.shadowCascades == defaults.shadowCascades);
        assert(off.reflections == defaults.reflections);
        assert(off.rayQueryShadows == defaults.rayQueryShadows);
        assert(off.fluidParticleLimit == defaults.fluidParticleLimit);
        assert(mentions(off, "꺼져 있어"));
    }

    // ---- 낮은 등급은 배율을 내리고 무거운 기능을 끈다 ----
    {
        gfx::HardwareProfile low = gfx::chooseProfile(discrete(2), gfx::AutoTune::SAFE);
        assert(low.renderScale < 1.0F && "낮은 등급은 작게 그린다");
        assert(low.upscaler == 2 && "시간축 업스케일로 키운다");
        assert(!low.reflections && !low.rayQueryShadows && "광선 질의가 있어도 켜지 않는다");
        assert(low.fluidParticleLimit < 32768);
        assert(low.shadowCascades == 2);
    }

    // 더 나은 시간축 업스케일러가 있으면 그것을 고른다.
    {
        gfx::ProfileInputs inputs = discrete(2);
        inputs.bestTemporalUpscaler = 3;
        assert(gfx::chooseProfile(inputs, gfx::AutoTune::SAFE).upscaler == 3);
    }

    // 넓은 화면 규칙은 낮은 등급을 더 내리지 않는다.
    {
        gfx::ProfileInputs wideLow = discrete(2);
        wideLow.displayWidth = 3840;
        wideLow.displayHeight = 2160;
        gfx::HardwareProfile profile = gfx::chooseProfile(wideLow, gfx::AutoTune::SAFE);
        assert(profile.tier == gfx::HardwareTier::LOW);
        assert(!mentions(profile, "한 등급 낮춘다") && "이미 가장 낮으면 근거도 남기지 않는다");
    }

    // ---- 중간 등급 ----
    {
        gfx::HardwareProfile safe = gfx::chooseProfile(discrete(4), gfx::AutoTune::SAFE);
        assert(safe.renderScale < 1.0F && safe.upscaler == 2 && "안전 단계는 여유를 남긴다");
        assert(safe.shadowCascades == 3 && safe.ssaoSamples == 16);
        assert(!safe.reflections && "중간 등급의 안전 단계는 반사를 켜지 않는다");

        gfx::HardwareProfile aggressive = gfx::chooseProfile(discrete(4), gfx::AutoTune::AGGRESSIVE);
        assert(aggressive.renderScale == 1.0F && "공격적 단계는 그대로 그린다");
        assert(aggressive.upscaler == 1 && "시간축 업스케일을 쓰지 않으면 공간 업스케일로 남는다");
        assert(aggressive.reflections && "광선 기능이 있으면 반사를 켠다");
    }

    // ---- 높은 등급 ----
    {
        gfx::HardwareProfile high = gfx::chooseProfile(discrete(12), gfx::AutoTune::SAFE);
        assert(high.renderScale == 1.0F && "높은 등급은 그대로 그린다");
        assert(high.reflections && "광선 질의가 있으면 반사를 켠다");
        assert(!high.rayQueryShadows && "안전 단계는 광선 그림자까지 켜지 않는다");
        assert(high.shadowCascades == 4);

        gfx::HardwareProfile aggressive = gfx::chooseProfile(discrete(12), gfx::AutoTune::AGGRESSIVE);
        assert(aggressive.rayQueryShadows && "공격적 단계는 광선 그림자를 켠다");
        assert(aggressive.ssaoSamples > high.ssaoSamples && "공격적 단계가 화질을 더 올린다");
    }

    // 광선 기능이 없으면 어떤 등급에서도 켜지 않는다. 광선 질의만 있고 추적 파이프라인이 없는
    // 장치도 마찬가지다. 하위 가속 구조를 RayTracer 가 세우는데 그것이 파이프라인을 요구한다.
    {
        gfx::ProfileInputs raster = discrete(12);
        raster.rayQuery = false;
        raster.rayTracingPipeline = false;
        gfx::HardwareProfile profile = gfx::chooseProfile(raster, gfx::AutoTune::AGGRESSIVE);
        assert(!profile.reflections && !profile.rayQueryShadows);
        assert(mentions(profile, "광선 기능이 없어"));

        gfx::ProfileInputs queryOnly = discrete(12);
        queryOnly.rayTracingPipeline = false;
        gfx::HardwareProfile partial = gfx::chooseProfile(queryOnly, gfx::AutoTune::AGGRESSIVE);
        assert(!partial.reflections && !partial.rayQueryShadows && "광선 질의만으로는 켜지지 않는다");
    }

    // mesh shader 가 없으면 근거에 적는다.
    {
        gfx::ProfileInputs classic = discrete(12);
        classic.meshShader = false;
        assert(mentions(gfx::chooseProfile(classic, gfx::AutoTune::SAFE), "mesh shader"));
    }

    // ---- 공격적 단계가 낮은 등급을 더 나쁘게 만들지는 않는다 ----
    {
        gfx::HardwareProfile safe = gfx::chooseProfile(discrete(2), gfx::AutoTune::SAFE);
        gfx::HardwareProfile aggressive = gfx::chooseProfile(discrete(2), gfx::AutoTune::AGGRESSIVE);
        assert(aggressive.renderScale >= safe.renderScale);
        assert(aggressive.ssaoSamples >= safe.ssaoSamples);
    }

    // ---- 이름 ----
    {
        // 편집기와 로그가 그대로 보여 주므로 빈 문자열이나 «알 수 없음» 이 새면 안 된다.
        for (uint32_t i = 0; i <= static_cast<uint32_t>(gfx::HardwareTier::HIGH); ++i) {
            std::string name = gfx::tierName(static_cast<gfx::HardwareTier>(i));
            assert(!name.empty() && name != "알 수 없음");
        }
        for (uint32_t i = 0; i <= static_cast<uint32_t>(gfx::AutoTune::AGGRESSIVE); ++i) {
            std::string name = gfx::autoTuneName(static_cast<gfx::AutoTune>(i));
            assert(!name.empty() && name != "알 수 없음");
        }
    }

    // ---- 인자 해석 ----
    {
        gfx::AutoTune level = gfx::AutoTune::SAFE;
        assert(gfx::parseAutoTune("off", level) && level == gfx::AutoTune::OFF);
        assert(gfx::parseAutoTune("aggressive", level) && level == gfx::AutoTune::AGGRESSIVE);
        assert(gfx::parseAutoTune("safe", level) && level == gfx::AutoTune::SAFE);
        gfx::AutoTune untouched = gfx::AutoTune::AGGRESSIVE;
        assert(!gfx::parseAutoTune("빠르게", untouched) && untouched == gfx::AutoTune::AGGRESSIVE &&
               "모르는 값은 건드리지 않는다");
    }

    std::printf("하드웨어 프로파일 자체 점검 통과\n");
    return 0;
}
