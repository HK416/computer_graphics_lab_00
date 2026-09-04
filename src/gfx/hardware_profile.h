#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace gfx {

// 자동 튜닝의 세기. 실행 인자 --auto-tune 이 정한다.
enum class AutoTune : uint32_t {
    // 아무것도 바꾸지 않는다. 코드의 기본값 그대로 시작한다.
    OFF = 0,
    // 이 기기에서 프레임이 흔들리지 않을 만한 설정.
    SAFE,
    // 여유가 있으면 화질을 더 올린다. 프레임이 흔들릴 수 있다.
    AGGRESSIVE,
};

// 기기 등급.
enum class HardwareTier : uint32_t { LOW = 0, MEDIUM, HIGH };

// 판정에 쓰는 하드웨어 사실. Vulkan 타입을 쓰지 않아 테스트에서 손으로 채울 수 있다. 그러려고
// 이 파일과 hardware_profile.cpp 를 Vulkan 과 떼어 두었다.
struct ProfileInputs {
    enum class DeviceType : uint32_t { OTHER = 0, INTEGRATED, DISCRETE, VIRTUAL, CPU };

    DeviceType deviceType = DeviceType::OTHER;
    // 장치 전용 힙의 «크기» 합(바이트). 남은 예산이 아니라 크기여야 한다. 예산은 다른 프로세스와
    // 이미 만든 자원에 따라 흔들려 같은 기계가 실행마다 다른 등급을 받는다.
    uint64_t deviceMemoryBytes = 0;
    bool meshShader = false;
    // 광선 반사와 광선 그림자는 광선 질의만으로 켜지지 않는다. 이 저장소는 하위 가속 구조를
    // RayTracer 가 세우고 그것이 rayTracingPipeline 을 요구한다.
    bool rayQuery = false;
    bool rayTracingPipeline = false;
    // 이 기기에서 쓸 수 있는 가장 나은 시간축 업스케일러(gfx::Upscaler 번호). 내장 TAAU(2)는 늘
    // 있으므로 FSR(3)이나 DLSS(4)를 쓸 수 있으면 그 번호가 온다.
    uint32_t bestTemporalUpscaler = 2;
    // 모니터 해상도. 창 크기가 아니라 «이 화면이 얼마나 넓어질 수 있는가» 다. 넓으면 한 등급
    // 보수적으로 고른다.
    uint32_t displayWidth = 1920;
    uint32_t displayHeight = 1080;
};

// 고른 설정. 이름과 뜻은 Renderer 의 같은 이름 필드와 같다.
struct HardwareProfile {
    HardwareTier tier = HardwareTier::MEDIUM;
    float renderScale = 1.0F;
    // gfx::Upscaler 와 같은 번호다(0 통과, 1 공간, 2 내장 시간축, 3 FSR, 4 DLSS). 헤더가 Vulkan 을
    // 끌어오지 않도록 숫자로 둔다.
    uint32_t upscaler = 1;
    uint32_t ssaoSamples = 16;
    uint32_t shadowCascades = 4;
    bool reflections = false;
    bool rayQueryShadows = false;
    // 유체 부품이 요청해도 이보다 많은 입자는 뿌리지 않는다.
    uint32_t fluidParticleLimit = 32768;
    // 사람이 읽을 판정 근거. 편집기가 그대로 보여 준다.
    std::vector<std::string> reasons;
};

// 하드웨어 사실에서 설정을 고른다. level 이 OFF 면 기본값 그대로인 프로파일을 돌려준다. 적용을
// 건너뛰는 것은 부르는 쪽 몫이다.
HardwareProfile chooseProfile(const ProfileInputs& inputs, AutoTune level);

const char* tierName(HardwareTier tier);
const char* autoTuneName(AutoTune level);
// 실행 인자 문자열을 해석한다. 모르는 값이면 거짓을 돌려주고 out 을 건드리지 않는다.
bool parseAutoTune(std::string_view text, AutoTune& out);

} // namespace gfx
