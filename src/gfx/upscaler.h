#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <glm/vec2.hpp>
#include <vulkan/vulkan.h>

#include "gfx/resources.h"

namespace gfx {

struct Context;
class BindlessTextures;

// 업스케일 방식. 벤더 SDK 가 필요한 것들은 감지만 하고 사용 가능 여부를 보고한다.
enum class Upscaler : uint32_t {
    NONE = 0,
    SPATIAL = 1,
    TAAU = 2,
    FSR = 3,
    DLSS = 4,
    // 경로 추적의 1표본 결과를 디노이즈하면서 확대한다. 경로 추적에서만 쓴다.
    DLSS_RR = 5,
};

// 시간축 업스케일러는 지터와 모션 벡터를 요구하고 톤 매핑 앞에서 돈다. 공간 업스케일은 톤 매핑
// 뒤에서 도므로 렌더러가 두 경로를 다르게 엮는다.
inline bool isTemporal(Upscaler kind) {
    return kind == Upscaler::TAAU || kind == Upscaler::FSR || kind == Upscaler::DLSS || kind == Upscaler::DLSS_RR;
}

struct UpscalerInfo {
    Upscaler kind;
    const char* name;
    bool available;
    // 쓸 수 없을 때의 이유. 사용 가능하면 비어 있다.
    const char* reason;
};

// 한 번 평가하는 데 필요한 입력. 이미지는 모두 슬롯으로 넘긴다. 색상/깊이/변위는 읽을 수 있는
// 상태여야 하고 출력은 GENERAL 이어야 한다.
struct UpscaleInputs {
    // 벤더 SDK 는 VkImage 를 직접 받는다. 내장 경로는 아래 bindless 슬롯을 쓴다.
    const Image* color = nullptr;    // 렌더 해상도 선형 HDR
    const Image* depth = nullptr;    // 렌더 해상도 reverse-Z
    const Image* velocity = nullptr; // 렌더 해상도 화면 UV 변위
    const Image* output = nullptr;   // 표시 해상도 선형 HDR
    uint32_t colorTexture = 0;
    uint32_t depthTexture = 0;
    uint32_t velocityTexture = 0;
    // 표시 해상도 결과를 쓸 rgba16f 스토리지 슬롯.
    uint32_t outputStorage = 0;
    VkDescriptorSet bindlessSet = VK_NULL_HANDLE;
    // Ray Reconstruction 이 요구하는 안내 버퍼. 그 방식이 아니면 널이다.
    const Image* guideDiffuseAlbedo = nullptr;
    const Image* guideSpecularAlbedo = nullptr;
    const Image* guideNormal = nullptr;
    const Image* guideRoughness = nullptr;
    const Image* guideDepth = nullptr;
    // 이번 프레임 투영에 들어간 지터. 렌더 해상도 픽셀 단위다.
    glm::vec2 jitter{0.0F};
    float deltaSeconds = 0.0F;
    float nearPlane = 0.05F;
    float verticalFovRadians = 1.0F;
    // 히스토리를 버린다. 장면 전환이나 카메라 순간 이동 직후 켠다.
    bool reset = false;
};

class TemporalUpscaler {
public:
    TemporalUpscaler() = default;
    virtual ~TemporalUpscaler() = default;
    TemporalUpscaler(const TemporalUpscaler&) = delete;
    TemporalUpscaler& operator=(const TemporalUpscaler&) = delete;

    // 렌더나 표시 해상도가 바뀌면 부른다. 히스토리는 버려진다.
    virtual void resize(VkExtent2D render, VkExtent2D display) = 0;
    virtual void evaluate(VkCommandBuffer commandBuffer, const UpscaleInputs& inputs) = 0;
    // 마지막 resize 가 실패했으면 거짓이다. 렌더러는 그때 지터도 끄고 공간 경로로 돌아간다.
    virtual bool ready() const { return true; }
};

// 이 장치에서 쓸 수 있는지와, 쓸 수 없다면 그 이유.
UpscalerInfo upscalerInfo(Upscaler kind, const Context& context);
// 시간축 업스케일러를 만든다. 쓸 수 없는 방식이면 nullptr 이다.
std::unique_ptr<TemporalUpscaler> createUpscaler(Upscaler kind, Context& context, BindlessTextures& bindless);

// 구현별 생성 함수와 사용 불가 사유. createUpscaler 와 upscalerInfo 가 골라 부른다.
std::unique_ptr<TemporalUpscaler> createTaauUpscaler(Context& context, BindlessTextures& bindless);
std::unique_ptr<TemporalUpscaler> createFsrUpscaler(Context& context, BindlessTextures& bindless);
std::unique_ptr<TemporalUpscaler> createDlssUpscaler(Context& context, BindlessTextures& bindless);
std::unique_ptr<TemporalUpscaler> createDlssRayReconstruction(Context& context, BindlessTextures& bindless);
// 쓸 수 있으면 nullptr 을 돌려준다.
const char* fsrUnavailableReason();
const char* dlssUnavailableReason(const Context& context);
const char* dlssRayReconstructionUnavailableReason(const Context& context);

// NGX 가 요구하는 인스턴스/장치 확장. 장치를 만들기 전에 불러야 한다. NGX 는 자기 셰이더를 직접
// 올리느라 NVX 확장 몇 개가 있어야 하고, 그건 인스턴스/장치 생성 시점에만 켤 수 있다. DLSS 가
// 빠진 빌드에서는 빈 목록이다.
void dlssRequiredExtensions(std::vector<const char*>& instanceExtensions,
                            std::vector<const char*>& deviceExtensions);
// NGX 를 미리 띄워 편집기가 가용성을 물어볼 수 있게 한다. NVIDIA 장치가 아니면 아무것도 하지 않는다.
void startDlssRuntime(Context& context);
// NGX 가 잡은 자원을 푼다. 장치를 지우기 전에 불러야 한다. 정적 소멸자에 맡기면 이미 없어진
// 장치를 붙들고 종료 중에 죽는다.
void shutdownDlssRuntime();

} // namespace gfx
