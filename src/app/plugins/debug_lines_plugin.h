#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "app/plugin.h"
#include "gfx/debug_lines.h"
#include "gfx/resources.h"

namespace app {

// 콜라이더·유체 경계 표시. --no-colliders 를 적용하고 편집기 절을 그리며, 선 파이프라인과 «디버그 선» 노드를
// 소유한다. 톤 매핑과 공간 업스케일이 끝난 표시 해상도에 덧그리므로 «업스케일» 뒤에 끼운다(이유는 README).
class DebugLinesPlugin : public Plugin {
public:
    ~DebugLinesPlugin() override;
    const char* name() const override { return "디버그 선"; }
    void build(Services& services) override;
    void ui(Services& services) override;

private:
    void createPipeline(gfx::Renderer& renderer, gfx::BindlessTextures& bindless);
    void reserve(uint32_t slot, uint32_t vertexCount);
    void record(VkCommandBuffer commandBuffer, const gfx::Renderer::FrameInfo& info);

    // Services 는 훅이 도는 프레임에 살아 있지 않을 수 있어(registerPlugins 의 지역 변수) 필요한 것만 잡아 둔다.
    // 가리키는 객체는 Application 이 소유해 플러그인보다 오래 산다.
    gfx::Context* context = nullptr;
    gfx::BindlessTextures* bindless = nullptr;
    gfx::Renderer* renderer = nullptr;
    editor::Editor* editor = nullptr;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    // 프레임 슬롯마다 정점 버퍼 하나. 콜라이더 표시를 켰을 때만 채운다.
    std::array<gfx::Buffer, gfx::FRAMES_IN_FLIGHT> buffers{};
    std::array<uint32_t, gfx::FRAMES_IN_FLIGHT> capacities{};
    // 프레임마다 다시 채우는 선분 목록. 벡터를 그대로 두어 할당을 되쓴다.
    std::vector<gfx::DebugLineVertex> vertices;
};

} // namespace app
