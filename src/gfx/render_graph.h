#pragma once

#include <functional>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

namespace gfx {

class GpuProfiler;

// 노드가 이미지를 어떤 레이아웃·단계·접근으로 쓰는지. 그래프가 앞 상태에서 여기로 옮기는 배리어를 낸다.
struct ImageUse {
    VkImage image = VK_NULL_HANDLE;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_NONE;
    VkAccessFlags2 access = VK_ACCESS_2_NONE;
    // 지난 내용이 필요 없다(첨부물을 CLEAR/DONT_CARE 로 연다). 옛 레이아웃을 UNDEFINED 로 두어 전이가
    // 내용을 보존하지 않아도 되게 한다. 프레임마다 처음부터 채우는 대상이 이렇다.
    bool discard = false;
};

// 프레임의 패스 하나. 이름은 수명이 프로그램 전체인 리터럴이어야 한다(ran·addAfter 가 이름으로 찾는다).
struct RenderNode {
    const char* name = nullptr;
    // 프로파일러 구간 이름. nullptr 면 재지 않는다. 프레임당 구간은 MAX_PROFILER_ZONES(32) 까지다.
    const char* zone = nullptr;
    // 실행 직전에 평가한다. 앞 노드가 정한 값(가속 구조 준비 여부 등)에 기댈 수 있다. 비어 있으면 늘 돈다.
    std::function<bool()> enabled;
    // 기록 전에 이 상태로 옮긴다. reads 는 같은 레이아웃의 읽기가 이어지면 배리어를 내지 않고, writes 는 늘 낸다.
    std::vector<ImageUse> reads;
    std::vector<ImageUse> writes;
    // 기록 뒤의 상태. 패스 안에서 스스로 전이하고 나온 이미지(층·밉 단위 전이, 컴퓨트가 도로 첨부물로 돌린
    // 것)를 그래프에 알린다. 배리어는 내지 않는다.
    std::vector<ImageUse> leaves;
    std::function<void(VkCommandBuffer)> record;
};

// 프레임의 패스 목록. Renderer::recordCommands 가 프레임마다 다시 짜고, 플러그인이 앵커 뒤에 자기 패스를
// 끼운다. 노드는 조건과 무관하게 늘 등록하고 `enabled` 가 실행을 정한다 — 그래야 앵커가 늘 있다.
//
// 이미지 레이아웃은 이미지 전체 단위로 추적한다. 노드가 선언한 reads/writes 로 앞 상태에서의 전이를 만들고,
// 프레임을 넘어 상태가 남는다(지난 프레임 톤 매핑이 읽던 것을 이번 업스케일이 덮어쓰는 식).
//
// ponytail: 순서는 등록 순서 하나뿐이다. 래스터·경로 추적의 갈림과 업스케일 두 경로는 enabled 로 다
// 표현되어 DAG 정렬이 필요 없다.
// ponytail: 층·밉 단위 전이(그림자 아틀라스, HZB, Bloom)는 패스 안에 남는다. 그런 패스는 leaves 로 자기가
// 남긴 상태를 알려야 한다. 스왑체인 이미지는 프레임마다 새로 받으므로 추적하지 않고 노드 안에서 전이한다.
// ponytail: 노드 목록을 프레임마다 다시 짠다(std::function 30여 개). 프레임 CPU 시간에서 보이지 않는다.
class RenderGraph {
public:
    void clear();
    void add(RenderNode node);
    // anchor 바로 뒤에 끼운다. 앵커가 없으면 core::fatal — 오타를 조용히 넘기지 않는다.
    void addAfter(const char* anchor, RenderNode node);
    // 등록 순서대로 enabled 인 노드를 기록한다. zone 이 있으면 프로파일러 구간으로 감싼다.
    void execute(VkCommandBuffer commandBuffer, GpuProfiler& profiler);
    // 마지막 execute 에서 그 이름의 노드가 돌았는지. 디버그 뷰어가 어느 대상이 채워졌는지 볼 때 쓴다.
    bool ran(const char* name) const;
    // 추적 중인 이미지의 현재 레이아웃. 모르면 UNDEFINED.
    VkImageLayout layout(VkImage image) const;
    // 렌더 대상을 다시 만들었을 때. 모든 이미지를 UNDEFINED 로 되돌린다.
    void resetStates();

private:
    struct State {
        VkImageLayout layout = VK_IMAGE_LAYOUT_UNDEFINED;
        VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
        VkAccessFlags2 access = VK_ACCESS_2_NONE;
    };
    void transition(VkCommandBuffer commandBuffer, const ImageUse& use, bool write);

    std::vector<RenderNode> nodes;
    std::vector<const char*> executed;
    std::unordered_map<VkImage, State> states;
};

} // namespace gfx
