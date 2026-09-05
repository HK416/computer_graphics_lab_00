#pragma once

#include <functional>
#include <vector>

#include <vulkan/vulkan.h>

namespace gfx {

class GpuProfiler;

// 프레임의 패스 하나. 이름은 수명이 프로그램 전체인 리터럴이어야 한다(ran·addAfter 가 이름으로 찾는다).
struct RenderNode {
    const char* name = nullptr;
    // 프로파일러 구간 이름. nullptr 면 재지 않는다. 프레임당 구간은 MAX_PROFILER_ZONES(32) 까지다.
    const char* zone = nullptr;
    // 실행 직전에 평가한다. 앞 노드가 정한 값(가속 구조 준비 여부 등)에 기댈 수 있다. 비어 있으면 늘 돈다.
    std::function<bool()> enabled;
    std::function<void(VkCommandBuffer)> record;
};

// 프레임의 패스 목록. Renderer::recordCommands 가 프레임마다 다시 짜고, 플러그인이 앵커 뒤에 자기 패스를
// 끼운다. 노드는 조건과 무관하게 늘 등록하고 `enabled` 가 실행을 정한다 — 그래야 앵커가 늘 있다.
//
// ponytail: 순서는 등록 순서 하나뿐이다. 래스터·경로 추적의 갈림과 업스케일 두 경로는 enabled 로 다
// 표현되어 DAG 정렬이 필요 없다. 이미지 레이아웃 전이는 아직 노드 안에 인라인이다(다음 단계).
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

private:
    std::vector<RenderNode> nodes;
    std::vector<const char*> executed;
};

} // namespace gfx
