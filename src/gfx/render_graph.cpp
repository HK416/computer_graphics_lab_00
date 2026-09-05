#include "gfx/render_graph.h"

#include <cstring>

#include "core/error.h"
#include "gfx/profiler.h"
#include "gfx/resources.h"

namespace gfx {

namespace {

constexpr VkAccessFlags2 WRITE_ACCESS = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;

} // namespace

void RenderGraph::clear() {
    nodes.clear();
    executed.clear();
}

void RenderGraph::add(RenderNode node) {
    nodes.push_back(std::move(node));
}

void RenderGraph::addAfter(const char* anchor, RenderNode node) {
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (std::strcmp(nodes[i].name, anchor) == 0) {
            nodes.insert(nodes.begin() + static_cast<std::ptrdiff_t>(i) + 1, std::move(node));
            return;
        }
    }
    core::fatal("렌더 그래프에 '{}' 노드가 없어 '{}' 를 끼울 수 없다", anchor, node.name);
}

void RenderGraph::transition(VkCommandBuffer commandBuffer, const ImageUse& use, bool write) {
    State& state = states[use.image];
    // 같은 레이아웃의 읽기가 이어지면 배리어가 필요 없다. 뒤에 올 쓰기가 이 읽기도 기다리도록 단계만 모아 둔다.
    bool previousWrote = (state.access & WRITE_ACCESS) != 0;
    if (!write && !use.discard && state.layout == use.layout && !previousWrote) {
        state.stage |= use.stage;
        state.access |= use.access;
        return;
    }
    VkImageLayout oldLayout = use.discard ? VK_IMAGE_LAYOUT_UNDEFINED : state.layout;
    VkPipelineStageFlags2 sourceStage = use.discard ? VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT : state.stage;
    VkAccessFlags2 sourceAccess = use.discard ? VK_ACCESS_2_NONE : state.access;
    imageBarrier(
        commandBuffer, use.image, use.aspect, oldLayout, use.layout, sourceStage, sourceAccess, use.stage, use.access);
    state.layout = use.layout;
    state.stage = use.stage;
    state.access = use.access;
}

void RenderGraph::execute(VkCommandBuffer commandBuffer, GpuProfiler& profiler) {
    executed.clear();
    for (RenderNode& node : nodes) {
        if (node.enabled && !node.enabled()) {
            continue;
        }
        executed.push_back(node.name);
        uint32_t zone = node.zone != nullptr ? profiler.begin(node.zone, commandBuffer) : 0;
        for (const ImageUse& use : node.reads) {
            transition(commandBuffer, use, false);
        }
        for (const ImageUse& use : node.writes) {
            transition(commandBuffer, use, true);
        }
        node.record(commandBuffer);
        for (const ImageUse& use : node.leaves) {
            states[use.image] = State{use.layout, use.stage, use.access};
        }
        if (node.zone != nullptr) {
            profiler.end(zone, commandBuffer);
        }
    }
}

bool RenderGraph::ran(const char* name) const {
    for (const char* done : executed) {
        if (std::strcmp(done, name) == 0) {
            return true;
        }
    }
    return false;
}

VkImageLayout RenderGraph::layout(VkImage image) const {
    auto found = states.find(image);
    return found == states.end() ? VK_IMAGE_LAYOUT_UNDEFINED : found->second.layout;
}

void RenderGraph::resetStates() {
    states.clear();
}

} // namespace gfx
