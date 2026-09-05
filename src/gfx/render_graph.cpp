#include "gfx/render_graph.h"

#include <cstring>

#include "core/error.h"
#include "gfx/profiler.h"

namespace gfx {

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

void RenderGraph::execute(VkCommandBuffer commandBuffer, GpuProfiler& profiler) {
    executed.clear();
    for (RenderNode& node : nodes) {
        if (node.enabled && !node.enabled()) {
            continue;
        }
        executed.push_back(node.name);
        uint32_t zone = node.zone != nullptr ? profiler.begin(node.zone, commandBuffer) : 0;
        node.record(commandBuffer);
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

} // namespace gfx
