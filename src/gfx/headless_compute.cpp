#include "gfx/headless_compute.h"

#include "gfx/context.h"
#include "gfx/vk_check.h"

namespace gfx {

HeadlessCompute::HeadlessCompute(Context& context) : context(context) {
    VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = context.queueFamilies.graphics;
    VK_CHECK(vkCreateCommandPool(context.device, &poolInfo, nullptr, &pool));

    VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    allocateInfo.commandPool = pool;
    allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocateInfo.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(context.device, &allocateInfo, &commandBuffer));

    VkFenceCreateInfo fenceInfo{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(context.device, &fenceInfo, nullptr, &fence));
}

HeadlessCompute::~HeadlessCompute() {
    vkDestroyFence(context.device, fence, nullptr);
    vkDestroyCommandPool(context.device, pool, nullptr);
}

uint64_t HeadlessCompute::submit(const std::function<void(VkCommandBuffer, uint64_t)>& record) {
    uint64_t index = submitted++;
    VK_CHECK(vkResetCommandBuffer(commandBuffer, 0));
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
    record(commandBuffer, index);
    VK_CHECK(vkEndCommandBuffer(commandBuffer));

    VK_CHECK(vkResetFences(context.device, 1, &fence));
    VkSubmitInfo submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    VK_CHECK(vkQueueSubmit(context.graphicsQueue, 1, &submitInfo, fence));
    VK_CHECK(vkWaitForFences(context.device, 1, &fence, VK_TRUE, UINT64_MAX));

    // 기다렸으니 이 제출이 붙들고 있던 것은 아무도 읽지 않는다. 편집기 없이 도는 경로라 자원이
    // 사라지는 일은 드물지만, 맡긴 것이 있으면 여기서 지운다.
    if (context.hasRetired()) {
        context.collectRetired();
    }
    return index;
}

} // namespace gfx
