#include "gfx/uploader.h"

#include <cstring>

#include "core/error.h"
#include "gfx/context.h"
#include "gfx/vk_check.h"

namespace gfx {
namespace {

VkCommandPool createPool(VkDevice device, uint32_t queueFamily) {
    VkCommandPoolCreateInfo info{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    info.queueFamilyIndex = queueFamily;
    VkCommandPool pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateCommandPool(device, &info, nullptr, &pool));
    return pool;
}

VkCommandBuffer allocateCommands(VkDevice device, VkCommandPool pool) {
    VkCommandBufferAllocateInfo info{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    info.commandPool = pool;
    info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    info.commandBufferCount = 1;
    VkCommandBuffer commands = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device, &info, &commands));
    return commands;
}

} // namespace

Uploader::Uploader(Context& context) : context(context) {
    needsOwnershipTransfer = context.queueFamilies.transfer != context.queueFamilies.graphics;

    transferPool = createPool(context.device, context.queueFamilies.transfer);
    transferCommands = allocateCommands(context.device, transferPool);
    if (needsOwnershipTransfer) {
        graphicsPool = createPool(context.device, context.queueFamilies.graphics);
        graphicsCommands = allocateCommands(context.device, graphicsPool);
    }

    VkSemaphoreTypeCreateInfo typeInfo{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    typeInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphoreInfo.pNext = &typeInfo;
    VK_CHECK(vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &timeline));
}

Uploader::~Uploader() {
    for (Buffer& buffer : stagingBuffers) {
        destroyBuffer(context, buffer);
    }
    vkDestroySemaphore(context.device, timeline, nullptr);
    vkDestroyCommandPool(context.device, graphicsPool, nullptr);
    vkDestroyCommandPool(context.device, transferPool, nullptr);
}

void Uploader::beginRecording() {
    if (recording) {
        return;
    }
    VK_CHECK(vkResetCommandPool(context.device, transferPool, 0));
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(transferCommands, &beginInfo));
    if (needsOwnershipTransfer) {
        VK_CHECK(vkResetCommandPool(context.device, graphicsPool, 0));
        VK_CHECK(vkBeginCommandBuffer(graphicsCommands, &beginInfo));
    }
    recording = true;
}

void Uploader::uploadBuffer(const Buffer& target, VkDeviceSize offset, const void* data, VkDeviceSize size) {
    if (size == 0) {
        return;
    }
    if (offset + size > target.size) {
        core::fatal("업로드 범위가 대상 버퍼를 벗어납니다 ({} + {} > {})", offset, size, target.size);
    }
    beginRecording();

    Buffer staging =
        createBuffer(context, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryLocation::HOST_WRITE, "업로드 스테이징");
    std::memcpy(staging.mapped, data, size);

    VkBufferCopy2 region{VK_STRUCTURE_TYPE_BUFFER_COPY_2};
    region.dstOffset = offset;
    region.size = size;
    VkCopyBufferInfo2 copyInfo{VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2};
    copyInfo.srcBuffer = staging.handle;
    copyInfo.dstBuffer = target.handle;
    copyInfo.regionCount = 1;
    copyInfo.pRegions = &region;
    vkCmdCopyBuffer2(transferCommands, &copyInfo);

    if (needsOwnershipTransfer) {
        VkBufferMemoryBarrier2 release{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
        release.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        release.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        release.srcQueueFamilyIndex = context.queueFamilies.transfer;
        release.dstQueueFamilyIndex = context.queueFamilies.graphics;
        release.buffer = target.handle;
        release.offset = offset;
        release.size = size;

        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.bufferMemoryBarrierCount = 1;
        dependency.pBufferMemoryBarriers = &release;
        vkCmdPipelineBarrier2(transferCommands, &dependency);

        VkBufferMemoryBarrier2 acquire = release;
        acquire.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        acquire.srcAccessMask = 0;
        acquire.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        acquire.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
        dependency.pBufferMemoryBarriers = &acquire;
        vkCmdPipelineBarrier2(graphicsCommands, &dependency);
    }

    stagingBuffers.push_back(staging);
}

void Uploader::uploadImage(const Image& target, const void* data, VkDeviceSize size, VkImageLayout finalLayout) {
    if (size == 0) {
        return;
    }
    beginRecording();

    Buffer staging = createBuffer(
        context, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryLocation::HOST_WRITE, "이미지 업로드 스테이징");
    std::memcpy(staging.mapped, data, size);

    imageBarrier(transferCommands,
                 target.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_2_NONE,
                 0,
                 VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT);

    VkBufferImageCopy2 region{VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2};
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = target.arrayLayers;
    region.imageExtent = target.extent;
    VkCopyBufferToImageInfo2 copyInfo{VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2};
    copyInfo.srcBuffer = staging.handle;
    copyInfo.dstImage = target.handle;
    copyInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    copyInfo.regionCount = 1;
    copyInfo.pRegions = &region;
    vkCmdCopyBufferToImage2(transferCommands, &copyInfo);

    uint32_t sourceFamily = needsOwnershipTransfer ? context.queueFamilies.transfer : VK_QUEUE_FAMILY_IGNORED;
    uint32_t destinationFamily = needsOwnershipTransfer ? context.queueFamilies.graphics : VK_QUEUE_FAMILY_IGNORED;

    imageBarrier(transferCommands,
                 target.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 finalLayout,
                 VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 needsOwnershipTransfer ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 needsOwnershipTransfer ? 0 : VK_ACCESS_2_MEMORY_READ_BIT,
                 sourceFamily,
                 destinationFamily);

    if (needsOwnershipTransfer) {
        imageBarrier(graphicsCommands,
                     target.handle,
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     finalLayout,
                     VK_PIPELINE_STAGE_2_NONE,
                     0,
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                     VK_ACCESS_2_MEMORY_READ_BIT,
                     sourceFamily,
                     destinationFamily);
    }

    stagingBuffers.push_back(staging);
}

void Uploader::flush() {
    if (!recording) {
        return;
    }
    VK_CHECK(vkEndCommandBuffer(transferCommands));

    VkCommandBufferSubmitInfo transferCommandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    transferCommandInfo.commandBuffer = transferCommands;

    VkSemaphoreSubmitInfo transferSignal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    transferSignal.semaphore = timeline;
    transferSignal.value = ++timelineValue;
    transferSignal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkSubmitInfo2 transferSubmit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    transferSubmit.commandBufferInfoCount = 1;
    transferSubmit.pCommandBufferInfos = &transferCommandInfo;
    transferSubmit.signalSemaphoreInfoCount = 1;
    transferSubmit.pSignalSemaphoreInfos = &transferSignal;
    VK_CHECK(vkQueueSubmit2(context.transferQueue, 1, &transferSubmit, VK_NULL_HANDLE));

    if (needsOwnershipTransfer) {
        VK_CHECK(vkEndCommandBuffer(graphicsCommands));

        VkSemaphoreSubmitInfo graphicsWait = transferSignal;
        VkSemaphoreSubmitInfo graphicsSignal{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        graphicsSignal.semaphore = timeline;
        graphicsSignal.value = ++timelineValue;
        graphicsSignal.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

        VkCommandBufferSubmitInfo graphicsCommandInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
        graphicsCommandInfo.commandBuffer = graphicsCommands;

        VkSubmitInfo2 graphicsSubmit{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        graphicsSubmit.waitSemaphoreInfoCount = 1;
        graphicsSubmit.pWaitSemaphoreInfos = &graphicsWait;
        graphicsSubmit.commandBufferInfoCount = 1;
        graphicsSubmit.pCommandBufferInfos = &graphicsCommandInfo;
        graphicsSubmit.signalSemaphoreInfoCount = 1;
        graphicsSubmit.pSignalSemaphoreInfos = &graphicsSignal;
        VK_CHECK(vkQueueSubmit2(context.graphicsQueue, 1, &graphicsSubmit, VK_NULL_HANDLE));
    }

    VkSemaphoreWaitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &timeline;
    waitInfo.pValues = &timelineValue;
    VK_CHECK(vkWaitSemaphores(context.device, &waitInfo, UINT64_MAX));

    for (Buffer& buffer : stagingBuffers) {
        destroyBuffer(context, buffer);
    }
    stagingBuffers.clear();
    recording = false;
}

} // namespace gfx
