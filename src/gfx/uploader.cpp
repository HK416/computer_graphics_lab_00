#include "gfx/uploader.h"

#include <algorithm>
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
    graphicsPool = createPool(context.device, context.queueFamilies.graphics);
    graphicsCommands = allocateCommands(context.device, graphicsPool);

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
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkResetCommandPool(context.device, transferPool, 0));
    VK_CHECK(vkBeginCommandBuffer(transferCommands, &beginInfo));
    VK_CHECK(vkResetCommandPool(context.device, graphicsPool, 0));
    VK_CHECK(vkBeginCommandBuffer(graphicsCommands, &beginInfo));
    recording = true;
}

VkBuffer Uploader::createStaging(const void* data, VkDeviceSize size, const char* debugName) {
    Buffer staging =
        createBuffer(context, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryLocation::HOST_WRITE, debugName);
    std::memcpy(staging.mapped, data, size);
    stagingBuffers.push_back(staging);
    return staging.handle;
}

void Uploader::uploadBuffer(const Buffer& target, VkDeviceSize offset, const void* data, VkDeviceSize size) {
    if (size == 0) {
        return;
    }
    if (offset + size > target.size) {
        core::fatal("업로드 범위가 대상 버퍼를 벗어납니다 ({} + {} > {})", offset, size, target.size);
    }
    beginRecording();

    VkBuffer staging = createStaging(data, size, "업로드 스테이징");

    VkBufferCopy2 region{VK_STRUCTURE_TYPE_BUFFER_COPY_2};
    region.dstOffset = offset;
    region.size = size;
    VkCopyBufferInfo2 copyInfo{VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2};
    copyInfo.srcBuffer = staging;
    copyInfo.dstBuffer = target.handle;
    copyInfo.regionCount = 1;
    copyInfo.pRegions = &region;
    vkCmdCopyBuffer2(transferCommands, &copyInfo);

    if (!needsOwnershipTransfer) {
        return;
    }

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

void Uploader::copyBuffer(const Buffer& source, const Buffer& target, VkDeviceSize size) {
    if (size == 0) {
        return;
    }
    if (size > source.size || size > target.size) {
        core::fatal("복사 범위가 버퍼를 벗어납니다 ({} > {} 또는 {})", size, source.size, target.size);
    }
    beginRecording();

    VkBufferCopy2 region{VK_STRUCTURE_TYPE_BUFFER_COPY_2};
    region.size = size;
    VkCopyBufferInfo2 copyInfo{VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2};
    copyInfo.srcBuffer = source.handle;
    copyInfo.dstBuffer = target.handle;
    copyInfo.regionCount = 1;
    copyInfo.pRegions = &region;
    vkCmdCopyBuffer2(graphicsCommands, &copyInfo);

    // 같은 제출 안의 스테이징 획득 배리어와는 구간이 겹치지 않는다. 뒤따르는 프레임이 읽도록 열어 둔다.
    VkBufferMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = target.handle;
    barrier.offset = 0;
    barrier.size = size;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.bufferMemoryBarrierCount = 1;
    dependency.pBufferMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(graphicsCommands, &dependency);
}

void Uploader::uploadImage(const Image& target, const void* data, VkDeviceSize size, VkImageLayout finalLayout) {
    if (size == 0) {
        return;
    }
    beginRecording();

    VkBuffer staging = createStaging(data, size, "이미지 업로드 스테이징");

    uint32_t transferFamily = needsOwnershipTransfer ? context.queueFamilies.transfer : VK_QUEUE_FAMILY_IGNORED;
    uint32_t graphicsFamily = needsOwnershipTransfer ? context.queueFamilies.graphics : VK_QUEUE_FAMILY_IGNORED;

    // 전송 큐: 밉 0 을 채우고 그래픽스 큐로 소유권을 넘긴다.
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
    copyInfo.srcBuffer = staging;
    copyInfo.dstImage = target.handle;
    copyInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    copyInfo.regionCount = 1;
    copyInfo.pRegions = &region;
    vkCmdCopyBufferToImage2(transferCommands, &copyInfo);

    imageBarrier(transferCommands,
                 target.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_NONE,
                 0,
                 transferFamily,
                 graphicsFamily);

    // 밉 생성은 blit 이라 그래픽스 큐에서만 가능하다.
    imageBarrier(graphicsCommands,
                 target.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_2_NONE,
                 0,
                 VK_PIPELINE_STAGE_2_BLIT_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 transferFamily,
                 graphicsFamily);

    auto levelWidth = static_cast<int32_t>(target.extent.width);
    auto levelHeight = static_cast<int32_t>(target.extent.height);
    for (uint32_t level = 1; level < target.mipLevels; ++level) {
        imageBarrier(graphicsCommands,
                     target.handle,
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_PIPELINE_STAGE_2_BLIT_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_BLIT_BIT,
                     VK_ACCESS_2_TRANSFER_READ_BIT,
                     VK_QUEUE_FAMILY_IGNORED,
                     VK_QUEUE_FAMILY_IGNORED,
                     level - 1,
                     1);

        int32_t nextWidth = std::max(levelWidth / 2, 1);
        int32_t nextHeight = std::max(levelHeight / 2, 1);

        VkImageBlit2 blit{VK_STRUCTURE_TYPE_IMAGE_BLIT_2};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = level - 1;
        blit.srcSubresource.layerCount = target.arrayLayers;
        blit.srcOffsets[1] = {levelWidth, levelHeight, 1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = level;
        blit.dstSubresource.layerCount = target.arrayLayers;
        blit.dstOffsets[1] = {nextWidth, nextHeight, 1};

        VkBlitImageInfo2 blitInfo{VK_STRUCTURE_TYPE_BLIT_IMAGE_INFO_2};
        blitInfo.srcImage = target.handle;
        blitInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        blitInfo.dstImage = target.handle;
        blitInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        blitInfo.regionCount = 1;
        blitInfo.pRegions = &blit;
        blitInfo.filter = VK_FILTER_LINEAR;
        vkCmdBlitImage2(graphicsCommands, &blitInfo);

        levelWidth = nextWidth;
        levelHeight = nextHeight;
    }

    if (target.mipLevels > 1) {
        // 마지막 레벨을 뺀 나머지는 TRANSFER_SRC 상태이므로 두 번에 나누어 전이시킨다.
        imageBarrier(graphicsCommands,
                     target.handle,
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     finalLayout,
                     VK_PIPELINE_STAGE_2_BLIT_BIT,
                     VK_ACCESS_2_TRANSFER_READ_BIT,
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                     VK_ACCESS_2_MEMORY_READ_BIT,
                     VK_QUEUE_FAMILY_IGNORED,
                     VK_QUEUE_FAMILY_IGNORED,
                     0,
                     target.mipLevels - 1);
    }
    imageBarrier(graphicsCommands,
                 target.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 finalLayout,
                 VK_PIPELINE_STAGE_2_BLIT_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT,
                 VK_QUEUE_FAMILY_IGNORED,
                 VK_QUEUE_FAMILY_IGNORED,
                 target.mipLevels - 1,
                 1);
}

void Uploader::uploadImageLevels(const Image& target,
                                 const void* data,
                                 const std::vector<VkDeviceSize>& levelBytes,
                                 VkImageLayout finalLayout) {
    VkDeviceSize total = 0;
    for (VkDeviceSize bytes : levelBytes) {
        total += bytes;
    }
    if (total == 0 || levelBytes.size() != target.mipLevels) {
        core::fatal(
            "이미지 밉 단계 수가 맞지 않습니다 ({} 단계 데이터, 이미지는 {})", levelBytes.size(), target.mipLevels);
    }
    beginRecording();

    VkBuffer staging = createStaging(data, total, "이미지 업로드 스테이징");
    uint32_t transferFamily = needsOwnershipTransfer ? context.queueFamilies.transfer : VK_QUEUE_FAMILY_IGNORED;
    uint32_t graphicsFamily = needsOwnershipTransfer ? context.queueFamilies.graphics : VK_QUEUE_FAMILY_IGNORED;

    imageBarrier(transferCommands,
                 target.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_2_NONE,
                 0,
                 VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT);

    std::vector<VkBufferImageCopy2> regions(levelBytes.size());
    VkDeviceSize offset = 0;
    for (uint32_t level = 0; level < levelBytes.size(); ++level) {
        VkBufferImageCopy2& region = regions[level];
        region = VkBufferImageCopy2{VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2};
        region.bufferOffset = offset;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = level;
        region.imageSubresource.layerCount = target.arrayLayers;
        region.imageExtent = {
            std::max(target.extent.width >> level, 1U), std::max(target.extent.height >> level, 1U), 1};
        offset += levelBytes[level];
    }
    VkCopyBufferToImageInfo2 copyInfo{VK_STRUCTURE_TYPE_COPY_BUFFER_TO_IMAGE_INFO_2};
    copyInfo.srcBuffer = staging;
    copyInfo.dstImage = target.handle;
    copyInfo.dstImageLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    copyInfo.regionCount = static_cast<uint32_t>(regions.size());
    copyInfo.pRegions = regions.data();
    vkCmdCopyBufferToImage2(transferCommands, &copyInfo);

    // uploadImage 와 같은 순서다. 놓기와 받기는 레이아웃을 바꾸지 않고 소유권만 넘기고, 최종 레이아웃
    // 전이는 그래픽스 큐가 따로 한다. 큐 패밀리가 같으면 앞의 둘은 아무것도 하지 않는 배리어가 된다.
    imageBarrier(transferCommands,
                 target.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COPY_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_NONE,
                 0,
                 transferFamily,
                 graphicsFamily);
    imageBarrier(graphicsCommands,
                 target.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_2_NONE,
                 0,
                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 transferFamily,
                 graphicsFamily);
    imageBarrier(graphicsCommands,
                 target.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 finalLayout,
                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_TRANSFER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT);
}

void Uploader::flush() {
    if (!recording) {
        return;
    }
    VK_CHECK(vkEndCommandBuffer(transferCommands));
    VK_CHECK(vkEndCommandBuffer(graphicsCommands));

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
