#include "gfx/resources.h"

#include "core/error.h"
#include "gfx/context.h"
#include "gfx/vk_check.h"

namespace gfx {
namespace {

void setDebugName(Context& context, uint64_t handle, VkObjectType type, const char* name) {
    if (name == nullptr) {
        return;
    }
    auto setName = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetInstanceProcAddr(context.instance, "vkSetDebugUtilsObjectNameEXT"));
    if (setName == nullptr) {
        return;
    }
    VkDebugUtilsObjectNameInfoEXT info{VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT};
    info.objectType = type;
    info.objectHandle = handle;
    info.pObjectName = name;
    setName(context.device, &info);
}

} // namespace

Buffer createBuffer(
    Context& context, VkDeviceSize size, VkBufferUsageFlags usage, MemoryLocation location, const char* debugName) {
    if (size == 0) {
        core::fatal("크기가 0 인 버퍼를 만들 수 없습니다: {}", debugName != nullptr ? debugName : "이름 없음");
    }

    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
    switch (location) {
    case MemoryLocation::DEVICE:
        allocationInfo.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
        allocationInfo.priority = 1.0F;
        break;
    case MemoryLocation::HOST_WRITE:
        allocationInfo.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        break;
    case MemoryLocation::HOST_READ:
        allocationInfo.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        break;
    }

    Buffer buffer;
    buffer.size = size;
    VmaAllocationInfo allocationResult{};
    VK_CHECK(vmaCreateBuffer(
        context.allocator, &bufferInfo, &allocationInfo, &buffer.handle, &buffer.allocation, &allocationResult));
    buffer.mapped = allocationResult.pMappedData;

    VkBufferDeviceAddressInfo addressInfo{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    addressInfo.buffer = buffer.handle;
    buffer.address = vkGetBufferDeviceAddress(context.device, &addressInfo);

    setDebugName(context, reinterpret_cast<uint64_t>(buffer.handle), VK_OBJECT_TYPE_BUFFER, debugName);
    return buffer;
}

void destroyBuffer(Context& context, Buffer& buffer) {
    if (buffer.handle == VK_NULL_HANDLE) {
        return;
    }
    vmaDestroyBuffer(context.allocator, buffer.handle, buffer.allocation);
    buffer = {};
}

Image createImage(Context& context, const ImageDesc& desc, const char* debugName) {
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = desc.extent.depth > 1 ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
    imageInfo.format = desc.format;
    imageInfo.extent = desc.extent;
    imageInfo.mipLevels = desc.mipLevels;
    imageInfo.arrayLayers = desc.arrayLayers;
    imageInfo.samples = desc.samples;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = desc.usage;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (desc.viewType == VK_IMAGE_VIEW_TYPE_CUBE || desc.viewType == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) {
        imageInfo.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    }

    VmaAllocationCreateInfo allocationInfo{};
    allocationInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocationInfo.priority = 1.0F;

    Image image;
    image.extent = desc.extent;
    image.format = desc.format;
    image.mipLevels = desc.mipLevels;
    image.arrayLayers = desc.arrayLayers;
    VK_CHECK(vmaCreateImage(context.allocator, &imageInfo, &allocationInfo, &image.handle, &image.allocation, nullptr));

    VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    viewInfo.image = image.handle;
    viewInfo.viewType = desc.viewType;
    viewInfo.format = desc.format;
    viewInfo.subresourceRange.aspectMask = desc.aspect;
    viewInfo.subresourceRange.levelCount = desc.mipLevels;
    viewInfo.subresourceRange.layerCount = desc.arrayLayers;
    VK_CHECK(vkCreateImageView(context.device, &viewInfo, nullptr, &image.view));

    setDebugName(context, reinterpret_cast<uint64_t>(image.handle), VK_OBJECT_TYPE_IMAGE, debugName);
    return image;
}

void destroyImage(Context& context, Image& image) {
    if (image.handle == VK_NULL_HANDLE) {
        return;
    }
    vkDestroyImageView(context.device, image.view, nullptr);
    vmaDestroyImage(context.allocator, image.handle, image.allocation);
    image = {};
}

void imageBarrier(VkCommandBuffer commandBuffer,
                  VkImage image,
                  VkImageAspectFlags aspect,
                  VkImageLayout oldLayout,
                  VkImageLayout newLayout,
                  VkPipelineStageFlags2 sourceStage,
                  VkAccessFlags2 sourceAccess,
                  VkPipelineStageFlags2 destinationStage,
                  VkAccessFlags2 destinationAccess,
                  uint32_t sourceQueueFamily,
                  uint32_t destinationQueueFamily) {
    VkImageMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    barrier.srcStageMask = sourceStage;
    barrier.srcAccessMask = sourceAccess;
    barrier.dstStageMask = destinationStage;
    barrier.dstAccessMask = destinationAccess;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = sourceQueueFamily;
    barrier.dstQueueFamilyIndex = destinationQueueFamily;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = aspect;
    barrier.subresourceRange.levelCount = VK_REMAINING_MIP_LEVELS;
    barrier.subresourceRange.layerCount = VK_REMAINING_ARRAY_LAYERS;

    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

} // namespace gfx
