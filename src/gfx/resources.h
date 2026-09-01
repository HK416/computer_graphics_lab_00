#pragma once

#include <cstdint>

#include <vma/vk_mem_alloc.h>
#include <vulkan/vulkan.h>

namespace gfx {

struct Context;

struct Buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    // bufferDeviceAddress 로 만든 버퍼만 채워진다. GPU-Driven 경로는 디스크립터 대신 이 주소를 쓴다.
    VkDeviceAddress address = 0;
    // 호스트에서 보이는 메모리에 할당된 경우에만 채워진다.
    void* mapped = nullptr;
};

struct Image {
    VkImage handle = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkExtent3D extent{};
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
};

enum class MemoryLocation {
    DEVICE,     // GPU 전용. 업로드는 스테이징을 거친다.
    HOST_WRITE, // 매 프레임 CPU 가 쓰는 상수/인스턴스 버퍼. 매핑된 채로 유지한다.
    HOST_READ,  // GPU 결과를 CPU 가 읽어가는 리드백 버퍼.
};

Buffer createBuffer(
    Context& context, VkDeviceSize size, VkBufferUsageFlags usage, MemoryLocation location, const char* debugName);
void destroyBuffer(Context& context, Buffer& buffer);

struct ImageDesc {
    VkExtent3D extent{1, 1, 1};
    VkFormat format = VK_FORMAT_R8G8B8A8_UNORM;
    VkImageUsageFlags usage = 0;
    uint32_t mipLevels = 1;
    uint32_t arrayLayers = 1;
    VkImageAspectFlags aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    VkImageViewType viewType = VK_IMAGE_VIEW_TYPE_2D;
    VkSampleCountFlagBits samples = VK_SAMPLE_COUNT_1_BIT;
};

Image createImage(Context& context, const ImageDesc& desc, const char* debugName);
void destroyImage(Context& context, Image& image);

// 이미지 배리어 한 장을 기록한다. 레이아웃 전이와 큐 패밀리 소유권 이전에 모두 쓴다.
void imageBarrier(VkCommandBuffer commandBuffer,
                  VkImage image,
                  VkImageAspectFlags aspect,
                  VkImageLayout oldLayout,
                  VkImageLayout newLayout,
                  VkPipelineStageFlags2 sourceStage,
                  VkAccessFlags2 sourceAccess,
                  VkPipelineStageFlags2 destinationStage,
                  VkAccessFlags2 destinationAccess,
                  uint32_t sourceQueueFamily = VK_QUEUE_FAMILY_IGNORED,
                  uint32_t destinationQueueFamily = VK_QUEUE_FAMILY_IGNORED,
                  uint32_t baseMipLevel = 0,
                  uint32_t mipLevelCount = VK_REMAINING_MIP_LEVELS,
                  uint32_t baseArrayLayer = 0,
                  uint32_t arrayLayerCount = VK_REMAINING_ARRAY_LAYERS);

} // namespace gfx
