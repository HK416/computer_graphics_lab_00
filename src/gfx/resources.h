#pragma once

#include <cstdint>
#include <string>

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

// minAlignment 가 0 이 아니면 그 배수 주소로 할당한다. 가속 구조 스크래치처럼 장치가 정렬을
// 요구하는 버퍼에 쓴다.
Buffer createBuffer(Context& context,
                    VkDeviceSize size,
                    VkBufferUsageFlags usage,
                    MemoryLocation location,
                    const char* debugName,
                    VkDeviceSize minAlignment = 0);
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

// 빌드가 내놓은 SPIR-V 를 이름으로 읽어 셰이더 모듈을 만든다. 실패하면 즉시 중단한다.
VkShaderModule createShaderModule(VkDevice device, const std::string& name);
// 같은 일을 하되 실패하면 사유를 로그에 남기고 VK_NULL_HANDLE 을 돌려준다. 없으면 다른 경로로
// 내려갈 수 있는 선택 기능이 쓴다.
VkShaderModule tryCreateShaderModule(VkDevice device, const std::string& name);
void destroyImage(Context& context, Image& image);

// 메모리 배리어 한 장을 기록한다. 버퍼를 쓰는 컴퓨트·전송과 그것을 읽는 단계 사이에 쓴다.
void memoryBarrier(VkCommandBuffer commandBuffer,
                   VkPipelineStageFlags2 sourceStage,
                   VkAccessFlags2 sourceAccess,
                   VkPipelineStageFlags2 destinationStage,
                   VkAccessFlags2 destinationAccess);

// 정점·인스턴스·가속 구조 입력 버퍼를 읽는 단계 전부. 스킨 패스와 유체·입자 컴퓨트가 지난 프레임의 읽기를
// 기다리고 이번 프레임의 읽기에 앞설 때 같은 집합을 쓴다.
inline constexpr VkPipelineStageFlags2 READER_STAGES =
    VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
    VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
    VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;

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

// 파이프라인·동적 렌더링에서 자주 쓰는 짧은 꼴. 렌더러와 자기 패스를 가진 플러그인이 함께 쓴다.
inline VkPipelineShaderStageCreateInfo shaderStage(VkShaderStageFlagBits stage, VkShaderModule module) {
    VkPipelineShaderStageCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    info.stage = stage;
    info.module = module;
    info.pName = "main";
    return info;
}

inline VkRenderingAttachmentInfo colorAttachment(VkImageView view, VkAttachmentLoadOp loadOp, VkClearColorValue clear) {
    VkRenderingAttachmentInfo info{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    info.imageView = view;
    info.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    info.loadOp = loadOp;
    info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    info.clearValue.color = clear;
    return info;
}

inline void setFullViewport(VkCommandBuffer commandBuffer, VkExtent2D extent) {
    VkViewport viewport{};
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.maxDepth = 1.0F;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = extent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
}

} // namespace gfx
