#pragma once

#include <vector>

#include <vulkan/vulkan.h>

#include "gfx/resources.h"

namespace gfx {

struct Context;

// 전송 큐로 스테이징 복사를 수행하고, 큐 패밀리가 다르면 그래픽스 큐로 소유권을 넘긴다.
// 적재 시점 업로드용이며, flush() 는 완료까지 대기한다.
class Uploader {
public:
    explicit Uploader(Context& context);
    ~Uploader();
    Uploader(const Uploader&) = delete;
    Uploader& operator=(const Uploader&) = delete;

    void uploadBuffer(const Buffer& target, VkDeviceSize offset, const void* data, VkDeviceSize size);
    void uploadImage(const Image& target, const void* data, VkDeviceSize size, VkImageLayout finalLayout);

    void flush();

private:
    void beginRecording();
    VkBuffer createStaging(const void* data, VkDeviceSize size, const char* debugName);

    Context& context;
    VkCommandPool transferPool = VK_NULL_HANDLE;
    VkCommandPool graphicsPool = VK_NULL_HANDLE;
    VkCommandBuffer transferCommands = VK_NULL_HANDLE;
    VkCommandBuffer graphicsCommands = VK_NULL_HANDLE;
    VkSemaphore timeline = VK_NULL_HANDLE;
    uint64_t timelineValue = 0;
    std::vector<Buffer> stagingBuffers;
    bool recording = false;
    // 전송 큐와 그래픽스 큐가 같은 패밀리면 소유권 이전이 필요 없다.
    bool needsOwnershipTransfer = false;
};

} // namespace gfx
