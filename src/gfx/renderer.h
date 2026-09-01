#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "gfx/resources.h"

struct SDL_Window;

namespace gfx {

struct Context;
struct Swapchain;

inline constexpr uint32_t FRAMES_IN_FLIGHT = 2;

class Renderer {
public:
    Renderer(Context& context, SDL_Window* window);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void drawFrame();
    void requestResize() { resizeRequested = true; }
    void waitIdle();

private:
    struct Frame {
        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
    };

    void createFrames();
    void createPresentSemaphores();
    void destroyPresentSemaphores();
    void createTriangleResources();
    void createTrianglePipeline();
    void recreateSwapchain();
    void recordCommands(VkCommandBuffer commandBuffer, uint32_t imageIndex);

    Context& context;
    std::unique_ptr<Swapchain> swapchain;

    std::array<Frame, FRAMES_IN_FLIGHT> frames{};
    // 표시 완료 세마포어는 프레임이 아니라 스왑체인 이미지마다 하나씩 두어야 재사용 충돌이 없다.
    std::vector<VkSemaphore> presentReady;
    VkSemaphore frameTimeline = VK_NULL_HANDLE;
    uint64_t frameIndex = 0;

    VkPipelineLayout trianglePipelineLayout = VK_NULL_HANDLE;
    VkPipeline trianglePipeline = VK_NULL_HANDLE;
    Buffer triangleVertices;

    bool resizeRequested = false;
    bool vsync = true;
};

} // namespace gfx
