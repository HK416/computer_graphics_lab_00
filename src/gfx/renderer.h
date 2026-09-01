#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "gfx/resources.h"

struct SDL_Window;

namespace scene {
struct Scene;
}

namespace gfx {

struct Context;
struct Swapchain;
class BindlessTextures;
class GeometryStore;

inline constexpr uint32_t FRAMES_IN_FLIGHT = 2;
inline constexpr uint32_t ALPHA_MODE_COUNT = 3;

// 간접 그리기 버퍼 안의 연속 구간. 재질 경로와 면 방향 조합마다 하나씩 둔다.
struct DrawBatch {
    uint32_t first = 0;
    uint32_t count = 0;
};
using DrawBatches = std::array<std::array<DrawBatch, 2>, ALPHA_MODE_COUNT>;

class Renderer {
public:
    Renderer(Context& context, GeometryStore& geometry, BindlessTextures& bindless, SDL_Window* window);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void drawFrame(const scene::Scene& scene);
    void requestResize() { resizeRequested = true; }
    // 다음 프레임의 색상 버퍼를 PNG 로 저장한다. 렌더 결과 검증에 쓴다.
    void requestCapture(std::filesystem::path path) { capturePath = std::move(path); }
    void waitIdle();

private:
    struct Frame {
        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        Buffer cameraBuffer;
        Buffer instanceBuffer;
        Buffer drawBuffer;
        uint32_t instanceCapacity = 0;
    };

    void createFrames();
    void createPresentSemaphores();
    void destroyPresentSemaphores();
    void createDepthImage();
    void createMeshPipelines();
    void recreateSwapchain();
    void reserveInstances(Frame& frame, uint32_t instanceCount);
    // 장면을 순회하며 인스턴스와 간접 그리기 명령을 재질 경로별 구간으로 채운다.
    DrawBatches buildDrawCommands(Frame& frame, const scene::Scene& scene);
    void recordCommands(Frame& frame, uint32_t imageIndex, const DrawBatches& batches);
    void writeCapture();

    Context& context;
    GeometryStore& geometry;
    BindlessTextures& bindless;
    std::unique_ptr<Swapchain> swapchain;
    Image depthImage;

    std::array<Frame, FRAMES_IN_FLIGHT> frames{};
    // 표시 완료 세마포어는 재사용 충돌을 피하려고 스왑체인 이미지마다 하나씩 둔다.
    std::vector<VkSemaphore> presentReady;
    VkSemaphore frameTimeline = VK_NULL_HANDLE;
    uint64_t frameIndex = 0;

    VkPipelineLayout meshPipelineLayout = VK_NULL_HANDLE;
    std::array<VkPipeline, ALPHA_MODE_COUNT> meshPipelines{};

    std::filesystem::path capturePath;
    Buffer captureBuffer;

    bool resizeRequested = false;
    bool vsync = true;
};

} // namespace gfx
