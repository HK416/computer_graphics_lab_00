#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
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

// 한 프레임의 그리기 구간. 고전 경로는 간접 그리기 명령, mesh shader 경로는 태스크 그룹 단위다.
struct FrameBatches {
    DrawBatches draws;
    DrawBatches groups;
    // 컴퓨트 컬링이 쓸 버킷별 간접 그리기 명령 구간. count 는 상한이고 실제 개수는 GPU 가 센다.
    DrawBatches meshletDraws;
    uint32_t instanceCount = 0;
};

// 화면 크기에 맞춰 다시 만들어지는 오프스크린 대상들. 셰이더에서 읽으려고 bindless 슬롯도 함께 잡는다.
struct RenderTargets {
    Image color; // 선형 HDR
    Image depth;
    Image oitAccumulation;
    Image oitRevealage;
    Image present; // 톤 매핑 결과. 편집기 뷰포트에 그대로 표시한다.
    // 이전 프레임 깊이로 만든 계층적 Z 버퍼. 오클루전 컬링이 읽는다.
    Image hzb;
    std::vector<VkImageView> hzbMipViews;
    std::vector<uint32_t> hzbStorageSlots;
    VkExtent2D hzbExtent{};
    uint32_t hzbSampledSlot = 0;
    uint32_t depthSlot = 0;
    uint32_t colorSlot = 0;
    uint32_t accumulationSlot = 0;
    uint32_t revealageSlot = 0;
    bool slotsAllocated = false;
};

class Renderer {
public:
    Renderer(Context& context, GeometryStore& geometry, BindlessTextures& bindless, SDL_Window* window);
    ~Renderer();
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void drawFrame(const scene::Scene& scene);
    void requestResize() { resizeRequested = true; }

    // 장면을 그릴 해상도. 편집기 뷰포트 크기에 맞춰 바뀐다.
    void setRenderExtent(VkExtent2D extent);
    VkExtent2D renderExtent() const { return currentRenderExtent; }
    // 스왑체인에 UI 를 기록할 콜백. 편집기가 채운다.
    void setUiCallback(std::function<void(VkCommandBuffer)> callback) { uiCallback = std::move(callback); }

    // 디버그 뷰어가 보여줄 오프스크린 대상.
    struct TargetView {
        const char* name;
        VkImageView view;
        VkImageLayout layout;
    };
    std::vector<TargetView> targetViews() const;
    VkImageView presentView() const { return targets.present.view; }
    // 대상이 다시 만들어질 때마다 증가한다. 편집기가 디스크립터를 다시 잡는 기준이다.
    uint64_t targetsGeneration() const { return generation; }
    VkFormat swapchainFormat() const;
    uint32_t swapchainImageCount() const;
    // 다음 프레임의 색상 버퍼를 PNG 로 저장한다. 렌더 결과 검증에 쓴다.
    void requestCapture(std::filesystem::path path) { capturePath = std::move(path); }
    void waitIdle();

    float exposure = 1.0F;
    bool wireframe = false;
    // shaders/scene_data.glsl 의 DEBUG_MODE_* 값.
    uint32_t debugMode = 0;
    // 자동 LOD 선정을 끄면 이 단계를 강제한다.
    bool automaticLod = true;
    uint32_t lodLevel = 0;
    // 허용할 화면 공간 오차(픽셀). 클수록 낮은 단계를 고른다.
    float lodErrorThreshold = 1.0F;

    // GPU 컴퓨트가 meshlet 단위로 컬링하고 간접 그리기 명령을 만든다.
    bool useComputeCulling = true;
    bool frustumCulling = true;
    bool coneCulling = true;
    bool occlusionCulling = true;
    // mesh shader 미지원 장치에서는 켤 수 없다.
    bool useMeshShader = false;
    bool meshShaderAvailable() const { return meshShaderPipelines[0] != VK_NULL_HANDLE; }

    bool vsyncEnabled() const { return vsync; }
    void setVsync(bool enabled);

private:
    struct Frame {
        VkCommandPool commandPool = VK_NULL_HANDLE;
        VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        Buffer cameraBuffer;
        Buffer instanceBuffer;
        Buffer drawBuffer;
        Buffer meshletGroupBuffer;
        Buffer meshTaskIndirectBuffer;
        // 컴퓨트 컬링이 채우는 meshlet 단위 간접 그리기 명령과 버킷별 개수.
        Buffer meshletDrawBuffer;
        Buffer drawCountBuffer;
        uint32_t instanceCapacity = 0;
        uint32_t groupCapacity = 0;
        uint32_t meshletDrawCapacity = 0;
    };

    void createFrames();
    void createPresentSemaphores();
    void destroyPresentSemaphores();
    void createRenderTargets();
    void createMeshPipelines();
    void createPostPipelines();
    void recreateSwapchain();
    void reserveInstances(Frame& frame, uint32_t instanceCount);
    void reserveMeshletGroups(Frame& frame, uint32_t groupCount);
    void reserveMeshletDraws(Frame& frame, uint32_t drawCount);
    void createCullPipeline();
    void recordCullPass(VkCommandBuffer commandBuffer, const FrameBatches& batches);
    void recordHzbPass(VkCommandBuffer commandBuffer);
    // 장면을 순회하며 인스턴스와 간접 그리기 명령을 재질 경로별 구간으로 채운다.
    FrameBatches buildDrawCommands(Frame& frame, const scene::Scene& scene);
    void recordCommands(Frame& frame, uint32_t imageIndex, const FrameBatches& batches);
    void recordGeometryPass(VkCommandBuffer commandBuffer, const FrameBatches& batches, bool translucentPass);
    void recordUiPass(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    void writeCapture();

    Context& context;
    GeometryStore& geometry;
    BindlessTextures& bindless;
    std::unique_ptr<Swapchain> swapchain;
    RenderTargets targets;
    VkSampler postSampler = VK_NULL_HANDLE;
    VkExtent2D currentRenderExtent{};
    uint64_t generation = 0;
    bool oitTargetsValid = false;
    std::function<void(VkCommandBuffer)> uiCallback;

    std::array<Frame, FRAMES_IN_FLIGHT> frames{};
    // 표시 완료 세마포어는 재사용 충돌을 피하려고 스왑체인 이미지마다 하나씩 둔다.
    std::vector<VkSemaphore> presentReady;
    VkSemaphore frameTimeline = VK_NULL_HANDLE;
    uint64_t frameIndex = 0;

    VkPipelineLayout meshPipelineLayout = VK_NULL_HANDLE;
    std::array<VkPipeline, ALPHA_MODE_COUNT> meshPipelines{};
    VkPipeline wireframePipeline = VK_NULL_HANDLE;
    std::array<VkPipeline, ALPHA_MODE_COUNT> meshShaderPipelines{};
    PFN_vkCmdDrawMeshTasksIndirectEXT drawMeshTasksIndirect = nullptr;
    VkShaderStageFlags scenePushStages = 0;
    VkPipelineLayout cullPipelineLayout = VK_NULL_HANDLE;
    VkPipeline cullPipeline = VK_NULL_HANDLE;
    VkPipelineLayout hzbPipelineLayout = VK_NULL_HANDLE;
    VkPipeline hzbPipeline = VK_NULL_HANDLE;
    PFN_vkCmdDrawIndexedIndirectCount drawIndexedIndirectCount = nullptr;
    VkPipelineLayout postPipelineLayout = VK_NULL_HANDLE;
    VkPipeline compositePipeline = VK_NULL_HANDLE;
    VkPipeline tonemapPipeline = VK_NULL_HANDLE;

    std::filesystem::path capturePath;
    Buffer captureBuffer;

    bool hzbNeedsClear = true;
    bool resizeRequested = false;
    bool vsync = true;
};

} // namespace gfx
