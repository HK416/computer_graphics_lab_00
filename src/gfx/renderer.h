#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <vulkan/vulkan.h>

#include "gfx/lod_network.h"
#include "gfx/profiler.h"
#include "gfx/raytracing.h"
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

// shaders/scene_types.glsl 의 Light 와 배치가 같아야 한다.
struct GpuLight {
    glm::vec4 positionRange;      // xyz 위치, w 도달 거리
    glm::vec4 directionIntensity; // xyz 앞 방향, w 세기
    glm::vec4 colorType;          // xyz 색, w 종류
    glm::vec4 coneSize;           // xy 원뿔 cos(안/바깥), zw 영역광 반크기
    glm::vec4 rightShadow;        // xyz 가로축, w 그림자 아틀라스 첫 타일(-1 이면 없음)
    glm::vec4 up;                 // xyz 세로축
};

// 그림자 아틀라스 한 변의 픽셀 수. shaders/shadow.glsl 의 SHADOW_ATLAS_SIZE 와 같아야 한다.
inline constexpr uint32_t SHADOW_ATLAS_SIZE = 4096;
inline constexpr uint32_t SHADOW_TILES_PER_SIDE = 4;
inline constexpr uint32_t SHADOW_TILE_SIZE = SHADOW_ATLAS_SIZE / SHADOW_TILES_PER_SIDE;
inline constexpr uint32_t MAX_SHADOW_VIEWS = SHADOW_TILES_PER_SIDE * SHADOW_TILES_PER_SIDE;

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

// 업스케일 방식. 벤더 SDK 가 필요한 것들은 감지만 하고 사용 가능 여부를 보고한다.
enum class Upscaler : uint32_t {
    NONE = 0,
    SPATIAL = 1,
    FSR = 2,
    DLSS = 3,
};

struct UpscalerInfo {
    Upscaler kind;
    const char* name;
    bool available;
    // 쓸 수 없을 때의 이유. 사용 가능하면 비어 있다.
    const char* reason;
};

// 화면 크기에 맞춰 다시 만들어지는 오프스크린 대상들. 셰이더에서 읽으려고 bindless 슬롯도 함께 잡는다.
struct RenderTargets {
    Image color; // 선형 HDR
    Image depth;
    Image oitAccumulation;
    Image oitRevealage;
    Image tonemapped; // 렌더 해상도의 톤 매핑 결과. 업스케일 입력이다.
    Image present;    // 표시 해상도. 편집기 뷰포트가 그대로 보여준다.
    // 이전 프레임 깊이로 만든 계층적 Z 버퍼. 오클루전 컬링이 읽는다.
    Image hzb;
    // 조명별 그림자 깊이를 담는 타일 아틀라스. 화면 크기와 무관해 한 번만 만든다.
    Image shadowAtlas;
    uint32_t shadowAtlasSlot = 0;
    // 반해상도 SSAO. 잡음이 많아 흐린 결과를 따로 둔다.
    Image ssaoRaw;
    Image ssao;
    VkExtent2D ssaoExtent{};
    uint32_t ssaoRawSlot = 0;
    uint32_t ssaoRawStorageSlot = 0;
    uint32_t ssaoSlot = 0;
    uint32_t ssaoStorageSlot = 0;
    // 경로 추적 누적 버퍼. 카메라가 멈춰 있는 동안 표본을 쌓는다.
    Image pathAccumulation;
    uint32_t pathAccumulationStorageSlot = 0;
    uint32_t pathAccumulationSampledSlot = 0;
    std::vector<VkImageView> hzbMipViews;
    std::vector<uint32_t> hzbStorageSlots;
    VkExtent2D hzbExtent{};
    uint32_t hzbSampledSlot = 0;
    uint32_t depthSlot = 0;
    uint32_t colorSlot = 0;
    uint32_t tonemappedSlot = 0;
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

    // 프레임을 시작하기 전에 밀린 크기 변경을 처리한다. UI 가 렌더 타겟을 참조하기 전에
    // 재생성이 끝나야 파괴된 이미지 뷰를 가리키는 디스크립터가 남지 않는다.
    void prepareFrame();
    // 프레임 맨 앞에서 프로파일러 슬롯을 연다. CPU 구간이 그리기보다 먼저 기록되기 때문이다.
    void beginProfilerFrame() { frameProfiler.beginFrame(static_cast<uint32_t>(frameIndex % FRAMES_IN_FLIGHT)); }
    // 지오메트리 저장소에 모델이 더해진 뒤 불린다. 가속 구조를 다시 만든다.
    void onGeometryChanged();
    void drawFrame(const scene::Scene& scene);
    void requestResize() { resizeRequested = true; }

    // 표시 해상도. 편집기 뷰포트 크기에 맞춰 바뀐다. 장면은 여기에 렌더 배율을 곱한 해상도로 그린다.
    void setDisplayExtent(VkExtent2D extent);
    VkExtent2D displayExtent() const { return currentDisplayExtent; }
    VkExtent2D renderExtent() const { return currentRenderExtent; }

    // 업스케일 설정. 배율을 낮추면 장면을 작게 그린 뒤 확대한다.
    float renderScale = 1.0F;
    Upscaler upscaler = Upscaler::SPATIAL;
    float upscaleSharpness = 0.25F;
    std::vector<UpscalerInfo> upscalers() const;
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

    // 그림자. 방향광과 스폿광은 시점 하나, 점광은 여섯 면을 아틀라스 타일에 담는다.
    bool shadowsEnabled = true;

    // 화면 공간 주변광 차폐.
    bool useSsao = true;
    // 장면 반지름에 대한 비율. 장면 크기가 제각각이라 절대 길이로 두지 않는다.
    float ssaoRadius = 0.04F;
    float ssaoIntensity = 1.0F;
    float ssaoBias = 0.002F;
    uint32_t ssaoSamples = 16;

    // GPU 컴퓨트가 meshlet 단위로 컬링하고 간접 그리기 명령을 만든다.
    bool useComputeCulling = true;
    bool frustumCulling = true;
    bool coneCulling = true;
    bool occlusionCulling = true;

    // 신경망이 LOD 임계값을 보정해 삼각형 예산을 맞춘다.
    bool useNeuralLod = false;
    bool trainLodNetwork = true;
    float triangleBudget = 60000.0F;
    LodNetwork lodNetwork;
    uint32_t lastSelectedTriangles = 0;

    // 경로 추적. 하드웨어가 지원할 때만 켤 수 있다.
    bool usePathTracing = false;
    PathTraceOptions pathTrace;
    bool pathTracingAvailable() const { return rayTracer != nullptr; }
    uint32_t pathTraceSamples() const { return pathSampleCount; }
    // 설정을 바꾸면 쌓인 표본이 섞이므로 편집기가 이걸 눌러 처음부터 다시 쌓게 한다.
    void resetPathAccumulation() { pathSampleCount = 0; }
    // mesh shader 미지원 장치에서는 켤 수 없다.
    bool useMeshShader = false;
    bool meshShaderAvailable() const { return meshShaderPipelines[0] != VK_NULL_HANDLE; }
    // 와이어프레임 디버그 뷰는 고전 경로에만 있으므로 그때는 mesh shader 경로를 쓰지 않는다.
    bool useMeshPath() const { return useMeshShader && meshShaderAvailable() && !wireframe; }

    // GPU/CPU 구간 계측. 편집기가 켜고 끄며, 꺼져 있으면 기록 자체를 하지 않는다.
    GpuProfiler& profiler() { return frameProfiler; }

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
        Buffer lodNetworkBuffer;
        // 스킨 인스턴스의 조인트 행렬을 이어 붙인다. 인스턴스마다 jointOffset 으로 자기 구간을 찾는다.
        Buffer jointBuffer;
        uint32_t jointCapacity = 0;
        // 장면의 조명과 그림자 시점 행렬. 둘 다 매 프레임 다시 채운다.
        Buffer lightBuffer;
        Buffer shadowMatrixBuffer;
        uint32_t lightCapacity = 0;
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
    void updateRenderExtent();
    void recreateSwapchain();
    void reserveInstances(Frame& frame, uint32_t instanceCount);
    void reserveMeshletGroups(Frame& frame, uint32_t groupCount);
    void reserveMeshletDraws(Frame& frame, uint32_t drawCount);
    void reserveJoints(Frame& frame, uint32_t jointCount);
    void reserveLights(Frame& frame, uint32_t lightCount);
    // 장면의 조명을 GPU 배치로 옮기고 그림자 시점을 정한다.
    void buildLights(Frame& frame, const scene::Scene& scene);
    void createCullPipeline();
    void createShadowPipeline();
    void createSsaoPipelines();
    void recordShadowPass(VkCommandBuffer commandBuffer, const FrameBatches& batches);
    void recordSsaoPass(VkCommandBuffer commandBuffer, const Frame& frame);
    void recordCullPass(VkCommandBuffer commandBuffer, const FrameBatches& batches);
    void recordHzbPass(VkCommandBuffer commandBuffer);
    void updateLodNetwork(const scene::Scene& scene, Frame& frame, float projectionScale);
    // 장면을 순회하며 인스턴스와 간접 그리기 명령을 재질 경로별 구간으로 채운다.
    FrameBatches buildDrawCommands(Frame& frame, const scene::Scene& scene);
    void recordCommands(Frame& frame, uint32_t imageIndex, const FrameBatches& batches, const scene::Scene& scene);
    void recordPathTracePass(VkCommandBuffer commandBuffer, Frame& frame, const scene::Scene& scene);
    void recordGeometryPass(VkCommandBuffer commandBuffer, const FrameBatches& batches, bool translucentPass);
    void recordUiPass(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    void writeCapture();

    Context& context;
    GeometryStore& geometry;
    BindlessTextures& bindless;
    std::unique_ptr<Swapchain> swapchain;
    RenderTargets targets;
    VkSampler postSampler = VK_NULL_HANDLE;
    VkExtent2D currentDisplayExtent{};
    VkExtent2D currentRenderExtent{};
    uint64_t generation = 0;
    bool oitTargetsValid = false;
    std::function<void(VkCommandBuffer)> uiCallback;

    // 오브젝트 인덱스 -> 인스턴스 슬롯. 그리지 않는 오브젝트는 INVALID_INSTANCE_SLOT.
    // 인스턴스는 버킷 순서로 채워지므로 장면 순서와 다르고, TLAS 가 이 표를 그대로 써야 한다.
    std::vector<uint32_t> objectInstanceSlots;
    // 장면이 바뀌었는지 판단하는 기준. 경로 추적 누적과 TLAS 재빌드가 함께 본다.
    const scene::Scene* lastScene = nullptr;
    uint64_t lastSceneRevision = 0;
    bool sceneChangedThisFrame = true;

    // buildLights 가 채운다. 그림자 패스와 푸시 상수가 함께 쓴다.
    std::vector<GpuLight> frameLights;
    std::vector<glm::mat4> shadowViews;

    GpuProfiler frameProfiler;

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
    VkPipelineLayout depthPipelineLayout = VK_NULL_HANDLE;
    VkPipeline shadowPipeline = VK_NULL_HANDLE;
    // 컷오프 캐스터는 프래그먼트 셰이더에서 discard 해야 실루엣이 맞는다.
    VkPipeline shadowCutoffPipeline = VK_NULL_HANDLE;
    VkPipelineLayout ssaoPipelineLayout = VK_NULL_HANDLE;
    VkPipeline ssaoPipeline = VK_NULL_HANDLE;
    VkPipelineLayout ssaoBlurPipelineLayout = VK_NULL_HANDLE;
    VkPipeline ssaoBlurPipeline = VK_NULL_HANDLE;
    // buildLights 가 재는 장면 반지름. SSAO 반지름을 장면 크기에 맞추는 데 쓴다.
    float sceneRadius = 1.0F;
    bool ssaoNeedsClear = true;
    VkPipelineLayout cullPipelineLayout = VK_NULL_HANDLE;
    VkPipeline cullPipeline = VK_NULL_HANDLE;
    VkPipelineLayout hzbPipelineLayout = VK_NULL_HANDLE;
    VkPipeline hzbPipeline = VK_NULL_HANDLE;
    PFN_vkCmdDrawIndexedIndirectCount drawIndexedIndirectCount = nullptr;
    VkPipelineLayout postPipelineLayout = VK_NULL_HANDLE;
    VkPipeline compositePipeline = VK_NULL_HANDLE;
    VkPipeline tonemapPipeline = VK_NULL_HANDLE;
    std::array<VkPipeline, 2> upscalePipelines{};

    std::filesystem::path capturePath;
    Buffer captureBuffer;

    std::unique_ptr<RayTracer> rayTracer;
    glm::mat4 lastViewProjection{0.0F};
    uint32_t pathSampleCount = 0;

    bool hzbNeedsClear = true;
    bool resizeRequested = false;
    bool vsync = true;
};

} // namespace gfx
