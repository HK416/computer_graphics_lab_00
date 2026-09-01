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

#include "gfx/environment.h"
#include "gfx/lod_network.h"
#include "gfx/profiler.h"
#include "gfx/raytracing.h"
#include "gfx/resources.h"
#include "gfx/shadow_math.h"

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
    glm::vec4 rightShadow;        // xyz 가로축, w 그림자 첫 층(-1 이면 없음)
    glm::vec4 up;                 // xyz 세로축, w 이 조명이 쓰는 그림자 시점 수
    glm::vec4 cascadeSplits;      // 캐스케이드 i 의 끝 거리. 방향광이 아니면 쓰지 않는다
    glm::vec4 cascadeTexelSizes;  // 캐스케이드 i 의 월드 텍셀 크기. 노멀 오프셋 배율이다
};

// 그림자 시점 하나. 컬링에 쓸 정보까지 함께 담는다.
struct ShadowView {
    glm::mat4 viewProjection{1.0F};
    // 그림자가 뻗어 나가는 방향. 방향광은 고정이고, 점광/스폿광은 캐스터마다 다시 구한다.
    glm::vec3 sweepDirection{0.0F};
    glm::vec3 origin{0.0F};
    bool directional = false;
};

// 층마다의 캐싱 상태. 실제로 그려 둔 시점 행렬과 비교해 다시 그릴지 정한다.
struct ShadowLayerState {
    glm::mat4 drawnViewProjection{0.0F};
    bool valid = false;
};

// 그림자 맵 한 장의 크기와 층 수. 아틀라스를 타일로 자르지 않고 2D 배열의 층 하나씩 쓴다.
// 그래야 다시 그릴 필요 없는 시점은 렌더 패스를 아예 시작하지 않아, 타일 기반 GPU 에서
// 첨부물을 통째로 읽어 오는 비용이 생기지 않는다.
inline constexpr uint32_t SHADOW_MAP_SIZE = 1024;
inline constexpr uint32_t MAX_SHADOW_VIEWS = 16;

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
    // 조명별 그림자 깊이. 층 하나가 시점 하나다. 화면 크기와 무관해 한 번만 만든다.
    Image shadowAtlas;
    // 층마다의 2D 뷰. 그 층만 렌더 패스 대상으로 잡을 때 쓴다.
    std::vector<VkImageView> shadowLayerViews;
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
    // 시점별 절두체 컬링과, 그림자가 화면에 닿을 수 없는 캐스터를 버리는 스윕 컬링.
    bool shadowViewCulling = true;
    bool shadowCasterCulling = true;
    // 방향광 캐스케이드. 층이 모자라면 자동으로 줄어든다.
    uint32_t shadowCascades = 4;
    float shadowSplitLambda = 0.85F;
    // 0 이면 장면 크기에서 자동으로 정한다.
    float shadowDistance = 0.0F;
    // 광원과 캐스터가 그대로인 시점은 다시 그리지 않는다.
    bool shadowCaching = true;
    uint32_t shadowLayersDrawn() const { return shadowLayersRedrawn; }
    uint32_t shadowDrawCount() const { return shadowDrawsIssued; }
    uint32_t shadowDrawCandidates() const { return shadowDrawsTotal; }

    // 화면 공간 주변광 차폐.
    bool useSsao = true;
    // 환경광을 IBL 로 계산한다. 끄면 균일 환경광만 남는다.
    bool useIbl = true;
    // 하이브리드 그림자: 카메라에서 이 거리 안쪽은 광선으로 가시성을 판정하고 나머지는 그림자 맵을
    // 그대로 쓴다. 광선 질의를 지원하는 장치에서만 켤 수 있다.
    bool useRayQueryShadows = false;
    float rayShadowDistance = 12.0F;
    bool rayQueryShadowsAvailable() const;
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
    void invalidateEnvironment() {
        if (environment) {
            environment->invalidate();
        }
    }
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
        // 시점별로 컬링해 압축한 그림자 그리기 명령. 시점 하나가 알파 경로 둘을 쓴다.
        Buffer shadowDrawBuffer;
        uint32_t shadowDrawCapacity = 0;
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
    void reserveShadowDraws(Frame& frame, uint32_t drawCount);
    // 그림자 시점마다 절두체와 캐스터 스윕으로 걸러 압축한 그리기 명령을 만든다.
    void buildShadowDraws(Frame& frame, const FrameBatches& batches, const glm::mat4& cameraViewProjection);
    // 장면의 조명을 GPU 배치로 옮기고 그림자 시점을 정한다.
    void buildLights(Frame& frame, const scene::Scene& scene);
    void createCullPipeline();
    void createShadowPipeline();
    void createSsaoPipelines();
    void recordShadowPass(VkCommandBuffer commandBuffer);
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
    std::vector<ShadowView> shadowViews;
    // 시점 × 알파 경로마다의 그리기 구간. shadowDrawBuffer 기준이다.
    std::vector<DrawBatch> shadowBatches;
    // 인스턴스 슬롯별 세계 경계 구. 시점 컬링이 쓴다.
    std::vector<glm::vec4> instanceBounds;
    // 컬링 전후 그리기 수. 편집기에 보여준다.
    uint32_t shadowDrawsIssued = 0;
    uint32_t shadowDrawsTotal = 0;
    std::array<ShadowLayerState, MAX_SHADOW_VIEWS> shadowLayers{};
    std::vector<uint8_t> shadowLayerDirty;
    uint32_t shadowLayersRedrawn = 0;
    uint64_t lastShadowSettings = 0;

    GpuProfiler frameProfiler;

    std::array<Frame, FRAMES_IN_FLIGHT> frames{};
    // 표시 완료 세마포어는 재사용 충돌을 피하려고 스왑체인 이미지마다 하나씩 둔다.
    std::vector<VkSemaphore> presentReady;
    VkSemaphore frameTimeline = VK_NULL_HANDLE;
    uint64_t frameIndex = 0;

    VkPipelineLayout meshPipelineLayout = VK_NULL_HANDLE;
    std::array<VkPipeline, ALPHA_MODE_COUNT> meshPipelines{};
    // 광선 질의로 그림자를 판정하는 같은 파이프라인. 하드웨어가 지원할 때만 만든다.
    std::array<VkPipeline, ALPHA_MODE_COUNT> meshRayQueryPipelines{};
    VkPipelineLayout meshRayQueryPipelineLayout = VK_NULL_HANDLE;
    // 이번 프레임 장면 패스가 광선 질의 파이프라인을 쓰는지. 기록 중에만 뜻이 있다.
    bool rayQueryPass = false;
    VkPipeline wireframePipeline = VK_NULL_HANDLE;
    std::array<VkPipeline, ALPHA_MODE_COUNT> meshShaderPipelines{};
    std::array<VkPipeline, ALPHA_MODE_COUNT> meshShaderRayQueryPipelines{};
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
    // 그림자 층을 한 번 읽기 좋은 레이아웃으로 옮겨 둔다. 그 뒤로는 층마다 따로 전이한다.
    bool shadowNeedsInit = true;
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
    // 누적을 언제 버려야 하는지 판단하려고 지난 프레임 값을 들고 있다. 카메라와 장면만 보면
    // 렌더 설정을 바꿔도 화면이 그대로여서 멈춘 것처럼 보인다.
    PathTraceOptions lastPathTrace{};
    bool lastUseIbl = true;
    std::unique_ptr<EnvironmentMap> environment;
    // 첫 방향광의 진행 방향. 하늘의 태양을 그림자와 맞추는 데 쓴다.
    glm::vec3 sunDirection{0.0F, -1.0F, 0.0F};
    glm::mat4 lastViewProjection{0.0F};
    uint32_t pathSampleCount = 0;

    bool hzbNeedsClear = true;
    bool resizeRequested = false;
    bool vsync = true;
};

} // namespace gfx
