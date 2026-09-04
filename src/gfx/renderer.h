#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec4.hpp>
#include <vulkan/vulkan.h>

#include "gfx/debug_lines.h"
#include "gfx/environment.h"
#include "gfx/fluid.h"
#include "gfx/lod_network.h"
#include "gfx/profiler.h"
#include "gfx/raytracing.h"
#include "gfx/resources.h"
#include "gfx/shadow_math.h"
#include "gfx/upscaler.h"

struct SDL_Window;

namespace core {
class JobSystem;
}

namespace scene {
struct Scene;
}

namespace gfx {

struct Context;
struct Swapchain;
class BindlessTextures;
class GeometryStore;

inline constexpr uint32_t FRAMES_IN_FLIGHT = 2;

// shaders/scene_types.glsl 의 DEBUG_MODE_* 와 값이 같아야 한다.
inline constexpr uint32_t DEBUG_MODE_SHADED = 0;
inline constexpr uint32_t DEBUG_MODE_NORMAL = 2;
inline constexpr uint32_t DEBUG_MODE_UV = 3;
inline constexpr uint32_t DEBUG_MODE_DEPTH = 4;
inline constexpr uint32_t DEBUG_MODE_SHADOW = 7;
inline constexpr uint32_t DEBUG_MODE_VELOCITY = 8;
inline constexpr uint32_t DEBUG_MODE_CULL_PHASE = 9;
inline constexpr uint32_t DEBUG_MODE_REFLECTION_RAW = 10;
inline constexpr uint32_t DEBUG_MODE_REFLECTION = 11;

// 경로 추적이 그릴 수 있는 디버그 뷰인지. meshlet 과 LOD 는 하위 가속 구조가 메쉬 단위 LOD 0 이라
// 개념 자체가 없고, 캐스케이드는 그림자 맵을 읽지 않으며, 모션 벡터는 경로 추적 프레임에 갱신되지
// 않는다.
inline bool pathTraceSupportsDebugMode(uint32_t mode) {
    return mode == DEBUG_MODE_SHADED || mode == DEBUG_MODE_NORMAL || mode == DEBUG_MODE_UV ||
           mode == DEBUG_MODE_DEPTH || mode == DEBUG_MODE_SHADOW || mode == DEBUG_MODE_VELOCITY;
}
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

// 스킨 컴퓨트 디스패치 하나. 오브젝트 하나가 자기 구간을 통째로 변형한다. destinationOffset 은
// 반쪽 안의 상대 위치라, 지난 프레임 목록과 같으면 다른 반쪽이 그대로 지난 포즈다.
struct SkinDispatch {
    uint32_t sourceOffset;
    uint32_t destinationOffset;
    uint32_t jointOffset;
    uint32_t vertexCount;
    // 스킨 가중치 버퍼에서 이 메쉬의 구간. 없으면 NO_SKIN_WEIGHTS.
    uint32_t skinWeightOffset;
    // 경계 구를 다시 잴 meshlet 구간과 결과 위치.
    uint32_t meshletOffset;
    uint32_t meshletCount;
    uint32_t boundsOffset;

    bool operator==(const SkinDispatch&) const = default;
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
    // 유체마다 인스턴스 드로우 하나(instanceCount = 입자 수). 오브젝트 명령 뒤에 이어진다. 버킷 구간은 0 부터
    // 연속이라 거기 끼워 넣을 수 없고, 입자마다 meshlet 그룹을 두면 디스패치 한도에 걸리므로 고전
    // 인스턴스 드로우로만 그린다.
    DrawBatch fluidDraws;
    // 유체 입자 인스턴스가 시작하는 슬롯. 오브젝트 인스턴스 바로 뒤다.
    uint32_t fluidInstanceBase = 0;
};

// 화면 크기에 맞춰 다시 만들어지는 오프스크린 대상들. 셰이더에서 읽으려고 bindless 슬롯도 함께 잡는다.
struct RenderTargets {
    Image color; // 선형 HDR
    Image depth;
    Image oitAccumulation;
    Image oitRevealage;
    // 화면 UV 단위 모션 벡터. 불투명 패스가 함께 기록한다.
    Image velocity;
    Image tonemapped; // 렌더 해상도의 톤 매핑 결과. 공간 업스케일 입력이다.
    // 시간축 업스케일이 내놓는 표시 해상도 선형 HDR. 톤 매핑이 이걸 읽는다.
    Image upscaledColor;
    Image present; // 표시 해상도. 편집기 뷰포트가 그대로 보여준다.
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
    // DLSS Ray Reconstruction 이 요구하는 안내 버퍼. 경로 추적이 1차 히트에서 채운다.
    // 깊이까지 여기 두는 이유는 경로 추적이 깊이 첨부물을 쓰지 않기 때문이다.
    Image guideDiffuseAlbedo;
    Image guideSpecularAlbedo;
    Image guideNormal;
    Image guideRoughness;
    Image guideDepth;
    uint32_t guideDiffuseAlbedoStorageSlot = 0;
    uint32_t guideSpecularAlbedoStorageSlot = 0;
    uint32_t guideNormalStorageSlot = 0;
    // 래스터 불투명 패스가 첨부물로 채운 노멀·거칠기와 반사 가중치를 반사 컴퓨트가 읽는 슬롯.
    // 경로 추적 프레임에는 이 이미지가 GENERAL 스토리지라 이 슬롯을 읽으면 안 된다.
    uint32_t guideNormalSlot = 0;
    uint32_t guideSpecularAlbedoSlot = 0;
    uint32_t guideRoughnessStorageSlot = 0;
    uint32_t guideDepthStorageSlot = 0;

    // Bloom 밉 사슬. 절반 해상도에서 시작해 단계마다 반으로 줄이고, 올라오며 더한다. 컴퓨트가
    // 쓰고 읽으므로 계속 GENERAL 이다.
    Image bloom;
    VkExtent2D bloomExtent{};
    std::vector<VkImageView> bloomMipViews;
    std::vector<uint32_t> bloomStorageSlots;
    // 밉마다 선형 샘플러로 읽는 슬롯. 업샘플 텐트 필터가 이중 선형 보간을 필요로 한다.
    std::vector<uint32_t> bloomSampledSlots;

    // 광선 반사. 이번 프레임 추적 결과와 시간축 누적 히스토리(더블 버퍼). 컴퓨트가 쓰고 읽으므로
    // 계속 GENERAL 이다.
    Image reflectionRaw;
    std::array<Image, 2> reflectionHistory;
    uint32_t reflectionRawSlot = 0;
    uint32_t reflectionRawStorageSlot = 0;
    std::array<uint32_t, 2> reflectionHistorySlots{};
    std::array<uint32_t, 2> reflectionHistoryStorageSlots{};
    // 반사 컴퓨트가 HDR 색상에 직접 더할 때 쓰는 rgba16f 스토리지 슬롯.
    uint32_t colorStorageSlot = 0;

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
    uint32_t velocitySlot = 0;
    // 경로 추적이 모션 벡터를 직접 쓸 때 쓰는 rg16f 스토리지 슬롯.
    uint32_t velocityStorageSlot = 0;
    uint32_t upscaledColorSlot = 0;
    uint32_t upscaledColorStorageSlot = 0;
    uint32_t accumulationSlot = 0;
    uint32_t revealageSlot = 0;
    bool slotsAllocated = false;
};

class Renderer {
public:
    // jobs 는 프레임마다 인스턴스 채우기와 그림자 시점별 컬링을 워커에 나누는 데 쓴다.
    Renderer(Context& context,
             GeometryStore& geometry,
             BindlessTextures& bindless,
             SDL_Window* window,
             core::JobSystem& jobs);
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

    // 디버그 뷰어가 보여줄 오프스크린 대상. 층이나 밉이 여럿이면 views 에 하나씩 담고 sliceLabel 로
    // 무엇을 고르는지 알려 준다. ImGui 는 2D 뷰만 그릴 수 있어 배열 뷰를 통째로 넘기지 않는다.
    struct TargetView {
        const char* name;
        std::vector<VkImageView> views;
        VkImageLayout layout;
        VkExtent2D extent;
        const char* sliceLabel = nullptr;
        // 지금 렌더 모드에서 이번 프레임에 실제로 채워지는 대상인지. 아니면 내용이 낡았거나
        // 레이아웃이 달라 보여 주면 안 된다.
        bool available = true;
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
    // 강체 콜라이더와 유체 경계를 선으로 덧그린다. Unity 의 기즈모처럼 물체 뒤로 숨는다.
    bool showColliders = true;
    bool colliderOcclusion = true;
    // 밝게 그릴 오브젝트. 편집기가 고른 것을 넣는다.
    int32_t selectedObject = -1;
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
    // 광선 기능이 처음 필요할 때 하위 가속 구조를 세운다. 예산을 넘으면 사유를 남기고 광선 기능을 끈다.
    bool ensureBottomLevel();
    // 광선 반사: 거칠기가 상한 이하인 불투명 표면의 스페큘러 IBL 을 추적한 반사로 바꾼다. 광선
    // 질의와 IBL 이 있어야 하고, 경로 추적 중에는 그쪽이 반사를 직접 계산하므로 꺼진다.
    bool useReflections = false;
    float reflectionRoughnessCutoff = 0.6F;
    float reflectionIntensity = 1.0F;
    uint32_t reflectionMaxSamples = 16;
    bool reflectionsActive() const;
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

    // 경로 추적. 하드웨어가 지원하고 가속 구조가 예산에 들어갈 때만 켤 수 있다.
    bool usePathTracing = false;
    PathTraceOptions pathTrace;
    bool pathTracingAvailable() const { return rayTracer != nullptr && rayTracingBlockedReason.empty(); }
    // 하위 가속 구조를 세우지 못한 사유. 비어 있으면 광선 기능을 쓸 수 있다. 편집기가 보여 준다.
    const std::string& rayTracingBlocked() const { return rayTracingBlockedReason; }
    void invalidateEnvironment() {
        if (environment) {
            environment->invalidate();
        }
    }
    uint32_t pathTraceSamples() const { return pathSampleCount; }
    // 설정을 바꾸면 쌓인 표본이 섞이므로 편집기가 이걸 눌러 처음부터 다시 쌓게 한다.
    void resetPathAccumulation() { pathSampleCount = 0; }
    // 유체 입자를 그릴 내장 구 메쉬. 애플리케이션이 등록한 번호를 넣는다. 없으면 유체를 그리지 않는다.
    uint32_t fluidSphereMesh = 0xFFFFFFFFU;
    // 유체 부품이 요청해도 이보다 많은 입자는 뿌리지 않는다. 하드웨어 프로파일이 정한다.
    uint32_t fluidParticleLimit = FLUID_MAX_PARTICLES;
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
        // drawBuffer 의 명령마다 그리는 meshlet. LOD 단위 명령이라 그 단계의 첫 meshlet 을 적는다.
        Buffer drawMeshletBuffer;
        Buffer meshletGroupBuffer;
        Buffer meshTaskIndirectBuffer;
        // 컴퓨트 컬링이 채우는 meshlet 단위 간접 그리기 명령과 버킷별 개수, 명령마다의 meshlet 번호.
        Buffer meshletDrawBuffer;
        Buffer meshletDrawMeshletBuffer;
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
        // 디버그 선의 정점. 콜라이더 표시를 켰을 때만 채운다.
        Buffer debugLineBuffer;
        uint32_t debugLineCapacity = 0;
        uint32_t lightCapacity = 0;
        uint32_t instanceCapacity = 0;
        uint32_t groupCapacity = 0;
        uint32_t meshletDrawCapacity = 0;
    };

    void createFrames();
    void createPresentSemaphores();
    void destroyPresentSemaphores();
    void createRenderTargets();
    // 편집기가 방식을 바꾸면 시간축 업스케일러를 다시 만든다.
    void updateUpscaler();
    // 시간축 업스케일러가 붙어 있고 마지막 크기 변경도 성공했는지. 벤더 SDK 는 컨텍스트 생성이
    // 실패할 수 있어, 그때는 지터도 끄고 공간 경로로 돌아가야 한다.
    bool temporalReady() const { return temporalUpscaler != nullptr && temporalUpscaler->ready(); }
    // RR 은 경로 추적 전용이다. 경로 추적이 꺼져 있으면 같은 DLSS 의 초해상으로 돌린다. 이렇게
    // 두지 않으면 RR 이 안내 버퍼가 없어 아무것도 쓰지 않고 돌아가, 표시 이미지가 비어 버린다.
    Upscaler effectiveUpscaler() const {
        return upscaler == Upscaler::DLSS_RR && !usePathTracing ? Upscaler::DLSS : upscaler;
    }
    // Ray Reconstruction 이 도는 프레임인지. 이때만 경로 추적이 누적하지 않고 1표본과 안내
    // 버퍼를 내놓으며, 지터도 들어간다.
    bool rayReconstructionActive() const {
        return usePathTracing && rayTracer != nullptr && upscaler == Upscaler::DLSS_RR && temporalReady();
    }
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
    // 스킨 정점을 포즈 공간으로 옮겨 따로 뽑아 두는 컴퓨트. 래스터와 광선 경로가 모두 이 결과를
    // 읽고, 경계 구 컴퓨트가 같은 결과로 meshlet 경계를 다시 잰다.
    void createSkinPipeline();
    void createShadowPipeline();
    void createSsaoPipelines();
    void createBloomPipelines();
    // 편집기의 콜라이더·유체 경계 표시. 톤 매핑과 공간 업스케일 뒤, UI 앞에 표시 해상도로 그린다.
    void createDebugLinePipeline();
    void reserveDebugLines(Frame& frame, uint32_t vertexCount);
    void recordDebugLines(VkCommandBuffer commandBuffer, Frame& frame, const scene::Scene& scene, VkExtent2D extent);
    // 광선 질의 컴퓨트로 반사를 추적하고 시간축으로 누적해 색상에 더한다. 광선 질의가 있을 때만 만든다.
    void createReflectionPipelines();
    void recordReflectionPass(VkCommandBuffer commandBuffer, const Frame& frame);
    // Bloom 밉 사슬과 자동 노출. 톤 매핑이 읽을 이미지를 원본으로 받는다.
    void recordPostEffects(VkCommandBuffer commandBuffer,
                           const scene::PostProcess& post,
                           uint32_t sourceSlot,
                           VkExtent2D sourceExtent,
                           uint32_t sampleCount);
    void recordShadowPass(VkCommandBuffer commandBuffer);
    void recordSsaoPass(VkCommandBuffer commandBuffer, const Frame& frame);
    // phase 는 culling.glsl 의 CULL_PHASE_*.
    void recordCullPass(VkCommandBuffer commandBuffer, const FrameBatches& batches, uint32_t phase);
    void reserveMeshletVisibility(uint32_t meshletCount);
    // 스킨 인스턴스의 변형 정점과 meshlet 경계 구를 만든다. 그림자 패스보다 먼저 온다.
    void recordSkinPass(VkCommandBuffer commandBuffer, const Frame& frame);
    // 유체를 진행하고 입자 인스턴스를 쓴다. 스킨 패스 뒤, 그림자 패스 앞. wantsTlas 면 상위 가속 구조
    // 인스턴스도 앞쪽에 써 두고 updateAccelerationStructures 가 그 뒤에 오브젝트를 잇는다.
    void recordFluidPass(VkCommandBuffer commandBuffer,
                         const Frame& frame,
                         const FrameBatches& batches,
                         const scene::Scene& scene,
                         bool wantsTlas);
    // 광선 경로가 이번 프레임에 쓸 가속 구조를 최신으로 맞춘다.
    void updateAccelerationStructures(VkCommandBuffer commandBuffer, const scene::Scene& scene);
    void recordHzbPass(VkCommandBuffer commandBuffer);
    void updateLodNetwork(const scene::Scene& scene, Frame& frame, float projectionScale);
    // 장면을 순회하며 인스턴스와 간접 그리기 명령을 재질 경로별 구간으로 채운다.
    FrameBatches buildDrawCommands(Frame& frame, const scene::Scene& scene);
    void recordCommands(Frame& frame, uint32_t imageIndex, const FrameBatches& batches, const scene::Scene& scene);
    void recordPathTracePass(VkCommandBuffer commandBuffer, Frame& frame, const scene::Scene& scene);
    void recordGeometryPass(VkCommandBuffer commandBuffer,
                            const FrameBatches& batches,
                            bool translucentPass,
                            uint32_t cullPhase);
    void recordUiPass(VkCommandBuffer commandBuffer, uint32_t imageIndex);
    void writeCapture();

    Context& context;
    GeometryStore& geometry;
    BindlessTextures& bindless;
    core::JobSystem& jobs;
    std::unique_ptr<Swapchain> swapchain;
    RenderTargets targets;
    VkSampler postSampler = VK_NULL_HANDLE;
    // Bloom 처럼 보간이 필요한 후처리가 쓴다. 밉을 명시적으로 고르므로 maxLod 를 열어 둔다.
    VkSampler linearSampler = VK_NULL_HANDLE;
    VkPipelineLayout reflectionPipelineLayout = VK_NULL_HANDLE;
    VkPipeline reflectionTracePipeline = VK_NULL_HANDLE;
    VkPipeline reflectionResolvePipeline = VK_NULL_HANDLE;
    // 지난 프레임에 반사 히스토리를 남겼는지. 아니면 이번 해결은 히스토리를 버린다.
    bool reflectionHistoryValid = false;
    VkPipelineLayout bloomPipelineLayout = VK_NULL_HANDLE;
    VkPipeline bloomDownsamplePipeline = VK_NULL_HANDLE;
    VkPipeline bloomUpsamplePipeline = VK_NULL_HANDLE;
    VkPipelineLayout histogramPipelineLayout = VK_NULL_HANDLE;
    VkPipeline histogramPipeline = VK_NULL_HANDLE;
    VkPipelineLayout exposurePipelineLayout = VK_NULL_HANDLE;
    VkPipeline exposurePipeline = VK_NULL_HANDLE;
    VkPipelineLayout debugLinePipelineLayout = VK_NULL_HANDLE;
    VkPipeline debugLinePipeline = VK_NULL_HANDLE;
    // 프레임마다 다시 채우는 선분 목록. 벡터를 그대로 두어 할당을 되쓴다.
    std::vector<DebugLineVertex> debugLineVertices;
    // 자동 노출. 히스토그램은 프레임마다 지우고, 노출 값은 프레임을 넘어 적응한다.
    Buffer histogramBuffer;
    Buffer exposureBuffer;
    // 참이면 다음 자동 노출은 적응 없이 목표 값으로 바로 간다.
    bool exposureNeedsReset = true;
    // 이번 프레임에 Bloom 밉 사슬을 채웠는지. 타깃 뷰어가 낡은 밉을 보여 주지 않게 한다.
    bool bloomActive = false;
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
    VkPipelineLayout skinPipelineLayout = VK_NULL_HANDLE;
    VkPipeline skinPipeline = VK_NULL_HANDLE;
    VkPipelineLayout skinBoundsPipelineLayout = VK_NULL_HANDLE;
    VkPipeline skinBoundsPipeline = VK_NULL_HANDLE;
    // 스킨 인스턴스의 변형 정점. 인스턴스마다 메쉬 정점 수만큼 이어 붙인 구간이 반쪽 둘로 있고,
    // 프레임마다 번갈아 쓴다. 쓰지 않는 반쪽이 지난 포즈다.
    Buffer skinnedVertexBuffer;
    uint32_t skinnedVertexCapacity = 0;
    // 스킨 인스턴스의 meshlet 경계 구. 변형 정점에서 프레임마다 다시 잰다.
    Buffer skinnedBoundsBuffer;
    uint32_t skinnedBoundsCapacity = 0;
    // 이번 프레임이 쓰는 반쪽. 정점 버퍼 안의 절대 위치는 이 값에 용량을 곱한 만큼 밀린다.
    uint32_t skinnedHalf = 0;
    // buildDrawCommands 가 채운다. 스킨 컴퓨트 디스패치와 하위 가속 구조 구축이 함께 읽는다.
    std::vector<SkinDispatch> skinDispatches;
    // 지난 프레임의 목록. 같으면 다른 반쪽이 그대로 지난 포즈다.
    std::vector<SkinDispatch> previousSkinDispatches;
    std::vector<SkinnedInstance> skinnedInstances;
    // 오브젝트 인덱스 -> skinnedInstances 번호. 스킨이 아니면 RayTracer::NO_SKINNED_BLAS.
    std::vector<uint32_t> objectSkinnedBlas;
    // GPU SPH. 장면의 유체 부품마다 상태를 들고, 입자 인스턴스를 프레임 인스턴스 버퍼 꼬리에 쓴다.
    std::unique_ptr<FluidSimulator> fluid;
    // 유체마다 용기의 경계 구. 그림자 시점 컬링이 쓴다.
    std::vector<glm::vec4> fluidBounds;
    // 이번 프레임 유체 패스가 상위 가속 구조 인스턴스 버퍼 앞쪽에 써 둔 입자 수.
    uint32_t fluidTlasPrepended = 0;
    VkPipelineLayout hzbPipelineLayout = VK_NULL_HANDLE;
    VkPipeline hzbPipeline = VK_NULL_HANDLE;
    // meshlet 가시성 비트. 프레임을 넘어 살아남으므로 프레임별 버퍼가 아니다. 낡은 비트는 1차
    // 패스의 드로우 낭비만 낳고 정확성은 2차 패스가 보장하므로 인스턴스 슬롯이 재배치돼도 지우지
    // 않는다. 커질 때만 0 으로 채운다.
    Buffer meshletVisibilityBuffer;
    uint32_t meshletVisibilityCapacity = 0;
    bool visibilityNeedsClear = true;
    PFN_vkCmdDrawIndexedIndirectCount drawIndexedIndirectCount = nullptr;
    std::string rayTracingBlockedReason;
    VkPipelineLayout postPipelineLayout = VK_NULL_HANDLE;
    VkPipeline compositePipeline = VK_NULL_HANDLE;
    VkPipeline tonemapPipeline = VK_NULL_HANDLE;
    std::array<VkPipeline, 2> upscalePipelines{};
    // 아무것도 그려지지 않은 화소를 하늘로 채운다. 깊이 판정이 걸러 주므로 셰이더는 분기가 없다.
    VkPipeline skyPipeline = VK_NULL_HANDLE;
    std::unique_ptr<TemporalUpscaler> temporalUpscaler;
    // temporalUpscaler 가 어느 방식으로 만들어졌는지. upscaler 와 어긋나면 다시 만든다.
    Upscaler activeUpscaler = Upscaler::NONE;
    // 이번 프레임 투영에 들어간 렌더 픽셀 단위 지터와, 히스토리를 버려야 하는지.
    glm::vec2 currentJitter{0.0F};
    bool temporalResetThisFrame = true;
    uint32_t jitterIndex = 0;
    // 프레임 간격. 벤더 업스케일러가 히스토리 감쇠에 쓴다. 편집기의 시간과 따로 재도 무방하다.
    std::chrono::steady_clock::time_point lastFrameTime{};
    float frameDeltaSeconds = 1.0F / 60.0F;
    // 경로에 따라 한쪽만 쓰이는 후처리 대상의 첫 레이아웃을 맞춘다.
    bool postTargetsNeedInit = true;

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
    // 안개가 바뀌면 경로 추적 누적을 버린다. 안개는 광선 생성 셰이더가 직접 걸기 때문이다.
    glm::vec4 lastFog{0.0F};
    glm::vec4 lastFogParameters{0.0F};
    // 모션 벡터용 지난 프레임 상태. 장면 구성이 바뀌면 현재 값으로 덮어 변위를 0 으로 만든다.
    glm::mat4 previousViewProjection{1.0F};
    std::vector<glm::mat4> previousWorld;
    std::vector<glm::mat4> jointMatrices;
    // 매핑된 버퍼는 캐시가 없어 필드 단위로 흩어 쓰거나 되읽으면 오브젝트 만 개에서 프레임당 수십 ms
    // 가 든다. CPU 사본을 채운 뒤 한 번에 복사한다. 그림자 시점별 압축도 이 사본에서 읽는다.
    std::vector<GpuInstance> instanceData;
    std::vector<VkDrawIndexedIndirectCommand> drawCommands;
    // drawCommands 와 나란히 간다. 명령마다 그리는 LOD 단계의 첫 meshlet.
    std::vector<uint32_t> drawMeshletData;
    std::vector<VkDrawIndexedIndirectCommand> shadowDrawData;
    uint64_t lastTopologyRevision = 0;
    uint32_t pathSampleCount = 0;

    bool hzbNeedsClear = true;
    bool resizeRequested = false;
    bool vsync = true;
};

} // namespace gfx
