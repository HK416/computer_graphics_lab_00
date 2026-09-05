#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <vulkan/vulkan.h>

#include "gfx/resources.h"
#include "physics/fluid_sph.h"
#include "scene/scene.h"

namespace core {
class JobSystem;
} // namespace core

namespace gfx {

struct Context;
class BindlessTextures;

inline constexpr uint32_t FLUID_MAX_PARTICLES = 32768;
inline constexpr uint32_t FLUID_CELL_CAPACITY = 32;
inline constexpr uint32_t FLUID_MAX_COLLIDERS = 8;
// CPU 백엔드와 셰이더가 같은 값을 써야 격자 규칙이 갈리지 않는다. physics 쪽이 커지면 fillParams 가
// GPU 배열을 넘어 쓴다.
static_assert(FLUID_MAX_COLLIDERS == physics::FLUID_MAX_COLLIDERS, "콜라이더 상한이 CPU 백엔드와 어긋난다");
static_assert(FLUID_CELL_CAPACITY == physics::FLUID_CELL_CAPACITY, "격자 버킷 용량이 CPU 백엔드와 어긋난다");
inline constexpr uint32_t FLUID_GROUP_SIZE = 128;
inline constexpr uint32_t FLUID_FRAMES = 2;
// 물 표면 정점 상한. 한 유체가 이보다 많은 삼각형을 내면 거기서 잘린다. 16 바이트 × 이 값이 곧
// 정점 버퍼 크기다(3 MB).
inline constexpr uint32_t FLUID_MAX_SURFACE_VERTICES = 196608;
// 표면 격자 해상도의 상한. 축마다의 셀 수라 표본 수는 세제곱으로 는다.
inline constexpr uint32_t FLUID_MAX_SURFACE_RESOLUTION = 128;
// CPU 백엔드의 상한. 표본마다 입자 전부를 훑으므로 GPU 와 같은 해상도를 줄 수 없다.
inline constexpr uint32_t FLUID_MAX_CPU_SURFACE_RESOLUTION = 40;
inline constexpr uint32_t FLUID_SURFACE_GROUP_SIZE = 4;

// 아래 셋은 shaders/fluid_common.glsl 의 FluidCollider / FluidParams / FluidPushConstants 와 배치가 같아야
// 한다(scalar).
struct GpuFluidCollider {
    glm::vec4 data0{0.0F};
    glm::mat4 inverseWorld{1.0F};
    glm::mat4 world{1.0F};
    uint32_t type = 0;
    uint32_t pad0 = 0;
    uint32_t pad1 = 0;
    uint32_t pad2 = 0;
};

struct GpuFluidParams {
    glm::mat4 emitterWorld{1.0F};
    glm::vec4 emitterHalfExtents{0.0F}; // xyz 반쪽 크기, w 입자 간격
    glm::vec4 containerMin{0.0F};       // xyz, w 입자 반지름
    glm::vec4 containerMax{0.0F};       // xyz, w 벽 반발
    glm::vec4 gravity{0.0F};            // xyz, w 입자 질량
    float smoothingRadius = 0.1F;
    float restDensity = 1000.0F;
    float stiffness = 50.0F;
    float viscosity = 0.5F;
    glm::uvec4 lattice{1U}; // xyz 축별 방출 개수, w 셀 수(2 의 거듭제곱)
    uint32_t colliderCount = 0;
    uint32_t pad0 = 0;
    uint32_t pad1 = 0;
    uint32_t pad2 = 0;
    std::array<GpuFluidCollider, FLUID_MAX_COLLIDERS> colliders{};
};

struct FluidPushConstants {
    VkDeviceAddress params = 0;
    VkDeviceAddress positionsIn = 0;
    VkDeviceAddress positionsOut = 0;
    VkDeviceAddress velocitiesIn = 0;
    VkDeviceAddress velocitiesOut = 0;
    VkDeviceAddress cellCounts = 0;
    VkDeviceAddress cellParticles = 0;
    VkDeviceAddress previousRendered = 0;
    VkDeviceAddress instances = 0;
    VkDeviceAddress tlasInstances = 0;
    uint32_t particleCount = 0;
    uint32_t instanceBase = 0;
    uint32_t tlasBase = 0;
    uint32_t flags = 0;
    uint32_t sphereMesh = 0;
    float dt = 0.0F;
    uint32_t blasLow = 0;
    uint32_t blasHigh = 0;
};
static_assert(sizeof(FluidPushConstants) == 112, "유체 푸시 상수 배치가 셰이더와 어긋난다");

// shaders/fluid_surface_common.glsl 의 FluidSurfacePushConstants 와 배치가 같아야 한다(scalar).
// 시뮬레이션 푸시 상수와 나눈 것은 한 블록에 다 넣으면 규격이 보장하는 128 바이트를 넘기 때문이다.
struct FluidSurfacePushConstants {
    VkDeviceAddress params = 0;
    VkDeviceAddress positionsIn = 0;
    VkDeviceAddress cellCounts = 0;
    VkDeviceAddress cellParticles = 0;
    VkDeviceAddress surfaceField = 0;
    VkDeviceAddress surfaceVertices = 0;
    VkDeviceAddress surfaceCounter = 0;
    // 하위 가속 구조 간접 구축이 읽는 VkAccelerationStructureBuildRangeInfoKHR. [0] 이 삼각형 수.
    VkDeviceAddress surfaceRange = 0;
    uint32_t surfaceResolution = 0;
    uint32_t surfaceCapacity = 0;
    float surfaceIso = 0.5F;
    uint32_t pad0 = 0;
};
static_assert(sizeof(FluidSurfacePushConstants) == 80, "유체 표면 푸시 상수 배치가 셰이더와 어긋난다");

inline constexpr uint32_t FLUID_FLAG_RESET_HISTORY = 1U;
inline constexpr uint32_t FLUID_FLAG_WRITE_TLAS = 2U;
// 입자 대신 물 표면 하나를 TLAS 인스턴스로 쓴다. 유체 번호는 flags 의 상위 비트(FLUID_FLAG_INDEX_SHIFT)에 싣는다.
inline constexpr uint32_t FLUID_FLAG_SURFACE_TLAS = 4U;
inline constexpr uint32_t FLUID_FLAG_INDEX_SHIFT = 8U;
// 물 표면 TLAS 인스턴스의 customIndex 표식(비트 23). 적중 셰이더가 인스턴스 배열 대신 FluidSurfaceInfo 를 찾는다.
inline constexpr uint32_t FLUID_SURFACE_CUSTOM_INDEX = 0x800000U;
// 물 표면 인스턴스의 광선 마스크. 그림자 광선(마스크 0xFE)은 물을 지나가고 카메라·바운스 광선(0xFF)만 맞힌다.
inline constexpr uint32_t FLUID_SURFACE_RAY_MASK = 0x01U;

// shaders/fluid_types.glsl 의 FluidSurfaceInfo 와 배치가 같아야 한다(scalar). 경로 추적 적중 셰이더가 유체
// 번호로 찾아 물 표면 정점과 재질을 읽는다.
struct GpuFluidSurfaceInfo {
    VkDeviceAddress vertices = 0;
    // rgb 물 색, w 표면 거칠기.
    glm::vec4 waterColor{0.0F};
    // rgb 흡수 계수(1/m), w 두께 배율.
    glm::vec4 absorption{0.0F};
};
static_assert(sizeof(GpuFluidSurfaceInfo) == 40, "물 표면 정보 배치가 셰이더와 어긋난다");

// GPU 컴퓨트 SPH. 입자 상태는 GPU 에만 있고 장면의 Fluid 부품은 설정만 갖는다. 서브스텝마다 해시 격자를
// 원자 카운터로 다시 짓고(잠금 없음), 밀도·압력, 힘·적분·충돌을 돈 뒤 프레임 끝에 입자마다 그리기
// 인스턴스를 렌더러의 인스턴스 버퍼에 직접 쓴다. 래스터와 경로 추적이 같은 인스턴스를 본다.
class FluidSimulator {
public:
    FluidSimulator(Context& context, BindlessTextures& bindless, core::JobSystem& jobs);
    ~FluidSimulator();
    FluidSimulator(const FluidSimulator&) = delete;
    FluidSimulator& operator=(const FluidSimulator&) = delete;

    // 장면의 유체 부품에 GPU 상태를 맞춘다. 설정이나 오브젝트 변환이 바뀌었거나 재생을 시작·정지했으면
    // 다시 뿌리도록 표시한다. 돌려주는 값은 이번 프레임에 입자가 움직이거나 다시 뿌려지는지다.
    bool prepare(const scene::Scene& scene, bool sceneSwitched);
    // 유체 index 의 입자 수. 그릴 수 없으면 0.
    uint32_t particleCount(uint32_t index) const;
    // 유체 index 를 CPU 에서 푸는지. 편집기가 «지금 도는 백엔드» 를 보여 주는 데 쓴다.
    bool onCpu(uint32_t index) const;
    // 컴퓨트 파이프라인을 다 만들었는지. 거짓이면 GPU 백엔드를 고를 수 없다.
    bool gpuAvailable() const { return gpuReady; }
    // 표면 컴퓨트를 만들었는지. 거짓이면 GPU 백엔드에서는 표면을 그릴 수 없다.
    bool surfaceAvailable() const { return surfaceReady; }
    // CPU 백엔드가 인스턴스를 직접 쓴다. 명령 기록 전에, 프레임 버퍼가 매핑된 채로 부른다.
    // instances 와 tlasInstances 는 매핑 포인터이며 tlasInstances 가 널이면 쓰지 않는다.
    //
    // ponytail: 시뮬레이션이 여기서 동기로 돌아 주 스레드를 막는다. 기본 입자 수(8192)에 12 스레드로
    // 프레임당 9 ms 쯤이고 그동안 GPU 는 논다. 겹치려면 지난 프레임 결과를 그리고 이번 프레임 계산을
    // 백그라운드로 돌려야 한다(한 프레임 늦은 물이 된다).
    void writeCpuInstances(uint32_t index,
                           const scene::Scene& scene,
                           float deltaSeconds,
                           void* instances,
                           uint32_t instanceBase,
                           void* tlasInstances,
                           uint32_t tlasBase,
                           VkDeviceAddress sphereBlas,
                           uint32_t sphereMesh,
                           VkDeviceAddress surfaceBlas,
                           bool resetHistory);
    // 유체 index 를 표면으로 그리는지. 부품 설정이 표면이고 정점 버퍼가 서 있어야 참이다.
    bool surfaceActive(uint32_t index) const;
    // 표면 정점 버퍼의 주소와 간접 그리기 인자 버퍼. surfaceActive 가 참일 때만 의미가 있다.
    // 프레임마다 한 벌이라 지난 프레임이 아직 그리고 있는 것을 덮어쓰지 않는다.
    VkDeviceAddress surfaceVertexAddress(uint32_t frameSlot, uint32_t index) const;
    VkBuffer surfaceDrawBuffer(uint32_t frameSlot, uint32_t index) const;
    // 간접 구축용 범위 구조체(삼각형 수)의 주소. surfaceActive 가 참일 때만 의미가 있다.
    VkDeviceAddress surfaceRangeAddress(uint32_t frameSlot, uint32_t index) const;

    uint32_t fluidCount() const { return static_cast<uint32_t>(states.size()); }
    // 부품이 더 달라고 해도 이보다 많이 뿌리지 않는다. 하드웨어 프로파일이 정한다.
    //
    // ponytail: 상한이 «줄어드는» 프레임에는 다시 뿌리지 않아 앞쪽 입자만 남고 잘린다. 버퍼는
    // 늘어날 때만 다시 잡기 때문이다. 기동 시 한 번만 정해지는 지금은 드러나지 않는다.
    void setParticleLimit(uint32_t limit) { particleLimit = std::min(limit, FLUID_MAX_PARTICLES); }
    uint32_t totalParticles() const;
    // 용기의 경계 구. 그림자 시점 컬링이 쓴다.
    glm::vec4 bounds(uint32_t index) const;

    // 유체 하나를 진행하고 인스턴스를 쓴다. instanceBase 는 렌더러 인스턴스 버퍼에서 이 유체의 첫 슬롯,
    // tlasInstances 가 0 이 아니면 tlasBase 부터 상위 가속 구조 인스턴스도 쓴다.
    void record(VkCommandBuffer commandBuffer,
                uint32_t frameSlot,
                uint32_t index,
                const scene::Scene& scene,
                float deltaSeconds,
                VkDeviceAddress instances,
                uint32_t instanceBase,
                VkDeviceAddress tlasInstances,
                uint32_t tlasBase,
                VkDeviceAddress sphereBlas,
                uint32_t sphereMesh,
                // 0 이 아니면 입자 대신 물 표면 하위 구조 하나를 TLAS 인스턴스로 쓴다.
                VkDeviceAddress surfaceBlas,
                bool resetHistory);

    // 유체 index 의 물 표면을 만든다. GPU 백엔드는 컴퓨트 두 패스, CPU 백엔드는 작업 큐로 같은 표를
    // 돌려 같은 정점 버퍼를 채운다. 시뮬레이션(record/writeCpuInstances) 뒤에 부른다.
    void recordSurface(VkCommandBuffer commandBuffer, uint32_t frameSlot, uint32_t index, const scene::Scene& scene);
    void buildCpuSurface(uint32_t frameSlot, uint32_t index, const scene::Scene& scene);

private:
    struct State {
        std::array<Buffer, 2> positions;
        std::array<Buffer, 2> velocities;
        Buffer previousRendered;
        Buffer cellCounts;
        Buffer cellParticles;
        std::array<Buffer, FLUID_FRAMES> params;
        // 물 표면. 장은 (해상도+1)³ 개의 float, 정점 버퍼는 마칭이 이어 붙이는 자리다.
        // drawArgs 는 VkDrawIndirectCommand 그대로라 마칭의 원자 카운터가 vertexCount 를 직접 쓴다.
        // 정점과 인자는 프레임마다 한 벌이다. 한 벌만 두면 지난 프레임이 그리는 중에 덮어쓴다.
        Buffer surfaceField;
        std::array<Buffer, FLUID_FRAMES> surfaceVertices;
        std::array<Buffer, FLUID_FRAMES> surfaceDrawArgs;
        // 하위 가속 구조 간접 구축이 읽는 삼각형 수. 마칭이 drawArgs 와 함께 채운다.
        std::array<Buffer, FLUID_FRAMES> surfaceRanges;
        uint32_t surfaceResolution = 0;
        uint32_t surfaceCapacity = 0;
        bool surfaceReady = false;
        std::vector<float> cpuField;
        uint32_t capacity = 0;
        uint32_t cellCount = 0;
        // 이번 스텝이 읽는 반쪽.
        uint32_t current = 0;
        scene::Fluid lastSettings;
        glm::mat4 lastWorld{0.0F};
        bool needsEmit = true;
        // 이번 프레임 prepare 가 정한 것.
        uint32_t objectIndex = 0;
        uint32_t count = 0;
        // CPU 백엔드면 참이다. GPU 버퍼는 만들지 않고 solver 가 상태를 든다.
        bool cpu = false;
        physics::FluidSolver solver;
    };

    void createPipelines();
    void destroyState(State& state);
    void ensureCapacity(State& state, uint32_t count);
    // 표면 버퍼를 부품 설정에 맞춰 잡는다. 표시가 입자면 있던 것을 놓아준다.
    void ensureSurface(State& state, const scene::Fluid& settings);
    // 두 백엔드가 함께 쓰는 시뮬레이션 상수.
    physics::FluidParams deriveParams(const State& state, const scene::Scene& scene) const;
    void fillParams(GpuFluidParams& params, const State& state, const scene::Scene& scene) const;

    Context& context;
    BindlessTextures& bindless;
    core::JobSystem& jobs;
    std::vector<State> states;
    const scene::Scene* lastScene = nullptr;
    // 마지막으로 본 부품 배치 번호. 부품을 붙이거나 떼면 유체 첨자가 밀려 states 의 슬롯이 다른
    // 유체에게 넘어가므로, 값이 달라진 프레임에는 전부 다시 뿌린다.
    uint64_t lastComponentRevision = 0;
    bool wasSimulating = false;
    uint32_t particleLimit = FLUID_MAX_PARTICLES;
    // 컴퓨트 파이프라인을 다 만들었는지. 하나라도 없으면 모든 유체가 CPU 로 내려간다.
    bool gpuReady = false;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline emitPipeline = VK_NULL_HANDLE;
    VkPipeline gridPipeline = VK_NULL_HANDLE;
    VkPipeline densityPipeline = VK_NULL_HANDLE;
    VkPipeline integratePipeline = VK_NULL_HANDLE;
    VkPipeline instancesPipeline = VK_NULL_HANDLE;
    // 표면 패스는 푸시 상수가 달라 파이프라인 배치를 따로 쓴다.
    VkPipelineLayout surfaceLayout = VK_NULL_HANDLE;
    VkPipeline fieldPipeline = VK_NULL_HANDLE;
    VkPipeline marchingPipeline = VK_NULL_HANDLE;
    bool surfaceReady = false;
};

} // namespace gfx
