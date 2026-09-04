#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <vulkan/vulkan.h>

#include "gfx/resources.h"
#include "scene/scene.h"

namespace gfx {

struct Context;
class BindlessTextures;

inline constexpr uint32_t FLUID_MAX_PARTICLES = 32768;
inline constexpr uint32_t FLUID_CELL_CAPACITY = 32;
inline constexpr uint32_t FLUID_MAX_COLLIDERS = 8;
inline constexpr uint32_t FLUID_GROUP_SIZE = 128;
inline constexpr uint32_t FLUID_FRAMES = 2;

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

inline constexpr uint32_t FLUID_FLAG_RESET_HISTORY = 1U;
inline constexpr uint32_t FLUID_FLAG_WRITE_TLAS = 2U;

// GPU 컴퓨트 SPH. 입자 상태는 GPU 에만 있고 장면의 Fluid 부품은 설정만 갖는다. 서브스텝마다 해시 격자를
// 원자 카운터로 다시 짓고(잠금 없음), 밀도·압력, 힘·적분·충돌을 돈 뒤 프레임 끝에 입자마다 그리기
// 인스턴스를 렌더러의 인스턴스 버퍼에 직접 쓴다. 래스터와 경로 추적이 같은 인스턴스를 본다.
class FluidSimulator {
public:
    FluidSimulator(Context& context, BindlessTextures& bindless);
    ~FluidSimulator();
    FluidSimulator(const FluidSimulator&) = delete;
    FluidSimulator& operator=(const FluidSimulator&) = delete;

    // 장면의 유체 부품에 GPU 상태를 맞춘다. 설정이나 오브젝트 변환이 바뀌었거나 재생을 시작·정지했으면
    // 다시 뿌리도록 표시한다. 돌려주는 값은 이번 프레임에 입자가 움직이거나 다시 뿌려지는지다.
    bool prepare(const scene::Scene& scene, bool sceneSwitched);
    // 유체 index 의 입자 수. 그릴 수 없으면 0.
    uint32_t particleCount(uint32_t index) const;
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
                bool resetHistory);

private:
    struct State {
        std::array<Buffer, 2> positions;
        std::array<Buffer, 2> velocities;
        Buffer previousRendered;
        Buffer cellCounts;
        Buffer cellParticles;
        std::array<Buffer, FLUID_FRAMES> params;
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
    };

    void createPipelines();
    void destroyState(State& state);
    void ensureCapacity(State& state, uint32_t count);
    void fillParams(GpuFluidParams& params, const State& state, const scene::Scene& scene) const;

    Context& context;
    BindlessTextures& bindless;
    std::vector<State> states;
    const scene::Scene* lastScene = nullptr;
    // 마지막으로 본 부품 배치 번호. 부품을 붙이거나 떼면 유체 첨자가 밀려 states 의 슬롯이 다른
    // 유체에게 넘어가므로, 값이 달라진 프레임에는 전부 다시 뿌린다.
    uint64_t lastComponentRevision = 0;
    bool wasSimulating = false;
    uint32_t particleLimit = FLUID_MAX_PARTICLES;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline emitPipeline = VK_NULL_HANDLE;
    VkPipeline gridPipeline = VK_NULL_HANDLE;
    VkPipeline densityPipeline = VK_NULL_HANDLE;
    VkPipeline integratePipeline = VK_NULL_HANDLE;
    VkPipeline instancesPipeline = VK_NULL_HANDLE;
};

} // namespace gfx
