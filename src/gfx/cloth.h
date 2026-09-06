#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <vulkan/vulkan.h>

#include "gfx/fluid.h"
#include "gfx/resources.h"
#include "physics/cloth.h"
#include "scene/scene.h"

namespace core {
class JobSystem;
} // namespace core

namespace gfx {

struct Context;
class BindlessTextures;
class GeometryStore;

inline constexpr uint32_t CLOTH_GROUP_SIZE = 128;
inline constexpr uint32_t CLOTH_FRAMES = 2;
// 천 인스턴스는 상위 가속 구조에서 이 비트를 내린 마스크(0xFD)로 올라간다. 천의 메쉬 충돌 광선은 이 비트만으로
// 쏘아(0x02) 자기 자신과 다른 천은 지나가고 나머지 인스턴스(0xFF)만 맞힌다. 그림자(0xFE)·카메라(0xFF) 광선은
// 천을 그대로 맞힌다.
inline constexpr uint32_t CLOTH_SELF_RAY_MASK = 0x02U;

// 아래 둘은 shaders/cloth_common.glsl 의 ClothParams / ClothPushConstants 와 배치가 같아야 한다(scalar).
struct GpuClothParams {
    // xyz 중력 + 바람, w 감쇠.
    glm::vec4 accelerationDamping{0.0F};
    // x 두께, y 마찰, z 정지 시간 간격(서브스텝), w 예약.
    glm::vec4 thicknessFriction{0.0F};
    uint32_t vertexCount = 0;
    uint32_t constraintCount = 0;
    uint32_t resolution = 0;
    uint32_t colliderCount = 0;
    std::array<GpuFluidCollider, FLUID_MAX_COLLIDERS> colliders{};
};

struct ClothPushConstants {
    VkDeviceAddress params = 0;
    VkDeviceAddress positions = 0;
    VkDeviceAddress predicted = 0;
    VkDeviceAddress velocities = 0;
    VkDeviceAddress constraints = 0;
    VkDeviceAddress lambdas = 0;
    VkDeviceAddress rest = 0;
    VkDeviceAddress info = 0;
    VkDeviceAddress grid = 0;
    // 렌더러의 변형 정점 버퍼. write 패스가 destinationOffset 부터 asset::Vertex 로 쓴다.
    VkDeviceAddress vertices = 0;
    uint32_t destinationOffset = 0;
    uint32_t vertexCount = 0;
    // 이번 디스패치가 맡는 제약 구간(색 하나).
    uint32_t constraintFirst = 0;
    uint32_t constraintCount = 0;
    float dt = 0.0F;
    uint32_t flags = 0;
};
static_assert(sizeof(ClothPushConstants) == 104, "천 푸시 상수 배치가 셰이더와 어긋난다");

inline constexpr uint32_t CLOTH_FLAG_RESET = 1U;
inline constexpr uint32_t CLOTH_FLAG_RAY_QUERY = 2U;

// XPBD 천. 장면의 천 부품마다 상태를 들고 스킨 패스 안에서 변형 정점 버퍼에 결과를 쓴다. GPU 백엔드는 컴퓨트
// 넷(예측·제약(색마다)·마무리·쓰기), CPU 백엔드는 physics::ClothSolver 가 prepare 에서 돌고 스테이징 복사로 같은
// 자리에 쓴다. 두 백엔드는 같은 알고리즘·같은 색 순서라 거동이 같다.
class ClothSimulator {
public:
    ClothSimulator(Context& context,
                   BindlessTextures& bindless,
                   core::JobSystem& jobs,
                   VkDescriptorSetLayout accelerationLayout);
    ~ClothSimulator();
    ClothSimulator(const ClothSimulator&) = delete;
    ClothSimulator& operator=(const ClothSimulator&) = delete;

    // 장면의 천 부품에 상태를 맞춘다. CPU 백엔드는 여기서 한 프레임을 진행한다. 돌려주는 값은 이번 프레임에 정점이
    // 바뀐 천(진행 또는 리셋)이 있는지다.
    bool prepare(const scene::Scene& scene, bool sceneSwitched, float deltaSeconds, const GeometryStore& geometry);
    // 천 index 가 돌 수 있는지(메쉬가 격자이고 콜라이더 메쉬 사본이 있다).
    bool active(uint32_t index) const;
    // 이번 프레임에 정점이 바뀌는지. 스킨 디스패치의 «바뀜» 플래그가 된다.
    bool stepped(uint32_t index) const;
    bool onCpu(uint32_t index) const;
    uint32_t clothCount() const { return static_cast<uint32_t>(states.size()); }
    // 현재 자세의 경계 구(월드). CPU 는 정확하고 GPU 는 정지 자세에서 넉넉히 잡은 근사다.
    glm::vec4 bounds(uint32_t index) const;
    bool gpuAvailable() const { return gpuReady; }
    // 광선 질의 변종을 만들었는지. 참이면 GPU 백엔드가 임의 메쉬와 부딪힌다.
    bool meshCollisionAvailable() const { return finishRayQueryPipeline != VK_NULL_HANDLE; }

    // 스킨 패스 안에서 부른다. destinationVertexOffset 은 변형 정점 버퍼(현재 반쪽) 안의 이 천 구간 시작이다.
    void record(VkCommandBuffer commandBuffer,
                uint32_t frameSlot,
                uint32_t index,
                VkBuffer skinnedVertexBuffer,
                VkDeviceAddress skinnedVertexAddress,
                uint32_t destinationVertexOffset,
                VkDescriptorSet accelerationSet,
                bool rayQuery);

private:
    struct State {
        bool cpu = false;
        physics::ClothTopology topology;
        physics::ClothSolver solver;
        std::vector<physics::ClothTriangle> triangles;
        // GPU 상태.
        Buffer positions;
        Buffer predicted;
        Buffer velocities;
        Buffer lambdas;
        // 리셋 때 채우는 정적 자료. 호스트에서 바로 쓴다.
        Buffer rest;
        Buffer info;
        Buffer grid;
        Buffer constraints;
        std::array<Buffer, CLOTH_FRAMES> params;
        // CPU 백엔드가 정점을 채워 복사하는 스테이징. 프레임마다 한 벌.
        std::array<Buffer, CLOTH_FRAMES> staging;
        uint32_t vertexCapacity = 0;
        uint32_t constraintCapacity = 0;
        uint32_t objectIndex = UINT32_MAX;
        uint32_t vertexCount = 0;
        bool needsReset = true;
        bool steppedThisFrame = false;
        bool valid = false;
        scene::Cloth lastSettings;
        glm::mat4 lastWorld{0.0F};
        physics::ClothParams frameParams;
    };

    void createPipelines(VkDescriptorSetLayout accelerationLayout);
    void destroyState(State& state);
    void ensureBuffers(State& state);
    void uploadStatic(State& state);
    void packCpuVertices(State& state, uint32_t frameSlot);

    Context& context;
    BindlessTextures& bindless;
    core::JobSystem& jobs;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout rayQueryLayout = VK_NULL_HANDLE;
    VkPipeline predictPipeline = VK_NULL_HANDLE;
    VkPipeline constraintPipeline = VK_NULL_HANDLE;
    VkPipeline finishPipeline = VK_NULL_HANDLE;
    VkPipeline finishRayQueryPipeline = VK_NULL_HANDLE;
    VkPipeline writePipeline = VK_NULL_HANDLE;
    bool gpuReady = false;
    std::vector<State> states;
    const scene::Scene* lastScene = nullptr;
    uint64_t lastComponentRevision = UINT64_MAX;
    bool wasSimulating = false;
};

} // namespace gfx
