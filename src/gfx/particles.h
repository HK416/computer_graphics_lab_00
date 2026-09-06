#pragma once

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

inline constexpr uint32_t PARTICLE_MAX_PARTICLES = 65536;
inline constexpr uint32_t PARTICLE_GROUP_SIZE = 128;
inline constexpr uint32_t PARTICLE_FRAMES = 2;

// 아래 셋은 shaders/particle_common.glsl 의 Particle / ParticleParams / ParticlePushConstants 와 배치가 같아야
// 한다(scalar).
struct GpuParticle {
    glm::vec3 position{0.0F};
    // 살아온 시간. lifetime 이상이면 죽은 것이다. 버퍼를 0 으로 채우면 lifetime 0 이라 전부 죽어 있다.
    float age = 0.0F;
    glm::vec3 velocity{0.0F};
    float lifetime = 0.0F;
};
static_assert(sizeof(GpuParticle) == 32, "입자 배치가 셰이더와 어긋난다");

struct GpuParticleParams {
    glm::mat4 emitterWorld{1.0F};
    // rgb 색, w 알파.
    glm::vec4 color{1.0F};
    // rgb 발광, w 반발 계수.
    glm::vec4 emissive{0.0F};
    // x 시작 지름, y 끝 지름, z 초기 속력, w 원뿔 반각의 코사인.
    glm::vec4 sizeSpeedSpread{0.0F};
    // xyz 중력, w 항력.
    glm::vec4 gravityDrag{0.0F};
    uint32_t particleCount = 0;
    // 이번 프레임에 뿌릴 링 버퍼 구간. spawnFirst 부터 spawnCount 개(끝에서 앞으로 감는다).
    uint32_t spawnFirst = 0;
    uint32_t spawnCount = 0;
    uint32_t frameIndex = 0;
    float dt = 0.0F;
    float lifetime = 0.0F;
    uint32_t flags = 0;
    uint32_t pad0 = 0;
    // 충돌 법선을 구할 장면 버퍼. 히트 삼각형의 정점 셋을 읽는다.
    VkDeviceAddress vertices = 0;
    VkDeviceAddress skinnedVertices = 0;
    VkDeviceAddress indices = 0;
    VkDeviceAddress meshes = 0;
    VkDeviceAddress lods = 0;
    VkDeviceAddress instances = 0;
};
static_assert(sizeof(GpuParticleParams) == 208, "입자 설정 배치가 셰이더와 어긋난다");

// 컴퓨트와 스프라이트 정점·프래그먼트가 같은 블록을 쓴다.
struct ParticlePushConstants {
    VkDeviceAddress particles = 0;
    VkDeviceAddress params = 0;
    VkDeviceAddress camera = 0;
    // 장면 깊이의 bindless 샘플 슬롯. 소프트 깊이 감쇠에 쓴다.
    uint32_t depthTexture = 0;
    uint32_t particleCount = 0;
};
static_assert(sizeof(ParticlePushConstants) == 32, "입자 푸시 상수 배치가 셰이더와 어긋난다");

// 상위 가속 구조에 광선 질의로 부딪힌다.
inline constexpr uint32_t PARTICLE_FLAG_COLLIDE = 1U;

// GPU 입자(불꽃·파편). 상태는 GPU 에만 있고 장면의 ParticleSystem 부품은 설정만 갖는다. 방출은 CPU 가 링
// 버퍼 구간으로 정해 원자 연산 없이 결정적이고, 진행 컴퓨트는 입자 하나를 스레드 하나가 제자리에서 갱신한다.
// 충돌은 진행 컴퓨트 안의 광선 질의(TLAS)다. 그리기는 렌더러의 스프라이트 패스가 맡는다.
class ParticleSimulator {
public:
    // accelerationLayout 이 널이 아니면 광선 질의 변종도 만든다.
    ParticleSimulator(Context& context, BindlessTextures& bindless, VkDescriptorSetLayout accelerationLayout);
    ~ParticleSimulator();
    ParticleSimulator(const ParticleSimulator&) = delete;
    ParticleSimulator& operator=(const ParticleSimulator&) = delete;

    // 장면의 입자 부품에 GPU 상태를 맞추고 이번 프레임의 방출 구간을 정한다. 설정·오브젝트 변환이 바뀌었거나
    // 재생을 시작·정지했으면 다시 뿌리도록 표시한다. 돌려주는 값은 그릴 입자가 있는지다.
    bool prepare(const scene::Scene& scene, bool sceneSwitched, float deltaSeconds);

    struct SceneBuffers {
        VkDeviceAddress vertices = 0;
        VkDeviceAddress skinnedVertices = 0;
        VkDeviceAddress indices = 0;
        VkDeviceAddress meshes = 0;
        VkDeviceAddress lods = 0;
        VkDeviceAddress instances = 0;
    };
    // 시스템 index 를 한 프레임 진행한다. rayQuery 가 참이면 집합 1 에 accelerationSet 을 묶고 충돌한다.
    void record(VkCommandBuffer commandBuffer,
                uint32_t frameSlot,
                uint32_t index,
                const scene::Scene& scene,
                uint64_t frameIndex,
                bool rayQuery,
                VkDescriptorSet accelerationSet,
                const SceneBuffers& buffers);

    uint32_t systemCount() const { return static_cast<uint32_t>(states.size()); }
    // 시스템 index 의 입자 수. 그릴 수 없으면 0.
    uint32_t count(uint32_t index) const;
    // 충돌을 켠 시스템이 하나라도 있는지. 있으면 렌더러가 가속 구조를 세운다.
    bool wantsCollision() const;
    VkDeviceAddress particleAddress(uint32_t index) const;
    VkDeviceAddress paramsAddress(uint32_t frameSlot, uint32_t index) const;
    // 컴퓨트 파이프라인을 만들었는지. 거짓이면 입자를 돌리지 않는다.
    bool gpuAvailable() const { return pipeline != VK_NULL_HANDLE; }
    // 광선 질의 변종을 만들었는지.
    bool collisionAvailable() const { return rayQueryPipeline != VK_NULL_HANDLE; }

private:
    struct State {
        Buffer particles;
        std::array<Buffer, PARTICLE_FRAMES> params;
        uint32_t capacity = 0;
        // 이번 프레임 prepare 가 정한 것.
        uint32_t objectIndex = 0;
        uint32_t count = 0;
        uint32_t spawnFirst = 0;
        uint32_t spawnCount = 0;
        float dt = 0.0F;
        // 링 버퍼의 다음 방출 자리와 방출 누산기(개 단위의 소수부).
        uint32_t cursor = 0;
        float emitAccumulator = 0.0F;
        scene::ParticleSystem lastSettings;
        glm::mat4 lastWorld{0.0F};
        bool needsReset = true;
    };

    void createPipelines(VkDescriptorSetLayout accelerationLayout);
    void destroyState(State& state);
    void ensureCapacity(State& state, uint32_t count);

    Context& context;
    BindlessTextures& bindless;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout rayQueryLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipeline rayQueryPipeline = VK_NULL_HANDLE;
    std::vector<State> states;
    uint64_t lastSceneId = 0;
    uint64_t lastComponentRevision = UINT64_MAX;
    bool wasSimulating = false;
};

} // namespace gfx
