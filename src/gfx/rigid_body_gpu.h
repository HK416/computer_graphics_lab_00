#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <glm/gtc/quaternion.hpp>
#include <glm/vec4.hpp>
#include <vulkan/vulkan.h>

#include "gfx/resources.h"
#include "physics/rigid_body.h"
#include "scene/scene.h"

namespace gfx {

struct Context;
class BindlessTextures;

// Jacobi 반복 수. 접촉을 하나씩 순서대로 푸는 CPU 순차 임펄스(8회)보다 수렴이 느려 넉넉히 돈다.
inline constexpr uint32_t RIGID_SOLVER_ITERATIONS = 16;
// 위치 보정 반복 수. CPU 와 같다. 한 번만 돌면 쌓인 물체가 서서히 가라앉는다. CPU 는 이미 밀어낸
// 만큼을 빼고 다시 밀지만 여기서는 매번 접촉을 다시 만든다.
inline constexpr uint32_t RIGID_POSITION_ITERATIONS = 8;
inline constexpr uint32_t RIGID_GROUP_SIZE = 64;
// 되읽기 버퍼 개수. FRAMES_IN_FLIGHT 보다 하나 많아야 이번 프레임이 덮어쓰기 전에 다 읽은 것을
// 고를 수 있다. 어긋나면 record 의 static_assert 가 잡는다.
inline constexpr uint32_t RIGID_READBACK_SLOTS = 3;
// 광역 격자 버킷 용량. shaders/rigid_common.glsl 의 RIGID_CELL_CAPACITY 와 같아야 한다.
inline constexpr uint32_t RIGID_CELL_CAPACITY = 64;

// shaders/rigid_common.glsl 의 RigidBody 와 배치가 같아야 한다(scalar).
struct GpuRigidBody {
    glm::vec4 position{0.0F};          // xyz 위치, w 질량 역수
    glm::vec4 rotation{0, 0, 0, 1.0F}; // 쿼터니언 (x, y, z, w)
    glm::vec4 velocity{0.0F};          // xyz 속도, w 콜라이더 반지름
    glm::vec4 angularVelocity{0.0F};   // xyz 각속도, w 경계 반지름
    glm::vec4 preVelocity{0.0F};
    glm::vec4 preAngularVelocity{0.0F};
    glm::vec4 inverseInertia{0.0F}; // xyz 지역 축 관성 역수, w 반발 계수
    glm::vec4 halfExtents{0.0F};    // xyz 상자 반쪽 크기(원기둥·캡슐은 y 반높이), w 마찰 계수
    uint32_t shape = 0;
    uint32_t flags = 0;
    // 메쉬 콜라이더의 세계 공간 삼각형 구간(triangles 버퍼 기준).
    uint32_t triangleOffset = 0;
    uint32_t triangleCount = 0;
};
static_assert(sizeof(GpuRigidBody) == 144, "강체 배치가 셰이더와 어긋난다");

// shaders/rigid_common.glsl 의 RigidPushConstants 와 배치가 같아야 한다(scalar).
struct RigidPushConstants {
    VkDeviceAddress bodiesIn = 0;
    VkDeviceAddress bodiesOut = 0;
    VkDeviceAddress triangles = 0;
    // 광역 격자: 버킷 개수·번호 버퍼와 평면 번호 목록.
    VkDeviceAddress cellCounts = 0;
    VkDeviceAddress cellBodies = 0;
    VkDeviceAddress planes = 0;
    uint32_t bodyCount = 0;
    float dt = 0.0F;
    float gravity = -9.81F;
    float positionCorrection = 0.2F;
    float penetrationSlop = 0.005F;
    float restitutionThreshold = 1.0F;
    uint32_t cellCount = 0;
    uint32_t planeCount = 0;
    float cellSize = 1.0F;
};
static_assert(sizeof(RigidPushConstants) == 88, "강체 푸시 상수 배치가 셰이더와 어긋난다");

// 강체 GPU 솔버. CPU 솔버와 같은 함수(physics::collectRigidBodies)로 세계 상태를 펴 컴퓨트로 풀고,
// 결과를 되읽어 오브젝트 변환에 되쓴다.
//
// CPU 와 다른 점 둘:
//
// 1. 접촉 해결이 순차 임펄스가 아니라 **Jacobi** 다. 물체마다 이웃 전부와의 접촉을 «같은 속도로»
//    풀어 한 번에 더한다. 순서 의존이 없어 나눠 풀 수 있는 대신 수렴이 느려 쌓인 물체가 조금 더
//    물렁하다.
// 2. 상태가 **GPU 에 머문다.** 매 프레임 장면에서 다시 올리면 되읽기 지연만큼 뒤로 감기기 때문이다.
//    장면 쪽 값이 우리가 되쓴 것과 달라졌을 때만 (부품 추가/삭제, 편집기 조작, 재생 시작) 다시
//    올린다.
//
// 되읽기는 몇 프레임 늦다. 기다리면 GPU 와 CPU 가 번갈아 멈춘다.
class RigidBodySimulator {
public:
    RigidBodySimulator(Context& context, BindlessTextures& bindless);
    ~RigidBodySimulator();
    RigidBodySimulator(const RigidBodySimulator&) = delete;
    RigidBodySimulator& operator=(const RigidBodySimulator&) = delete;

    // 컴퓨트 파이프라인을 다 만들었는지. 거짓이면 GPU 백엔드를 골라도 아무 것도 돌지 않는다.
    bool available() const { return ready; }
    // 지난 프레임에 GPU 로 푼 강체 수.
    uint32_t bodyCount() const { return static_cast<uint32_t>(bodies.size()); }

    // GPU 에 남은 상태를 버리고 다음 prepare 에서 장면 값을 다시 올리게 한다. 아직 읽지 않은 되읽기도
    // 함께 버린다. 그러지 않으면 정지하면서 되돌린 스냅샷을 몇 프레임 뒤 옛 결과가 덮어쓴다.
    void invalidate();

    // 완료된 결과가 있으면 장면에 되쓴다. `completedFrames` 보다 작은 번호의 프레임은 GPU 가 끝냈다는
    // 뜻이다. scene.refresh 앞에 불러야 세계 변환 캐시가 이 값으로 다시 만들어진다.
    bool applyReadback(scene::Scene& scene, uint64_t completedFrames);
    // 장면에서 GPU 강체를 모은다. steps 는 이번 프레임에 풀 고정 간격 스텝 수이고 0 이면 멈춘 것이다.
    void prepare(const scene::Scene& scene, uint32_t steps, float stepSeconds);
    // 컴퓨트를 기록하고 결과를 되읽기 버퍼로 복사한다.
    void record(VkCommandBuffer commandBuffer, uint64_t frameIndex);

private:
    void createPipelines();
    void reserveBuffers(uint32_t count);
    void reserveTriangles(uint32_t count);
    // 광역 격자 버퍼. 물체 수로 정한 셀 수가 커질 때만 다시 잡는다.
    void reserveGrid(uint32_t cellCount);
    // 이번에 모은 상태를 GPU 로 보낼 모습으로 편다.
    void buildUpload();
    // 편집기가 손댄 것이 있는지. 적분이 바꾸는 값(위치·회전·속도)은 오차를 봐주고 나머지는 그대로
    // 견준다. 인스펙터의 모양·질량·반발 같은 값도 여기서 걸러야 GPU 에 반영된다.
    bool sceneEdited() const;

    Context& context;
    BindlessTextures& bindless;
    bool ready = false;

    // 이번 프레임에 푸는 강체의 세계 상태. 되쓰기가 오브젝트 번호를 여기서 읽는다.
    std::vector<physics::RigidBodyState> bodies;
    // 메쉬 콜라이더의 세계 공간 삼각형. 강체와 함께 올린다(메쉬는 운동학이라 편집기가 손댈 때만 바뀐다).
    std::vector<physics::Triangle> triangles;
    std::vector<GpuRigidBody> upload;
    // GPU 가 들고 있다고 믿는 상태. 되읽기 결과와 마지막 업로드를 합친 것이다. 이번에 모은 것과 견줘
    // 편집기가 손댄 것을 알아챈다.
    std::vector<GpuRigidBody> resident;
    // 되읽기가 어느 장면·부품 구성의 것인지. 달라졌으면 그 결과는 남의 것이다.
    const scene::Scene* readbackScene = nullptr;
    uint64_t readbackComponents = 0;

    bool uploadPending = false;
    uint32_t stepCount = 0;
    float stepSeconds = 0.0F;
    uint32_t capacity = 0;
    uint32_t triangleCapacity = 0;

    // 읽는 쪽과 쓰는 쪽을 번갈아 쓴다.
    std::array<Buffer, 2> bodyBuffers;
    // 업로드 스테이징은 프레임마다 하나다. 하나로 두면 아직 복사 중인 지난 프레임의 것을 덮어쓴다.
    std::array<Buffer, RIGID_READBACK_SLOTS> stagings;
    std::array<Buffer, RIGID_READBACK_SLOTS> readbacks;
    // 메쉬 콜라이더 삼각형. 장치 버퍼 하나와 프레임마다의 스테이징.
    Buffer triangleBuffer;
    std::array<Buffer, RIGID_READBACK_SLOTS> triangleStagings;
    // 광역 격자. 셀 크기는 2·최대 경계 반지름(CPU collectPairs 와 같은 규칙), 셀 수는 물체 수의 두 배를 2 의
    // 거듭제곱으로 올린 것. 평면 번호는 프레임마다 호스트가 쓰는 작은 버퍼로 넘긴다.
    Buffer cellCountBuffer;
    Buffer cellBodyBuffer;
    std::array<Buffer, RIGID_READBACK_SLOTS> planeBuffers;
    uint32_t gridCellCount = 0;
    uint32_t gridCapacity = 0;
    float gridCellSize = 1.0F;
    std::vector<uint32_t> planeIndices;
    // 슬롯마다 «어느 프레임이 채웠는지»와 «몇 개인지». 프레임이 끝난 슬롯만 읽는다.
    std::array<uint64_t, RIGID_READBACK_SLOTS> readbackFrame{};
    std::array<uint32_t, RIGID_READBACK_SLOTS> readbackCount{};
    // 마지막으로 장면에 적용한 결과가 어느 프레임 것인지. 이보다 오래된 것은 버린다. 시뮬레이션은
    // 앞으로만 가므로 오래된 결과를 뒤늦게 적용하면 화면에서 공이 뒤로 되감겨 덜덜거린다.
    uint64_t appliedFrame = 0;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline integratePipeline = VK_NULL_HANDLE;
    VkPipeline solvePipeline = VK_NULL_HANDLE;
    VkPipeline finishPipeline = VK_NULL_HANDLE;
    VkPipeline gridPipeline = VK_NULL_HANDLE;
};

} // namespace gfx
