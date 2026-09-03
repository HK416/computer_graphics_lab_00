#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "asset/model.h"
#include "scene/camera.h"

namespace core {
class JobSystem;
} // namespace core

namespace scene {

inline constexpr uint32_t INVALID_MESH = 0xFFFFFFFFU;

struct Transform {
    glm::vec3 position{0.0F};
    glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
    glm::vec3 scale{1.0F};

    glm::mat4 matrix() const;
    static Transform fromMatrix(const glm::mat4& matrix);

    bool operator==(const Transform&) const = default;
};

// 모델 하나의 스켈레톤과 재생 상태. Unity 의 Animator 처럼 오브젝트가 인덱스로 가리킨다.
struct Animator {
    std::string name;
    asset::Skeleton skeleton;
    // 이 스켈레톤이 온 모델의 번호. 장면을 저장하고 다시 읽을 때 스켈레톤을 되찾는 데 쓴다.
    int32_t model = -1;
    uint32_t clip = 0;
    float clipTime = 0.0F;
    bool playing = true;
    float speed = 1.0F;
    // 스킨마다의 조인트 행렬. Scene::update 가 채우고 렌더러가 그대로 올린다.
    std::vector<std::vector<glm::mat4>> jointMatrices;

private:
    friend struct Scene;
    std::vector<glm::mat4> nodeWorlds;
    // 마지막으로 실제 포즈를 만든 클립과 시각. 같으면 다시 만들지 않는다.
    uint32_t posedClip = 0xFFFFFFFFU;
    float posedTime = -1.0F;
};

enum class LightType : uint32_t {
    DIRECTIONAL = 0,
    POINT = 1,
    SPOT = 2,
    AREA = 3,
};

// 조명은 오브젝트에 달리는 부품이다. 위치와 방향은 오브젝트의 세계 변환에서 가져오며,
// 방향광과 스폿광은 -Z 를 앞으로 본다.
struct Light {
    LightType type = LightType::DIRECTIONAL;
    glm::vec3 color{1.0F};
    float intensity = 3.0F;
    // 점광, 스폿광, 영역광이 닿는 거리.
    float range = 20.0F;
    // 스폿광 원뿔의 안쪽과 바깥쪽 반각(도).
    float innerConeDegrees = 20.0F;
    float outerConeDegrees = 30.0F;
    // 영역광 직사각형의 가로세로 크기.
    glm::vec2 size{2.0F, 2.0F};
    bool castsShadow = true;

    bool operator==(const Light&) const = default;
};

// 오브젝트가 그리는 메쉬. 부품이라 오브젝트가 첨자로 가리킨다.
struct MeshRenderer {
    // GeometryStore 의 전역 메쉬 인덱스.
    uint32_t mesh = INVALID_MESH;
    // 이 메쉬가 쓰는 스켈레톤의 스킨 번호. 스킨이 없으면 -1.
    int32_t skin = -1;

    bool operator==(const MeshRenderer&) const = default;
};

enum class ColliderShape : uint32_t {
    SPHERE = 0,
    BOX = 1,
    // 오브젝트의 +Y 를 법선으로 하는 무한 평면. 바닥으로 쓴다.
    PLANE = 2,
};

// 강체 부품. 재생 중에 physics 가 세계 공간에서 적분해 오브젝트 변환에 되돌려 쓴다. 운동학 물체는
// 힘을 받지 않고 다른 물체만 밀어낸다. 평면은 늘 운동학으로 다룬다.
struct RigidBody {
    ColliderShape shape = ColliderShape::SPHERE;
    float mass = 1.0F;
    bool useGravity = true;
    bool kinematic = false;
    float restitution = 0.3F;
    float friction = 0.5F;
    // 구 반지름과 상자 반쪽 크기(오브젝트 지역 공간, 크기 변환 전).
    float radius = 0.5F;
    glm::vec3 halfExtents{0.5F};
    // 시뮬레이션 상태. 재생을 멈추면 스냅샷 복귀로 함께 되돌아간다.
    glm::vec3 velocity{0.0F};
    glm::vec3 angularVelocity{0.0F};

    bool operator==(const RigidBody&) const = default;
};

// 유체 부품. 입자 상태는 GPU 에만 있고 여기에는 방출·시뮬레이션 설정만 둔다. 방출 상자는 오브젝트
// 지역 공간이고 용기는 월드 공간이다. 값이 바뀌면 렌더러가 입자를 다시 뿌린다.
struct Fluid {
    glm::vec3 emitterHalfExtents{0.5F};
    uint32_t particleCount = 8192;
    float particleRadius = 0.025F;
    // SPH 커널 반지름. 입자 간격(2r)의 두 배쯤이 무난하다.
    float smoothingRadius = 0.1F;
    float restDensity = 1000.0F;
    // 압력 = 강성 × (밀도 − 기준 밀도). 크면 덜 눌리지만 시간 간격을 줄여야 한다.
    float stiffness = 50.0F;
    float viscosity = 0.5F;
    glm::vec3 containerMin{-1.0F, 0.0F, -1.0F};
    glm::vec3 containerMax{1.0F, 2.0F, 1.0F};
    glm::vec3 gravity{0.0F, -9.81F, 0.0F};

    bool operator==(const Fluid&) const = default;
};

struct Object {
    std::string name;
    // 부모 기준 지역 변환. 세계 변환은 Scene::worldMatrix 가 부모를 거슬러 올라가 만든다.
    Transform transform;
    // Scene::objects 인덱스. 뿌리면 -1.
    int32_t parent = -1;
    bool visible = true;
    // 붙어 있는 부품의 첨자. 없으면 -1. 종류마다 Scene 이 배열을 따로 들고 있다.
    int32_t meshRenderer = -1;
    int32_t animator = -1;
    int32_t light = -1;
    int32_t rigidBody = -1;
    int32_t fluid = -1;

    bool operator==(const Object&) const = default;
};

// 환경 맵(IBL)을 만들 재료. 태양 방향은 첫 방향광에서 받으므로 여기 두지 않는다.
struct Environment {
    // 참이고 경로가 비어 있지 않으면 HDR 파일을, 아니면 절차적 하늘을 쓴다.
    bool useHdr = false;
    std::filesystem::path hdrPath;
    glm::vec3 sunColor{1.0F, 0.95F, 0.85F};
    float sunIntensity = 1.0F;
    glm::vec3 zenithColor{0.18F, 0.32F, 0.62F};
    glm::vec3 horizonColor{0.62F, 0.72F, 0.86F};
    glm::vec3 groundColor{0.16F, 0.14F, 0.12F};
    float intensity = 1.0F;
    float yawDegrees = 0.0F;

    bool operator==(const Environment&) const = default;
};

// 톤 매핑 앞뒤의 후처리. Environment 와 따로 두는 이유는 그쪽 비교가 환경 맵을 다시 굽는
// 조건이라, 여기 값을 만질 때마다 굽기가 돌면 안 되기 때문이다.
struct PostProcess {
    // 0 이면 Bloom 을 끈다. 흐린 이미지를 원본에 섞는 비율이다.
    float bloomIntensity = 0.1F;
    bool autoExposure = false;
    // 자동 노출이 목표 값으로 옮겨 가는 속도(1/초).
    float adaptationSpeed = 2.0F;
    // 자동 노출이 고를 수 있는 EV100 범위.
    float exposureMinEv = -4.0F;
    float exposureMaxEv = 12.0F;
    // 지수 높이 안개. 밀도가 0 이면 끈다. 기준 높이에서 밀도가 가장 크고 위로 갈수록 감쇠한다.
    glm::vec3 fogColor{0.62F, 0.70F, 0.80F};
    float fogDensity = 0.0F;
    float fogHeight = 0.0F;
    float fogFalloff = 0.5F;

    bool operator==(const PostProcess&) const = default;
};

// 되돌리기가 되살리는 장면 상태. 카메라와 애니메이션 재생 시각처럼 매 프레임 스스로 변하는
// 것은 담지 않는다. 담으면 재생 중에 되돌리기 기록이 프레임마다 쌓인다.
struct SceneSnapshot {
    std::string name;
    std::vector<Object> objects;
    std::vector<MeshRenderer> meshRenderers;
    std::vector<Animator> animators;
    std::vector<Light> lights;
    std::vector<RigidBody> rigidBodies;
    std::vector<Fluid> fluids;
    glm::vec3 ambientColor{0.25F};
    float ambientIntensity = 1.0F;
    Environment environment;
    PostProcess post;
};

struct Scene {
    std::string name;
    std::vector<Object> objects;
    std::vector<MeshRenderer> meshRenderers;
    std::vector<Animator> animators;
    std::vector<Light> lights;
    std::vector<RigidBody> rigidBodies;
    std::vector<Fluid> fluids;
    Camera camera;
    // 재생 중인지. 참일 때만 물리가 돌고, 편집기는 되돌리기 기록을 멈춘다. 저장하지 않는다.
    bool simulating = false;
    // 조명이 닿지 않는 곳을 채우는 균일 환경광.
    glm::vec3 ambientColor{0.25F};
    float ambientIntensity = 1.0F;
    Environment environment;
    PostProcess post;

    // 애니메이션 시간을 진행시키고 조인트 행렬을 다시 만든다. 재생 중이 아니고 클립도 그대로면
    // 포즈 계산 자체를 건너뛴다. jobs 가 있으면 애니메이터마다 독립이라 워커에 나눈다.
    void update(float deltaSeconds, core::JobSystem* jobs = nullptr);

    // 프레임에 한 번, 장면을 읽기 직전에 부른다. 지난 사본과 비교해 변한 것을 찾고 세계 변환과
    // 가시성 캐시를 다시 만든다.
    //
    // 훅을 거는 대신 비교하는 이유: Object::transform 이 public 이고 편집기 여러 곳에서 직접
    // 대입한다. 훅을 하나라도 빠뜨리면 화면이 조용히 틀리는데, 비교는 빠뜨릴 수가 없다.
    // 덤으로 오브젝트별 더티 플래그가 나와 그림자 시점 무효화에 그대로 쓰인다.
    void refresh();

    // 마지막 refresh 에서 무엇이든 바뀌었으면 증가한다. 소비자는 자기가 본 값과 비교만 하면 된다.
    uint64_t revision() const { return anyRevision; }
    uint64_t transformRevision() const { return transformRev; }
    uint64_t lightRevision() const { return lightRev; }
    // 오브젝트 개수나 부모 관계가 바뀌면 증가한다. 인덱스가 통째로 재배치될 수 있다는 뜻이다.
    uint64_t topologyRevision() const { return topologyRev; }

    // 아래 셋은 refresh 이후에만 유효하다.
    const glm::mat4& world(uint32_t index) const { return cachedWorlds[index]; }
    bool visibleCached(uint32_t index) const { return cachedVisible[index] != 0; }
    bool objectDirty(uint32_t index) const { return cachedDirty[index] != 0; }
    bool animatorPosed(uint32_t index) const { return animatorPosedFlags[index] != 0; }

    // 캐시를 쓰지 않는 즉시 계산. refresh 사이에 부르는 편집기(기즈모, 재부모화)가 쓴다.
    glm::mat4 worldMatrix(uint32_t index) const;
    // 조상 중 하나라도 숨겨져 있으면 보이지 않는다.
    bool visibleInTree(uint32_t index) const;

    // 되돌리기용 사본을 뜨고 되살린다.
    SceneSnapshot capture() const;
    void restore(const SceneSnapshot& snapshot);
    // 되돌리기 기록에 남길 만한 차이가 있는지. 애니메이션 재생 시각은 빼고 본다.
    bool differsFrom(const SceneSnapshot& snapshot) const;

    // 메쉬 부품을 붙이고 그 첨자를 돌려준다. 이미 붙어 있으면 값만 바꾼다.
    int32_t attachMeshRenderer(uint32_t index, uint32_t mesh, int32_t skin = -1);
    // 오브젝트가 그리는 전역 메쉬 번호. 메쉬 부품이 없으면 INVALID_MESH.
    uint32_t meshOf(uint32_t index) const;
    // 그 메쉬가 쓰는 스킨 번호. 없으면 -1.
    int32_t skinOf(uint32_t index) const;
    // 부품을 붙이고 첨자를 돌려준다. 이미 붙어 있으면 그 첨자를 그대로 돌려준다.
    int32_t attachLight(uint32_t index, const Light& light = {});
    int32_t attachRigidBody(uint32_t index, const RigidBody& body = {});
    int32_t attachFluid(uint32_t index, const Fluid& fluid = {});
    // 부품을 뗀다. 아무도 가리키지 않게 된 부품은 배열에서 빠지고 첨자가 다시 맞춰진다.
    void detachComponent(uint32_t index, int32_t Object::* handle);

    // candidate 가 ancestor 자신이거나 그 자손인지. 순환하는 부모 관계를 막는 데 쓴다.
    bool isDescendant(uint32_t candidate, uint32_t ancestor) const;
    // 대상과 그 자손을 모두 지운다. 남은 오브젝트의 부모 인덱스는 다시 맞춘다.
    void removeObject(uint32_t index);
    // 여러 개를 한 번에 지운다. 하나씩 지우면 첫 번째 삭제가 인덱스를 밀어 나머지가 엉뚱한
    // 오브젝트를 가리킨다. 자손 관계로 겹쳐도 안전하다.
    void removeObjects(const std::vector<uint32_t>& indices);
    // 대상과 그 자손을 복제하고 새로 만든 뿌리의 인덱스를 돌려준다.
    uint32_t duplicateObject(uint32_t index);

private:
    void resolveWorld(uint32_t index);

    std::vector<glm::mat4> cachedWorlds;
    std::vector<uint8_t> cachedVisible;
    std::vector<uint8_t> cachedDirty;
    std::vector<uint8_t> cachedResolved;
    std::vector<uint8_t> animatorPosedFlags;

    // 지난 refresh 때의 사본. 이것과 비교해 변경을 찾는다.
    std::vector<Transform> previousTransforms;
    std::vector<int32_t> previousParents;
    std::vector<uint8_t> previousVisible;
    std::vector<Light> previousLights;

    uint64_t anyRevision = 1;
    uint64_t transformRev = 1;
    uint64_t lightRev = 1;
    uint64_t topologyRev = 1;
};

// 여러 장면을 담아 두고 전환한다.
class SceneManager {
public:
    Scene& create(std::string name);
    void setActive(size_t index);

    Scene& active() { return scenes[activeIndex]; }
    const Scene& active() const { return scenes[activeIndex]; }
    size_t count() const { return scenes.size(); }
    size_t current() const { return activeIndex; }
    Scene& at(size_t index) { return scenes[index]; }
    const Scene& at(size_t index) const { return scenes[index]; }

private:
    std::vector<Scene> scenes;
    size_t activeIndex = 0;
};

} // namespace scene
