#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

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
    // 크로스페이드. nextClip 이 0 이상이면 clip 에서 nextClip 으로 blendSeconds 동안 섞어 가며, 끝나면 nextClip 이
    // clip 이 된다. 두 클립의 시각은 각자 흐른다. blendSeconds 가 0 이면 바로 바꾼다.
    int32_t nextClip = -1;
    float nextClipTime = 0.0F;
    float blendSeconds = 0.3F;
    float blendElapsed = 0.0F;
    // 스킨마다의 조인트 행렬. Scene::update 가 채우고 렌더러가 그대로 올린다.
    std::vector<std::vector<glm::mat4>> jointMatrices;

    // clip 을 target 으로 크로스페이드한다. 같은 클립이면 아무 일도 없다.
    void crossfadeTo(uint32_t target) {
        if (target == clip && nextClip < 0) {
            return;
        }
        nextClip = static_cast<int32_t>(target);
        nextClipTime = 0.0F;
        blendElapsed = 0.0F;
    }

private:
    friend struct Scene;
    std::vector<glm::mat4> nodeWorlds;
    std::vector<asset::Node> posedNodes;
    std::vector<asset::Node> nextNodes;
    std::vector<asset::Node> blendedNodes;
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

// 시뮬레이션을 어디서 돌릴지. AUTO 는 하드웨어와 자동 튜닝이 정한다.
enum class SimulationBackend : uint32_t {
    AUTO = 0,
    CPU = 1,
    GPU = 2,
};

// GLSL 의 RIGID_SHAPE_* / FLUID_COLLIDER_* 와 번호가 같아야 한다.
enum class ColliderShape : uint32_t {
    SPHERE = 0,
    BOX = 1,
    // 오브젝트의 +Y 를 법선으로 하는 무한 평면. 바닥으로 쓴다.
    PLANE = 2,
    // 축이 +Y 인 원기둥·캡슐. radius 와 halfExtents.y(반높이)를 쓴다.
    CYLINDER = 3,
    CAPSULE = 4,
    // 오브젝트의 메쉬 렌더러가 그리는 메쉬를 그대로 콜라이더로 쓴다. 늘 운동학이다(바닥·지형용).
    MESH = 5,
};
inline constexpr uint32_t COLLIDER_SHAPE_COUNT = 6;

// 강체 부품. 재생 중에 physics 가 세계 공간에서 적분해 오브젝트 변환에 되돌려 쓴다. 운동학 물체는
// 힘을 받지 않고 다른 물체만 밀어낸다. 평면은 늘 운동학으로 다룬다.
struct RigidBody {
    // 어디서 풀지. AUTO 는 CPU 를 쓴다. GPU 솔버는 Jacobi 라 CPU 의 순차 임펄스와 수치가 다르다.
    SimulationBackend backend = SimulationBackend::AUTO;
    ColliderShape shape = ColliderShape::SPHERE;
    float mass = 1.0F;
    bool useGravity = true;
    bool kinematic = false;
    float restitution = 0.3F;
    float friction = 0.5F;
    // 구·원기둥·캡슐 반지름과 상자 반쪽 크기(오브젝트 지역 공간, 크기 변환 전). 원기둥·캡슐은
    // halfExtents.y 를 반높이로 쓴다(캡슐은 반구를 뺀 몸통의 반높이).
    float radius = 0.5F;
    glm::vec3 halfExtents{0.5F};
    // 시뮬레이션 상태. 재생을 멈추면 스냅샷 복귀로 함께 되돌아간다.
    glm::vec3 velocity{0.0F};
    glm::vec3 angularVelocity{0.0F};

    bool operator==(const RigidBody&) const = default;
};

// 강체 부품이 세계 공간에서 차지하는 모양. 솔버와 편집기의 콜라이더 표시가 어긋나지 않도록 규칙을
// 여기서 한 번만 정한다. 구는 가장 큰 축의 배율을 받고 상자는 축마다 따로 받는다. 원기둥·캡슐은
// 반지름이 X·Z 중 큰 배율, 반높이(halfExtents.y)가 Y 배율을 받는다.
struct ColliderPose {
    glm::vec3 position{0.0F};
    glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
    float radius = 0.0F;
    glm::vec3 halfExtents{0.0F};
    // 메쉬 콜라이더만. 오브젝트 배율 그대로(메쉬 정점에 곱한다).
    glm::vec3 scale{1.0F};
};
ColliderPose colliderPose(const RigidBody& body, const glm::mat4& world);

// 메쉬 콜라이더가 읽는 CPU 측 삼각형. 전역 메쉬 번호로 찾는다. 모델을 올릴 때 app 이 채우고 내릴 때
// 비운다. 물리가 gfx 를 보지 않으므로 여기(scene)에 둔다.
//
// ponytail: 삼각형 수가 MAX_TRIANGLES 이하인 가장 고운 LOD 단계를 담는다. 굵은 단계는 화면의 메쉬와
// 조금 어긋나지만 삼각형마다 훑는 협역 검사가 감당할 크기다. BVH 를 넣으면 0단계를 쓸 수 있다.
struct ColliderMesh {
    static constexpr uint32_t MAX_TRIANGLES = 2048;
    std::vector<glm::vec3> positions;
    std::vector<uint32_t> indices;
    glm::vec3 boundsCenter{0.0F};
    float boundsRadius = 0.0F;
    bool empty() const { return indices.empty(); }
};

// 유체를 화면에 어떻게 낼지. 입자는 내장 구 인스턴스, 표면은 마칭 큐브로 뽑은 등치면이다.
enum class FluidDisplay : uint8_t { PARTICLES, SURFACE };

// 유체 부품. 입자 상태는 GPU 에만 있고 여기에는 방출·시뮬레이션·표시 설정만 둔다. 방출 상자는
// 오브젝트 지역 공간이고 용기는 월드 공간이다. 값이 바뀌면 렌더러가 입자를 다시 뿌린다.
struct Fluid {
    // 어디서 풀지. AUTO 는 GPU 를 쓰되 만들 수 없으면 CPU 로 내려간다.
    SimulationBackend backend = SimulationBackend::AUTO;
    FluidDisplay display = FluidDisplay::PARTICLES;
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

    // 아래는 표면 표시에만 쓰인다. 입자 표시에서는 무시된다.
    // 용기를 덮는 격자의 축별 셀 수. 값이 클수록 곱하기 세제곱으로 비싸진다.
    uint32_t surfaceResolution = 48;
    // 등치값. 커널을 합친 무차원 값이라 입자 배치에 따라 손으로 맞춘다. 작으면 표면이 부풀고
    // 크면 물이 얇아져 구멍이 뚫린다.
    float surfaceIso = 0.5F;
    // 통과한 빛에 남는 색. 확산 색이 아니라 흡수를 거치고 남은 색이다.
    glm::vec3 waterColor{0.35F, 0.65F, 0.75F};
    float surfaceRoughness = 0.05F;
    // 미터당 흡수 계수. 빨강이 크면 두꺼운 곳이 푸르게 보인다.
    glm::vec3 absorption{2.2F, 0.5F, 0.25F};
    // 두께에 곱하는 값. 용기가 작을 때 색을 살리는 손잡이다.
    float thicknessScale = 1.0F;

    bool operator==(const Fluid&) const = default;
};

// 입자 부품(불꽃·파편). 입자 상태는 GPU 에만 있고 여기에는 방출·운동·표시 설정만 둔다. 방출 자세는
// 오브젝트의 월드 변환(위치와 +Y 축)이다. GPU 전용이라 백엔드 선택이 없다.
struct ParticleSystem {
    // 살아 있을 수 있는 입자 수의 상한. 링 버퍼라 이보다 많이 뿌리면 가장 오래된 것이 덮인다.
    uint32_t maxParticles = 4096;
    // 초당 방출 수.
    float emitRate = 200.0F;
    float lifetime = 3.0F;
    float initialSpeed = 3.0F;
    // +Y 둘레 원뿔의 반각(도).
    float spreadAngleDegrees = 20.0F;
    // (0, -9.81, 0) 에 곱한다.
    float gravityScale = 1.0F;
    // 초당 속도 감쇠 비율.
    float drag = 0.1F;
    // 스프라이트 지름(월드). 나이에 따라 시작에서 끝으로 선형 보간한다.
    float sizeStart = 0.1F;
    float sizeEnd = 0.02F;
    // 미리 곱하지 않은 색과 알파. 알파는 나이에 따라 0 으로 줄어든다.
    glm::vec4 color{1.0F, 0.8F, 0.4F, 1.0F};
    glm::vec3 emissive{0.0F};
    // 장면과 부딪혔을 때 법선 방향 속도에 남는 비율.
    float restitution = 0.4F;
    // 상위 가속 구조에 광선 질의로 부딪힌다. 광선 질의가 없는 장치에서는 무시된다.
    bool collide = true;

    bool operator==(const ParticleSystem&) const = default;
};

// 천의 고정 방식. 월드 Y 평균이 가장 높은 격자 변을 «윗변» 으로 본다(오브젝트 방향에 무관).
enum class ClothPin : uint32_t {
    NONE = 0,
    TOP_EDGE = 1,
    TWO_CORNERS = 2,
};

// 천 부품(XPBD). 오브젝트의 메쉬가 내장 «천 격자»(resolution 과 같은 분할)여야 돈다. 크기는 오브젝트 배율,
// 정지 자세는 오브젝트 월드 변환이다. 상태(정점 위치)는 렌더러의 변형 정점 버퍼에만 있고 저장하지 않는다.
struct Cloth {
    // AUTO 는 GPU 를 쓰되 만들 수 없으면 CPU 로 내려간다(유체와 같은 규칙).
    SimulationBackend backend = SimulationBackend::AUTO;
    // 격자 분할 수. 16, 32, 64 만 있다(내장 도형).
    uint32_t resolution = 32;
    // 천 전체 질량(kg).
    float mass = 1.0F;
    // XPBD 컴플라이언스(m/N). 0 이면 늘어나지 않는다.
    float stretchCompliance = 0.0F;
    float shearCompliance = 1.0e-3F;
    float bendCompliance = 1.0e-2F;
    // 초당 속도 감쇠 비율.
    float damping = 0.5F;
    uint32_t substeps = 4;
    uint32_t iterations = 8;
    ClothPin pin = ClothPin::TOP_EDGE;
    // 콜라이더에서 띄우는 두께.
    float thickness = 0.02F;
    // 콜라이더에 닿은 정점의 접선 이동을 이 비율만큼 죽인다(0 미끄러움, 1 달라붙음).
    float friction = 0.5F;
    glm::vec3 gravity{0.0F, -9.81F, 0.0F};
    // 단위 질량당 힘으로 더하는 바람.
    glm::vec3 wind{0.0F};

    bool operator==(const Cloth&) const = default;
};

// 힘 마당의 종류. 바람은 오브젝트의 앞(-Z) 방향으로 일정하게 밀고, 소용돌이는 오브젝트 +Y 축 둘레로 돌리며,
// 점은 오브젝트 위치로 끌어당긴다(세기가 음수면 밀어낸다).
enum class ForceFieldType : uint32_t {
    WIND = 0,
    VORTEX = 1,
    POINT = 2,
};

// 힘 마당 부품. 천·유체·GPU 입자가 같은 식으로 읽는다(physics/force_field.h ↔ shaders/force_field.glsl). 강체는
// 읽지 않는다. 방향과 중심은 오브젝트 월드 변환에서 온다.
struct ForceField {
    ForceFieldType type = ForceFieldType::WIND;
    // 단위 질량당 힘(가속도, m/s²).
    float strength = 5.0F;
    // 닿는 반지름(월드). 0 이면 무한이다.
    float radius = 0.0F;
    // 가장자리로 갈수록 (1 - d/r)^falloff 로 약해진다. 0 이면 반지름 안에서 일정하다.
    float falloff = 1.0F;

    bool operator==(const ForceField&) const = default;
};

// DDGI 프로브 볼륨 부품. 오브젝트 위치를 중심으로 배율만큼(±scale)의 축 정렬 상자에 축마다 probes 개의 프로브를
// 깐다. 회전은 무시한다(격자는 축 정렬). 켜진 첫 볼륨이 장면 경계 격자를 대신한다. 장면에 하나만 쓴다.
struct DdgiVolume {
    uint32_t probes = 8;
    bool enabled = true;

    bool operator==(const DdgiVolume&) const = default;
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
    int32_t particleSystem = -1;
    int32_t cloth = -1;
    int32_t forceField = -1;
    int32_t ddgiVolume = -1;

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
    // 0 이면 Bloom 을 끈다. 임계값을 넘은 부분을 흐려서 «더할» 세기다. 원본과 섞는 것이 아니라
    // 더하므로 1 을 넘겨도 된다.
    float bloomIntensity = 0.6F;
    // 이 밝기를 넘는 부분만 번진다. 노출을 곱하기 «전»의 HDR 값에 걸리므로, 자동 노출을 켜면
    // 화면에서 날아가는 밝기와 임계값이 어긋날 수 있다.
    float bloomThreshold = 1.0F;
    // 임계값 언저리를 부드럽게 넘기는 폭. 0 이면 경계가 딱 끊겨 화면이 깜빡인다. 임계값보다 크면
    // 곡선이 뒤집혀 어두운 곳까지 번지므로 렌더러가 임계값으로 자른다.
    float bloomKnee = 0.5F;
    // 올라오며 섞을 때 넓은 밉 쪽에 두는 무게. 크면 더 멀리 퍼진다.
    float bloomScatter = 0.7F;
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
    // 첫 방향광 쪽 안개 인스캐터 세기. 0 이면 안개 색이 방향과 무관한 상수다.
    float fogSunScatter = 0.0F;

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
    std::vector<ParticleSystem> particleSystems;
    std::vector<Cloth> cloths;
    std::vector<ForceField> forceFields;
    std::vector<DdgiVolume> ddgiVolumes;
    glm::vec3 ambientColor{0.25F};
    float ambientIntensity = 1.0F;
    Environment environment;
    PostProcess post;
};

struct Scene {
    std::string name;
    // SceneManager 가 붙이는 고유 번호(1부터). 렌더러·시뮬레이터가 «지난 프레임과 같은 장면인가»를 이 번호로
    // 판정한다. 주소로 판정하면 장면을 닫고 그 자리에 새 장면이 놓였을 때 같은 장면으로 보인다. 저장하지 않는다.
    uint64_t id = 0;
    std::vector<Object> objects;
    std::vector<MeshRenderer> meshRenderers;
    std::vector<Animator> animators;
    std::vector<Light> lights;
    std::vector<RigidBody> rigidBodies;
    std::vector<Fluid> fluids;
    std::vector<ParticleSystem> particleSystems;
    std::vector<Cloth> cloths;
    std::vector<ForceField> forceFields;
    std::vector<DdgiVolume> ddgiVolumes;
    Camera camera;
    // 재생 중인지. 참일 때만 물리가 돌고, 편집기는 되돌리기 기록을 멈춘다. 저장하지 않는다.
    bool simulating = false;
    // 전역 메쉬 번호로 찾는 메쉬 콜라이더 표. app 이 소유하고 모든 장면이 같은 것을 가리킨다.
    // 저장하지 않는다.
    const std::vector<ColliderMesh>* colliderMeshes = nullptr;
    // 오브젝트의 메쉬 콜라이더. 메쉬 렌더러가 없거나 표에 없으면 nullptr.
    const ColliderMesh* colliderMesh(uint32_t index) const;
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
    void refresh(core::JobSystem* jobs = nullptr);

    // 마지막 refresh 에서 무엇이든 바뀌었으면 증가한다. 소비자는 자기가 본 값과 비교만 하면 된다.
    uint64_t revision() const { return anyRevision; }
    uint64_t transformRevision() const { return transformRev; }
    uint64_t lightRevision() const { return lightRev; }
    // 오브젝트 개수나 부모 관계가 바뀌면 증가한다. 인덱스가 통째로 재배치될 수 있다는 뜻이다.
    uint64_t topologyRevision() const { return topologyRev; }
    // 부품을 붙이거나 떼어 부품 배열의 배치가 바뀌면 증가한다. 부품 하나를 떼면 모든 종류의 배열이
    // 압축되므로(detachComponent) 첨자로 GPU 상태를 짝지어 둔 쪽은 이 값을 보고 다시 맞춰야 한다.
    // 강체 속도처럼 재생 중 매 프레임 변하는 «값»은 보지 않으므로 재생만으로는 오르지 않는다.
    uint64_t componentRevision() const { return componentRev; }

    // 배열이 재배치되었음을 알린다. refresh 의 비교만으로는 «지우고 같은 수만큼 새로 만들기»를
    // 알아챌 수 없어 덧붙인 보수적 표식이다. 빠뜨려도 지금보다 나빠지지 않는다.
    void markStructureDirty() { structureDirty = true; }

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
    int32_t attachParticleSystem(uint32_t index, const ParticleSystem& system = {});
    int32_t attachCloth(uint32_t index, const Cloth& cloth = {});
    int32_t attachForceField(uint32_t index, const ForceField& field = {});
    int32_t attachDdgiVolume(uint32_t index, const DdgiVolume& volume = {});
    // 부품을 뗀다. 아무도 가리키지 않게 된 부품은 배열에서 빠지고 첨자가 다시 맞춰진다.
    void detachComponent(uint32_t index, int32_t Object::* handle);
    // 오브젝트에 붙은 T 부품. 없거나 첨자가 범위 밖이면 nullptr. 첨자를 손으로 가드하는 관용구를 대신한다.
    template <typename T> T* component(uint32_t index);
    template <typename T> const T* component(uint32_t index) const;

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
    // 오브젝트를 깊이 순으로 늘어놓는다. 부모가 자식보다 먼저 오므로 깊이 한 단계씩 나눠 풀면 재귀도
    // 메모 표식도 필요 없고, 같은 단계끼리는 서로를 보지 않아 워커에 나눌 수 있다. 계층이 바뀐
    // 프레임에만 다시 만든다.
    void rebuildDepthOrder();

    std::vector<glm::mat4> cachedWorlds;
    std::vector<uint8_t> cachedVisible;
    std::vector<uint8_t> cachedDirty;
    // 오브젝트마다의 깊이(뿌리가 0)와 깊이 순 나열, 그리고 단계마다의 시작 위치.
    std::vector<uint32_t> depths;
    std::vector<uint32_t> depthOrder;
    std::vector<uint32_t> depthOffsets;
    std::vector<uint8_t> animatorPosedFlags;

    // 지난 refresh 때의 사본. 이것과 비교해 변경을 찾는다.
    std::vector<Transform> previousTransforms;
    std::vector<int32_t> previousParents;
    std::vector<uint8_t> previousVisible;
    std::vector<Light> previousLights;
    // 부품 배열 크기 다섯 개와 오브젝트별 부품 첨자 다섯 개를 이어 붙인 배치표. 값이 아니라 배치만
    // 담으므로 재생 중에는 변하지 않는다. 아래는 refresh 전용 스크래치로, 둘을 맞바꿔 쓴다.
    std::vector<int32_t> previousComponentLayout;
    std::vector<int32_t> componentLayout;
    bool structureDirty = true;

    uint64_t anyRevision = 1;
    uint64_t transformRev = 1;
    uint64_t lightRev = 1;
    uint64_t topologyRev = 1;
    uint64_t componentRev = 1;
};

// 부품 종류 표. 종류마다 (Scene 의 배열, Object 의 첨자 멤버) 한 쌍을 f 에 넘긴다. 부품을 떼고 옮기고 복제하고
// 배치를 비교하는 코드가 전부 이 표를 돌므로, 부품 종류를 더할 때 고칠 곳은 여기와 ComponentSlot 특수화뿐이다.
template <typename SceneType, typename F> void forEachComponentKind(SceneType& scene, F&& f) {
    f(scene.meshRenderers, &Object::meshRenderer);
    f(scene.animators, &Object::animator);
    f(scene.lights, &Object::light);
    f(scene.rigidBodies, &Object::rigidBody);
    f(scene.fluids, &Object::fluid);
    f(scene.particleSystems, &Object::particleSystem);
    f(scene.cloths, &Object::cloth);
    f(scene.forceFields, &Object::forceField);
    f(scene.ddgiVolumes, &Object::ddgiVolume);
}

// 부품 타입 → Object 의 첨자 멤버와 Scene 의 배열.
template <typename T> struct ComponentSlot;
#define CG_LAB_COMPONENT_SLOT(Type, member, array)                                                                     \
    template <> struct ComponentSlot<Type> {                                                                           \
        static constexpr int32_t Object::* HANDLE = &Object::member;                                                   \
        static std::vector<Type>& items(Scene& scene) { return scene.array; }                                          \
        static const std::vector<Type>& items(const Scene& scene) { return scene.array; }                              \
    }
CG_LAB_COMPONENT_SLOT(MeshRenderer, meshRenderer, meshRenderers);
CG_LAB_COMPONENT_SLOT(Animator, animator, animators);
CG_LAB_COMPONENT_SLOT(Light, light, lights);
CG_LAB_COMPONENT_SLOT(RigidBody, rigidBody, rigidBodies);
CG_LAB_COMPONENT_SLOT(Fluid, fluid, fluids);
CG_LAB_COMPONENT_SLOT(ParticleSystem, particleSystem, particleSystems);
CG_LAB_COMPONENT_SLOT(Cloth, cloth, cloths);
CG_LAB_COMPONENT_SLOT(ForceField, forceField, forceFields);
CG_LAB_COMPONENT_SLOT(DdgiVolume, ddgiVolume, ddgiVolumes);
#undef CG_LAB_COMPONENT_SLOT

template <typename T> T* Scene::component(uint32_t index) {
    int32_t slot = objects[index].*ComponentSlot<T>::HANDLE;
    std::vector<T>& items = ComponentSlot<T>::items(*this);
    return slot >= 0 && static_cast<size_t>(slot) < items.size() ? &items[static_cast<size_t>(slot)] : nullptr;
}
template <typename T> const T* Scene::component(uint32_t index) const {
    int32_t slot = objects[index].*ComponentSlot<T>::HANDLE;
    const std::vector<T>& items = ComponentSlot<T>::items(*this);
    return slot >= 0 && static_cast<size_t>(slot) < items.size() ? &items[static_cast<size_t>(slot)] : nullptr;
}

// 여러 장면을 담아 두고 전환한다.
//
// 장면을 포인터로 담는 이유: 장면을 더 만들거나 닫을 때 벡터가 재배치되어도 편집기·적재 스레드가 잡아 둔 Scene&
// 가 살아 있어야 해서다. «같은 장면인가»는 주소가 아니라 Scene::id 로 판정한다 — 닫힌 자리에 새 장면이 놓여도
// 번호는 다르다.
class SceneManager {
public:
    Scene& create(std::string name);
    void setActive(size_t index);
    // 장면을 닫는다. 마지막 하나는 닫지 않는다(거짓). 활성 장면이 닫히면 앞 장면이 활성이 된다.
    bool close(size_t index);
    // 번호로 찾는다. 닫혔으면 nullptr.
    Scene* find(uint64_t id);

    Scene& active() { return *scenes[activeIndex]; }
    const Scene& active() const { return *scenes[activeIndex]; }
    size_t count() const { return scenes.size(); }
    size_t current() const { return activeIndex; }
    Scene& at(size_t index) { return *scenes[index]; }
    const Scene& at(size_t index) const { return *scenes[index]; }

private:
    std::vector<std::unique_ptr<Scene>> scenes;
    size_t activeIndex = 0;
    uint64_t nextId = 1;
};

} // namespace scene
