#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

#include "asset/model.h"
#include "scene/camera.h"

namespace scene {

inline constexpr uint32_t INVALID_MESH = 0xFFFFFFFFU;

struct Transform {
    glm::vec3 position{0.0F};
    glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
    glm::vec3 scale{1.0F};

    glm::mat4 matrix() const;
    static Transform fromMatrix(const glm::mat4& matrix);
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
};

struct Object {
    std::string name;
    // 부모 기준 지역 변환. 세계 변환은 Scene::worldMatrix 가 부모를 거슬러 올라가 만든다.
    Transform transform;
    // Scene::objects 인덱스. 뿌리면 -1.
    int32_t parent = -1;
    // GeometryStore 의 전역 메쉬 인덱스. 변환만 담는 노드면 INVALID_MESH.
    uint32_t meshIndex = INVALID_MESH;
    bool visible = true;
    // Scene::animators 인덱스와 그 스켈레톤의 스킨 번호. 없으면 -1.
    int32_t animator = -1;
    int32_t skin = -1;
    // Scene::lights 인덱스. 조명이 아니면 -1.
    int32_t light = -1;
};

struct Scene {
    std::string name;
    std::vector<Object> objects;
    std::vector<Animator> animators;
    std::vector<Light> lights;
    Camera camera;
    // 조명이 닿지 않는 곳을 채우는 균일 환경광.
    glm::vec3 ambientColor{0.25F};
    float ambientIntensity = 1.0F;

    // 애니메이션 시간을 진행시키고 조인트 행렬을 다시 만든다.
    void update(float deltaSeconds);

    // 부모를 거슬러 올라가며 곱한 세계 변환.
    //
    // ponytail: 오브젝트마다 부모 사슬을 다시 타므로 깊이에 비례한다. 편집기 규모에서는 문제가
    // 없지만, 깊은 계층을 대량으로 다루게 되면 프레임마다 한 번 훑어 캐시해야 한다.
    glm::mat4 worldMatrix(uint32_t index) const;
    // 조상 중 하나라도 숨겨져 있으면 보이지 않는다.
    bool visibleInTree(uint32_t index) const;

    // candidate 가 ancestor 자신이거나 그 자손인지. 순환하는 부모 관계를 막는 데 쓴다.
    bool isDescendant(uint32_t candidate, uint32_t ancestor) const;
    // 대상과 그 자손을 모두 지운다. 남은 오브젝트의 부모 인덱스는 다시 맞춘다.
    void removeObject(uint32_t index);
    // 대상과 그 자손을 복제하고 새로 만든 뿌리의 인덱스를 돌려준다.
    uint32_t duplicateObject(uint32_t index);
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
    const Scene& at(size_t index) const { return scenes[index]; }

private:
    std::vector<Scene> scenes;
    size_t activeIndex = 0;
};

} // namespace scene
