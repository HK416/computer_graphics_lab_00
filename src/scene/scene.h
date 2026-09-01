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

struct Transform {
    glm::vec3 position{0.0F};
    glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
    glm::vec3 scale{1.0F};

    glm::mat4 matrix() const;
    static Transform fromMatrix(const glm::mat4& matrix);
};

struct Object {
    std::string name;
    Transform transform;
    // GeometryStore 의 전역 메쉬 인덱스.
    uint32_t meshIndex = 0;
    bool visible = true;
    // Scene::skeleton 의 스킨 인덱스. 스킨이 없으면 -1.
    int32_t skin = -1;
};

struct Scene {
    std::string name;
    std::vector<Object> objects;
    Camera camera;

    // 장면 하나는 모델 하나에서 만들어지므로 스켈레톤도 하나면 충분하다.
    asset::Skeleton skeleton;
    uint32_t clip = 0;
    float clipTime = 0.0F;
    bool playAnimation = true;
    float animationSpeed = 1.0F;
    // 스킨마다의 조인트 행렬. update() 가 채우고 렌더러가 그대로 올린다.
    std::vector<std::vector<glm::mat4>> jointMatrices;

    // 애니메이션 시간을 진행시키고 조인트 행렬을 다시 만든다. 스킨이 없으면 아무것도 하지 않는다.
    void update(float deltaSeconds);

private:
    std::vector<glm::mat4> nodeWorlds;
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
