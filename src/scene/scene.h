#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

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
};

struct Scene {
    std::string name;
    std::vector<Object> objects;
    Camera camera;
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
