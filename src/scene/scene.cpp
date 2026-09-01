#include "scene/scene.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include "core/error.h"

namespace scene {

glm::mat4 Transform::matrix() const {
    glm::mat4 result = glm::translate(glm::mat4(1.0F), position);
    result *= glm::mat4_cast(rotation);
    return glm::scale(result, scale);
}

Transform Transform::fromMatrix(const glm::mat4& matrix) {
    Transform transform;
    glm::vec3 skew;
    glm::vec4 perspective;
    glm::decompose(matrix, transform.scale, transform.rotation, transform.position, skew, perspective);
    return transform;
}

Scene& SceneManager::create(std::string name) {
    Scene scene;
    scene.name = std::move(name);
    scenes.push_back(std::move(scene));
    return scenes.back();
}

void SceneManager::setActive(size_t index) {
    if (index >= scenes.size()) {
        core::fatal("존재하지 않는 장면으로 전환할 수 없습니다: {} / {}", index, scenes.size());
    }
    activeIndex = index;
}

} // namespace scene
