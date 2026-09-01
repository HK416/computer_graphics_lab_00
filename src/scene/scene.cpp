#include "scene/scene.h"

#include <cmath>

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

void Scene::update(float deltaSeconds) {
    if (skeleton.skins.empty()) {
        return;
    }
    if (playAnimation && clip < skeleton.animations.size()) {
        float duration = skeleton.animations[clip].duration;
        if (duration > 0.0F) {
            clipTime = std::fmod(clipTime + deltaSeconds * animationSpeed, duration);
            if (clipTime < 0.0F) {
                clipTime += duration;
            }
        }
    }

    asset::poseNodes(skeleton, clip, clipTime, nodeWorlds);
    jointMatrices.resize(skeleton.skins.size());
    for (uint32_t i = 0; i < skeleton.skins.size(); ++i) {
        asset::skinMatrices(skeleton, nodeWorlds, i, jointMatrices[i]);
    }
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
