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
    for (Animator& animator : animators) {
        if (animator.skeleton.skins.empty()) {
            continue;
        }
        if (animator.playing && animator.clip < animator.skeleton.animations.size()) {
            float duration = animator.skeleton.animations[animator.clip].duration;
            if (duration > 0.0F) {
                animator.clipTime = std::fmod(animator.clipTime + deltaSeconds * animator.speed, duration);
                if (animator.clipTime < 0.0F) {
                    animator.clipTime += duration;
                }
            }
        }

        asset::poseNodes(animator.skeleton, animator.clip, animator.clipTime, animator.nodeWorlds);
        animator.jointMatrices.resize(animator.skeleton.skins.size());
        for (uint32_t i = 0; i < animator.skeleton.skins.size(); ++i) {
            asset::skinMatrices(animator.skeleton, animator.nodeWorlds, i, animator.jointMatrices[i]);
        }
    }
}

glm::mat4 Scene::worldMatrix(uint32_t index) const {
    glm::mat4 world = objects[index].transform.matrix();
    for (int32_t parent = objects[index].parent; parent >= 0; parent = objects[static_cast<size_t>(parent)].parent) {
        world = objects[static_cast<size_t>(parent)].transform.matrix() * world;
    }
    return world;
}

bool Scene::visibleInTree(uint32_t index) const {
    for (int32_t current = static_cast<int32_t>(index); current >= 0;
         current = objects[static_cast<size_t>(current)].parent) {
        if (!objects[static_cast<size_t>(current)].visible) {
            return false;
        }
    }
    return true;
}

bool Scene::isDescendant(uint32_t candidate, uint32_t ancestor) const {
    for (int32_t current = static_cast<int32_t>(candidate); current >= 0;
         current = objects[static_cast<size_t>(current)].parent) {
        if (static_cast<uint32_t>(current) == ancestor) {
            return true;
        }
    }
    return false;
}

void Scene::removeObject(uint32_t index) {
    std::vector<bool> doomed(objects.size(), false);
    doomed[index] = true;
    // 부모가 배열에서 항상 앞선다는 보장이 없어 더 번지지 않을 때까지 훑는다.
    for (bool spreading = true; spreading;) {
        spreading = false;
        for (size_t i = 0; i < objects.size(); ++i) {
            const Object& object = objects[i];
            if (!doomed[i] && object.parent >= 0 && doomed[static_cast<size_t>(object.parent)]) {
                doomed[i] = true;
                spreading = true;
            }
        }
    }

    std::vector<int32_t> remap(objects.size(), -1);
    std::vector<Object> kept;
    kept.reserve(objects.size());
    for (size_t i = 0; i < objects.size(); ++i) {
        if (doomed[i]) {
            continue;
        }
        remap[i] = static_cast<int32_t>(kept.size());
        kept.push_back(std::move(objects[i]));
    }
    for (Object& object : kept) {
        if (object.parent >= 0) {
            object.parent = remap[static_cast<size_t>(object.parent)];
        }
    }
    objects = std::move(kept);
}

uint32_t Scene::duplicateObject(uint32_t index) {
    std::vector<int32_t> remap(objects.size(), -1);
    auto original = static_cast<uint32_t>(objects.size());

    // 원본 순서를 지키며 복사하고, 부모가 복사 범위 안이면 새 인덱스로 옮긴다.
    for (uint32_t i = 0; i < original; ++i) {
        if (i != index && !isDescendant(i, index)) {
            continue;
        }
        remap[i] = static_cast<int32_t>(objects.size());
        Object copy = objects[i];
        if (i == index) {
            copy.name += " (복사)";
        }
        objects.push_back(std::move(copy));
    }
    for (uint32_t i = 0; i < original; ++i) {
        if (remap[i] < 0) {
            continue;
        }
        Object& copy = objects[static_cast<size_t>(remap[i])];
        if (copy.parent >= 0 && remap[static_cast<size_t>(copy.parent)] >= 0) {
            copy.parent = remap[static_cast<size_t>(copy.parent)];
        }
    }
    return static_cast<uint32_t>(remap[index]);
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
