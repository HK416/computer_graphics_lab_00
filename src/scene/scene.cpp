#include "scene/scene.h"

#include <cmath>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/matrix_decompose.hpp>

#include "core/error.h"
#include "core/job_system.h"

namespace scene {

namespace {

// 부품 종류 수. refresh 가 만드는 배치표의 폭이며 Object 의 부품 첨자 개수와 같아야 한다.
// 부품을 한 종류 더 넣으면 여기와 refresh 의 배치표 채우기도 함께 고쳐야 한다.
constexpr size_t COMPONENT_KINDS = 5;

// 살아남은 오브젝트가 하나도 가리키지 않는 부품을 버리고 첨자를 다시 맞춘다.
// 애니메이터처럼 여러 오브젝트가 함께 가리키는 부품도 있어 소유가 아니라 참조를 기준으로 센다.
template <typename T>
void compactComponents(std::vector<T>& items, std::vector<Object>& objects, int32_t Object::* handle) {
    std::vector<uint8_t> referenced(items.size(), 0);
    for (const Object& object : objects) {
        int32_t slot = object.*handle;
        if (slot >= 0 && static_cast<size_t>(slot) < items.size()) {
            referenced[static_cast<size_t>(slot)] = 1;
        }
    }

    std::vector<int32_t> remap(items.size(), -1);
    std::vector<T> kept;
    kept.reserve(items.size());
    for (size_t i = 0; i < items.size(); ++i) {
        if (referenced[i] == 0) {
            continue;
        }
        remap[i] = static_cast<int32_t>(kept.size());
        kept.push_back(std::move(items[i]));
    }
    for (Object& object : objects) {
        int32_t& slot = object.*handle;
        slot = slot >= 0 && static_cast<size_t>(slot) < remap.size() ? remap[static_cast<size_t>(slot)] : -1;
    }
    items = std::move(kept);
}

// 복사본들이 가리키는 부품마다 사본을 하나씩 만들고 그것을 가리키게 한다. 같은 부품을 함께
// 가리키던 것들은 사본에서도 함께 가리킨다. 스킨 모델의 뿌리와 자식이 애니메이터 하나를 공유하는
// 관계가 복제 뒤에도 유지되어야 하기 때문이다.
template <typename T>
void duplicateComponents(std::vector<T>& items,
                         std::vector<Object>& objects,
                         const std::vector<int32_t>& objectRemap,
                         int32_t Object::* handle) {
    std::vector<int32_t> copyOf(items.size(), -1);
    for (size_t i = 0; i < objectRemap.size(); ++i) {
        if (objectRemap[i] < 0) {
            continue;
        }
        int32_t& slot = objects[static_cast<size_t>(objectRemap[i])].*handle;
        if (slot < 0 || static_cast<size_t>(slot) >= copyOf.size()) {
            continue;
        }
        if (copyOf[static_cast<size_t>(slot)] < 0) {
            copyOf[static_cast<size_t>(slot)] = static_cast<int32_t>(items.size());
            T copy = items[static_cast<size_t>(slot)];
            items.push_back(std::move(copy));
        }
        slot = copyOf[static_cast<size_t>(slot)];
    }
}

} // namespace

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

void Scene::update(float deltaSeconds, core::JobSystem* jobs) {
    animatorPosedFlags.assign(animators.size(), 0);
    // 애니메이터마다 자기 배열만 만지므로 서로 독립이다. 캐릭터가 여럿이면 워커에 나눈다.
    auto updateAnimator = [this, deltaSeconds](uint32_t index) {
        Animator& animator = animators[index];
        if (animator.skeleton.skins.empty()) {
            return;
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

        // 재생 중이 아니고 클립도 시각도 그대로면 같은 포즈가 다시 나온다. 노드 배열 복사까지
        // 통째로 건너뛴다.
        if (animator.clip == animator.posedClip && animator.clipTime == animator.posedTime) {
            return;
        }
        animator.posedClip = animator.clip;
        animator.posedTime = animator.clipTime;
        animatorPosedFlags[index] = 1;

        asset::poseNodes(animator.skeleton, animator.clip, animator.clipTime, animator.nodeWorlds);
        animator.jointMatrices.resize(animator.skeleton.skins.size());
        for (uint32_t skin = 0; skin < animator.skeleton.skins.size(); ++skin) {
            asset::skinMatrices(animator.skeleton, animator.nodeWorlds, skin, animator.jointMatrices[skin]);
        }
    };
    auto count = static_cast<uint32_t>(animators.size());
    if (jobs != nullptr && count > 1) {
        jobs->parallelFor(count, 1, [&updateAnimator](uint32_t begin, uint32_t end) {
            for (uint32_t index = begin; index < end; ++index) {
                updateAnimator(index);
            }
        });
    } else {
        for (uint32_t index = 0; index < count; ++index) {
            updateAnimator(index);
        }
    }
}

void Scene::resolveWorld(uint32_t index) {
    if (cachedResolved[index] != 0) {
        return;
    }
    // 순환 부모는 편집기가 막지만, 만에 하나 들어와도 재귀가 무한히 돌지 않도록 먼저 표시한다.
    cachedResolved[index] = 1;

    const Object& object = objects[index];
    glm::mat4 local = object.transform.matrix();
    if (object.parent < 0) {
        cachedWorlds[index] = local;
        cachedVisible[index] = object.visible ? 1 : 0;
        return;
    }

    auto parent = static_cast<uint32_t>(object.parent);
    resolveWorld(parent);
    cachedWorlds[index] = cachedWorlds[parent] * local;
    cachedVisible[index] = (object.visible && cachedVisible[parent] != 0) ? 1 : 0;
    // 부모가 움직였으면 자손의 세계 변환도 바뀐 것이다.
    cachedDirty[index] |= cachedDirty[parent];
}

void Scene::refresh() {
    size_t count = objects.size();
    // 지우고 같은 수만큼 새로 만들면 개수 비교로는 알 수 없다. 그때는 첨자마다 남의 지난 값을 보게
    // 되므로, 배열이 재배치되었다는 표식이 서 있으면 지난 사본을 통째로 버린다.
    bool structureReset = structureDirty;
    structureDirty = false;
    if (structureReset) {
        previousTransforms.clear();
        previousParents.clear();
        previousVisible.clear();
    }
    // 사본을 버리면 크기가 0 이 되므로, 남은 오브젝트도 0 이면 크기 비교로는 아무것도 잡히지 않는다.
    // 마지막 오브젝트를 지운 프레임이 그렇다.
    bool sizeChanged = structureReset || previousTransforms.size() != count;
    bool topologyChanged = sizeChanged;
    bool transformChanged = sizeChanged;

    previousTransforms.resize(count);
    previousParents.resize(count, -2);
    previousVisible.resize(count, 2);
    cachedWorlds.assign(count, glm::mat4{1.0F});
    cachedVisible.assign(count, 0);
    cachedResolved.assign(count, 0);
    cachedDirty.assign(count, sizeChanged ? 1 : 0);

    for (uint32_t index = 0; index < count; ++index) {
        const Object& object = objects[index];
        auto visible = static_cast<uint8_t>(object.visible ? 1 : 0);
        bool reparented = previousParents[index] != object.parent;
        bool posed = object.animator >= 0 && static_cast<size_t>(object.animator) < animatorPosedFlags.size() &&
                     animatorPosedFlags[object.animator] != 0;
        if (reparented || previousTransforms[index] != object.transform || previousVisible[index] != visible || posed) {
            cachedDirty[index] = 1;
            transformChanged = true;
        }
        topologyChanged = topologyChanged || reparented;

        previousTransforms[index] = object.transform;
        previousParents[index] = object.parent;
        previousVisible[index] = visible;
    }

    for (uint32_t index = 0; index < count; ++index) {
        resolveWorld(index);
    }

    bool lightsChanged = previousLights != lights;
    previousLights = lights;

    // 부품 배열의 «배치»만 담는다. 크기 다섯 개를 앞에 두고 오브젝트마다 부품 첨자 다섯 개를 잇는다.
    // 값(강체 속도 등)은 담지 않으므로 재생 중에는 변하지 않는다.
    componentLayout.clear();
    componentLayout.reserve(COMPONENT_KINDS + count * COMPONENT_KINDS);
    componentLayout.push_back(static_cast<int32_t>(meshRenderers.size()));
    componentLayout.push_back(static_cast<int32_t>(animators.size()));
    componentLayout.push_back(static_cast<int32_t>(lights.size()));
    componentLayout.push_back(static_cast<int32_t>(rigidBodies.size()));
    componentLayout.push_back(static_cast<int32_t>(fluids.size()));
    for (const Object& object : objects) {
        componentLayout.push_back(object.meshRenderer);
        componentLayout.push_back(object.animator);
        componentLayout.push_back(object.light);
        componentLayout.push_back(object.rigidBody);
        componentLayout.push_back(object.fluid);
    }
    // 배치표가 같아도 배열이 재배치되었을 수 있다. 부품을 떼고 같은 자리에 다시 붙이면 배치는
    // 그대로지만 부품은 다른 것이다.
    bool componentsChanged = structureReset || componentLayout != previousComponentLayout;
    // 맞바꾸면 프레임마다 새로 잡지 않고 두 벌의 용량을 번갈아 쓴다. 다음 프레임에 clear 로 채운다.
    previousComponentLayout.swap(componentLayout);

    if (transformChanged) {
        ++transformRev;
    }
    if (lightsChanged) {
        ++lightRev;
    }
    if (topologyChanged) {
        ++topologyRev;
    }
    if (componentsChanged) {
        ++componentRev;
    }
    if (transformChanged || lightsChanged || topologyChanged || componentsChanged) {
        ++anyRevision;
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

SceneSnapshot Scene::capture() const {
    SceneSnapshot snapshot;
    snapshot.name = name;
    snapshot.objects = objects;
    snapshot.meshRenderers = meshRenderers;
    snapshot.animators = animators;
    snapshot.lights = lights;
    snapshot.rigidBodies = rigidBodies;
    snapshot.fluids = fluids;
    snapshot.ambientColor = ambientColor;
    snapshot.ambientIntensity = ambientIntensity;
    snapshot.environment = environment;
    snapshot.post = post;
    return snapshot;
}

void Scene::restore(const SceneSnapshot& snapshot) {
    // public 필드만 되돌린다. previous* 캐시와 리비전까지 되돌리면 다음 refresh 가 "변한 게
    // 없다"고 판단해 세계 변환과 그림자 캐시가 낡은 채로 남는다.
    // 오브젝트 수가 같아도 다른 오브젝트가 그 자리에 올 수 있으므로 지난 사본은 버리게 표시한다.
    markStructureDirty();
    name = snapshot.name;
    objects = snapshot.objects;
    meshRenderers = snapshot.meshRenderers;
    animators = snapshot.animators;
    lights = snapshot.lights;
    rigidBodies = snapshot.rigidBodies;
    fluids = snapshot.fluids;
    ambientColor = snapshot.ambientColor;
    ambientIntensity = snapshot.ambientIntensity;
    environment = snapshot.environment;
    post = snapshot.post;
}

bool Scene::differsFrom(const SceneSnapshot& snapshot) const {
    if (name != snapshot.name || objects != snapshot.objects || meshRenderers != snapshot.meshRenderers ||
        lights != snapshot.lights || rigidBodies != snapshot.rigidBodies || fluids != snapshot.fluids ||
        ambientColor != snapshot.ambientColor || ambientIntensity != snapshot.ambientIntensity ||
        !(environment == snapshot.environment) || !(post == snapshot.post)) {
        return true;
    }
    if (animators.size() != snapshot.animators.size()) {
        return true;
    }
    for (size_t i = 0; i < animators.size(); ++i) {
        const Animator& current = animators[i];
        const Animator& saved = snapshot.animators[i];
        // clipTime 은 재생 중 매 프레임 흐르므로 뺀다. 그것 때문에 기록이 쌓이면 되돌리기가
        // 쓸모없어진다.
        if (current.name != saved.name || current.model != saved.model || current.clip != saved.clip ||
            current.playing != saved.playing || current.speed != saved.speed) {
            return true;
        }
    }
    return false;
}

int32_t Scene::attachMeshRenderer(uint32_t index, uint32_t mesh, int32_t skin) {
    Object& object = objects[index];
    if (object.meshRenderer < 0 || static_cast<size_t>(object.meshRenderer) >= meshRenderers.size()) {
        object.meshRenderer = static_cast<int32_t>(meshRenderers.size());
        meshRenderers.push_back(MeshRenderer{mesh, skin});
    } else {
        meshRenderers[static_cast<size_t>(object.meshRenderer)] = MeshRenderer{mesh, skin};
    }
    return object.meshRenderer;
}

uint32_t Scene::meshOf(uint32_t index) const {
    int32_t slot = objects[index].meshRenderer;
    if (slot < 0 || static_cast<size_t>(slot) >= meshRenderers.size()) {
        return INVALID_MESH;
    }
    return meshRenderers[static_cast<size_t>(slot)].mesh;
}

int32_t Scene::skinOf(uint32_t index) const {
    int32_t slot = objects[index].meshRenderer;
    if (slot < 0 || static_cast<size_t>(slot) >= meshRenderers.size()) {
        return -1;
    }
    return meshRenderers[static_cast<size_t>(slot)].skin;
}

namespace {
template <typename T>
int32_t attachComponent(
    std::vector<Object>& objects, std::vector<T>& items, uint32_t index, int32_t Object::* handle, const T& value) {
    int32_t& slot = objects[index].*handle;
    if (slot >= 0 && static_cast<size_t>(slot) < items.size()) {
        return slot;
    }
    slot = static_cast<int32_t>(items.size());
    items.push_back(value);
    return slot;
}
} // namespace

int32_t Scene::attachLight(uint32_t index, const Light& light) {
    return attachComponent(objects, lights, index, &Object::light, light);
}

int32_t Scene::attachRigidBody(uint32_t index, const RigidBody& body) {
    return attachComponent(objects, rigidBodies, index, &Object::rigidBody, body);
}

int32_t Scene::attachFluid(uint32_t index, const Fluid& fluid) {
    return attachComponent(objects, fluids, index, &Object::fluid, fluid);
}

void Scene::detachComponent(uint32_t index, int32_t Object::* handle) {
    markStructureDirty();
    objects[index].*handle = -1;
    // 어느 배열의 첨자인지는 handle 이 정한다. 종류마다 배열이 달라 전부 다시 압축한다. 부품 수가 적어 싸다.
    compactComponents(meshRenderers, objects, &Object::meshRenderer);
    compactComponents(animators, objects, &Object::animator);
    compactComponents(lights, objects, &Object::light);
    compactComponents(rigidBodies, objects, &Object::rigidBody);
    compactComponents(fluids, objects, &Object::fluid);
}

void Scene::removeObject(uint32_t index) {
    removeObjects(std::vector<uint32_t>{index});
}

void Scene::removeObjects(const std::vector<uint32_t>& indices) {
    if (indices.empty()) {
        return;
    }
    markStructureDirty();
    std::vector<bool> doomed(objects.size(), false);
    for (uint32_t index : indices) {
        if (index < objects.size()) {
            doomed[index] = true;
        }
    }
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

    // 아무도 가리키지 않게 된 부품은 함께 사라진다. 예전에는 남아 고아가 됐다.
    compactComponents(meshRenderers, objects, &Object::meshRenderer);
    compactComponents(animators, objects, &Object::animator);
    compactComponents(lights, objects, &Object::light);
    compactComponents(rigidBodies, objects, &Object::rigidBody);
    compactComponents(fluids, objects, &Object::fluid);
}

uint32_t Scene::duplicateObject(uint32_t index) {
    markStructureDirty();
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

    // 부품까지 복제해야 사본이 원본과 독립적으로 편집된다. 예전에는 첨자만 복사되어 조명 하나를
    // 둘이 나눠 쓰고, 복제한 캐릭터가 원본과 같은 클립·같은 시각으로 붙어 움직였다.
    //
    // ponytail: 애니메이터 사본은 스켈레톤과 애니메이션 커브까지 통째로 복사한다. 리그가 큰
    // 모델을 여러 벌 복제하면 눈에 띌 수 있다. 필요하면 스켈레톤을 공유 포인터로 돌리면 된다.
    duplicateComponents(meshRenderers, objects, remap, &Object::meshRenderer);
    duplicateComponents(animators, objects, remap, &Object::animator);
    duplicateComponents(lights, objects, remap, &Object::light);
    duplicateComponents(rigidBodies, objects, remap, &Object::rigidBody);
    duplicateComponents(fluids, objects, remap, &Object::fluid);
    return static_cast<uint32_t>(remap[index]);
}

Scene& SceneManager::create(std::string name) {
    auto scene = std::make_unique<Scene>();
    scene->name = std::move(name);
    scenes.push_back(std::move(scene));
    return *scenes.back();
}

void SceneManager::setActive(size_t index) {
    if (index >= scenes.size()) {
        core::fatal("존재하지 않는 장면으로 전환할 수 없습니다: {} / {}", index, scenes.size());
    }
    activeIndex = index;
}

} // namespace scene
