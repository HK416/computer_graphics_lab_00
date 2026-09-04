#include <cassert>
#include <cstdio>

#include <glm/gtc/matrix_transform.hpp>

#include "scene/scene.h"

namespace {

// 뿌리 - 자식 - 손자로 이어지는 사슬. 각 단계가 X 축으로 1 씩 밀린다.
scene::Scene makeChain() {
    scene::Scene scene;
    for (int i = 0; i < 3; ++i) {
        scene::Object object;
        object.name = "노드" + std::to_string(i);
        object.parent = i - 1;
        object.transform.position = glm::vec3{1.0F, 0.0F, 0.0F};
        scene.objects.push_back(std::move(object));
    }
    return scene;
}

glm::vec3 origin(const scene::Scene& scene, uint32_t index) {
    return glm::vec3{scene.worldMatrix(index) * glm::vec4{0.0F, 0.0F, 0.0F, 1.0F}};
}

} // namespace

int main() {
    scene::Scene scene = makeChain();

    // 세계 변환은 부모를 거슬러 올라가며 누적된다.
    assert(origin(scene, 2).x == 3.0F && "손자는 부모들의 이동을 함께 받아야 한다");

    // 부모의 크기도 자식에게 전해진다.
    scene.objects[0].transform.scale = glm::vec3{2.0F};
    assert(origin(scene, 2).x == 5.0F && "부모의 크기가 자식 위치에 반영되어야 한다");
    scene.objects[0].transform.scale = glm::vec3{1.0F};

    // 조상을 숨기면 자손도 보이지 않는다.
    scene.objects[0].visible = false;
    assert(!scene.visibleInTree(2) && "숨긴 조상의 자손은 보이지 않아야 한다");
    scene.objects[0].visible = true;
    assert(scene.visibleInTree(2));

    // 자기 자손을 부모로 삼으려는 시도를 걸러내는 데 쓰는 판정.
    assert(scene.isDescendant(2, 0) && "손자는 뿌리의 자손이다");
    assert(!scene.isDescendant(0, 2) && "뿌리는 손자의 자손이 아니다");

    // 복제는 자손까지 따라오고 부모 관계가 새 인덱스로 옮겨진다.
    uint32_t copy = scene.duplicateObject(1);
    assert(scene.objects.size() == 5);
    assert(scene.objects[copy].parent == 0 && "복제본의 부모는 원본과 같아야 한다");
    assert(scene.objects[4].parent == static_cast<int32_t>(copy) && "복제된 손자는 복제된 부모를 가리켜야 한다");
    assert(origin(scene, 4).x == 3.0F);

    // 지우면 자손까지 사라지고 남은 부모 인덱스가 다시 맞춰진다. 원본 1, 2 만 사라지고 복제본은 남는다.
    scene.removeObject(1);
    assert(scene.objects.size() == 3 && "자식과 손자가 함께 지워져야 한다");
    assert(scene.objects[0].parent == -1);
    assert(scene.objects[1].parent == 0 && "남은 복제본이 여전히 뿌리를 가리켜야 한다");
    assert(scene.objects[2].parent == 1 && "복제본 사이의 부모 관계도 새 인덱스를 따라야 한다");
    assert(origin(scene, 2).x == 3.0F);

    // 부품은 오브젝트에 묶인다. 복제하면 사본이 제 부품을 갖고, 지우면 아무도 안 가리키는
    // 부품이 함께 사라진다. 예전에는 첨자만 복사되어 원본과 부품 하나를 나눠 썼다.
    scene::Scene parts = makeChain();
    parts.lights.push_back(scene::Light{});
    parts.objects[1].light = 0;
    parts.attachMeshRenderer(1, 7);

    uint32_t partsCopy = parts.duplicateObject(1);
    assert(parts.lights.size() == 2 && "복제본은 제 조명을 가져야 한다");
    assert(parts.objects[partsCopy].light == 1 && "복제본이 원본 조명을 가리키면 안 된다");
    assert(parts.meshOf(partsCopy) == 7 && "메쉬 부품도 함께 복제되어야 한다");
    parts.lights[static_cast<size_t>(parts.objects[partsCopy].light)].intensity = 99.0F;
    assert(parts.lights[0].intensity != 99.0F && "사본을 고쳐도 원본 조명은 그대로여야 한다");

    parts.removeObject(partsCopy);
    assert(parts.lights.size() == 1 && "가리키는 오브젝트가 사라지면 조명도 사라진다");
    assert(parts.objects[1].light == 0 && "남은 오브젝트의 조명 첨자는 새 자리를 가리켜야 한다");
    assert(parts.meshOf(1) == 7 && "남은 오브젝트의 메쉬 부품은 그대로여야 한다");

    // 강체·유체 부품도 같은 규칙을 따른다. 붙이면 첨자가 생기고, 떼면 배열에서 빠지며 남은 첨자가 당겨진다.
    {
        scene::Scene physics = makeChain();
        scene::RigidBody body;
        body.mass = 3.0F;
        assert(physics.attachRigidBody(0, body) == 0);
        assert(physics.attachRigidBody(2, scene::RigidBody{}) == 1);
        assert(physics.attachRigidBody(0, scene::RigidBody{}) == 0 && "이미 붙어 있으면 그 첨자를 돌려준다");
        assert(physics.rigidBodies.size() == 2 && physics.rigidBodies[0].mass == 3.0F);
        assert(physics.attachFluid(1) == 0 && physics.fluids.size() == 1);

        physics.detachComponent(0, &scene::Object::rigidBody);
        assert(physics.objects[0].rigidBody == -1);
        assert(physics.rigidBodies.size() == 1 && "아무도 가리키지 않는 강체는 빠진다");
        assert(physics.objects[2].rigidBody == 0 && "남은 강체의 첨자가 당겨진다");
        assert(physics.objects[1].fluid == 0 && "다른 종류의 부품은 그대로다");

        uint32_t physicsCopy = physics.duplicateObject(2);
        assert(physics.rigidBodies.size() == 2 && physics.objects[physicsCopy].rigidBody == 1);
        physics.removeObject(2);
        assert(physics.rigidBodies.size() == 1 && physics.objects[physicsCopy - 1].rigidBody == 0);

        scene::SceneSnapshot saved = physics.capture();
        physics.fluids[0].stiffness += 1.0F;
        assert(physics.differsFrom(saved) && "부품 값 변경은 되돌리기 대상이다");
        physics.restore(saved);
        assert(!physics.differsFrom(saved));
    }

    // 하나의 애니메이터를 뿌리와 자식이 함께 가리키는 관계는 복제 뒤에도 유지된다.
    scene::Scene rig = makeChain();
    rig.animators.push_back(scene::Animator{});
    rig.objects[0].animator = 0;
    rig.objects[1].animator = 0;
    rig.objects[2].animator = 0;
    uint32_t rigCopy = rig.duplicateObject(0);
    assert(rig.animators.size() == 2 && "복제된 사슬 전체가 애니메이터 하나를 새로 가져야 한다");
    assert(rig.objects[rigCopy].animator == 1);
    assert(rig.objects[rigCopy + 1].animator == 1 && "사본끼리는 계속 같은 애니메이터를 공유한다");
    assert(rig.objects[rigCopy + 2].animator == 1);

    // 되돌리기 스냅샷. public 필드만 되살리므로 복원 뒤에도 refresh 가 변경을 잡아야 한다.
    // Scene 을 통째로 대입하면 previous* 캐시까지 되돌아가 "변한 게 없다"고 판단해 버린다.
    {
        scene::Scene history = makeChain();
        history.refresh();
        scene::SceneSnapshot saved = history.capture();
        assert(!history.differsFrom(saved));

        history.objects[1].transform.position.x = 10.0F;
        assert(history.differsFrom(saved) && "변경은 스냅샷과의 차이로 잡혀야 한다");
        history.refresh();
        assert(origin(history, 2).x == 12.0F);

        uint64_t beforeRestore = history.transformRevision();
        history.restore(saved);
        history.refresh();
        assert(!history.differsFrom(saved) && "되살린 뒤에는 스냅샷과 같아야 한다");
        assert(history.transformRevision() != beforeRestore && "복원도 변경으로 잡혀야 한다");
        assert(origin(history, 2).x == 3.0F && "세계 변환 캐시가 복원된 값으로 다시 만들어져야 한다");
        assert(history.objectDirty(1) && "복원된 오브젝트는 더티로 표시되어야 한다");
    }

    // 재생 시각은 스냅샷 비교에서 빠진다. 안 그러면 재생 중에 기록이 프레임마다 쌓인다.
    {
        scene::Scene playing;
        scene::Object node;
        playing.objects.push_back(std::move(node));
        scene::Animator animator;
        animator.name = "재생";
        playing.animators.push_back(std::move(animator));
        playing.objects[0].animator = 0;

        scene::SceneSnapshot saved = playing.capture();
        playing.animators[0].clipTime = 1.25F;
        assert(!playing.differsFrom(saved) && "재생 시각만 흐른 것은 변경이 아니다");
        playing.animators[0].speed = 2.0F;
        assert(playing.differsFrom(saved) && "재생 속도는 편집이므로 변경이다");
    }

    // 부모가 자식보다 뒤에 있어도 세계 변환이 맞아야 한다.
    scene::Scene reversed;
    scene::Object child;
    child.parent = 1;
    child.transform.position = glm::vec3{0.0F, 1.0F, 0.0F};
    reversed.objects.push_back(child);
    scene::Object root;
    root.transform.position = glm::vec3{0.0F, 10.0F, 0.0F};
    reversed.objects.push_back(root);
    assert(origin(reversed, 0).y == 11.0F && "배열 순서와 무관하게 계층을 따라야 한다");

    // ---- 더티 검출과 세계 변환 캐시 ----
    scene::Scene tracked = makeChain();
    tracked.refresh();
    for (uint32_t i = 0; i < tracked.objects.size(); ++i) {
        assert(tracked.world(i) == tracked.worldMatrix(i) && "캐시가 즉시 계산과 같아야 한다");
        assert(tracked.visibleCached(i) == tracked.visibleInTree(i));
    }

    // 아무것도 안 바꾸면 리비전이 늘지 않는다.
    uint64_t revision = tracked.revision();
    tracked.refresh();
    assert(tracked.revision() == revision && "변한 것이 없으면 리비전도 그대로여야 한다");
    for (uint32_t i = 0; i < tracked.objects.size(); ++i) {
        assert(!tracked.objectDirty(i));
    }

    // mark 류 훅 없이 변환을 직접 대입해도 잡아내야 한다. 이 저장소의 편집기가 실제로 이렇게 쓴다.
    tracked.objects[1].transform.position.x = 5.0F;
    tracked.refresh();
    assert(tracked.revision() != revision && "직접 대입도 리비전을 올려야 한다");
    assert(tracked.objectDirty(1) && "직접 대입한 오브젝트가 더티여야 한다");
    assert(tracked.objectDirty(2) && "부모가 움직이면 자손도 더티여야 한다");
    assert(!tracked.objectDirty(0) && "움직이지 않은 조상까지 더티가 되면 안 된다");
    assert(tracked.world(2) == tracked.worldMatrix(2) && "캐시가 갱신되어야 한다");

    // 가시성만 바꿔도 잡아낸다.
    tracked.refresh();
    tracked.objects[0].visible = false;
    tracked.refresh();
    assert(tracked.objectDirty(0));
    assert(!tracked.visibleCached(2) && "숨긴 조상의 자손은 캐시에서도 안 보여야 한다");
    tracked.objects[0].visible = true;
    tracked.refresh();

    // 조명만 바꾸면 조명 리비전만 오른다.
    tracked.lights.push_back(scene::Light{});
    tracked.refresh();
    uint64_t lightRevision = tracked.lightRevision();
    uint64_t topologyRevision = tracked.topologyRevision();
    tracked.lights[0].intensity = 12.0F;
    tracked.refresh();
    assert(tracked.lightRevision() != lightRevision && "조명 변경은 조명 리비전을 올려야 한다");
    assert(tracked.topologyRevision() == topologyRevision && "조명 값만 바뀌면 구조는 그대로다");

    // 부모를 바꾸면 구조 리비전이 오른다.
    topologyRevision = tracked.topologyRevision();
    tracked.objects[2].parent = 0;
    tracked.refresh();
    assert(tracked.topologyRevision() != topologyRevision && "부모 변경은 구조 리비전을 올려야 한다");

    // 배열 순서와 무관하게 캐시가 맞아야 한다(부모가 뒤에 있는 경우).
    scene::Scene reversedCache;
    scene::Object back;
    back.parent = 1;
    back.transform.position = glm::vec3{0.0F, 1.0F, 0.0F};
    reversedCache.objects.push_back(back);
    scene::Object front;
    front.transform.position = glm::vec3{0.0F, 10.0F, 0.0F};
    reversedCache.objects.push_back(front);
    reversedCache.refresh();
    assert(reversedCache.world(0) == reversedCache.worldMatrix(0) && "부모가 뒤에 있어도 캐시가 맞아야 한다");

    // ---- 정지한 애니메이터는 포즈를 다시 만들지 않는다 ----
    scene::Scene animated;
    scene::Animator animator;
    animator.skeleton.nodes.resize(1);
    asset::Skin skin;
    skin.joints = {0};
    skin.inverseBind = {glm::mat4{1.0F}};
    animator.skeleton.skins.push_back(std::move(skin));
    animator.playing = false;
    animated.animators.push_back(std::move(animator));

    animated.update(0.016F);
    assert(animated.animatorPosed(0) && "첫 갱신은 바인드 포즈를 만들어야 한다");
    animated.update(0.016F);
    assert(!animated.animatorPosed(0) && "정지한 애니메이터는 다시 포즈하지 않아야 한다");
    animated.animators[0].clipTime = 0.25F;
    animated.update(0.016F);
    assert(animated.animatorPosed(0) && "시각이 바뀌면 다시 포즈해야 한다");

    // ---- 부품 배치 리비전 ----
    scene::Scene attached;
    attached.objects.push_back(scene::Object{});
    attached.objects.push_back(scene::Object{});
    attached.refresh();

    uint64_t componentBase = attached.componentRevision();
    attached.attachFluid(1);
    attached.refresh();
    assert(attached.componentRevision() != componentBase && "부품을 붙이면 배치 리비전이 올라야 한다");

    componentBase = attached.componentRevision();
    attached.refresh();
    assert(attached.componentRevision() == componentBase && "가만히 두면 배치 리비전이 그대로여야 한다");

    // 재생 중 강체 속도가 변하는 것은 «배치»가 아니다.
    attached.attachRigidBody(0);
    attached.refresh();
    componentBase = attached.componentRevision();
    uint64_t transformBase = attached.transformRevision();
    attached.rigidBodies[0].velocity = glm::vec3{0.0F, -3.0F, 0.0F};
    attached.refresh();
    assert(attached.componentRevision() == componentBase && "강체 속도만 바뀌면 배치 리비전은 그대로여야 한다");
    assert(attached.transformRevision() == transformBase && "값만 바뀌면 변환 리비전도 그대로여야 한다");

    // ---- 다른 종류를 떼도 유체 첨자가 정합해야 한다 ----
    // detachComponent 는 다섯 배열을 모두 압축한다. 강체를 떼면 유체 첨자가 밀릴 수 있다.
    scene::Scene mixed;
    for (int i = 0; i < 3; ++i) {
        mixed.objects.push_back(scene::Object{});
    }
    mixed.attachRigidBody(0);
    mixed.attachFluid(1);
    mixed.attachFluid(2);
    mixed.refresh();
    componentBase = mixed.componentRevision();
    mixed.detachComponent(0, &scene::Object::rigidBody);
    mixed.refresh();
    assert(mixed.rigidBodies.empty() && "가리키는 오브젝트가 없어진 강체는 사라져야 한다");
    assert(mixed.fluids.size() == 2 && "유체는 그대로 둘이어야 한다");
    assert(mixed.objects[1].fluid >= 0 && static_cast<size_t>(mixed.objects[1].fluid) < mixed.fluids.size() &&
           "유체 첨자가 배열 안이어야 한다");
    assert(mixed.objects[2].fluid >= 0 && static_cast<size_t>(mixed.objects[2].fluid) < mixed.fluids.size() &&
           "유체 첨자가 배열 안이어야 한다");
    assert(mixed.objects[1].fluid != mixed.objects[2].fluid && "두 유체가 같은 부품을 가리키면 안 된다");
    assert(mixed.componentRevision() != componentBase && "부품을 떼면 배치 리비전이 올라야 한다");

    // ---- 지우고 같은 수만큼 새로 만들어도 알아채야 한다 ----
    scene::Scene swapped;
    scene::Object first;
    first.transform.position = glm::vec3{1.0F, 0.0F, 0.0F};
    swapped.objects.push_back(first);
    swapped.refresh();
    uint64_t topologyBase = swapped.topologyRevision();
    swapped.removeObject(0);
    scene::Object replacement;
    replacement.transform.position = glm::vec3{1.0F, 0.0F, 0.0F};
    swapped.objects.push_back(replacement);
    swapped.refresh();
    assert(swapped.topologyRevision() != topologyBase && "같은 프레임에 지우고 만들어도 위상이 바뀐 것이다");
    assert(swapped.objectDirty(0) && "새 오브젝트는 더티여야 한다");

    // 마지막 하나까지 지우면 개수가 0 이라 «크기가 달라졌다»로도 잡히지 않는다. 미사용 모델 회수가
    // 위상 리비전을 보므로 여기서 놓치면 모델이 GPU 에 남는다.
    topologyBase = swapped.topologyRevision();
    swapped.removeObject(0);
    swapped.refresh();
    assert(swapped.objects.empty());
    assert(swapped.topologyRevision() != topologyBase && "모두 지운 것도 위상 변화다");

    // ---- 장면을 더 만들어도 이미 잡아 둔 참조가 살아 있어야 한다 ----
    scene::SceneManager manager;
    scene::Scene& kept = manager.create("첫 장면");
    kept.ambientIntensity = 0.5F;
    for (int i = 0; i < 8; ++i) {
        manager.create("추가 장면");
    }
    assert(&kept == &manager.at(0) && "장면을 더 만들어도 주소가 바뀌면 안 된다");
    assert(manager.at(0).ambientIntensity == 0.5F && "잡아 둔 참조로 쓴 값이 살아 있어야 한다");

    std::printf("장면 계층 자체 점검 통과\n");
    return 0;
}
