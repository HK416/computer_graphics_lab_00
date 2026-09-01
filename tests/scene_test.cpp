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

    std::printf("장면 계층 자체 점검 통과\n");
    return 0;
}
