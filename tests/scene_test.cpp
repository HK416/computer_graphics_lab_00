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

    std::printf("장면 계층 자체 점검 통과\n");
    return 0;
}
