#include <cassert>
#include <cmath>
#include <cstdio>

#include "scene/scene_io.h"

namespace {

// 모델 둘을 올린 상태를 흉내 낸다. 전역 메쉬 0~1 은 첫 모델, 2~4 는 두 번째 모델이다.
scene::ModelTable makeTable() {
    scene::ModelTable table;
    table.paths = {"/에셋/여우.glb", "/에셋/헬멧.glb"};
    table.meshBase = {0, 2};
    table.meshCount = {2, 3};
    return table;
}

scene::Scene makeScene() {
    scene::Scene scene;
    scene.name = "시험 장면";
    scene.camera.mode = scene::CameraMode::FLY;
    scene.camera.target = glm::vec3{1.0F, 3.0F, -2.0F};
    scene.camera.distance = 12.5F;
    scene.ambientColor = glm::vec3{0.1F, 0.2F, 0.3F};
    scene.ambientIntensity = 0.75F;
    scene.environment.useHdr = true;
    scene.environment.hdrPath = "sky/studio.hdr";
    scene.environment.yawDegrees = 37.5F;
    scene.environment.intensity = 2.25F;
    scene.environment.zenithColor = glm::vec3{0.05F, 0.06F, 0.07F};
    scene.camera.position = glm::vec3{1.0F, 2.0F, 3.0F};
    scene.camera.yawDegrees = 45.0F;

    scene::Light spot;
    spot.type = scene::LightType::SPOT;
    spot.color = glm::vec3{1.0F, 0.5F, 0.25F};
    spot.intensity = 7.5F;
    spot.range = 33.0F;
    spot.outerConeDegrees = 41.0F;
    spot.castsShadow = false;
    scene.lights.push_back(spot);

    scene::Animator animator;
    animator.name = "여우";
    animator.model = 0;
    animator.clip = 2;
    animator.clipTime = 0.5F;
    animator.speed = 1.5F;
    scene.animators.push_back(std::move(animator));

    scene::Object root;
    root.name = "여우";
    root.animator = 0;
    scene.objects.push_back(std::move(root));

    scene::Object child;
    child.name = "몸통";
    child.parent = 0;
    child.animator = 0;
    child.transform.position = glm::vec3{0.0F, 1.0F, 0.0F};
    child.transform.scale = glm::vec3{2.0F};
    scene.objects.push_back(std::move(child));
    // 첫 모델의 두 번째 메쉬, 스킨 0.
    scene.attachMeshRenderer(1, 1, 0);

    scene::Object helmet;
    helmet.name = "헬멧";
    helmet.visible = false;
    scene.objects.push_back(std::move(helmet));
    // 두 번째 모델의 두 번째 메쉬.
    scene.attachMeshRenderer(2, 3);

    scene::Object light;
    light.name = "스폿광";
    light.light = 0;
    scene.objects.push_back(std::move(light));
    return scene;
}

} // namespace

int main() {
    scene::ModelTable table = makeTable();
    scene::Scene original = makeScene();

    std::string text = scene::writeScene(original, table);
    scene::SceneFile loaded = scene::readScene(text);

    assert(loaded.scene.name == original.name);
    assert(loaded.models.size() == 2 && "모델 목록이 그대로 실려야 한다");
    assert(loaded.scene.objects.size() == original.objects.size());
    assert(loaded.scene.lights.size() == 1);
    assert(loaded.scene.animators.size() == 1);

    // 계층과 변환.
    assert(loaded.scene.objects[1].parent == 0);
    assert(loaded.scene.objects[1].transform.position.y == 1.0F);
    assert(loaded.scene.objects[1].transform.scale.x == 2.0F);
    assert(!loaded.scene.objects[2].visible && "숨김 상태가 남아야 한다");

    // 메쉬는 (모델, 모델 안의 번호) 로 나뉘어 저장된다.
    assert(loaded.objectModels[0] == -1 && "메쉬가 없는 뿌리는 모델을 가리키지 않는다");
    assert(loaded.objectModels[1] == 0 && loaded.objectLocalMeshes[1] == 1);
    assert(loaded.objectModels[2] == 1 && loaded.objectLocalMeshes[2] == 1);

    // 조명과 애니메이터.
    const scene::Light& light = loaded.scene.lights[0];
    assert(light.type == scene::LightType::SPOT);
    assert(std::abs(light.intensity - 7.5F) < 1e-5F);
    assert(std::abs(light.range - 33.0F) < 1e-5F);
    assert(std::abs(light.outerConeDegrees - 41.0F) < 1e-5F);
    assert(!light.castsShadow);
    assert(loaded.scene.objects[3].light == 0);
    assert(loaded.scene.skinOf(1) == 0 && "스킨 번호가 메쉬 부품에 실려 돌아와야 한다");

    assert(loaded.animatorModels[0] == 0 && "애니메이터는 어느 모델에서 왔는지 기억해야 한다");
    assert(loaded.scene.animators[0].clip == 2);
    assert(std::abs(loaded.scene.animators[0].speed - 1.5F) < 1e-5F);

    assert(std::abs(loaded.scene.ambientIntensity - 0.75F) < 1e-5F);

    // 환경 설정도 장면과 함께 저장되어야 한다. 안 그러면 장면을 다시 열 때 조명이 달라진다.
    const scene::Environment& environment = loaded.scene.environment;
    assert(environment.useHdr);
    assert(environment.hdrPath == std::filesystem::path{"sky/studio.hdr"});
    assert(std::abs(environment.yawDegrees - 37.5F) < 1e-5F);
    assert(std::abs(environment.intensity - 2.25F) < 1e-5F);
    assert(std::abs(environment.zenithColor.b - 0.07F) < 1e-5F);
    assert(environment.horizonColor == scene::Environment{}.horizonColor && "적지 않은 값은 기본값이어야 한다");
    assert(std::abs(loaded.scene.camera.yawDegrees - 45.0F) < 1e-5F);

    // 카메라 조작 방식도 장면과 함께 남아야 한다. 궤도 중심을 잃으면 다시 열 때 시점이 튄다.
    assert(loaded.scene.camera.mode == scene::CameraMode::FLY);
    assert(std::abs(loaded.scene.camera.distance - 12.5F) < 1e-5F);
    assert(std::abs(loaded.scene.camera.target.y - 3.0F) < 1e-5F);

    // 한 번 더 돌려도 같은 문자열이어야 한다.
    scene::Scene rebuilt = loaded.scene;
    // 메쉬 번호만 되꽂는다. 스킨은 이미 부품에 실려 왔으므로 그대로 둔다.
    rebuilt.attachMeshRenderer(1, 1, rebuilt.skinOf(1));
    rebuilt.attachMeshRenderer(2, 3, rebuilt.skinOf(2));
    assert(scene::writeScene(rebuilt, table) == text && "왕복해도 같은 파일이 나와야 한다");

    std::printf("장면 저장/불러오기 자체 점검 통과\n");
    return 0;
}
