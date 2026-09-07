#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "scene/scene.h"

namespace scene {

// 장면 파일 형식의 판. 읽는 쪽이 모르는(더 새로운) 판을 만나면 거절하고, 옛 판은 빠진 키를 기본값으로 읽는다.
// 2: 강체·유체 부품.
// 3: Bloom 임계값·무릎·번짐. bloomIntensity 의 뜻이 «섞는 비율»에서 «더하는 세기»로 바뀌었다.
// 4: 힘 마당 부품(forceFields).
// 5: DDGI 볼륨 부품(ddgiVolumes).
// 6: 카메라 부품(cameraComponents)과 카메라 경로(cameraPaths).
// 7: 관절 부품(joints).
inline constexpr uint32_t SCENE_FILE_VERSION = 8;

// 저장할 때 전역 메쉬 인덱스를 (모델, 모델 안의 메쉬) 로 나누는 데 쓰는 표.
struct ModelTable {
    std::vector<std::filesystem::path> paths;
    std::vector<uint32_t> meshBase;
    std::vector<uint32_t> meshCount;
};

// 읽어 들인 장면. 메쉬와 스켈레톤은 아직 모델을 적재하기 전이라 모델 번호로만 남는다.
struct SceneFile {
    Scene scene;
    std::vector<std::filesystem::path> models;
    // scene.objects 와 같은 길이. 오브젝트가 쓰는 모델 번호이며 메쉬가 없으면 -1.
    std::vector<int32_t> objectModels;
    // scene.objects 와 같은 길이. 모델 안에서 몇 번째 메쉬인지.
    std::vector<uint32_t> objectLocalMeshes;
    // scene.animators 와 같은 길이. 스켈레톤을 가져올 모델 번호.
    std::vector<int32_t> animatorModels;
};

// 장면을 JSON 문자열로 만든다. 모델 경로는 root 기준 상대 경로로 적어 옮겨 다닐 수 있게 한다.
std::string writeScene(const Scene& scene, const ModelTable& models, const std::filesystem::path& root = {});
// JSON 문자열을 읽는다. 형식이 잘못되었으면 core::fatal 로 끝낸다.
SceneFile readScene(const std::string& text);

// 오브젝트 roots 와 그 자손만 담은 장면 문자열(프리팹·클립보드). 형식은 장면 파일과 같아 readScene 으로 읽고,
// 뿌리는 parent 가 -1 이다. 카메라·환경 같은 장면 설정도 따라가지만 appendScene 은 오브젝트와 부품만 옮긴다.
std::string writeSubtree(const Scene& scene,
                         const std::vector<uint32_t>& roots,
                         const ModelTable& models,
                         const std::filesystem::path& root = {});
// source 의 오브젝트와 부품을 target 뒤에 붙인다. 뿌리(parent -1)는 parent 아래로 가고 부품 첨자·관절 상대 번호는
// 밀린다. 메쉬 번호·스켈레톤은 source 에 이미 되꽂혀 있어야 한다. 첫 새 오브젝트 번호를 돌려준다.
uint32_t appendScene(Scene& target, const Scene& source, int32_t parent);

} // namespace scene
