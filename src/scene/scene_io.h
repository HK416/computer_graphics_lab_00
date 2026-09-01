#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "scene/scene.h"

namespace scene {

// 장면 파일 형식의 판. 읽는 쪽이 모르는 판을 만나면 거절한다.
inline constexpr uint32_t SCENE_FILE_VERSION = 1;

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

} // namespace scene
