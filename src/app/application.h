#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "asset/model.h"
#include "core/job_system.h"
#include "editor/editor.h"
#include "gfx/bindless.h"
#include "gfx/context.h"
#include "gfx/geometry.h"
#include "gfx/renderer.h"
#include "gfx/texture.h"
#include "scene/scene.h"

struct SDL_Window;

namespace app {

inline constexpr uint32_t AUTOMATIC_LOD = 0xFFFFFFFFU;

struct Options {
    // 지정하면 몇 프레임 뒤에 화면을 PNG 로 저장하고 종료한다. 렌더 결과 검증용이다.
    std::filesystem::path screenshotPath;
    // 몇 프레임째를 저장할지. 시간축 업스케일처럼 여러 프레임을 쌓는 기능은 뒤쪽을 봐야 한다.
    uint64_t screenshotFrame = 8;
    size_t initialScene = 0;
    // 지정하면 시작할 때 이 장면 파일을 연다.
    std::filesystem::path scenePath;
    // 시작할 때 불러올 glTF 모델들. 편집기의 "모델" 단추와 같은 경로를 탄다.
    std::vector<std::filesystem::path> modelPaths;
    // shaders/scene_data.glsl 의 DEBUG_MODE_* 값.
    uint32_t debugMode = 0;
    // 0 이면 하드웨어 동시성에 맞춰 정한다.
    unsigned threadCount = 0;
    // AUTOMATIC_LOD 면 오차 기반 자동 선정을 쓴다.
    uint32_t lodLevel = AUTOMATIC_LOD;
    float lodErrorThreshold = 1.0F;
    bool neuralLod = false;
    // 시작하자마자 구간 계측을 켠다. 스크린샷으로 확인할 때 쓴다.
    bool profile = false;
    float renderScale = 1.0F;
    // 0 통과, 1 내장 공간 업스케일
    uint32_t upscaler = 1;
    // 시작할 때 경로 추적을 켠다. 하드웨어가 지원하지 않으면 사유를 남기고 무시한다.
    bool pathTracing = false;
    float triangleBudget = 0.0F;
};

class Application {
public:
    explicit Application(const Options& options);
    ~Application();
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;

    void run();

private:
    // 한 번 적재한 모델. 같은 파일을 두 번 올리지 않고, 장면 파일이 가리킬 대상이 된다.
    struct LoadedModel {
        std::filesystem::path path;
        uint32_t meshBase = 0;
        uint32_t meshCount = 0;
        asset::Skeleton skeleton;
        std::vector<asset::Instance> instances;
    };

    void loadScenes();
    // 이미 해석해 둔 모델을 GPU 에 올리고 등록 번호를 돌려준다. 지오메트리 재구축은 부르는 쪽 몫이다.
    uint32_t registerModel(const std::filesystem::path& path, asset::Model& model);
    // 모델을 아직 올리지 않았으면 해석해서 올리고 등록 번호를 돌려준다.
    uint32_t ensureModel(const std::filesystem::path& path);
    // 등록된 모델로 장면에 뿌리 오브젝트와 자식들을 만든다.
    void instantiateModel(uint32_t modelIndex, scene::Scene& scene);
    // 편집기가 부르는 런타임 적재. 지오메트리 버퍼를 다시 만들고 활성 장면에 붙인다.
    void loadModel(const std::filesystem::path& path);
    // 장면을 커스텀 JSON 으로 저장하고 읽는다. 읽은 장면은 새 장면으로 추가한 뒤 전환한다.
    void saveScene(const std::filesystem::path& path);
    void openScene(const std::filesystem::path& path);

    SDL_Window* window = nullptr;
    std::unique_ptr<gfx::Context> context;
    std::unique_ptr<gfx::BindlessTextures> bindless;
    std::unique_ptr<gfx::TextureCache> textures;
    std::unique_ptr<gfx::GeometryStore> geometry;
    std::unique_ptr<gfx::Renderer> renderer;
    std::unique_ptr<editor::Editor> editorUi;
    core::JobSystem jobs;
    scene::SceneManager scenes;
    std::filesystem::path assetRoot;
    std::filesystem::path sceneRoot;
    std::vector<LoadedModel> loadedModels;
    Options options;
};

} // namespace app
