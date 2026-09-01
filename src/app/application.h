#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>

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
    size_t initialScene = 0;
    // shaders/scene_data.glsl 의 DEBUG_MODE_* 값.
    uint32_t debugMode = 0;
    // 0 이면 하드웨어 동시성에 맞춰 정한다.
    unsigned threadCount = 0;
    // AUTOMATIC_LOD 면 오차 기반 자동 선정을 쓴다.
    uint32_t lodLevel = AUTOMATIC_LOD;
    float lodErrorThreshold = 1.0F;
    bool neuralLod = false;
    float renderScale = 1.0F;
    // 0 통과, 1 내장 공간 업스케일
    uint32_t upscaler = 1;
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
    void loadScenes();
    // 모델 하나를 지오메트리 저장소에 넣고 장면에 뿌리 오브젝트와 자식들을 만든다.
    void addModelToScene(asset::Model& model, scene::Scene& scene);
    // 편집기가 부르는 런타임 적재. 지오메트리 버퍼를 다시 만들고 활성 장면에 붙인다.
    void loadModel(const std::filesystem::path& path);

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
    Options options;
};

} // namespace app
