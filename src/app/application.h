#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>

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
    // AUTOMATIC_LOD 면 오차 기반 자동 선정을 쓴다.
    uint32_t lodLevel = AUTOMATIC_LOD;
    float lodErrorThreshold = 1.0F;
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

    SDL_Window* window = nullptr;
    std::unique_ptr<gfx::Context> context;
    std::unique_ptr<gfx::BindlessTextures> bindless;
    std::unique_ptr<gfx::TextureCache> textures;
    std::unique_ptr<gfx::GeometryStore> geometry;
    std::unique_ptr<gfx::Renderer> renderer;
    std::unique_ptr<editor::Editor> editorUi;
    scene::SceneManager scenes;
    Options options;
};

} // namespace app
