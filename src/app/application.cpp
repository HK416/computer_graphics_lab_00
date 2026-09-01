#include "app/application.h"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <vector>

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>

#include "asset/model.h"
#include "core/error.h"
#include "gfx/uploader.h"

namespace app {
namespace {

constexpr int DEFAULT_WINDOW_WIDTH = 1600;
constexpr int DEFAULT_WINDOW_HEIGHT = 900;
constexpr float NANOSECONDS_PER_SECOND = 1.0e9F;
// 캡처 전에 스왑체인 크기가 안정될 시간을 준다.
constexpr uint64_t SCREENSHOT_FRAME = 8;

// 장면 전체가 화면에 들어오도록 카메라를 뒤로 물린다.
void frameCamera(scene::Scene& scene, const gfx::GeometryStore& geometry) {
    if (scene.objects.empty()) {
        return;
    }
    glm::vec3 minimum{std::numeric_limits<float>::max()};
    glm::vec3 maximum{std::numeric_limits<float>::lowest()};
    for (const scene::Object& object : scene.objects) {
        glm::vec4 sphere = geometry.mesh(object.meshIndex).boundingSphere;
        glm::vec3 center = glm::vec3(object.transform.matrix() * glm::vec4{glm::vec3(sphere), 1.0F});
        float scale = std::max({object.transform.scale.x, object.transform.scale.y, object.transform.scale.z});
        float radius = sphere.w * scale;
        minimum = glm::min(minimum, center - radius);
        maximum = glm::max(maximum, center + radius);
    }

    glm::vec3 center = (minimum + maximum) * 0.5F;
    float radius = glm::length(maximum - minimum) * 0.5F;
    scene.camera.position = center + glm::vec3{0.0F, radius * 0.35F, -radius * 2.4F};
    scene.camera.yawDegrees = 90.0F;
    scene.camera.pitchDegrees = -8.0F;
    scene.camera.moveSpeed = std::max(radius, 0.5F);
}

} // namespace

Application::Application(const Options& options) : options(options) {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        core::fatal("SDL 초기화에 실패했습니다: {}", SDL_GetError());
    }

    window = SDL_CreateWindow("Computer Graphics Lab",
                              DEFAULT_WINDOW_WIDTH,
                              DEFAULT_WINDOW_HEIGHT,
                              SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (window == nullptr) {
        core::fatal("윈도우 생성에 실패했습니다: {}", SDL_GetError());
    }

    context = std::make_unique<gfx::Context>(window);
    bindless = std::make_unique<gfx::BindlessTextures>(*context);
    textures = std::make_unique<gfx::TextureCache>(*context, *bindless);
    geometry = std::make_unique<gfx::GeometryStore>(*context);
    loadScenes();
    renderer = std::make_unique<gfx::Renderer>(*context, *geometry, *bindless, window);
    editorUi = std::make_unique<editor::Editor>(*context, *renderer, window);
    renderer->debugMode = options.debugMode;
    if (options.lodLevel != AUTOMATIC_LOD) {
        renderer->automaticLod = false;
        renderer->lodLevel = options.lodLevel;
    }
    renderer->lodErrorThreshold = options.lodErrorThreshold;
    renderer->useNeuralLod = options.neuralLod;
    if (options.triangleBudget > 0.0F) {
        renderer->triangleBudget = options.triangleBudget;
    }
    renderer->setUiCallback([this](VkCommandBuffer commandBuffer) { editorUi->record(commandBuffer); });
}

Application::~Application() {
    // 렌더러가 쓰는 서피스는 윈도우보다 먼저 파괴되어야 한다.
    editorUi.reset();
    renderer.reset();
    geometry.reset();
    textures.reset();
    bindless.reset();
    context.reset();
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void Application::loadScenes() {
    std::filesystem::path assetRoot = std::filesystem::path(CG_LAB_ASSET_ROOT) / "assets";

    std::vector<std::filesystem::path> files;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(assetRoot, error)) {
        if (entry.is_regular_file() && (entry.path().extension() == ".glb" || entry.path().extension() == ".gltf")) {
            files.push_back(entry.path());
        }
    }
    if (files.empty()) {
        core::fatal("{} 에서 glTF 에셋을 찾지 못했습니다", assetRoot.string());
    }
    std::ranges::sort(files);

    gfx::Uploader uploader(*context);
    for (const std::filesystem::path& file : files) {
        asset::Model model = asset::loadGltf(file);

        std::vector<uint32_t> textureSlots;
        textureSlots.reserve(model.textures.size());
        for (const asset::Texture& texture : model.textures) {
            textureSlots.push_back(textures->add(uploader, texture));
        }
        uint32_t meshBase = geometry->addModel(model, textureSlots);

        scene::Scene& created = scenes.create(model.name);
        created.objects.reserve(model.instances.size());
        for (const asset::Instance& instance : model.instances) {
            scene::Object object;
            object.name = instance.name;
            object.meshIndex = meshBase + instance.meshIndex;
            object.transform = scene::Transform::fromMatrix(instance.transform);
            created.objects.push_back(std::move(object));
        }
    }

    uploader.flush();
    geometry->build();
    for (size_t i = 0; i < scenes.count(); ++i) {
        scenes.setActive(i);
        frameCamera(scenes.active(), *geometry);
    }
    scenes.setActive(std::min(options.initialScene, scenes.count() - 1));
    spdlog::info("장면 {}개 준비 완료, 현재 장면: {}", scenes.count(), scenes.active().name);
}

void Application::run() {
    bool running = true;
    uint64_t previousTicks = SDL_GetTicksNS();
    uint64_t frameCount = 0;

    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_EVENT_QUIT:
                running = false;
                break;
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                renderer->requestResize();
                break;
            case SDL_EVENT_KEY_DOWN:
                // 숫자 키로 장면을 전환한다. ImGui 가 들어오기 전까지의 임시 조작이다.
                if (event.key.key >= SDLK_1 && event.key.key < SDLK_1 + static_cast<int>(scenes.count())) {
                    scenes.setActive(static_cast<size_t>(event.key.key - SDLK_1));
                    spdlog::info("장면 전환: {}", scenes.active().name);
                } else if (event.key.key == SDLK_ESCAPE) {
                    running = false;
                }
                break;
            default:
                break;
            }
            editorUi->processEvent(event);
            // 카메라 조작은 장면 뷰 위에서 시작한 경우에만 받는다.
            if (editorUi->viewportHovered() || scenes.active().camera.isLooking()) {
                scenes.active().camera.handleEvent(event);
            }
        }

        uint64_t currentTicks = SDL_GetTicksNS();
        float deltaSeconds = static_cast<float>(currentTicks - previousTicks) / NANOSECONDS_PER_SECOND;
        previousTicks = currentTicks;

        if ((SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED) != 0) {
            SDL_Delay(10);
            continue;
        }

        scenes.active().camera.update(deltaSeconds);
        renderer->setRenderExtent(editorUi->desiredRenderExtent());
        editorUi->build(scenes, *geometry, deltaSeconds);
        renderer->drawFrame(scenes.active());
        ++frameCount;

        if (!options.screenshotPath.empty()) {
            if (frameCount == SCREENSHOT_FRAME) {
                renderer->requestCapture(options.screenshotPath);
            } else if (frameCount > SCREENSHOT_FRAME) {
                running = false;
            }
        }
    }
    renderer->waitIdle();
}

} // namespace app
