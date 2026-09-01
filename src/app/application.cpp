#include "app/application.h"

#include <algorithm>
#include <cmath>
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
    glm::vec3 minimum{std::numeric_limits<float>::max()};
    glm::vec3 maximum{std::numeric_limits<float>::lowest()};
    bool found = false;
    for (uint32_t i = 0; i < scene.objects.size(); ++i) {
        const scene::Object& object = scene.objects[i];
        if (object.meshIndex >= geometry.meshCount()) {
            continue;
        }
        glm::mat4 world = scene.worldMatrix(i);
        glm::vec4 sphere = geometry.mesh(object.meshIndex).boundingSphere;
        glm::vec3 center = glm::vec3(world * glm::vec4{glm::vec3(sphere), 1.0F});
        // 비균등 스케일은 가장 긴 축으로 보수적으로 잡는다.
        float scale = std::sqrt(std::max({glm::dot(glm::vec3(world[0]), glm::vec3(world[0])),
                                          glm::dot(glm::vec3(world[1]), glm::vec3(world[1])),
                                          glm::dot(glm::vec3(world[2]), glm::vec3(world[2]))}));
        float radius = sphere.w * scale;
        minimum = glm::min(minimum, center - radius);
        maximum = glm::max(maximum, center + radius);
        found = true;
    }
    if (!found) {
        return;
    }

    glm::vec3 center = (minimum + maximum) * 0.5F;
    float radius = glm::length(maximum - minimum) * 0.5F;
    scene.camera.position = center + glm::vec3{0.0F, radius * 0.35F, -radius * 2.4F};
    scene.camera.yawDegrees = 90.0F;
    scene.camera.pitchDegrees = -8.0F;
    scene.camera.moveSpeed = std::max(radius, 0.5F);
}

} // namespace

Application::Application(const Options& options) : jobs(options.threadCount), options(options) {
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
    editorUi->workerCount = jobs.workerCount();
    renderer->debugMode = options.debugMode;
    if (options.lodLevel != AUTOMATIC_LOD) {
        renderer->automaticLod = false;
        renderer->lodLevel = options.lodLevel;
    }
    renderer->lodErrorThreshold = options.lodErrorThreshold;
    renderer->useNeuralLod = options.neuralLod;
    renderer->renderScale = options.renderScale;
    renderer->upscaler = static_cast<gfx::Upscaler>(options.upscaler);
    if (options.triangleBudget > 0.0F) {
        renderer->triangleBudget = options.triangleBudget;
    }
    renderer->setUiCallback([this](VkCommandBuffer commandBuffer) { editorUi->record(commandBuffer); });
    editorUi->setModelLoader(assetRoot, [this](const std::filesystem::path& path) { loadModel(path); });
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
    assetRoot = std::filesystem::path(CG_LAB_ASSET_ROOT) / "assets";

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

    // glTF 파싱과 LOD 계층 구성은 서로 독립이므로 워커에 흩뿌린다. 적재 시간의 대부분이 여기다.
    uint64_t loadStart = SDL_GetTicksNS();
    std::vector<asset::Model> models(files.size());
    jobs.parallelFor(static_cast<uint32_t>(files.size()), 1, [&files, &models](uint32_t begin, uint32_t end) {
        for (uint32_t i = begin; i < end; ++i) {
            models[i] = asset::loadGltf(files[i]);
        }
    });

    // 텍스처 디코딩과 LOD 구성 모두 모델 경계를 넘어 하나의 목록으로 펼쳐야 워커에 고르게 퍼진다.
    std::vector<asset::Texture*> allTextures;
    for (asset::Model& model : models) {
        for (asset::Texture& texture : model.textures) {
            allTextures.push_back(&texture);
        }
    }
    jobs.parallelFor(static_cast<uint32_t>(allTextures.size()), 1, [&allTextures](uint32_t begin, uint32_t end) {
        for (uint32_t i = begin; i < end; ++i) {
            asset::decodeTexture(*allTextures[i]);
        }
    });

    std::vector<asset::Mesh*> allMeshes;
    for (asset::Model& model : models) {
        for (asset::Mesh& mesh : model.meshes) {
            allMeshes.push_back(&mesh);
        }
    }
    // 메쉬가 적으면 메쉬 단위 분배만으로는 워커가 놀기 때문에 계층 구성 안쪽까지 나눈다.
    if (allMeshes.size() >= jobs.workerCount()) {
        jobs.parallelFor(static_cast<uint32_t>(allMeshes.size()), 1, [&allMeshes](uint32_t begin, uint32_t end) {
            for (uint32_t i = begin; i < end; ++i) {
                asset::buildLodHierarchy(*allMeshes[i]);
            }
        });
    } else {
        for (asset::Mesh* mesh : allMeshes) {
            asset::buildLodHierarchy(*mesh, &jobs);
        }
    }

    size_t meshletTotal = 0;
    size_t maxLodLevels = 0;
    for (const asset::Mesh* mesh : allMeshes) {
        meshletTotal += mesh->meshlets.size();
        maxLodLevels = std::max(maxLodLevels, mesh->lods.size());
    }
    spdlog::info("에셋 적재 완료: 메쉬 {}, meshlet {}, LOD {}단계, {:.1f} ms, 워커 {}",
                 allMeshes.size(),
                 meshletTotal,
                 maxLodLevels,
                 static_cast<double>(SDL_GetTicksNS() - loadStart) / 1.0e6,
                 jobs.workerCount());

    // GPU 자원 생성은 순서를 지켜 한 스레드에서만 한다.
    for (asset::Model& model : models) {
        scene::Scene& created = scenes.create(model.name);
        addModelToScene(model, created);
    }
    geometry->build();
    for (size_t i = 0; i < scenes.count(); ++i) {
        scenes.setActive(i);
        frameCamera(scenes.active(), *geometry);
    }
    scenes.setActive(std::min(options.initialScene, scenes.count() - 1));
    spdlog::info("장면 {}개 준비 완료, 현재 장면: {}", scenes.count(), scenes.active().name);
}

void Application::addModelToScene(asset::Model& model, scene::Scene& scene) {
    gfx::Uploader uploader(*context);
    std::vector<uint32_t> textureSlots;
    textureSlots.reserve(model.textures.size());
    for (const asset::Texture& texture : model.textures) {
        textureSlots.push_back(textures->add(uploader, texture));
    }
    uint32_t meshBase = geometry->addModel(model, textureSlots);
    uploader.flush();

    // 모델 하나가 계층의 뿌리 하나를 이룬다. 기즈모로 뿌리를 옮기면 자식이 함께 따라간다.
    auto root = static_cast<int32_t>(scene.objects.size());
    scene::Object rootObject;
    rootObject.name = model.name;
    scene.objects.push_back(std::move(rootObject));

    int32_t animator = -1;
    if (!model.skeleton.skins.empty()) {
        animator = static_cast<int32_t>(scene.animators.size());
        scene::Animator created;
        created.name = model.name;
        created.skeleton = std::move(model.skeleton);
        scene.animators.push_back(std::move(created));
        // 애니메이션 컨트롤러는 Unity 처럼 뿌리 오브젝트에 붙인다.
        scene.objects[static_cast<size_t>(root)].animator = animator;
    }

    for (const asset::Instance& instance : model.instances) {
        scene::Object object;
        object.name = instance.name;
        object.parent = root;
        object.meshIndex = meshBase + instance.meshIndex;
        object.transform = scene::Transform::fromMatrix(instance.transform);
        if (instance.skin >= 0) {
            object.animator = animator;
            object.skin = instance.skin;
        }
        scene.objects.push_back(std::move(object));
    }

    // 바인드 포즈라도 조인트 행렬이 있어야 스킨 메쉬가 제자리에 선다.
    scene.update(0.0F);
}

void Application::loadModel(const std::filesystem::path& path) {
    // 지오메트리 버퍼를 통째로 다시 만들기 때문에 진행 중인 프레임이 끝난 뒤에 손대야 한다.
    renderer->waitIdle();

    uint64_t start = SDL_GetTicksNS();
    asset::Model model = asset::loadGltf(path);
    jobs.parallelFor(static_cast<uint32_t>(model.textures.size()), 1, [&model](uint32_t begin, uint32_t end) {
        for (uint32_t i = begin; i < end; ++i) {
            asset::decodeTexture(model.textures[i]);
        }
    });
    for (asset::Mesh& mesh : model.meshes) {
        asset::buildLodHierarchy(mesh, &jobs);
    }

    addModelToScene(model, scenes.active());
    geometry->build();
    renderer->onGeometryChanged();
    spdlog::info(
        "모델 적재: {} ({:.1f} ms)", path.filename().string(), static_cast<double>(SDL_GetTicksNS() - start) / 1.0e6);
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
        // 애니메이션은 그리기 전에 진행시켜야 이번 프레임의 조인트 행렬이 올라간다.
        scenes.active().update(deltaSeconds);
        // 밀린 크기 변경은 UI 가 렌더 타겟을 참조하기 전에 끝내야 한다.
        renderer->prepareFrame();
        renderer->setDisplayExtent(editorUi->desiredRenderExtent());
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
