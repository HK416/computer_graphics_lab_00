#include "app/application.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <limits>
#include <optional>
#include <vector>

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>

#include "asset/model.h"
#include "asset/primitives.h"
#include "core/error.h"
#include "gfx/uploader.h"
#include "physics/rigid_body.h"
#include "scene/scene_io.h"

namespace app {
namespace {

constexpr int DEFAULT_WINDOW_WIDTH = 1600;
constexpr int DEFAULT_WINDOW_HEIGHT = 900;
constexpr float NANOSECONDS_PER_SECOND = 1.0e9F;
// 캡처 전에 스왑체인 크기가 안정될 시간을 준다.

// Unity 처럼 새 장면에는 방향광 하나를 기본으로 둔다.
void addDefaultLight(scene::Scene& scene) {
    scene::Light light;
    light.type = scene::LightType::DIRECTIONAL;
    scene.lights.push_back(light);

    scene::Object object;
    object.name = "방향광";
    // -Z 가 앞이므로 위에서 비스듬히 내려오도록 돌려 둔다.
    object.transform.rotation = glm::quat(glm::radians(glm::vec3{-50.0F, -30.0F, 0.0F}));
    object.light = static_cast<int32_t>(scene.lights.size()) - 1;
    scene.objects.push_back(std::move(object));
}

// 장면 전체가 화면에 들어오도록 카메라를 뒤로 물린다.
void frameCamera(scene::Scene& scene, const gfx::GeometryStore& geometry) {
    glm::vec3 minimum{std::numeric_limits<float>::max()};
    glm::vec3 maximum{std::numeric_limits<float>::lowest()};
    bool found = false;
    for (uint32_t i = 0; i < scene.objects.size(); ++i) {
        if (scene.meshOf(i) >= geometry.meshCount()) {
            continue;
        }
        glm::mat4 world = scene.worldMatrix(i);
        glm::vec4 sphere = geometry.mesh(scene.meshOf(i)).boundingSphere;
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
    scene.camera.yawDegrees = 90.0F;
    scene.camera.pitchDegrees = -8.0F;
    scene.camera.moveSpeed = std::max(radius, 0.5F);
    // 궤도 중심을 장면 한가운데로 잡는다. 자유 모드였다면 위치만 같은 자리로 옮긴다.
    scene.camera.focusOn(center, std::max(radius * 2.4F, 0.5F));
    if (scene.camera.mode != scene::CameraMode::ORBIT) {
        scene.camera.position = center - scene.camera.forward() * scene.camera.distance;
    }
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
    context->memoryBudgetOverride = options.gpuBudgetMegabytes * 1024ULL * 1024ULL;
    bindless = std::make_unique<gfx::BindlessTextures>(*context);
    textures = std::make_unique<gfx::TextureCache>(*context, *bindless);
    geometry = std::make_unique<gfx::GeometryStore>(*context);
    registerBuiltinModels();
    loadScenes();
    renderer = std::make_unique<gfx::Renderer>(*context, *geometry, *bindless, window, jobs);
    // 입자는 수만 개가 그려지므로 극에 삼각형이 몰리지 않는 정이십면체 구를 쓴다.
    renderer->fluidSphereMesh = primitiveMeshes[static_cast<size_t>(asset::Primitive::ICO_SPHERE)];
    editorUi = std::make_unique<editor::Editor>(*context, *renderer, window);
    editorUi->workerCount = jobs.workerCount();
    editorUi->primitiveMeshes = primitiveMeshes;
    renderer->debugMode = options.debugMode;
    renderer->showColliders = options.showColliders;
    if (options.lodLevel != AUTOMATIC_LOD) {
        renderer->automaticLod = false;
        renderer->lodLevel = options.lodLevel;
    }
    renderer->lodErrorThreshold = options.lodErrorThreshold;
    renderer->useNeuralLod = options.neuralLod;
    renderer->profiler().enabled = options.profile;
    renderer->renderScale = options.renderScale;
    renderer->upscaler = static_cast<gfx::Upscaler>(options.upscaler);
    if (options.pathTracing) {
        // 편집기 체크박스와 같은 게이트를 탄다.
        if (renderer->pathTracingAvailable()) {
            renderer->usePathTracing = true;
        } else {
            spdlog::warn("이 장치는 경로 추적을 지원하지 않아 --pathtrace 를 무시한다");
        }
    }
    if (options.triangleBudget > 0.0F) {
        renderer->triangleBudget = options.triangleBudget;
    }
    renderer->occlusionCulling = options.occlusionCulling;
    renderer->useReflections = options.reflections;
    if (!options.meshShader) {
        renderer->useMeshShader = false;
    }
    applyHardwareProfile();
    orbitDegreesPerFrame = options.orbitDegreesPerFrame;
    renderer->setUiCallback([this](VkCommandBuffer commandBuffer) { editorUi->record(commandBuffer); });
    editorUi->setModelLoader(assetRoot, [this](const std::filesystem::path& path) { requestModel(path); });
    editorUi->setSceneIo(
        sceneRoot,
        [this](const std::filesystem::path& path) { saveScene(path); },
        [this](const std::filesystem::path& path) { openScene(path); });
    editorUi->setModelCollector([this] { collectUnusedModels(true); });
    // 렌더러가 있어야 지오메트리 재구축을 알릴 수 있으므로 여기서 연다.
    for (const std::filesystem::path& path : options.modelPaths) {
        loadModel(path);
    }
    if (!options.modelPaths.empty()) {
        frameCamera(scenes.active(), *geometry);
    }
    if (!options.scenePath.empty()) {
        openScene(options.scenePath);
    }
    if (options.play) {
        scenes.active().simulating = true;
    }
}

Application::~Application() {
    // 백그라운드 해석이 남아 있으면 워커 큐가 살아 있을 때 먼저 끝낸다. 결과는 버린다.
    pendingLoad.reset();
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
    sceneRoot = std::filesystem::path(CG_LAB_ASSET_ROOT) / "scenes";

    // Unity 처럼 빈 장면 하나로 시작한다. public/assets 의 glTF 는 편집기 «모델 불러오기» 나 --model 로 올린다.
    scene::Scene& created = scenes.create("GameScene");
    addDefaultLight(created);
    created.refresh();
    geometry->build();
    scenes.setActive(0);
    spdlog::info("기본 장면 준비 완료: {}", created.name);
}

void Application::applyHardwareProfile() {
    // HardwareProfile::upscaler 는 헤더가 Vulkan 을 끌어오지 않도록 숫자로 둔다. 그 번호가
    // gfx::Upscaler 와 어긋나면 조용히 엉뚱한 업스케일러가 켜진다.
    static_assert(static_cast<uint32_t>(gfx::Upscaler::SPATIAL) == 1);
    static_assert(static_cast<uint32_t>(gfx::Upscaler::TAAU) == 2);
    static_assert(static_cast<uint32_t>(gfx::Upscaler::FSR) == 3);
    static_assert(static_cast<uint32_t>(gfx::Upscaler::DLSS) == 4);

    gfx::ProfileInputs inputs;
    switch (context->properties.deviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        inputs.deviceType = gfx::ProfileInputs::DeviceType::INTEGRATED;
        break;
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        inputs.deviceType = gfx::ProfileInputs::DeviceType::DISCRETE;
        break;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        inputs.deviceType = gfx::ProfileInputs::DeviceType::VIRTUAL;
        break;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        inputs.deviceType = gfx::ProfileInputs::DeviceType::CPU;
        break;
    default:
        inputs.deviceType = gfx::ProfileInputs::DeviceType::OTHER;
        break;
    }
    // 남은 예산이 아니라 힙 «크기» 를 본다. 예산은 다른 프로세스와 이미 만든 자원에 따라 흔들려
    // 같은 기계가 실행마다 다른 등급을 받는다.
    inputs.deviceMemoryBytes = context->deviceLocalMemoryBytes();
    inputs.meshShader = context->caps.meshShader;
    inputs.rayQuery = context->caps.rayQuery;
    inputs.rayTracingPipeline = context->caps.rayTracingPipeline;
    // 창 크기가 아니라 모니터 해상도를 본다. 창은 기동 시 고정 크기라 «이 화면이 얼마나 넓어질 수
    // 있는가» 를 말해 주지 못한다.
    if (const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(SDL_GetDisplayForWindow(window)); mode != nullptr) {
        inputs.displayWidth = static_cast<uint32_t>(mode->w);
        inputs.displayHeight = static_cast<uint32_t>(mode->h);
    }
    // 내장 TAAU 는 늘 있으므로 더 나은 것이 있으면 그것을 고른다.
    for (const gfx::UpscalerInfo& info : renderer->upscalers()) {
        if (info.available && (info.kind == gfx::Upscaler::FSR || info.kind == gfx::Upscaler::DLSS)) {
            inputs.bestTemporalUpscaler = static_cast<uint32_t>(info.kind);
        }
    }

    hardwareProfile = gfx::chooseProfile(inputs, options.autoTune);
    editorUi->hardwareProfile = hardwareProfile;
    editorUi->autoTune = options.autoTune;
    if (options.autoTune == gfx::AutoTune::OFF) {
        spdlog::info("자동 튜닝 꺼짐: 기본값을 그대로 쓴다");
        return;
    }

    // 명령줄이 정한 것은 자동 튜닝보다 세다. 스크린샷 비교가 인자대로 도는 것이 중요하다.
    if (!options.renderScaleGiven) {
        renderer->renderScale = hardwareProfile.renderScale;
    }
    if (!options.upscalerGiven) {
        renderer->upscaler = static_cast<gfx::Upscaler>(hardwareProfile.upscaler);
    }
    renderer->ssaoSamples = hardwareProfile.ssaoSamples;
    renderer->shadowCascades = hardwareProfile.shadowCascades;
    renderer->fluidParticleLimit = hardwareProfile.fluidParticleLimit;
    // 반사는 --reflections 로 켠 것을 끄지 않고 --no-reflections 로 끈 것을 켜지 않는다.
    if (!options.reflectionsGiven) {
        renderer->useReflections = hardwareProfile.reflections;
    }
    renderer->useRayQueryShadows = hardwareProfile.rayQueryShadows;

    spdlog::info("자동 튜닝({}): 등급 {}", gfx::autoTuneName(options.autoTune), gfx::tierName(hardwareProfile.tier));
    for (const std::string& reason : hardwareProfile.reasons) {
        spdlog::info("  - {}", reason);
    }
}

void Application::registerBuiltinModels() {
    auto count = static_cast<uint32_t>(asset::Primitive::COUNT);
    primitiveMeshes.assign(count, scene::INVALID_MESH);
    for (uint32_t i = 0; i < count; ++i) {
        auto primitive = static_cast<asset::Primitive>(i);
        asset::Model model = asset::makePrimitive(primitive);
        asset::buildLodHierarchy(model.meshes[0], &jobs);
        uint32_t registered = registerModel(std::filesystem::path{asset::primitiveAssetName(primitive)}, model, true);
        primitiveMeshes[i] = loadedModels[registered].meshBase;
    }
}

Application::PrepareTimings Application::prepareAssets(std::vector<asset::Texture*>& allTextures,
                                                       std::vector<asset::Mesh*>& allMeshes,
                                                       asset::LoadProgress* progress) {
    PrepareTimings timings;

    uint64_t textureStart = SDL_GetTicksNS();
    if (progress != nullptr) {
        progress->begin(asset::LoadProgress::Stage::TEXTURES, allTextures.size());
    }
    jobs.parallelFor(
        static_cast<uint32_t>(allTextures.size()), 1, [&allTextures, progress](uint32_t begin, uint32_t end) {
            for (uint32_t i = begin; i < end; ++i) {
                asset::decodeTexture(*allTextures[i]);
                if (progress != nullptr) {
                    progress->advance();
                }
            }
        });
    timings.textureMs = static_cast<double>(SDL_GetTicksNS() - textureStart) / 1.0e6;

    uint64_t lodStart = SDL_GetTicksNS();
    if (progress != nullptr) {
        uint64_t work = 0;
        for (const asset::Mesh* mesh : allMeshes) {
            work += asset::lodWorkEstimate(*mesh);
        }
        progress->begin(asset::LoadProgress::Stage::LOD, work);
    }
    // 메쉬끼리 나누고, 큰 메쉬는 그 안의 그룹도 나눈다. 큐가 여러 생산자를 받으므로 워커 안에서 다시
    // 나눠도 된다. 메쉬가 적고 하나가 크면 메쉬 단위만으로는 워커가 놀고, 메쉬가 많으면 그룹 단위만으로는
    // 0 단계 분할처럼 메쉬 안에서 직렬인 부분이 줄을 서므로 둘 다 필요하다. 작은 메쉬는 중첩하지 않는다.
    // 그룹을 기다리는 워커가 큐에 먼저 들어온 다른 메쉬 작업을 집어 들어 대기가 겹겹이 쌓이는데, 작은
    // 메쉬 수만 개가 그러면 스택이 깊어진다.
    constexpr size_t NESTED_LOD_INDEX_THRESHOLD = 300000;
    core::JobSystem* jobsPointer = &jobs;
    jobs.parallelFor(
        static_cast<uint32_t>(allMeshes.size()), 1, [&allMeshes, progress, jobsPointer](uint32_t begin, uint32_t end) {
            for (uint32_t i = begin; i < end; ++i) {
                bool large = allMeshes[i]->indices.size() >= NESTED_LOD_INDEX_THRESHOLD;
                asset::buildLodHierarchy(*allMeshes[i], large ? jobsPointer : nullptr, progress);
            }
        });
    timings.lodMs = static_cast<double>(SDL_GetTicksNS() - lodStart) / 1.0e6;
    return timings;
}

Application::PrepareTimings Application::prepareModel(asset::Model& model, asset::LoadProgress* progress) {
    std::vector<asset::Texture*> allTextures;
    std::vector<asset::Mesh*> allMeshes;
    for (asset::Texture& texture : model.textures) {
        allTextures.push_back(&texture);
    }
    for (asset::Mesh& mesh : model.meshes) {
        allMeshes.push_back(&mesh);
    }
    return prepareAssets(allTextures, allMeshes, progress);
}

asset::LoadSettings Application::loadSettings() const {
    asset::LoadSettings settings;
    settings.weldSmoothingDegrees = options.weldAngleDegrees;
    return settings;
}

bool Application::fitsGpuBudget(const asset::Model& model) const {
    VkDeviceSize geometryBytes = gfx::GeometryStore::estimateModelBytes(model);
    VkDeviceSize textureBytes = 0;
    for (const asset::Texture& texture : model.textures) {
        textureBytes += gfx::TextureCache::estimateBytes(texture);
    }
    // 지오메트리 버퍼를 키우는 동안 옛 버퍼가 새 버퍼와 함께 살아 있다. 장면 파일이 모델 여럿을 build 하나로
    // 올릴 때는 앞 모델의 꼬리가 아직 GPU 사용량에 잡히지 않으므로 그것도 더한다. 하위 가속 구조는
    // 광선 기능을 켤 때 따로 재서 넘으면 그 기능만 끈다.
    VkDeviceSize overlapBytes = geometry->residentBytes() + geometry->pendingBytes();
    VkDeviceSize needed = geometryBytes + textureBytes + overlapBytes;

    gfx::Context::MemoryBudget budget = context->deviceMemoryBudget();
    VkDeviceSize available = budget.budget > budget.usage ? budget.budget - budget.usage : 0;
    constexpr double MB = 1024.0 * 1024.0;
    if (needed > available) {
        spdlog::error(
            "GPU 메모리가 모자라 모델을 올리지 않습니다: {} (필요 {:.0f} MB = 지오메트리 {:.0f} + 텍스처 {:.0f} "
            "+ 재할당 겹침 {:.0f}, 남은 예산 {:.0f} MB, 예산 {:.0f} MB 중 사용 {:.0f} MB)",
            model.name,
            static_cast<double>(needed) / MB,
            static_cast<double>(geometryBytes) / MB,
            static_cast<double>(textureBytes) / MB,
            static_cast<double>(overlapBytes) / MB,
            static_cast<double>(available) / MB,
            static_cast<double>(budget.budget) / MB,
            static_cast<double>(budget.usage) / MB);
        return false;
    }
    spdlog::info(
        "GPU 메모리: {} 에 {:.0f} MB (지오메트리 {:.0f}, 텍스처 {:.0f}, 재할당 겹침 {:.0f}), 남은 예산 {:.0f} MB",
        model.name,
        static_cast<double>(needed) / MB,
        static_cast<double>(geometryBytes) / MB,
        static_cast<double>(textureBytes) / MB,
        static_cast<double>(overlapBytes) / MB,
        static_cast<double>(available) / MB);
    return true;
}

uint32_t Application::findModel(const std::filesystem::path& path) const {
    std::error_code error;
    for (uint32_t i = 0; i < loadedModels.size(); ++i) {
        if (loadedModels[i].unloaded) {
            continue;
        }
        // 내장 모델은 파일이 없어 equivalent 가 실패하므로 이름을 먼저 견준다.
        if (loadedModels[i].path == path || std::filesystem::equivalent(loadedModels[i].path, path, error)) {
            return i;
        }
    }
    return static_cast<uint32_t>(loadedModels.size());
}

uint32_t Application::registerModel(const std::filesystem::path& path, asset::Model& model, bool builtin) {
    gfx::Uploader uploader(*context);
    std::vector<uint32_t> textureSlots;
    textureSlots.reserve(model.textures.size());
    for (const asset::Texture& texture : model.textures) {
        textureSlots.push_back(textures->add(uploader, texture));
    }

    LoadedModel entry;
    entry.path = path;
    entry.builtin = builtin;
    entry.range = geometry->addModel(model, textureSlots);
    entry.meshBase = entry.range.meshBase;
    entry.meshCount = entry.range.meshCount;
    entry.textureSlots = std::move(textureSlots);
    entry.skeleton = std::move(model.skeleton);
    entry.instances = std::move(model.instances);
    uploader.flush();

    // 해제된 자리가 있으면 거기 넣는다. 애니메이터가 모델 번호를 들고 있어 항목을 지우지 않는다.
    for (uint32_t i = 0; i < loadedModels.size(); ++i) {
        if (loadedModels[i].unloaded) {
            loadedModels[i] = std::move(entry);
            return i;
        }
    }
    loadedModels.push_back(std::move(entry));
    return static_cast<uint32_t>(loadedModels.size()) - 1;
}

void Application::collectUnusedModels(bool force) {
    std::vector<uint8_t> used(loadedModels.size(), 0);
    auto markMesh = [&](uint32_t mesh) {
        for (uint32_t i = 0; i < loadedModels.size(); ++i) {
            const LoadedModel& model = loadedModels[i];
            if (!model.unloaded && mesh >= model.meshBase && mesh < model.meshBase + model.meshCount) {
                used[i] = 1;
                return;
            }
        }
    };
    auto markModel = [&](int32_t model) {
        if (model >= 0 && static_cast<size_t>(model) < used.size()) {
            used[static_cast<size_t>(model)] = 1;
        }
    };
    for (size_t s = 0; s < scenes.count(); ++s) {
        const scene::Scene& scene = scenes.at(s);
        for (const scene::MeshRenderer& renderer : scene.meshRenderers) {
            if (renderer.mesh != scene::INVALID_MESH) {
                markMesh(renderer.mesh);
            }
        }
        for (const scene::Animator& animator : scene.animators) {
            markModel(animator.model);
        }
    }

    std::vector<uint32_t> doomed;
    for (uint32_t i = 0; i < loadedModels.size(); ++i) {
        const LoadedModel& model = loadedModels[i];
        if (used[i] != 0 || model.unloaded || model.builtin) {
            continue;
        }
        // 되돌리기 기록이 아직 이 모델을 가리키면 되살릴 수 있으므로 남긴다. 기록에서 밀려나면 그때 해제된다.
        if (!force && editorUi->referencesModel(model.meshBase, model.meshCount, static_cast<int32_t>(i))) {
            continue;
        }
        doomed.push_back(i);
    }
    if (doomed.empty()) {
        return;
    }

    // 버퍼를 다시 잡고 이미지를 지우므로 진행 중인 프레임이 끝난 뒤에 한다.
    renderer->waitIdle();
    constexpr double MB = 1024.0 * 1024.0;
    double before = static_cast<double>(geometry->residentBytes()) / MB;
    for (uint32_t index : doomed) {
        LoadedModel& model = loadedModels[index];
        for (uint32_t slot : model.textureSlots) {
            if (slot != asset::INVALID_TEXTURE) {
                textures->remove(slot);
            }
        }
        geometry->removeModel(model.range);
        spdlog::info("모델 해제: {} (메쉬 {}개, 텍스처 {}장)",
                     model.path.filename().string(),
                     model.meshCount,
                     model.textureSlots.size());
        model.unloaded = true;
        model.textureSlots.clear();
        model.skeleton = {};
        model.instances.clear();
    }
    geometry->build();
    renderer->onGeometryChanged();
    spdlog::info(
        "지오메트리 GPU 메모리 {:.1f} MB -> {:.1f} MB", before, static_cast<double>(geometry->residentBytes()) / MB);
}

uint32_t Application::ensureModel(const std::filesystem::path& path) {
    uint32_t existing = findModel(path);
    if (existing < loadedModels.size()) {
        return existing;
    }
    std::optional<asset::Model> loaded = asset::loadGltf(path, nullptr, &jobs, loadSettings());
    if (!loaded) {
        return static_cast<uint32_t>(loadedModels.size());
    }
    prepareModel(*loaded, nullptr);
    if (!fitsGpuBudget(*loaded)) {
        return static_cast<uint32_t>(loadedModels.size());
    }
    return registerModel(path, *loaded);
}

void Application::instantiateModel(uint32_t modelIndex, scene::Scene& scene) {
    const LoadedModel& entry = loadedModels[modelIndex];

    // 모델 하나가 계층의 뿌리 하나를 이룬다. 기즈모로 뿌리를 옮기면 자식이 함께 따라간다.
    auto root = static_cast<int32_t>(scene.objects.size());
    std::string name = entry.path.stem().string();
    scene::Object rootObject;
    rootObject.name = name;
    scene.objects.push_back(std::move(rootObject));

    int32_t animator = -1;
    if (!entry.skeleton.skins.empty()) {
        animator = static_cast<int32_t>(scene.animators.size());
        scene::Animator created;
        created.name = name;
        created.skeleton = entry.skeleton;
        created.model = static_cast<int32_t>(modelIndex);
        scene.animators.push_back(std::move(created));
        // 애니메이션 컨트롤러는 Unity 처럼 뿌리 오브젝트에 붙인다.
        scene.objects[static_cast<size_t>(root)].animator = animator;
    }

    for (const asset::Instance& instance : entry.instances) {
        scene::Object object;
        object.name = instance.name;
        object.parent = root;
        object.transform = scene::Transform::fromMatrix(instance.transform);
        if (instance.skin >= 0) {
            object.animator = animator;
        }
        scene.objects.push_back(std::move(object));
        scene.attachMeshRenderer(
            static_cast<uint32_t>(scene.objects.size() - 1), entry.meshBase + instance.meshIndex, instance.skin);
    }

    // 바인드 포즈라도 조인트 행렬이 있어야 스킨 메쉬가 제자리에 선다.
    scene.update(0.0F);
    scene.refresh();
}

void Application::requestModel(const std::filesystem::path& path) {
    loadQueue.push_back(path);
    startNextLoad();
}

void Application::startNextLoad() {
    while (pendingLoad == nullptr && !loadQueue.empty()) {
        std::filesystem::path path = std::move(loadQueue.front());
        loadQueue.pop_front();

        // 이미 올라간 모델은 해석도 업로드도 없이 장면에만 붙인다.
        uint32_t existing = findModel(path);
        if (existing < loadedModels.size()) {
            instantiateModel(existing, scenes.active());
            spdlog::info("모델 적재: {} (이미 올라가 있어 장면에만 붙임)", path.filename().string());
            continue;
        }

        pendingLoad = std::make_unique<PendingLoad>();
        pendingLoad->path = std::move(path);
        pendingLoad->startTicks = SDL_GetTicksNS();
        pendingLoad->sceneIndex = scenes.current();
        PendingLoad* load = pendingLoad.get();
        // 스레드는 load 가 가리키는 것만 만진다. pendingLoad 는 스레드를 합류한 뒤에만 비운다.
        load->worker = std::thread([this, load]() {
            std::optional<asset::Model> loaded = asset::loadGltf(load->path, &load->progress, &jobs, loadSettings());
            if (loaded) {
                load->model = std::move(*loaded);
                load->timings = prepareModel(load->model, &load->progress);
            } else {
                load->failed.store(true, std::memory_order_relaxed);
            }
            load->progress.begin(asset::LoadProgress::Stage::DONE);
            load->finished.store(true, std::memory_order_release);
        });
    }
}

void Application::pumpLoads() {
    startNextLoad();
    if (pendingLoad == nullptr || !pendingLoad->finished.load(std::memory_order_acquire)) {
        return;
    }
    if (!pendingLoad->uploadShown) {
        pendingLoad->progress.begin(asset::LoadProgress::Stage::UPLOAD);
        pendingLoad->uploadShown = true;
        return;
    }
    pendingLoad->worker.join();
    completeLoad();
    startNextLoad();
}

void Application::completeLoad() {
    PendingLoad& load = *pendingLoad;
    // 장면이 지워지는 일은 없지만, 방어적으로 범위를 벗어나면 활성 장면에 붙인다.
    scene::Scene& target = load.sceneIndex < scenes.count() ? scenes.at(load.sceneIndex) : scenes.active();

    if (load.failed.load(std::memory_order_relaxed)) {
        spdlog::error("모델 적재 실패: {}", load.path.string());
        pendingLoad.reset();
        return;
    }
    // 해석하는 동안 장면 파일이 같은 모델을 동기로 올렸을 수 있다. 그러면 두 번 올리지 않고 붙이기만 한다.
    uint32_t existing = findModel(load.path);
    if (existing < loadedModels.size()) {
        instantiateModel(existing, target);
        spdlog::info("모델 적재: {} (해석 중에 이미 올라가 장면에만 붙임)", load.path.filename().string());
        pendingLoad.reset();
        return;
    }

    if (!fitsGpuBudget(load.model)) {
        pendingLoad.reset();
        return;
    }

    // 지오메트리 버퍼를 통째로 다시 만들기 때문에 진행 중인 프레임이 끝난 뒤에 손대야 한다.
    renderer->waitIdle();
    uint64_t uploadStart = SDL_GetTicksNS();

    uint32_t modelIndex = registerModel(load.path, load.model);
    geometry->build();
    renderer->onGeometryChanged();
    instantiateModel(modelIndex, target);

    uint64_t end = SDL_GetTicksNS();
    spdlog::info("모델 적재: {} ({:.1f} ms; 텍스처 {:.1f}, LOD {:.1f}, 업로드 {:.1f})",
                 load.path.filename().string(),
                 static_cast<double>(end - load.startTicks) / 1.0e6,
                 load.timings.textureMs,
                 load.timings.lodMs,
                 static_cast<double>(end - uploadStart) / 1.0e6);
    pendingLoad.reset();
}

void Application::loadModel(const std::filesystem::path& path) {
    requestModel(path);
    while (pendingLoad != nullptr) {
        pendingLoad->worker.join();
        completeLoad();
        startNextLoad();
    }
}

editor::LoadStatus Application::loadStatus() const {
    editor::LoadStatus status;
    if (pendingLoad == nullptr) {
        return status;
    }
    status.active = true;
    status.file = pendingLoad->path.filename().string();
    status.queued = loadQueue.size();
    switch (pendingLoad->progress.stage.load(std::memory_order_acquire)) {
    case asset::LoadProgress::Stage::PARSE:
        status.stage = "파일 해석";
        break;
    case asset::LoadProgress::Stage::CONVERT:
        status.stage = "정점 변환";
        break;
    case asset::LoadProgress::Stage::TEXTURES:
        status.stage = "텍스처 디코딩";
        break;
    case asset::LoadProgress::Stage::LOD:
        status.stage = "LOD 계층 구축";
        break;
    case asset::LoadProgress::Stage::UPLOAD:
    case asset::LoadProgress::Stage::DONE:
        // 해석이 끝난 뒤 업로드가 시작되기까지의 한 프레임도 같은 글자로 보인다.
        status.stage = "GPU 업로드";
        break;
    default:
        status.stage = "준비";
        break;
    }
    status.fraction = pendingLoad->progress.fraction();
    status.indeterminate = pendingLoad->progress.total.load(std::memory_order_relaxed) == 0;
    return status;
}

void Application::saveScene(const std::filesystem::path& path) {
    const scene::Scene& active = scenes.active();

    // 이 장면이 실제로 쓰는 모델만 적는다. 그래야 다시 열 때 필요한 것만 올린다.
    std::vector<int32_t> fileIndex(loadedModels.size(), -1);
    scene::ModelTable table;
    auto useModel = [&](uint32_t modelIndex) {
        if (fileIndex[modelIndex] < 0) {
            fileIndex[modelIndex] = static_cast<int32_t>(table.paths.size());
            table.paths.push_back(loadedModels[modelIndex].path);
            table.meshBase.push_back(loadedModels[modelIndex].meshBase);
            table.meshCount.push_back(loadedModels[modelIndex].meshCount);
        }
    };
    for (uint32_t object = 0; object < active.objects.size(); ++object) {
        uint32_t mesh = active.meshOf(object);
        for (uint32_t i = 0; i < loadedModels.size(); ++i) {
            if (!loadedModels[i].unloaded && mesh >= loadedModels[i].meshBase &&
                mesh < loadedModels[i].meshBase + loadedModels[i].meshCount) {
                useModel(i);
                break;
            }
        }
    }
    for (const scene::Animator& animator : active.animators) {
        if (animator.model >= 0) {
            useModel(static_cast<uint32_t>(animator.model));
        }
    }

    // 애니메이터가 가리키는 번호를 파일 안의 번호로 옮긴다.
    scene::Scene remapped = active;
    for (scene::Animator& animator : remapped.animators) {
        animator.model = animator.model >= 0 ? fileIndex[static_cast<size_t>(animator.model)] : -1;
    }

    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    std::ofstream file(path);
    if (!file) {
        spdlog::error("장면을 저장하지 못했습니다: {}", path.string());
        return;
    }
    file << scene::writeScene(remapped, table, assetRoot);
    spdlog::info("장면 저장: {}", path.string());
}

void Application::openScene(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        spdlog::error("장면을 열지 못했습니다: {}", path.string());
        return;
    }
    std::string text((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    scene::SceneFile loaded = scene::readScene(text);

    // 적재는 지오메트리 버퍼를 다시 만드므로 진행 중인 프레임이 끝난 뒤에 한다.
    // ponytail: 장면 파일의 모델은 아직 동기로 올린다. 백그라운드 적재가 도는 중이면 같은 워커를 나눠
    // 써서 둘 다 느려질 뿐 결과는 옳다.
    renderer->waitIdle();
    std::vector<uint32_t> modelIndices;
    modelIndices.reserve(loaded.models.size());
    for (const std::filesystem::path& modelPath : loaded.models) {
        // 상대 경로는 에셋 뿌리 기준으로 푼다. 내장 모델 이름은 그대로 찾는다.
        // 내장 도형은 파일이 없어 이름 그대로 찾는다. 아는 이름만 그렇게 다뤄야, 깨진 파일의 엉뚱한
        // 이름이 «파일을 못 찾았다»로 정직하게 실패한다.
        bool builtin = asset::primitiveFromAssetName(modelPath.generic_string()) != asset::Primitive::COUNT;
        uint32_t index = ensureModel(modelPath.is_absolute() || builtin ? modelPath : assetRoot / modelPath);
        if (index >= loadedModels.size()) {
            spdlog::error("장면이 가리키는 모델을 읽지 못해 그 오브젝트는 메쉬 없이 둔다: {}", modelPath.string());
        }
        modelIndices.push_back(index);
    }
    if (!loaded.models.empty()) {
        geometry->build();
        renderer->onGeometryChanged();
    }

    scene::Scene& created = scenes.create(loaded.scene.name);
    created = std::move(loaded.scene);
    // 모델과 같은 규칙: 상대 경로는 에셋 뿌리 기준으로 푼다.
    if (!created.environment.hdrPath.empty() && !created.environment.hdrPath.is_absolute()) {
        created.environment.hdrPath = assetRoot / created.environment.hdrPath;
    }
    auto resolved = [&](int32_t model) {
        return model >= 0 && static_cast<size_t>(model) < modelIndices.size() &&
               modelIndices[static_cast<size_t>(model)] < loadedModels.size();
    };
    for (size_t i = 0; i < created.objects.size(); ++i) {
        int32_t model = loaded.objectModels[i];
        if (resolved(model)) {
            created.attachMeshRenderer(static_cast<uint32_t>(i),
                                       loadedModels[modelIndices[model]].meshBase + loaded.objectLocalMeshes[i],
                                       created.skinOf(static_cast<uint32_t>(i)));
        }
    }
    for (size_t i = 0; i < created.animators.size(); ++i) {
        int32_t model = loaded.animatorModels[i];
        if (resolved(model)) {
            created.animators[i].skeleton = loadedModels[modelIndices[model]].skeleton;
        }
    }
    created.update(0.0F);
    created.refresh();

    scenes.setActive(scenes.count() - 1);
    spdlog::info("장면 열기: {} (오브젝트 {}, 조명 {})", path.string(), created.objects.size(), created.lights.size());
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
            bool overViewport = editorUi->viewportHovered();
            scenes.active().camera.keyboardEnabled = overViewport;
            if (overViewport || scenes.active().camera.isLooking()) {
                scenes.active().camera.handleEvent(event);
            }
        }

        uint64_t currentTicks = SDL_GetTicksNS();
        float deltaSeconds = static_cast<float>(currentTicks - previousTicks) / NANOSECONDS_PER_SECOND;
        previousTicks = currentTicks;

        if (orbitDegreesPerFrame != 0.0F) {
            scenes.active().camera.yawDegrees += orbitDegreesPerFrame;
            scenes.active().camera.applyOrbit();
        }

        if ((SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED) != 0) {
            SDL_Delay(10);
            continue;
        }

        // 프로파일러 슬롯은 프레임 맨 앞에서 연다. 아래 CPU 구간이 그리기보다 먼저 기록된다.
        renderer->beginProfilerFrame();
        scenes.active().camera.update(deltaSeconds);
        {
            gfx::ProfilerScope scope(renderer->profiler(), "장면 갱신");
            // 애니메이션은 그리기 전에 진행시켜야 이번 프레임의 조인트 행렬이 올라간다.
            scenes.active().update(deltaSeconds, &jobs);
        }
        // 고정 간격으로 나눠 돈다. 프레임이 길어도 여덟 스텝까지만 따라잡아 나선형으로 느려지지 않는다.
        constexpr float PHYSICS_STEP = 1.0F / 120.0F;
        // GPU 솔버가 끝낸 결과를 먼저 장면에 되쓴다. 아래 refresh 가 이 값으로 세계 변환을 다시 만든다.
        renderer->applyRigidBodyReadback(scenes.active());
        uint32_t rigidSteps = 0;
        if (scenes.active().simulating) {
            gfx::ProfilerScope scope(renderer->profiler(), "강체 물리");
            physicsAccumulator = std::min(physicsAccumulator + deltaSeconds, PHYSICS_STEP * 8.0F);
            while (physicsAccumulator >= PHYSICS_STEP) {
                physics::stepRigidBodies(scenes.active(), PHYSICS_STEP, &jobs);
                physicsAccumulator -= PHYSICS_STEP;
                ++rigidSteps;
            }
        } else {
            physicsAccumulator = 0.0F;
        }
        // GPU 백엔드 강체는 같은 간격으로 렌더러가 푼다.
        renderer->setRigidBodySteps(rigidSteps, PHYSICS_STEP);
        // 밀린 크기 변경은 UI 가 렌더 타겟을 참조하기 전에 끝내야 한다.
        renderer->prepareFrame();
        renderer->setDisplayExtent(editorUi->desiredRenderExtent());
        // 백그라운드 해석이 끝난 모델을 올린다. 편집기보다 앞이라 "GPU 업로드" 표시가 올린 프레임에
        // 그려진 채로 남고, 그 다음 프레임의 pumpLoads 가 실제로 올리는 동안 화면에 보인다.
        pumpLoads();
        editorUi->setLoadStatus(loadStatus());
        {
            gfx::ProfilerScope scope(renderer->profiler(), "편집기 UI");
            editorUi->build(scenes, *geometry, deltaSeconds);
        }
        // 편집기가 장면을 바꾼 뒤, 렌더러가 읽기 전에 캐시를 다시 만든다.
        scenes.active().refresh();
        // 오브젝트가 지워지거나 장면이 바뀐 프레임에만 미사용 모델을 살핀다. 매 프레임 훑을 일은 아니다.
        if (scenes.current() != collectedScene || scenes.active().topologyRevision() != collectedTopology) {
            collectedScene = scenes.current();
            collectedTopology = scenes.active().topologyRevision();
            collectUnusedModels(false);
        }
        renderer->drawFrame(scenes.active());
        ++frameCount;

        if (!options.screenshotPath.empty()) {
            if (frameCount == options.screenshotFrame) {
                renderer->requestCapture(options.screenshotPath);
            } else if (frameCount > options.screenshotFrame) {
                running = false;
            }
        }
    }
    renderer->waitIdle();

    // 창을 못 보는 실행(스크린샷, CI)에서도 결과를 남긴다.
    if (renderer->profiler().enabled) {
        spdlog::info("구간 계측 결과 (CPU / GPU, ms)");
        for (const gfx::ProfilerZone& zone : renderer->profiler().zones()) {
            spdlog::info("  {:<28} {:7.3f}  {:>7}",
                         std::string(zone.depth * 2, ' ') + zone.name,
                         zone.cpuMilliseconds,
                         zone.hasGpu ? std::format("{:.3f}", zone.gpuMilliseconds) : std::string{"-"});
        }
    }
}

} // namespace app
