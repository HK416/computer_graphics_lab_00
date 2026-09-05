#include "editor/editor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <utility>

#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/quaternion.hpp>
#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <imgui_impl_vulkan.h>
#include <imgui_internal.h>
#include <ImGuizmo.h>
#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>

#include "asset/primitives.h"
#include "core/error.h"
#include "editor/log_sink.h"
#include "gfx/context.h"
#include "gfx/geometry.h"
#include "gfx/profiler.h"
#include "gfx/renderer.h"
#include "gfx/rigid_body_gpu.h"
#include "gfx/shadow_math.h"
#include "scene/scene.h"

namespace editor {

namespace {

// 광선과 구가 만나는 가장 가까운 거리. 만나지 않거나 뒤쪽이면 음수.
float raySphere(const scene::Ray& ray, const glm::vec4& sphere) {
    glm::vec3 toCenter = glm::vec3(sphere) - ray.origin;
    float alongRay = glm::dot(toCenter, ray.direction);
    float centerDistanceSq = glm::dot(toCenter, toCenter) - alongRay * alongRay;
    float radiusSq = sphere.w * sphere.w;
    if (centerDistanceSq > radiusSq) {
        return -1.0F;
    }
    float halfChord = std::sqrt(radiusSq - centerDistanceSq);
    float near = alongRay - halfChord;
    float far = alongRay + halfChord;
    // 구 안에서 쏘면 near 가 음수다. 그때는 반대쪽 교차를 쓴다.
    return near >= 0.0F ? near : (far >= 0.0F ? far : -1.0F);
}

// 광선에 걸리는 가장 가까운 오브젝트. 없으면 -1.
//
// ponytail: 메쉬 경계 구까지만 본다. 더 정확히 하려면 meshlet 경계 구로 한 단계 좁힌 뒤
// LOD 0 삼각형과 교차시키면 된다. 스킨 메쉬는 CPU 정점이 바인드 포즈라 구로만 다뤄야 한다.
int pickObject(const scene::Scene& scene, const gfx::GeometryStore& geometry, const scene::Ray& ray) {
    int best = -1;
    float bestDistance = std::numeric_limits<float>::max();
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        uint32_t mesh = scene.meshOf(index);
        if (!geometry.meshLive(mesh) || !scene.visibleInTree(index)) {
            continue;
        }
        const glm::mat4& world = scene.worldMatrix(index);
        glm::vec4 local = geometry.mesh(mesh).boundingSphere;
        glm::vec3 center = glm::vec3(world * glm::vec4{glm::vec3(local), 1.0F});
        // 비균등 스케일은 가장 긴 축으로 보수적으로 키운다. focusSelected 와 같은 계산이다.
        float scale = std::sqrt(std::max({glm::dot(glm::vec3(world[0]), glm::vec3(world[0])),
                                          glm::dot(glm::vec3(world[1]), glm::vec3(world[1])),
                                          glm::dot(glm::vec3(world[2]), glm::vec3(world[2]))}));
        float distance = raySphere(ray, glm::vec4{center, local.w * scale});
        if (distance >= 0.0F && distance < bestDistance) {
            bestDistance = distance;
            best = static_cast<int>(index);
        }
    }
    return best;
}

} // namespace
namespace {

constexpr float BASE_FONT_SIZE = 16.0F;
constexpr const char* WINDOW_HIERARCHY = "계층";
constexpr const char* WINDOW_INSPECTOR = "인스펙터";
constexpr const char* WINDOW_SCENE = "장면";
constexpr const char* WINDOW_CONSOLE = "콘솔";
constexpr const char* WINDOW_SETTINGS = "렌더 설정";
// 계층 패널에서 오브젝트를 끌 때 쓰는 페이로드 이름.
constexpr const char* HIERARCHY_PAYLOAD = "계층 오브젝트";

void checkVulkanResult(VkResult result) {
    if (result != VK_SUCCESS) {
        core::fatal("ImGui Vulkan 백엔드 오류: {}", static_cast<int>(result));
    }
}

ImVec4 levelColor(spdlog::level::level_enum level) {
    switch (level) {
    case spdlog::level::err:
    case spdlog::level::critical:
        return ImVec4{1.0F, 0.4F, 0.4F, 1.0F};
    case spdlog::level::warn:
        return ImVec4{1.0F, 0.8F, 0.3F, 1.0F};
    case spdlog::level::debug:
    case spdlog::level::trace:
        return ImVec4{0.6F, 0.6F, 0.6F, 1.0F};
    default:
        return ImVec4{0.85F, 0.85F, 0.85F, 1.0F};
    }
}

const char* alphaModeName(asset::AlphaMode mode) {
    switch (mode) {
    case asset::AlphaMode::CUTOFF:
        return "컷오프";
    case asset::AlphaMode::TRANSLUCENT:
        return "반투명";
    default:
        return "불투명";
    }
}

void applyDarkTheme() {
    ImGui::StyleColorsDark();
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 2.0F;
    style.FrameRounding = 2.0F;
    style.TabRounding = 2.0F;
    style.WindowPadding = ImVec2{8.0F, 8.0F};
    style.FramePadding = ImVec2{6.0F, 3.0F};
    style.ItemSpacing = ImVec2{8.0F, 5.0F};
    style.Colors[ImGuiCol_WindowBg] = ImVec4{0.22F, 0.22F, 0.22F, 1.0F};
    style.Colors[ImGuiCol_TitleBg] = ImVec4{0.16F, 0.16F, 0.16F, 1.0F};
    style.Colors[ImGuiCol_TitleBgActive] = ImVec4{0.16F, 0.16F, 0.16F, 1.0F};
    style.Colors[ImGuiCol_Tab] = ImVec4{0.19F, 0.19F, 0.19F, 1.0F};
    style.Colors[ImGuiCol_TabSelected] = ImVec4{0.27F, 0.27F, 0.27F, 1.0F};
    style.Colors[ImGuiCol_FrameBg] = ImVec4{0.16F, 0.16F, 0.16F, 1.0F};
    style.Colors[ImGuiCol_Header] = ImVec4{0.30F, 0.34F, 0.40F, 1.0F};
}

} // namespace

Editor::Editor(gfx::Context& context, gfx::Renderer& renderer, SDL_Window* window)
    : context(context), renderer(renderer) {
    logSink = std::make_shared<LogSink>();
    spdlog::default_logger()->sinks().push_back(logSink);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();

    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable | ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigDpiScaleFonts = true;
    // 배치는 코드로 만들어 두므로 ini 파일을 남기지 않는다.
    io.IniFilename = nullptr;

    std::filesystem::path fontPath = std::filesystem::path(CG_LAB_ASSET_ROOT) / "fonts" / "NotoSans-Bold.otf";
    if (io.Fonts->AddFontFromFileTTF(fontPath.string().c_str(), BASE_FONT_SIZE) == nullptr) {
        core::fatal("폰트를 읽을 수 없습니다: {}", fontPath.string());
    }

    applyDarkTheme();

    if (!ImGui_ImplSDL3_InitForVulkan(window)) {
        core::fatal("ImGui SDL3 백엔드 초기화에 실패했습니다");
    }

    VkFormat swapchainFormat = renderer.swapchainFormat();
    VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &swapchainFormat;

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_3;
    initInfo.Instance = context.instance;
    initInfo.PhysicalDevice = context.physicalDevice;
    initInfo.Device = context.device;
    initInfo.QueueFamily = context.queueFamilies.graphics;
    initInfo.Queue = context.graphicsQueue;
    initInfo.DescriptorPoolSize = 64;
    initInfo.MinImageCount = 2;
    initInfo.ImageCount = renderer.swapchainImageCount();
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = renderingInfo;
    initInfo.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.CheckVkResultFn = checkVulkanResult;
    if (!ImGui_ImplVulkan_Init(&initInfo)) {
        core::fatal("ImGui Vulkan 백엔드 초기화에 실패했습니다");
    }
}

Editor::~Editor() {
    vkDeviceWaitIdle(context.device);
    for (auto& entry : textures) {
        ImGui_ImplVulkan_RemoveTexture(entry.second);
    }
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplSDL3_Shutdown();
    ImGui::DestroyContext();

    auto& sinks = spdlog::default_logger()->sinks();
    sinks.erase(std::remove(sinks.begin(), sinks.end(), logSink), sinks.end());
}

void Editor::processEvent(const SDL_Event& event) {
    ImGui_ImplSDL3_ProcessEvent(&event);
}

VkDescriptorSet Editor::textureFor(VkImageView view, VkImageLayout layout) {
    // 렌더 대상이 다시 만들어졌으면 이전 디스크립터는 모두 버린다.
    if (cachedGeneration != renderer.targetsGeneration()) {
        for (auto& entry : textures) {
            ImGui_ImplVulkan_RemoveTexture(entry.second);
        }
        textures.clear();
        cachedGeneration = renderer.targetsGeneration();
    }

    auto found = textures.find(view);
    if (found != textures.end()) {
        return found->second;
    }
    VkDescriptorSet set = ImGui_ImplVulkan_AddTexture(view, layout);
    textures.emplace(view, set);
    return set;
}

void Editor::buildDockspace(scene::SceneManager& scenes, const gfx::GeometryStore& geometry) {
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::SetNextWindowViewport(viewport->ID);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0F);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0F);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0.0F, 0.0F});

    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                             ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                             ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus |
                             ImGuiWindowFlags_MenuBar;
    ImGui::Begin("##dockhost", nullptr, flags);
    ImGui::PopStyleVar(3);

    ImGuiID dockspaceId = ImGui::GetID("MainDockspace");
    if (!layoutBuilt) {
        layoutBuilt = true;
        ImGui::DockBuilderRemoveNode(dockspaceId);
        ImGui::DockBuilderAddNode(dockspaceId, ImGuiDockNodeFlags_DockSpace);
        ImGui::DockBuilderSetNodeSize(dockspaceId, viewport->WorkSize);

        ImGuiID center = dockspaceId;
        ImGuiID left = ImGui::DockBuilderSplitNode(center, ImGuiDir_Left, 0.18F, nullptr, &center);
        ImGuiID right = ImGui::DockBuilderSplitNode(center, ImGuiDir_Right, 0.26F, nullptr, &center);
        ImGuiID bottom = ImGui::DockBuilderSplitNode(center, ImGuiDir_Down, 0.28F, nullptr, &center);

        ImGui::DockBuilderDockWindow(WINDOW_HIERARCHY, left);
        ImGui::DockBuilderDockWindow(WINDOW_INSPECTOR, right);
        ImGui::DockBuilderDockWindow(WINDOW_CONSOLE, bottom);
        ImGui::DockBuilderDockWindow(WINDOW_SCENE, center);
        ImGui::DockBuilderFinish(dockspaceId);
    }
    ImGui::DockSpace(dockspaceId, ImVec2{0.0F, 0.0F}, ImGuiDockNodeFlags_None);

    buildMenuBar(scenes, geometry);
    buildPopups(scenes);
    ImGui::End();
}

void Editor::buildMenuBar(scene::SceneManager& scenes, const gfx::GeometryStore& geometry) {
    if (!ImGui::BeginMenuBar()) {
        return;
    }
    scene::Scene& active = scenes.active();
    bool anySelected = hasSelection();

    if (ImGui::BeginMenu("파일")) {
        if (ImGui::MenuItem("새 장면", "Ctrl+N")) {
            deferred = [this, &scenes] { newScene(scenes); };
        }
        if (ImGui::MenuItem("장면 열기...", "Ctrl+O")) {
            popupRequest = PopupRequest::OPEN_SCENE;
        }
        if (ImGui::MenuItem("장면 저장...", "Ctrl+S")) {
            popupRequest = PopupRequest::SAVE_SCENE;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("모델 불러오기...")) {
            popupRequest = PopupRequest::LOAD_MODEL;
        }
        if (ImGui::MenuItem("미사용 모델 해제", nullptr, false, static_cast<bool>(modelCollector))) {
            // 기록이 모델을 붙들고 있으면 해제되지 않으므로 먼저 비운다. 지운 오브젝트는 되돌릴 수 없게 된다.
            deferred = [this] {
                clearHistories();
                modelCollector();
            };
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip("어느 장면도 쓰지 않는 모델을 GPU 에서 내린다. 되돌리기 기록도 함께 비운다");
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("편집")) {
        const History* history = scenes.current() < histories.size() ? &histories[scenes.current()] : nullptr;
        if (ImGui::MenuItem("되돌리기", "Ctrl+Z", false, history != nullptr && !history->undoStack.empty())) {
            menuUndo = true;
        }
        if (ImGui::MenuItem("다시 실행", "Ctrl+Y", false, history != nullptr && !history->redoStack.empty())) {
            menuRedo = true;
        }
        ImGui::Separator();
        if (ImGui::MenuItem("복제", "Ctrl+D", false, anySelected)) {
            deferred = [this, &active] { duplicateSelection(active); };
        }
        if (ImGui::MenuItem("삭제", "Delete", false, anySelected)) {
            deferred = [this, &active] { deleteSelection(active); };
        }
        if (ImGui::MenuItem("부모 해제", nullptr, false, anySelected)) {
            deferred = [this, &active] { unparentSelection(active); };
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("오브젝트")) {
        buildCreateItems(active, geometry, -1);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("설정")) {
        ImGui::MenuItem("렌더 설정", nullptr, &showRenderSettings);
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("창")) {
        if (ImGui::MenuItem("배치 초기화")) {
            layoutBuilt = false;
        }
        ImGui::EndMenu();
    }

    // 재생/정지는 Unity 처럼 메뉴바 가운데에 둔다.
    const char* playLabel = active.simulating ? "정지" : "재생";
    float labelWidth = ImGui::CalcTextSize(playLabel).x + ImGui::GetStyle().FramePadding.x * 2.0F;
    ImGui::SetCursorPosX(std::max(ImGui::GetCursorPosX(), ImGui::GetWindowWidth() * 0.5F - labelWidth * 0.5F));
    if (active.simulating) {
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.55F, 0.25F, 0.2F, 1.0F});
    }
    if (ImGui::SmallButton(playLabel)) {
        deferred = [this, &scenes] {
            if (scenes.active().simulating) {
                stopSimulation(scenes);
            } else {
                startSimulation(scenes);
            }
        };
    }
    if (active.simulating) {
        ImGui::PopStyleColor();
    }
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Ctrl+P. 재생 중에는 강체·유체가 움직이고, 정지하면 재생 전 상태로 돌아간다");
    }
    ImGui::EndMenuBar();
}

void Editor::buildPopups(scene::SceneManager& scenes) {
    // 팝업은 요청한 다음 프레임에 연다. 메뉴나 우클릭 메뉴 안에서 바로 열면 그 메뉴의 ID 스택에 묶여
    // 메뉴가 닫히는 순간 함께 닫힌다.
    switch (popupRequest) {
    case PopupRequest::SAVE_SCENE: {
        std::string suggested = scenes.active().name + ".json";
        std::copy_n(suggested.c_str(), std::min(suggested.size() + 1, sceneNameInput.size()), sceneNameInput.begin());
        sceneNameInput.back() = '\0';
        ImGui::OpenPopup("장면 저장");
        break;
    }
    case PopupRequest::OPEN_SCENE: {
        sceneFiles.clear();
        std::error_code error;
        for (const auto& entry : std::filesystem::directory_iterator(sceneRoot, error)) {
            if (entry.is_regular_file() && entry.path().extension() == ".json") {
                sceneFiles.push_back(entry.path());
            }
        }
        std::ranges::sort(sceneFiles);
        ImGui::OpenPopup("장면 열기");
        break;
    }
    case PopupRequest::LOAD_MODEL: {
        // 팝업을 열 때마다 다시 훑는다. 실행 중에 파일이 늘어날 수 있다.
        modelFiles.clear();
        std::error_code error;
        for (const auto& entry : std::filesystem::directory_iterator(modelRoot, error)) {
            std::filesystem::path extension = entry.path().extension();
            if (entry.is_regular_file() && (extension == ".glb" || extension == ".gltf")) {
                modelFiles.push_back(entry.path());
            }
        }
        std::ranges::sort(modelFiles);
        ImGui::OpenPopup("모델 선택");
        break;
    }
    default:
        break;
    }
    popupRequest = PopupRequest::NONE;

    if (ImGui::BeginPopup("장면 저장")) {
        ImGui::TextDisabled("%s", sceneRoot.string().c_str());
        ImGui::SetNextItemWidth(280.0F);
        ImGui::InputText("파일 이름", sceneNameInput.data(), sceneNameInput.size());
        ImGui::SameLine();
        if (ImGui::Button("저장##확인") && sceneNameInput[0] != '\0') {
            pendingSceneSave = sceneRoot / sceneNameInput.data();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
    if (ImGui::BeginPopup("장면 열기")) {
        if (sceneFiles.empty()) {
            ImGui::TextDisabled("%s 에 저장된 장면이 없습니다", sceneRoot.string().c_str());
        }
        for (const std::filesystem::path& file : sceneFiles) {
            if (ImGui::Selectable(file.filename().string().c_str())) {
                pendingSceneOpen = file;
            }
        }
        ImGui::EndPopup();
    }
    if (ImGui::BeginPopup("모델 선택")) {
        if (modelFiles.empty()) {
            ImGui::TextDisabled("%s 에 glTF 파일이 없습니다", modelRoot.string().c_str());
        }
        for (const std::filesystem::path& file : modelFiles) {
            if (ImGui::Selectable(file.filename().string().c_str())) {
                pendingModel = file;
                ImGui::CloseCurrentPopup();
            }
        }
        ImGui::Separator();
        ImGui::SetNextItemWidth(320.0F);
        ImGui::InputText("경로", modelPathInput.data(), modelPathInput.size());
        ImGui::SameLine();
        if (ImGui::Button("열기") && modelPathInput[0] != '\0') {
            pendingModel = std::filesystem::path{modelPathInput.data()};
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

void Editor::buildCreateItems(scene::Scene& active, const gfx::GeometryStore& geometry, int parent) {
    if (ImGui::MenuItem("빈 오브젝트")) {
        deferred = [this, &active, parent] { createEmptyObject(active, parent); };
    }
    if (ImGui::BeginMenu("기본 도형")) {
        for (uint32_t i = 0; i < primitiveMeshes.size(); ++i) {
            uint32_t meshIndex = primitiveMeshes[i];
            ImGui::BeginDisabled(!geometry.meshLive(meshIndex));
            if (ImGui::MenuItem(asset::primitiveLabel(static_cast<asset::Primitive>(i)))) {
                deferred = [this, &active, &geometry, meshIndex, parent] {
                    createMeshObject(active, geometry, meshIndex, parent);
                };
            }
            ImGui::EndDisabled();
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("메쉬")) {
        // ponytail: 메쉬가 수백 개면 메뉴가 화면을 넘는다. 그때는 검색 칸을 둔다.
        if (geometry.meshCount() == 0) {
            ImGui::TextDisabled("올라온 메쉬가 없습니다");
        }
        for (uint32_t meshIndex = 0; meshIndex < geometry.meshCount(); ++meshIndex) {
            if (!geometry.meshLive(meshIndex)) {
                continue;
            }
            const asset::Material& material = geometry.material(geometry.mesh(meshIndex).materialIndex);
            std::string label = std::to_string(meshIndex) + ": " + geometry.meshName(meshIndex) + " / " + material.name;
            if (ImGui::MenuItem(label.c_str())) {
                deferred = [this, &active, &geometry, meshIndex, parent] {
                    createMeshObject(active, geometry, meshIndex, parent);
                };
            }
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("조명")) {
        constexpr std::array<const char*, 4> LIGHT_NAMES{"방향광", "점광", "스폿광", "영역광"};
        for (uint32_t type = 0; type < LIGHT_NAMES.size(); ++type) {
            if (ImGui::MenuItem(LIGHT_NAMES[type])) {
                deferred = [this, &active, type, parent] {
                    createLightObject(active, static_cast<scene::LightType>(type), parent);
                };
            }
        }
        ImGui::EndMenu();
    }
}

void Editor::createEmptyObject(scene::Scene& active, int parent) {
    scene::Object object;
    object.name = "빈 오브젝트";
    object.parent = parent;
    // 뿌리에 만들면 카메라 앞에 두고, 자식으로 만들면 부모 자리에 둔다.
    if (parent < 0) {
        object.transform.position = active.camera.position + active.camera.forward() * 2.0F;
    }
    active.objects.push_back(std::move(object));
    selectOnly(static_cast<int>(active.objects.size()) - 1);
}

void Editor::createMeshObject(scene::Scene& active,
                              const gfx::GeometryStore& geometry,
                              uint32_t meshIndex,
                              int parent) {
    if (meshIndex >= geometry.meshCount()) {
        return;
    }
    scene::Object object;
    object.name = geometry.meshName(meshIndex);
    object.parent = parent;
    if (parent < 0) {
        // 새 오브젝트는 카메라 앞쪽, 바운딩 반지름을 고려한 거리에 놓는다.
        float radius = std::max(geometry.mesh(meshIndex).boundingSphere.w, 0.1F);
        object.transform.position = active.camera.position + active.camera.forward() * (radius * 3.0F);
    }
    active.objects.push_back(std::move(object));
    auto index = static_cast<uint32_t>(active.objects.size() - 1);
    active.attachMeshRenderer(index, meshIndex);
    selectOnly(static_cast<int>(index));
}

void Editor::createLightObject(scene::Scene& active, scene::LightType type, int parent) {
    constexpr std::array<const char*, 4> LIGHT_NAMES{"방향광", "점광", "스폿광", "영역광"};
    scene::Light light;
    light.type = type;
    active.lights.push_back(light);

    scene::Object object;
    object.name = LIGHT_NAMES[static_cast<size_t>(type)];
    object.parent = parent;
    object.light = static_cast<int32_t>(active.lights.size()) - 1;
    if (type == scene::LightType::DIRECTIONAL) {
        object.transform.rotation = glm::quat(glm::radians(glm::vec3{-50.0F, -30.0F, 0.0F}));
    } else if (parent < 0) {
        // 카메라 앞쪽에 놓고 보고 있는 쪽을 비추게 한다.
        object.transform.position = active.camera.position + active.camera.forward();
        object.transform.rotation = glm::quatLookAt(active.camera.forward(), glm::vec3{0.0F, 1.0F, 0.0F});
    }
    active.objects.push_back(std::move(object));
    selectOnly(static_cast<int>(active.objects.size()) - 1);
}

void Editor::newScene(scene::SceneManager& scenes) {
    // Unity 처럼 새 장면에는 방향광 하나를 둔다.
    scene::Scene& created = scenes.create("GameScene");
    createLightObject(created, scene::LightType::DIRECTIONAL, -1);
    scenes.setActive(scenes.count() - 1);
    clearSelection();
}

void Editor::duplicateSelection(scene::Scene& active) {
    // 여러 개를 고른 채 복제하면 각각 복제하고 사본들을 새 선택으로 삼는다.
    std::vector<int> copies;
    for (int selected : selection) {
        if (selected >= 0 && selected < static_cast<int>(active.objects.size())) {
            copies.push_back(static_cast<int>(active.duplicateObject(static_cast<uint32_t>(selected))));
        }
    }
    selection = std::move(copies);
}

void Editor::deleteSelection(scene::Scene& active) {
    // 하나씩 지우면 첫 삭제가 인덱스를 밀어 나머지가 엉뚱한 것을 가리킨다. 한 번에 넘긴다.
    std::vector<uint32_t> doomed;
    for (int selected : selection) {
        if (selected >= 0) {
            doomed.push_back(static_cast<uint32_t>(selected));
        }
    }
    active.removeObjects(doomed);
    clearSelection();
}

void Editor::unparentSelection(scene::Scene& active) {
    for (int selected : selection) {
        if (selected >= 0 && selected < static_cast<int>(active.objects.size())) {
            reparent(active, selected, -1);
        }
    }
}

void Editor::reparent(scene::Scene& active, int child, int parent) {
    if (child < 0 || child >= static_cast<int>(active.objects.size()) || child == parent ||
        parent >= static_cast<int>(active.objects.size())) {
        return;
    }
    // 자기 자손 밑으로 들어가면 순환이다.
    if (parent >= 0 && active.isDescendant(static_cast<uint32_t>(parent), static_cast<uint32_t>(child))) {
        return;
    }
    // 부모가 바뀌어도 화면에서의 위치는 그대로 두려고 지역 변환을 다시 계산한다.
    glm::mat4 world = active.worldMatrix(static_cast<uint32_t>(child));
    glm::mat4 parentWorld = parent >= 0 ? active.worldMatrix(static_cast<uint32_t>(parent)) : glm::mat4{1.0F};
    scene::Object& object = active.objects[static_cast<size_t>(child)];
    object.parent = parent;
    object.transform = scene::Transform::fromMatrix(glm::inverse(parentWorld) * world);
}

void Editor::handleShortcuts(scene::SceneManager& scenes, const gfx::GeometryStore& geometry) {
    (void)geometry;
    ImGuiIO& io = ImGui::GetIO();
    if (io.WantTextInput) {
        return;
    }
    scene::Scene& active = scenes.active();
    if (io.KeyCtrl) {
        if (ImGui::IsKeyPressed(ImGuiKey_N, false)) {
            newScene(scenes);
        } else if (ImGui::IsKeyPressed(ImGuiKey_O, false)) {
            popupRequest = PopupRequest::OPEN_SCENE;
        } else if (ImGui::IsKeyPressed(ImGuiKey_S, false)) {
            popupRequest = PopupRequest::SAVE_SCENE;
        } else if (ImGui::IsKeyPressed(ImGuiKey_D, false) && hasSelection()) {
            duplicateSelection(active);
        } else if (ImGui::IsKeyPressed(ImGuiKey_P, false)) {
            if (active.simulating) {
                stopSimulation(scenes);
            } else {
                startSimulation(scenes);
            }
        }
    } else if (ImGui::IsKeyPressed(ImGuiKey_Delete, false) && hasSelection()) {
        deleteSelection(active);
    }
}

void Editor::setModelLoader(std::filesystem::path root, std::function<void(const std::filesystem::path&)> loader) {
    modelRoot = std::move(root);
    modelLoader = std::move(loader);
}

void Editor::setSceneIo(std::filesystem::path root,
                        std::function<void(const std::filesystem::path&)> saver,
                        std::function<void(const std::filesystem::path&)> opener) {
    sceneRoot = std::move(root);
    sceneSaver = std::move(saver);
    sceneOpener = std::move(opener);
}

void Editor::drawHierarchyNode(scene::Scene& active, const std::vector<std::vector<int>>& children, int index) {
    scene::Object& object = active.objects[static_cast<size_t>(index)];
    const std::vector<int>& own = children[static_cast<size_t>(index)];
    bool hasChildren = !own.empty();

    ImGui::PushID(index);
    ImGui::Checkbox("##visible", &object.visible);
    ImGui::SameLine();

    ImGuiTreeNodeFlags flags =
        ImGuiTreeNodeFlags_OpenOnArrow | ImGuiTreeNodeFlags_SpanAvailWidth | ImGuiTreeNodeFlags_DefaultOpen;
    if (!hasChildren) {
        flags |= ImGuiTreeNodeFlags_Leaf;
    }
    if (isSelected(index)) {
        flags |= ImGuiTreeNodeFlags_Selected;
    }
    bool open = ImGui::TreeNodeEx("##node", flags, "%s", object.name.c_str());
    if (ImGui::IsItemClicked() && !ImGui::IsItemToggledOpen()) {
        if (ImGui::GetIO().KeyCtrl) {
            toggleSelect(index);
        } else {
            selectOnly(index);
        }
    }

    // 끌어다 놓아 부모를 바꾼다. 실제 적용은 순회가 끝난 뒤에 한다.
    if (ImGui::BeginDragDropSource()) {
        ImGui::SetDragDropPayload(HIERARCHY_PAYLOAD, &index, sizeof(index));
        ImGui::TextUnformatted(object.name.c_str());
        ImGui::EndDragDropSource();
    }
    if (ImGui::BeginDragDropTarget()) {
        const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(HIERARCHY_PAYLOAD);
        if (payload != nullptr) {
            pendingChild = *static_cast<const int*>(payload->Data);
            pendingParent = index;
        }
        ImGui::EndDragDropTarget();
    }

    // 우클릭 메뉴. 고른 것에 없는 노드를 누르면 그것만 고른다. 배열을 바꾸는 동작은 순회가 끝난 뒤 한다.
    if (ImGui::BeginPopupContextItem("##nodeMenu")) {
        if (!isSelected(index)) {
            selectOnly(index);
        }
        if (ImGui::MenuItem("복제", "Ctrl+D")) {
            deferred = [this, &active] { duplicateSelection(active); };
        }
        if (ImGui::MenuItem("삭제", "Delete")) {
            deferred = [this, &active] { deleteSelection(active); };
        }
        if (ImGui::MenuItem("부모 해제", nullptr, false, object.parent >= 0)) {
            deferred = [this, &active] { unparentSelection(active); };
        }
        ImGui::Separator();
        if (ImGui::BeginMenu("자식 추가")) {
            buildCreateItems(active, *geometryStore, index);
            ImGui::EndMenu();
        }
        ImGui::EndPopup();
    }

    if (open) {
        for (int child : own) {
            drawHierarchyNode(active, children, child);
        }
        ImGui::TreePop();
    }
    ImGui::PopID();
}

void Editor::buildHierarchy(scene::SceneManager& scenes, const gfx::GeometryStore& geometry) {
    if (!ImGui::Begin(WINDOW_HIERARCHY)) {
        ImGui::End();
        return;
    }

    if (ImGui::BeginCombo("장면", scenes.active().name.c_str())) {
        for (size_t i = 0; i < scenes.count(); ++i) {
            bool selected = i == scenes.current();
            if (ImGui::Selectable(scenes.at(i).name.c_str(), selected)) {
                scenes.setActive(i);
                clearSelection();
            }
        }
        ImGui::EndCombo();
    }
    ImGui::Separator();

    scene::Scene& active = scenes.active();

    // 여기부터는 오브젝트의 부모-자식 구조만 보여준다. 만들기·복제·삭제는 우클릭 메뉴와 메뉴바에 있다.
    // 부모별 자식 목록을 한 번 만든다. 노드마다 전체를 훑으면 오브젝트 만 개에서 프레임당 수백 ms 다.
    std::vector<std::vector<int>> children(active.objects.size());
    for (int i = 0; i < static_cast<int>(active.objects.size()); ++i) {
        int parent = active.objects[static_cast<size_t>(i)].parent;
        if (parent >= 0 && parent < static_cast<int>(active.objects.size())) {
            children[static_cast<size_t>(parent)].push_back(i);
        }
    }
    for (int i = 0; i < static_cast<int>(active.objects.size()); ++i) {
        if (active.objects[static_cast<size_t>(i)].parent < 0) {
            drawHierarchyNode(active, children, i);
        }
    }

    // 노드 밖에 놓으면 뿌리로 끌어올린다. 창 전체를 대상으로 잡는다. 전에는 남는 자리에 Dummy 를 두었는데
    // 트리가 창을 채우면 크기가 0 이라 받지 못했다. ImGui 는 면적이 작은 대상을 우선하므로 노드 위에
    // 놓으면 그 노드가 받는다.
    ImGuiWindow* window = ImGui::GetCurrentWindow();
    if (ImGui::BeginDragDropTargetCustom(window->InnerRect, window->ID)) {
        const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(HIERARCHY_PAYLOAD);
        if (payload != nullptr) {
            pendingChild = *static_cast<const int*>(payload->Data);
            pendingParent = -1;
        }
        ImGui::EndDragDropTarget();
    }

    // 빈 공간 우클릭. 노드 위에서는 노드의 메뉴가 뜬다.
    if (ImGui::BeginPopupContextWindow("##hierarchyMenu",
                                       ImGuiPopupFlags_MouseButtonRight | ImGuiPopupFlags_NoOpenOverItems)) {
        buildCreateItems(active, geometry, -1);
        ImGui::Separator();
        if (ImGui::MenuItem("모델 불러오기...")) {
            popupRequest = PopupRequest::LOAD_MODEL;
        }
        ImGui::EndPopup();
    }

    reparent(active, pendingChild, pendingParent);
    pendingChild = -1;

    ImGui::End();
}

void Editor::buildInspector(scene::Scene& active, const gfx::GeometryStore& geometry) {
    if (!ImGui::Begin(WINDOW_INSPECTOR)) {
        ImGui::End();
        return;
    }

    int selectedObject = primarySelection();
    if (selectedObject < 0 || selectedObject >= static_cast<int>(active.objects.size())) {
        ImGui::TextDisabled("선택된 오브젝트가 없습니다");
        ImGui::End();
        return;
    }

    scene::Object& object = active.objects[static_cast<size_t>(selectedObject)];
    auto objectIndex = static_cast<uint32_t>(selectedObject);
    {
        std::array<char, 256> nameInput{};
        std::copy_n(object.name.c_str(), std::min(object.name.size() + 1, nameInput.size() - 1), nameInput.begin());
        ImGui::SetNextItemWidth(-1.0F);
        if (ImGui::InputText("##name", nameInput.data(), nameInput.size())) {
            object.name = nameInput.data();
        }
    }
    ImGui::Separator();

    // 부품 헤더. 오른쪽 끝에 «제거» 단추를 겹쳐 둔다. 떼는 일은 배열을 압축하므로 패널을 다 그린 뒤 한다.
    auto componentHeader = [this, &active, objectIndex](const char* label, int32_t scene::Object::* handle) {
        ImGui::PushID(label);
        bool open = ImGui::CollapsingHeader(label, ImGuiTreeNodeFlags_DefaultOpen | ImGuiTreeNodeFlags_AllowOverlap);
        float buttonWidth = ImGui::CalcTextSize("제거").x + ImGui::GetStyle().FramePadding.x * 2.0F;
        ImGui::SameLine(ImGui::GetContentRegionMax().x - buttonWidth);
        if (ImGui::SmallButton("제거")) {
            deferred = [&active, objectIndex, handle] { active.detachComponent(objectIndex, handle); };
        }
        ImGui::PopID();
        return open;
    };

    if (ImGui::CollapsingHeader("변환", ImGuiTreeNodeFlags_DefaultOpen)) {
        ImGui::DragFloat3("위치", glm::value_ptr(object.transform.position), 0.01F);
        glm::vec3 euler = glm::degrees(glm::eulerAngles(object.transform.rotation));
        if (ImGui::DragFloat3("회전", glm::value_ptr(euler), 0.5F)) {
            object.transform.rotation = glm::quat(glm::radians(euler));
        }
        ImGui::DragFloat3("크기", glm::value_ptr(object.transform.scale), 0.01F, 0.001F, 1000.0F);
        if (object.parent >= 0) {
            ImGui::TextDisabled("부모: %s", active.objects[static_cast<size_t>(object.parent)].name.c_str());
        }
    }

    // 조명 속성. 위치와 방향은 변환에서 오므로 여기서는 나머지만 다룬다.
    if (object.light >= 0 && object.light < static_cast<int>(active.lights.size()) &&
        componentHeader("조명", &scene::Object::light)) {
        scene::Light& light = active.lights[static_cast<size_t>(object.light)];
        constexpr std::array<const char*, 4> LIGHT_NAMES{"방향광", "점광", "스폿광", "영역광"};
        auto typeIndex = static_cast<int>(light.type);
        if (ImGui::Combo("종류", &typeIndex, LIGHT_NAMES.data(), static_cast<int>(LIGHT_NAMES.size()))) {
            light.type = static_cast<scene::LightType>(typeIndex);
        }
        ImGui::ColorEdit3("색", glm::value_ptr(light.color));
        ImGui::DragFloat("세기", &light.intensity, 0.05F, 0.0F, 1000.0F);
        if (light.type != scene::LightType::DIRECTIONAL) {
            ImGui::DragFloat("거리", &light.range, 0.1F, 0.01F, 10000.0F);
        }
        if (light.type == scene::LightType::SPOT) {
            ImGui::DragFloat("안쪽 각", &light.innerConeDegrees, 0.5F, 0.0F, 89.0F);
            ImGui::DragFloat("바깥 각", &light.outerConeDegrees, 0.5F, 0.0F, 89.0F);
            light.innerConeDegrees = std::min(light.innerConeDegrees, light.outerConeDegrees);
        }
        if (light.type == scene::LightType::AREA) {
            ImGui::DragFloat2("크기", glm::value_ptr(light.size), 0.05F, 0.01F, 1000.0F);
            ImGui::TextDisabled("영역광은 그림자를 만들지 않습니다");
        } else {
            ImGui::Checkbox("그림자", &light.castsShadow);
        }
    }

    // 애니메이션 컨트롤러. 스켈레톤을 가진 오브젝트에만 나온다.
    if (object.animator >= 0 && object.animator < static_cast<int>(active.animators.size())) {
        scene::Animator& animator = active.animators[static_cast<size_t>(object.animator)];
        if (ImGui::CollapsingHeader("애니메이션", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (animator.skeleton.animations.empty()) {
                ImGui::TextDisabled("클립이 없는 스켈레톤입니다");
            } else {
                animator.clip = std::min(animator.clip, static_cast<uint32_t>(animator.skeleton.animations.size()) - 1);
                const asset::Animation& clip = animator.skeleton.animations[animator.clip];
                if (ImGui::BeginCombo("클립", clip.name.c_str())) {
                    for (uint32_t i = 0; i < animator.skeleton.animations.size(); ++i) {
                        if (ImGui::Selectable(animator.skeleton.animations[i].name.c_str(), i == animator.clip)) {
                            animator.clip = i;
                            animator.clipTime = 0.0F;
                        }
                    }
                    ImGui::EndCombo();
                }
                ImGui::Checkbox("재생", &animator.playing);
                ImGui::SameLine();
                ImGui::SetNextItemWidth(90.0F);
                ImGui::DragFloat("속도", &animator.speed, 0.01F, -4.0F, 4.0F);
                ImGui::SliderFloat("시간", &animator.clipTime, 0.0F, std::max(clip.duration, 0.001F), "%.2f s");
            }
            ImGui::Text("조인트 %zu, 스킨 %zu", animator.skeleton.nodes.size(), animator.skeleton.skins.size());
        }
    }

    // Mesh Renderer. 메쉬는 바꿀 수 있고 재질은 메쉬에 딸려 온다.
    if (object.meshRenderer >= 0 && componentHeader("Mesh Renderer", &scene::Object::meshRenderer)) {
        uint32_t selectedMesh = active.meshOf(objectIndex);
        bool live = geometry.meshLive(selectedMesh);
        std::string current = live ? std::to_string(selectedMesh) + ": " + geometry.meshName(selectedMesh) : "(없음)";
        if (ImGui::BeginCombo("메쉬", current.c_str())) {
            for (uint32_t meshIndex = 0; meshIndex < geometry.meshCount(); ++meshIndex) {
                if (!geometry.meshLive(meshIndex)) {
                    continue;
                }
                std::string label = std::to_string(meshIndex) + ": " + geometry.meshName(meshIndex);
                if (ImGui::Selectable(label.c_str(), meshIndex == selectedMesh)) {
                    active.attachMeshRenderer(objectIndex, meshIndex, active.skinOf(objectIndex));
                }
            }
            ImGui::EndCombo();
        }
        if (live) {
            const gfx::GpuMesh& mesh = geometry.mesh(selectedMesh);
            ImGui::Text("삼각형 %u", mesh.indexCount / 3);
            ImGui::Text("바운딩 반지름 %.3f", mesh.boundingSphere.w);

            const asset::Material& material = geometry.material(mesh.materialIndex);
            ImGui::Text("재질: %s", material.name.c_str());
            ImGui::Text("알파 경로: %s%s", alphaModeName(material.alphaMode), material.doubleSided ? " (양면)" : "");
            ImGui::Text("금속성 %.2f / 거칠기 %.2f", material.metallicFactor, material.roughnessFactor);
            ImGui::ColorButton("기저 색",
                               ImVec4{material.baseColorFactor.r,
                                      material.baseColorFactor.g,
                                      material.baseColorFactor.b,
                                      material.baseColorFactor.a});
        } else {
            ImGui::TextDisabled("메쉬가 해제되었거나 아직 올라오지 않았다");
        }
    }

    if (object.rigidBody >= 0 && object.rigidBody < static_cast<int>(active.rigidBodies.size()) &&
        componentHeader("강체", &scene::Object::rigidBody)) {
        scene::RigidBody& body = active.rigidBodies[static_cast<size_t>(object.rigidBody)];
        constexpr std::array<const char*, scene::COLLIDER_SHAPE_COUNT> SHAPE_NAMES{
            "구", "상자", "평면", "원기둥", "캡슐", "메쉬"};
        auto shapeIndex = static_cast<int>(body.shape);
        if (ImGui::Combo("모양", &shapeIndex, SHAPE_NAMES.data(), static_cast<int>(SHAPE_NAMES.size()))) {
            body.shape = static_cast<scene::ColliderShape>(shapeIndex);
        }
        bool alwaysKinematic = body.shape == scene::ColliderShape::PLANE || body.shape == scene::ColliderShape::MESH;
        switch (body.shape) {
        case scene::ColliderShape::SPHERE:
            ImGui::DragFloat("반지름", &body.radius, 0.01F, 0.01F, 100.0F);
            break;
        case scene::ColliderShape::BOX:
            ImGui::DragFloat3("반쪽 크기", glm::value_ptr(body.halfExtents), 0.01F, 0.01F, 100.0F);
            break;
        case scene::ColliderShape::CYLINDER:
        case scene::ColliderShape::CAPSULE:
            ImGui::DragFloat("반지름", &body.radius, 0.01F, 0.01F, 100.0F);
            ImGui::DragFloat("반높이", &body.halfExtents.y, 0.01F, 0.01F, 100.0F);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip(body.shape == scene::ColliderShape::CAPSULE
                                      ? "축은 오브젝트의 +Y. 반구를 뺀 몸통의 반높이다"
                                      : "축은 오브젝트의 +Y");
            }
            break;
        case scene::ColliderShape::MESH:
            if (active.colliderMesh(objectIndex) != nullptr) {
                ImGui::TextDisabled("Mesh Renderer 의 메쉬(삼각형 %zu개, 굵은 LOD)를 그대로 쓴다. 늘 운동학이다",
                                    active.colliderMesh(objectIndex)->indices.size() / 3);
            } else {
                ImGui::TextDisabled("Mesh Renderer 가 없어 부딪히지 않는다. 늘 운동학이다");
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("바닥·지형용. 움직이는 물체가 이 삼각형에 닿는다. 유체는 통과한다");
            }
            break;
        case scene::ColliderShape::PLANE:
            ImGui::TextDisabled("오브젝트의 +Y 가 법선인 무한 평면. 늘 운동학이다");
            break;
        }
        ImGui::BeginDisabled(alwaysKinematic);
        ImGui::Checkbox("운동학", &body.kinematic);
        ImGui::SameLine();
        ImGui::Checkbox("중력", &body.useGravity);
        ImGui::DragFloat("질량", &body.mass, 0.1F, 0.01F, 10000.0F);
        ImGui::EndDisabled();
        ImGui::SliderFloat("반발", &body.restitution, 0.0F, 1.0F, "%.2f");
        ImGui::SliderFloat("마찰", &body.friction, 0.0F, 1.0F, "%.2f");

        constexpr std::array<const char*, 3> RIGID_BACKENDS{"자동", "CPU", "GPU"};
        // 컴퓨트 파이프라인을 만들지 못한 장치에서는 GPU 를 고를 수 없다.
        bool rigidGpuUsable = rigidStatus.gpuAvailable;
        if (ImGui::BeginCombo("백엔드", RIGID_BACKENDS[static_cast<size_t>(body.backend)])) {
            for (uint32_t i = 0; i < RIGID_BACKENDS.size(); ++i) {
                bool usable = rigidGpuUsable || i != static_cast<uint32_t>(scene::SimulationBackend::GPU);
                ImGui::BeginDisabled(!usable);
                if (ImGui::Selectable(RIGID_BACKENDS[i], static_cast<uint32_t>(body.backend) == i)) {
                    body.backend = static_cast<scene::SimulationBackend>(i);
                }
                ImGui::EndDisabled();
                if (!usable && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                    ImGui::SetTooltip("이 장치에서는 강체 컴퓨트 파이프라인을 만들지 못했다");
                }
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("자동은 CPU 다. GPU 는 접촉을 Jacobi 로 풀어 CPU 와 수치가 다르고,\n"
                              "결과가 몇 프레임 늦게 반영된다.\n"
                              "두 백엔드는 서로 부딪히지 않으므로 한 장면에서 섞지 않는다");
        }
        if (body.backend == scene::SimulationBackend::GPU) {
            ImGui::TextDisabled("Jacobi %u 회, GPU 강체 %u 개. 쌓인 물체가 CPU 보다 물렁하다",
                                gfx::RIGID_SOLVER_ITERATIONS,
                                rigidStatus.gpuBodies);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("CPU 백엔드 강체와는 서로 부딪히지 않는다. 바닥도 GPU 로 맞춰야 한다");
            }
        }
        if (active.simulating) {
            ImGui::Text("속도 (%.2f, %.2f, %.2f)", body.velocity.x, body.velocity.y, body.velocity.z);
        }
    }

    if (object.fluid >= 0 && object.fluid < static_cast<int>(active.fluids.size()) &&
        componentHeader("유체", &scene::Object::fluid)) {
        scene::Fluid& fluid = active.fluids[static_cast<size_t>(object.fluid)];
        constexpr std::array<const char*, 3> BACKEND_NAMES{"자동", "CPU", "GPU"};
        // 컴퓨트 파이프라인을 만들지 못한 장치에서는 GPU 를 고를 수 없다. 하드웨어 게이트 규약대로
        // 항목을 잠그고 사유를 보여 준다.
        bool gpuUsable = renderer.fluidGpuAvailable();
        if (ImGui::BeginCombo("백엔드", BACKEND_NAMES[static_cast<size_t>(fluid.backend)])) {
            for (uint32_t i = 0; i < BACKEND_NAMES.size(); ++i) {
                bool usable = gpuUsable || i != static_cast<uint32_t>(scene::SimulationBackend::GPU);
                ImGui::BeginDisabled(!usable);
                if (ImGui::Selectable(BACKEND_NAMES[i], static_cast<uint32_t>(fluid.backend) == i)) {
                    fluid.backend = static_cast<scene::SimulationBackend>(i);
                }
                ImGui::EndDisabled();
                if (!usable && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                    ImGui::SetTooltip("이 장치에서는 유체 컴퓨트 파이프라인을 만들지 못했다");
                }
            }
            ImGui::EndCombo();
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("자동은 GPU 를 쓰되 만들지 못하면 CPU 로 내려간다");
        }
        // 자동이 무엇을 골랐는지는 렌더러만 안다. 그것을 그대로 보여 준다.
        ImGui::SameLine();
        bool onCpu = renderer.fluidOnCpu(static_cast<uint32_t>(object.fluid));
        ImGui::TextDisabled("(지금 %s)", onCpu ? "CPU" : "GPU");

        int particles = static_cast<int>(fluid.particleCount);
        if (ImGui::SliderInt("입자 수", &particles, 64, gfx::FLUID_MAX_PARTICLES)) {
            fluid.particleCount = static_cast<uint32_t>(particles);
        }
        // 하드웨어 프로파일이 상한을 낮췄으면 슬라이더 값보다 적게 뿌린다. 그 사실을 여기서 말한다.
        if (fluid.particleCount > renderer.settings.fluidParticleLimit) {
            ImGui::SameLine();
            ImGui::TextDisabled("(상한 %u)", renderer.settings.fluidParticleLimit);
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("자동 튜닝이 이 기기의 상한을 정했다. --auto-tune off 로 풀 수 있다");
            }
        }
        constexpr std::array<const char*, 2> DISPLAY_NAMES{"입자", "표면"};
        bool surfaceUsable = renderer.fluidSurfaceAvailable() || onCpu;
        ImGui::BeginDisabled(!surfaceUsable);
        auto displayIndex = static_cast<int>(fluid.display);
        if (ImGui::Combo("표시", &displayIndex, DISPLAY_NAMES.data(), static_cast<int>(DISPLAY_NAMES.size()))) {
            fluid.display = static_cast<scene::FluidDisplay>(displayIndex);
        }
        ImGui::EndDisabled();
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
            ImGui::SetTooltip(surfaceUsable
                                  ? "표면은 마칭 큐브로 등치면을 뽑아 물처럼 그린다.\n"
                                    "Path Tracing 은 이 표면을 굴절·흡수로 추적한다. Ray Traced Reflections 와\n"
                                    "그림자는 물을 지나간다"
                                  : "이 장치에서는 표면 컴퓨트를 만들지 못했다. CPU 백엔드로는 쓸 수 있다");
        }
        if (fluid.display == scene::FluidDisplay::SURFACE) {
            uint32_t ceiling = onCpu ? gfx::FLUID_MAX_CPU_SURFACE_RESOLUTION : gfx::FLUID_MAX_SURFACE_RESOLUTION;
            auto resolution = static_cast<int>(fluid.surfaceResolution);
            if (ImGui::SliderInt("격자 해상도", &resolution, 8, static_cast<int>(gfx::FLUID_MAX_SURFACE_RESOLUTION))) {
                fluid.surfaceResolution = static_cast<uint32_t>(resolution);
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("축마다의 셀 수다. 표본 수는 세제곱으로 는다");
            }
            // CPU 백엔드는 표본마다 입자 전부를 훑으므로 렌더러가 더 낮게 묶는다. 그 사실을 말해 준다.
            if (fluid.surfaceResolution > ceiling) {
                ImGui::SameLine();
                ImGui::TextDisabled("(CPU 상한 %u)", ceiling);
            }
            ImGui::SliderFloat("등치값", &fluid.surfaceIso, 0.05F, 4.0F, "%.2f");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("작으면 표면이 부풀고 크면 물이 얇아져 구멍이 뚫린다");
            }
            ImGui::ColorEdit3("물 색", glm::value_ptr(fluid.waterColor));
            ImGui::SliderFloat("표면 거칠기", &fluid.surfaceRoughness, 0.01F, 0.5F, "%.3f");
            ImGui::DragFloat3("흡수 계수", glm::value_ptr(fluid.absorption), 0.05F, 0.0F, 20.0F);
            ImGui::DragFloat("두께 배율", &fluid.thicknessScale, 0.05F, 0.05F, 20.0F);
        }
        ImGui::DragFloat3("방출 반쪽 크기", glm::value_ptr(fluid.emitterHalfExtents), 0.01F, 0.01F, 50.0F);
        ImGui::DragFloat("입자 반지름", &fluid.particleRadius, 0.001F, 0.005F, 1.0F, "%.3f");
        ImGui::DragFloat("커널 반지름", &fluid.smoothingRadius, 0.005F, 0.01F, 2.0F, "%.3f");
        ImGui::DragFloat("기준 밀도", &fluid.restDensity, 1.0F, 1.0F, 10000.0F);
        ImGui::DragFloat("강성", &fluid.stiffness, 1.0F, 1.0F, 5000.0F);
        ImGui::DragFloat("점성", &fluid.viscosity, 0.01F, 0.0F, 50.0F);
        ImGui::DragFloat3("용기 최소", glm::value_ptr(fluid.containerMin), 0.05F, -100.0F, 100.0F);
        ImGui::DragFloat3("용기 최대", glm::value_ptr(fluid.containerMax), 0.05F, -100.0F, 100.0F);
        ImGui::DragFloat3("중력", glm::value_ptr(fluid.gravity), 0.1F, -100.0F, 100.0F);
        ImGui::TextDisabled("입자는 GPU 에서 계산해 내장 구로 그린다. Path Tracing에도 보인다");
    }

    ImGui::Separator();
    if (ImGui::Button("컴포넌트 추가", ImVec2{-1.0F, 0.0F})) {
        ImGui::OpenPopup("컴포넌트 추가");
    }
    if (ImGui::BeginPopup("컴포넌트 추가")) {
        ImGui::BeginDisabled(object.meshRenderer >= 0);
        if (ImGui::BeginMenu("Mesh Renderer")) {
            for (uint32_t meshIndex = 0; meshIndex < geometry.meshCount(); ++meshIndex) {
                if (!geometry.meshLive(meshIndex)) {
                    continue;
                }
                std::string label = std::to_string(meshIndex) + ": " + geometry.meshName(meshIndex);
                if (ImGui::MenuItem(label.c_str())) {
                    active.attachMeshRenderer(objectIndex, meshIndex);
                }
            }
            ImGui::EndMenu();
        }
        ImGui::EndDisabled();
        ImGui::BeginDisabled(object.light >= 0);
        if (ImGui::MenuItem("조명")) {
            active.attachLight(objectIndex);
        }
        ImGui::EndDisabled();
        ImGui::BeginDisabled(object.rigidBody >= 0);
        if (ImGui::MenuItem("강체")) {
            scene::RigidBody body;
            // 메쉬가 있으면 경계 구를 콜라이더 크기로 삼는다.
            uint32_t mesh = active.meshOf(objectIndex);
            if (geometry.meshLive(mesh)) {
                body.radius = std::max(geometry.mesh(mesh).boundingSphere.w, 0.01F);
                body.halfExtents = glm::vec3{body.radius * 0.7F};
            }
            active.attachRigidBody(objectIndex, body);
        }
        ImGui::EndDisabled();
        ImGui::BeginDisabled(object.fluid >= 0);
        if (ImGui::MenuItem("유체")) {
            active.attachFluid(objectIndex);
        }
        ImGui::EndDisabled();
        ImGui::EndPopup();
    }
    ImGui::End();
}

void Editor::buildSceneView(scene::Scene& active) {
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2{0.0F, 0.0F});
    bool open = ImGui::Begin(WINDOW_SCENE);
    ImGui::PopStyleVar();
    if (!open) {
        sceneHovered = false;
        ImGui::End();
        return;
    }

    // ImGui 좌표는 논리 단위이므로, 실제 픽셀 해상도로 그리려면 프레임버퍼 배율을 곱해야 한다.
    ImVec2 available = ImGui::GetContentRegionAvail();
    ImVec2 scale = ImGui::GetIO().DisplayFramebufferScale;
    if (available.x >= 1.0F && available.y >= 1.0F) {
        viewportExtent = {static_cast<uint32_t>(available.x * scale.x), static_cast<uint32_t>(available.y * scale.y)};
    }

    // 툴바에서 고른 렌더 타깃을 장면 자리에 보여 준다. 기본은 표시 이미지고, 다른 대상은 자기 비율로
    // 가운데에 맞춰 넣는다. 기즈모와 클릭 선택은 표시 이미지일 때만 받는다.
    std::vector<gfx::Renderer::TargetView> targets = renderer.targetViews();
    int targetCount = static_cast<int>(targets.size());
    int presentIndex = 0;
    for (int i = 0; i < targetCount; ++i) {
        if (targets[static_cast<size_t>(i)].views[0] == renderer.presentView()) {
            presentIndex = i;
        }
    }
    // 렌더 모드가 바뀌어 고른 대상이 더는 채워지지 않으면 표시 이미지로 돌아간다.
    if (selectedTarget < 0 || selectedTarget >= targetCount ||
        !targets[static_cast<size_t>(selectedTarget)].available) {
        selectedTarget = presentIndex;
    }
    const gfx::Renderer::TargetView& target = targets[static_cast<size_t>(selectedTarget)];
    bool presentView = selectedTarget == presentIndex;
    int sliceCount = static_cast<int>(target.views.size());
    selectedSlice = std::clamp(selectedSlice, 0, sliceCount - 1);

    ImVec2 imageArea = available;
    ImVec2 imageOrigin = ImGui::GetCursorPos();
    if (!presentView) {
        float aspect = static_cast<float>(target.extent.width) / static_cast<float>(std::max(target.extent.height, 1U));
        float height = std::min(available.y, available.x / aspect);
        imageArea = ImVec2{height * aspect, height};
        ImGui::SetCursorPos(ImVec2{imageOrigin.x + (available.x - imageArea.x) * 0.5F,
                                   imageOrigin.y + (available.y - imageArea.y) * 0.5F});
    }
    ImGui::Image(ImTextureRef{static_cast<ImTextureID>(reinterpret_cast<uintptr_t>(
                     textureFor(target.views[static_cast<size_t>(selectedSlice)], target.layout)))},
                 imageArea);
    sceneHovered = ImGui::IsItemHovered();
    bool imageClicked = presentView && ImGui::IsItemClicked(ImGuiMouseButton_Left);

    ImVec2 imagePosition = ImGui::GetItemRectMin();
    ImVec2 imageSize = ImGui::GetItemRectSize();
    // 툴바는 대상 크기와 무관하게 장면 창 왼쪽 위에 둔다.
    ImVec2 toolbarPosition = ImVec2{ImGui::GetWindowPos().x + imageOrigin.x + 8.0F,
                                    ImGui::GetWindowPos().y + imageOrigin.y - ImGui::GetScrollY() + 8.0F};

    gizmoUsing = false;
    bool anySelected = hasSelection() && presentView;
    if (anySelected && imageSize.x > 1.0F && imageSize.y > 1.0F) {
        // 텍스트 입력 중에는 W/E/R 이 글자다.
        if (!active.camera.isLooking() && !ImGui::GetIO().WantTextInput) {
            if (ImGui::IsKeyPressed(ImGuiKey_W)) {
                gizmoOperation = ImGuizmo::TRANSLATE;
            } else if (ImGui::IsKeyPressed(ImGuiKey_E)) {
                gizmoOperation = ImGuizmo::ROTATE;
            } else if (ImGui::IsKeyPressed(ImGuiKey_R)) {
                gizmoOperation = ImGuizmo::SCALE;
            }
        }

        ImGuizmo::SetOrthographic(false);
        ImGuizmo::SetDrawlist();
        ImGuizmo::SetRect(imagePosition.x, imagePosition.y, imageSize.x, imageSize.y);

        float aspect = imageSize.x / imageSize.y;
        glm::mat4 view = active.camera.viewMatrix();
        glm::mat4 projection = active.camera.gizmoProjectionMatrix(aspect);

        // 기즈모는 기준 오브젝트 자리에 서고, 끌어서 생긴 차이를 선택 전체에 똑같이 먹인다.
        auto primary = static_cast<uint32_t>(primarySelection());
        glm::mat4 pivotBefore = active.worldMatrix(primary);
        glm::mat4 model = pivotBefore;

        // Ctrl 을 누르면 스냅을 건다. 값은 툴바에서 고치고, «스냅» 체크는 늘 걸리게 한다.
        glm::vec3 snapValue{snapTranslate};
        if (gizmoOperation == ImGuizmo::ROTATE) {
            snapValue = glm::vec3{snapRotate};
        } else if (gizmoOperation == ImGuizmo::SCALE) {
            snapValue = glm::vec3{snapScale};
        }
        const float* snap = snapAlways || ImGui::GetIO().KeyCtrl ? glm::value_ptr(snapValue) : nullptr;

        if (ImGuizmo::Manipulate(glm::value_ptr(view),
                                 glm::value_ptr(projection),
                                 static_cast<ImGuizmo::OPERATION>(gizmoOperation),
                                 static_cast<ImGuizmo::MODE>(gizmoMode),
                                 glm::value_ptr(model),
                                 nullptr,
                                 snap)) {
            glm::mat4 delta = model * glm::inverse(pivotBefore);

            // 조상이 함께 선택돼 있으면 그쪽에서 이미 옮겨진다. 여기서 또 먹이면 두 배로 움직인다.
            auto ancestorSelected = [this, &active](int index) {
                int parent = active.objects[static_cast<size_t>(index)].parent;
                while (parent >= 0) {
                    if (isSelected(parent)) {
                        return true;
                    }
                    parent = active.objects[static_cast<size_t>(parent)].parent;
                }
                return false;
            };

            // 옮기기 전에 세계 변환을 모두 재어 둔다. 하나를 고치면 그 자손의 세계 변환이 따라
            // 움직여, 순회 도중에 재면 뒤쪽이 어긋난다.
            std::vector<std::pair<uint32_t, glm::mat4>> targets;
            for (int selected : selection) {
                if (selected < 0 || selected >= static_cast<int>(active.objects.size()) || ancestorSelected(selected)) {
                    continue;
                }
                auto index = static_cast<uint32_t>(selected);
                targets.emplace_back(index, delta * active.worldMatrix(index));
            }
            for (const auto& [index, world] : targets) {
                int parent = active.objects[static_cast<size_t>(index)].parent;
                glm::mat4 parentWorld =
                    parent >= 0 ? active.worldMatrix(static_cast<uint32_t>(parent)) : glm::mat4{1.0F};
                active.objects[static_cast<size_t>(index)].transform =
                    scene::Transform::fromMatrix(glm::inverse(parentWorld) * world);
            }
        }
        gizmoUsing = ImGuizmo::IsUsing();
    }

    // 장면 뷰를 왼쪽 단추로 누르면 그 아래 오브젝트를 고른다. 기즈모를 잡고 있거나 시선을 돌리는
    // 중에는 받지 않는다. 빈 곳을 누르면 선택이 풀린다.
    if (imageClicked && !gizmoUsing && !ImGuizmo::IsOver() && !active.camera.isLooking() && geometryStore != nullptr &&
        imageSize.x > 1.0F && imageSize.y > 1.0F) {
        ImVec2 mouse = ImGui::GetMousePos();
        glm::vec2 uv{(mouse.x - imagePosition.x) / imageSize.x, (mouse.y - imagePosition.y) / imageSize.y};
        scene::Ray ray = active.camera.screenToRay(uv, imageSize.x / imageSize.y);
        int picked = pickObject(active, *geometryStore, ray);
        if (ImGui::GetIO().KeyCtrl) {
            toggleSelect(picked);
        } else {
            selectOnly(picked);
        }
    }

    // 조작 도구 선택은 장면 뷰 위에 겹쳐 둔다. 맨 앞에서 무엇을 볼지 고른다. 위젯을 채널 1 에 그리고
    // 그 뒤(채널 0)에 배경을 깐다. 장면 위에 바로 놓으면 밝은 하늘에서 글자가 묻힌다.
    ImGui::SetCursorScreenPos(toolbarPosition);
    ImDrawList* toolbarDrawList = ImGui::GetWindowDrawList();
    toolbarDrawList->ChannelsSplit(2);
    toolbarDrawList->ChannelsSetCurrent(1);
    // 기본 프레임 색은 배경과 거의 같아 라디오·체크박스 테두리가 묻힌다. 툴바 안에서만 밝게 둔다.
    ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4{0.34F, 0.34F, 0.36F, 1.0F});
    ImGui::BeginGroup();
    ImGui::SetNextItemWidth(160.0F);
    if (ImGui::BeginCombo("##target", target.name)) {
        for (int i = 0; i < targetCount; ++i) {
            const gfx::Renderer::TargetView& candidate = targets[static_cast<size_t>(i)];
            ImGui::BeginDisabled(!candidate.available);
            if (ImGui::Selectable(candidate.name, selectedTarget == i)) {
                selectedTarget = i;
                selectedSlice = 0;
            }
            ImGui::EndDisabled();
            if (!candidate.available && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("지금 렌더 모드에서는 채워지지 않는 대상이다");
            }
        }
        ImGui::EndCombo();
    }
    if (sliceCount > 1) {
        ImGui::SameLine();
        ImGui::SetNextItemWidth(120.0F);
        ImGui::SliderInt(target.sliceLabel, &selectedSlice, 0, sliceCount - 1);
    }
    ImGui::SameLine();
    ImGui::TextUnformatted("|");
    ImGui::SameLine();
    if (ImGui::RadioButton("이동", gizmoOperation == ImGuizmo::TRANSLATE)) {
        gizmoOperation = ImGuizmo::TRANSLATE;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("회전", gizmoOperation == ImGuizmo::ROTATE)) {
        gizmoOperation = ImGuizmo::ROTATE;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("크기", gizmoOperation == ImGuizmo::SCALE)) {
        gizmoOperation = ImGuizmo::SCALE;
    }
    ImGui::SameLine();
    if (ImGui::Button(gizmoMode == ImGuizmo::LOCAL ? "로컬" : "월드")) {
        gizmoMode = gizmoMode == ImGuizmo::LOCAL ? ImGuizmo::WORLD : ImGuizmo::LOCAL;
    }
    // 스냅 값은 지금 고른 조작 종류의 것을 보여 준다.
    ImGui::SameLine();
    ImGui::Checkbox("스냅", &snapAlways);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("Ctrl 을 누르는 동안 스냅이 걸린다. 체크하면 늘 건다");
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(72.0F);
    if (gizmoOperation == ImGuizmo::ROTATE) {
        ImGui::DragFloat("##snapRotate", &snapRotate, 1.0F, 1.0F, 90.0F, "%.0f°");
    } else if (gizmoOperation == ImGuizmo::SCALE) {
        ImGui::DragFloat("##snapScale", &snapScale, 0.01F, 0.01F, 2.0F, "x%.2f");
    } else {
        ImGui::DragFloat("##snapTranslate", &snapTranslate, 0.05F, 0.01F, 100.0F, "%.2f");
    }

    // 카메라 조작 방식. 기본은 궤도이고, 자유 모드는 1인칭처럼 날아다닌다.
    ImGui::SameLine();
    ImGui::TextUnformatted("|");
    ImGui::SameLine();
    bool orbit = active.camera.mode == scene::CameraMode::ORBIT;
    if (ImGui::RadioButton("궤도", orbit)) {
        active.camera.setMode(scene::CameraMode::ORBIT);
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("자유", !orbit)) {
        active.camera.setMode(scene::CameraMode::FLY);
    }
    ImGui::SameLine();
    ImGui::TextDisabled(orbit ? "(?) 우클릭 회전, 휠 확대, 가운데 단추 이동"
                              : "(?) 우클릭 시선, WASD/방향키 이동, 휠 속도");
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(orbit ? "선택한 오브젝트로 궤도 중심을 옮기려면 계층에서 F 를 누른다"
                                : "Q/E 또는 PageUp/PageDown 으로 오르내리고, Shift 로 4배 빨라진다");
    }
    ImGui::EndGroup();
    ImGui::PopStyleColor();
    {
        const ImVec2 PAD{6.0F, 4.0F};
        ImVec2 minimum = ImGui::GetItemRectMin();
        ImVec2 maximum = ImGui::GetItemRectMax();
        toolbarDrawList->ChannelsSetCurrent(0);
        toolbarDrawList->AddRectFilled(ImVec2{minimum.x - PAD.x, minimum.y - PAD.y},
                                       ImVec2{maximum.x + PAD.x, maximum.y + PAD.y},
                                       IM_COL32(18, 18, 18, 210),
                                       4.0F);
        toolbarDrawList->ChannelsMerge();
    }

    ImGui::End();
}

void Editor::buildRenderSettings(scene::Scene& active, float deltaSeconds) {
    if (!showRenderSettings) {
        return;
    }
    // 도킹되지 않는 떠 있는 창. 열어 둔 채 장면을 돌려 보며 값을 만질 수 있다.
    ImGui::SetNextWindowSize(ImVec2{440.0F, 680.0F}, ImGuiCond_FirstUseEver);
    if (!ImGui::Begin(WINDOW_SETTINGS, &showRenderSettings, ImGuiWindowFlags_NoDocking)) {
        ImGui::End();
        return;
    }

    frameTimeMilliseconds = frameTimeMilliseconds * 0.9F + deltaSeconds * 1000.0F * 0.1F;
    ImGui::Text("프레임 %.2f ms (%.0f FPS)",
                frameTimeMilliseconds,
                frameTimeMilliseconds > 0.0F ? 1000.0F / frameTimeMilliseconds : 0.0F);
    ImGui::Text("렌더 해상도 %ux%u", renderer.renderExtent().width, renderer.renderExtent().height);
    ImGui::Text("작업 워커 %u", workerCount);

    // 설정이 길어 그룹을 접을 수 있게 하고, 검색어가 있으면 이름에 그 말이 든 그룹만 펼친다.
    ImGui::SetNextItemWidth(200.0F);
    ImGui::InputTextWithHint("##settingsFilter", "그룹 검색", settingsFilter.data(), settingsFilter.size());
    auto section = [this](const char* name) { return settingsSection(name); };

    // Path Tracing은 래스터 패스를 통째로 건너뛴다. 거기 딸린 설정은 눌러도 아무 일이 없으므로
    // 디버그 뷰와 같은 이유로 잠근다.
    bool rasterOnly = renderer.settings.usePathTracing;
    // 광선 그림자와 반사는 래스터 안에서 도는 것이라 Path Tracing과는 무관하다.
    bool rayQueryReady = renderer.rayQueryShadowsAvailable() && !rasterOnly;
    scene::PostProcess& post = active.post;

    if (section("하드웨어")) {
        if (autoTune == gfx::AutoTune::OFF) {
            ImGui::Text("자동 튜닝: %s", gfx::autoTuneName(autoTune));
        } else {
            ImGui::Text("자동 튜닝: %s · 등급 %s", gfx::autoTuneName(autoTune), gfx::tierName(hardwareProfile.tier));
        }
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("실행 인자 --auto-tune off|safe|aggressive 로 고른다");
        }
        // 왜 그렇게 골랐는지 그대로 보여 준다. 설정을 되돌리기 전에 근거를 알 수 있어야 한다.
        for (const std::string& reason : hardwareProfile.reasons) {
            ImGui::BulletText("%s", reason.c_str());
        }
        // 기동 시의 판정이라 그 뒤에 바뀐 것은 여기 나오지 않는다. 사용자가 고쳤거나, 가속 구조가
        // 예산을 넘어 렌더러가 스스로 광선 기능을 끈 경우가 그렇다.
        ImGui::TextDisabled("기동 시 판정이다. 지금 값은 아래 절들이 보여 준다");
    }

    if (section("후처리")) {
        ImGui::SliderFloat("노출", &renderer.settings.exposure, 0.05F, 8.0F, "%.2f");
        ImGui::SliderFloat("Bloom 세기", &post.bloomIntensity, 0.0F, 2.0F, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("임계값을 넘은 부분을 흐려서 «더할» 세기");
        }
        ImGui::BeginDisabled(post.bloomIntensity <= 0.0F);
        ImGui::SliderFloat("Bloom 임계값", &post.bloomThreshold, 0.0F, 8.0F, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("이 밝기를 넘는 곳만 번진다. 0 이면 화면 전체가 흐려진다");
        }
        ImGui::SliderFloat("Bloom 무릎", &post.bloomKnee, 0.0F, 1.0F, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("임계값 언저리를 부드럽게 넘기는 폭. 0 이면 경계에서 깜빡인다");
        }
        ImGui::SliderFloat("Bloom 번짐", &post.bloomScatter, 0.0F, 1.0F, "%.2f");
        if (ImGui::IsItemHovered()) {
            ImGui::SetTooltip("클수록 넓은 밉 쪽에 무게를 두어 멀리 퍼진다");
        }
        ImGui::EndDisabled();
        ImGui::Checkbox("Auto Exposure", &post.autoExposure);
        ImGui::BeginDisabled(!post.autoExposure);
        ImGui::SliderFloat("적응 속도", &post.adaptationSpeed, 0.1F, 10.0F, "%.1f /s");
        ImGui::DragFloatRange2("EV 범위", &post.exposureMinEv, &post.exposureMaxEv, 0.1F, -10.0F, 20.0F, "%.1f");
        ImGui::EndDisabled();
    }

    if (section("높이 안개")) {
        ImGui::SliderFloat("안개 밀도", &post.fogDensity, 0.0F, 2.0F, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::BeginDisabled(post.fogDensity <= 0.0F);
        ImGui::ColorEdit3("안개 색", glm::value_ptr(post.fogColor));
        ImGui::DragFloat("안개 높이", &post.fogHeight, 0.05F, -100.0F, 100.0F, "%.2f");
        ImGui::SliderFloat("높이 감쇠", &post.fogFalloff, 0.0F, 5.0F, "%.2f");
        ImGui::EndDisabled();
    }

    if (section("조명")) {
        ImGui::Text("장면 조명 %zu개", active.lights.size());
        ImGui::ColorEdit3("환경광", glm::value_ptr(active.ambientColor));
        ImGui::SliderFloat("환경광 세기", &active.ambientIntensity, 0.0F, 4.0F, "%.2f");
        ImGui::BeginDisabled(rasterOnly);
        ImGui::Checkbox("그림자", &renderer.settings.shadowsEnabled);
        ImGui::BeginDisabled(!renderer.settings.shadowsEnabled);
        ImGui::Checkbox("시점 Frustum Culling", &renderer.settings.shadowViewCulling);
        ImGui::SameLine();
        ImGui::Checkbox("Caster Culling", &renderer.settings.shadowCasterCulling);
        ImGui::Checkbox("시점 캐싱", &renderer.settings.shadowCaching);
        int cascades = static_cast<int>(renderer.settings.shadowCascades);
        if (ImGui::SliderInt("캐스케이드", &cascades, 1, static_cast<int>(gfx::MAX_SHADOW_CASCADES))) {
            renderer.settings.shadowCascades = static_cast<uint32_t>(cascades);
        }
        ImGui::SliderFloat("분할 혼합", &renderer.settings.shadowSplitLambda, 0.0F, 1.0F, "%.2f");
        ImGui::DragFloat("그림자 거리", &renderer.settings.shadowDistance, 1.0F, 0.0F, 10000.0F, "%.0f (0 이면 자동)");
        ImGui::Text("드로우 %u / %u, 다시 그린 층 %u",
                    renderer.shadowDrawCount(),
                    renderer.shadowDrawCandidates(),
                    renderer.shadowLayersDrawn());
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        ImGui::TextDisabled("그림자 시점 %u개까지 (방향광/스폿광 1, 점광 6)", gfx::MAX_SHADOW_VIEWS);

        ImGui::BeginDisabled(!rayQueryReady);
        ImGui::Checkbox("Ray Traced Shadows (하이브리드)", &renderer.settings.useRayQueryShadows);
        ImGui::BeginDisabled(!renderer.settings.useRayQueryShadows);
        ImGui::SliderFloat("광선 거리", &renderer.settings.rayShadowDistance, 1.0F, 200.0F, "%.0f");
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        if (rasterOnly) {
            ImGui::TextDisabled("Path Tracing 중에는 래스터 패스를 건너뛰므로 적용되지 않는다");
        } else if (!rayQueryReady) {
            if (!renderer.rayTracingBlocked().empty()) {
                ImGui::TextWrapped("%s", renderer.rayTracingBlocked().c_str());
            } else {
                ImGui::TextDisabled("이 장치는 Ray Query를 지원하지 않는다");
            }
        } else {
            ImGui::TextDisabled("이 거리 안쪽만 광선으로 판정하고 바깥은 그림자 맵을 쓴다");
        }
    }

    if (section("Ray Traced Reflections")) {
        bool reflectionReady = rayQueryReady && renderer.settings.useIbl;
        ImGui::BeginDisabled(!reflectionReady);
        ImGui::Checkbox("Ray Traced Reflections", &renderer.settings.useReflections);
        ImGui::BeginDisabled(!renderer.settings.useReflections);
        ImGui::SliderFloat("거칠기 상한", &renderer.settings.reflectionRoughnessCutoff, 0.03F, 1.0F, "%.2f");
        ImGui::SliderFloat("반사 세기", &renderer.settings.reflectionIntensity, 0.0F, 2.0F, "%.2f");
        int reflectionSamples = static_cast<int>(renderer.settings.reflectionMaxSamples);
        if (ImGui::SliderInt("누적 상한", &reflectionSamples, 1, 64)) {
            renderer.settings.reflectionMaxSamples = static_cast<uint32_t>(reflectionSamples);
        }
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        if (rasterOnly) {
            ImGui::TextDisabled("Path Tracing이 반사를 직접 계산한다");
        } else if (!rayQueryReady) {
            ImGui::TextDisabled("이 장치는 Ray Query를 지원하지 않는다");
        } else if (!renderer.settings.useIbl) {
            ImGui::TextDisabled("IBL 이 꺼져 있으면 스페큘러 항이 없어 반사도 쉰다");
        } else {
            ImGui::TextDisabled("거칠기 상한 이하의 불투명 표면만 추적한다. 픽셀당 광선 하나, Temporal 누적");
        }
    }

    if (section("환경 (IBL)")) {
        scene::Environment& env = active.environment;
        ImGui::Checkbox("IBL 사용", &renderer.settings.useIbl);
        int skySource = env.useHdr ? 1 : 0;
        if (ImGui::Combo("하늘", &skySource, "절차적\0HDR 파일\0")) {
            env.useHdr = skySource == 1;
        }
        if (env.useHdr) {
            ImGui::TextDisabled("%s", env.hdrPath.empty() ? "파일 없음" : env.hdrPath.filename().string().c_str());
            if (ImGui::Button("HDR 파일 고르기")) {
                hdrFiles.clear();
                std::error_code error;
                for (const auto& entry : std::filesystem::directory_iterator(modelRoot, error)) {
                    if (entry.is_regular_file() && entry.path().extension() == ".hdr") {
                        hdrFiles.push_back(entry.path());
                    }
                }
                std::ranges::sort(hdrFiles);
                ImGui::OpenPopup("HDR 선택");
            }
            if (ImGui::BeginPopup("HDR 선택")) {
                if (hdrFiles.empty()) {
                    ImGui::TextDisabled("%s 에 .hdr 파일이 없습니다", modelRoot.string().c_str());
                }
                for (const std::filesystem::path& file : hdrFiles) {
                    if (ImGui::Selectable(file.filename().string().c_str())) {
                        env.hdrPath = file;
                    }
                }
                ImGui::Separator();
                ImGui::SetNextItemWidth(320.0F);
                ImGui::InputText("경로", hdrPathInput.data(), hdrPathInput.size());
                ImGui::SameLine();
                if (ImGui::Button("열기") && hdrPathInput[0] != '\0') {
                    env.hdrPath = std::filesystem::path(hdrPathInput.data());
                }
                ImGui::EndPopup();
            }
        } else {
            ImGui::ColorEdit3("천정", glm::value_ptr(env.zenithColor));
            ImGui::ColorEdit3("지평", glm::value_ptr(env.horizonColor));
            ImGui::ColorEdit3("지면", glm::value_ptr(env.groundColor));
            ImGui::ColorEdit3("태양색", glm::value_ptr(env.sunColor));
            ImGui::SliderFloat("태양 세기", &env.sunIntensity, 0.0F, 8.0F, "%.2f");
        }
        ImGui::SliderFloat("환경 세기", &env.intensity, 0.0F, 4.0F, "%.2f");
        ImGui::SliderFloat("환경 회전", &env.yawDegrees, -180.0F, 180.0F, "%.0f°");
        if (ImGui::Button("다시 굽기")) {
            renderer.invalidateEnvironment();
        }
        ImGui::SameLine();
        ImGui::TextDisabled("태양 방향은 첫 방향광을 따라간다");
    }

    if (section("SSAO")) {
        ImGui::BeginDisabled(rasterOnly);
        ImGui::Checkbox("사용", &renderer.settings.useSsao);
        ImGui::BeginDisabled(!renderer.settings.useSsao);
        ImGui::SliderFloat("반지름", &renderer.settings.ssaoRadius, 0.005F, 0.3F, "장면의 %.3f배");
        ImGui::SliderFloat("세기", &renderer.settings.ssaoIntensity, 0.0F, 3.0F, "%.2f");
        ImGui::SliderFloat("편향", &renderer.settings.ssaoBias, 0.0F, 0.02F, "%.4f");
        int samples = static_cast<int>(renderer.settings.ssaoSamples);
        if (ImGui::SliderInt("표본", &samples, 4, 64)) {
            renderer.settings.ssaoSamples = static_cast<uint32_t>(samples);
        }
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        if (rasterOnly) {
            ImGui::TextDisabled("Path Tracing 중에는 래스터 패스를 건너뛰므로 적용되지 않는다");
        }

        // 컬 컴퓨트도 태스크 셰이더도 래스터 분기 안에서만 돈다. Path Tracing의 가속 구조는 0단계 LOD
        // 삼각형으로 세우므로 meshlet 컬링도 LOD 선정도 관여하지 않는다.
    }

    if (section("컬링과 LOD")) {
        ImGui::BeginDisabled(rasterOnly);
        ImGui::Checkbox("Compute Culling", &renderer.settings.useComputeCulling);
        // 오클루전은 Mesh Shader 경로의 태스크 셰이더에도 적용되므로 Compute Culling 잠금 밖에 둔다.
        ImGui::Checkbox("HZB Occlusion Culling (두 패스)", &renderer.settings.occlusionCulling);
        ImGui::BeginDisabled(!renderer.settings.useComputeCulling);
        ImGui::Checkbox("Frustum Culling", &renderer.settings.frustumCulling);
        ImGui::SameLine();
        ImGui::Checkbox("Normal Cone Culling", &renderer.settings.coneCulling);
        ImGui::EndDisabled();

        ImGui::Checkbox("자동 LOD 선정", &renderer.settings.automaticLod);
        if (renderer.settings.automaticLod) {
            ImGui::SliderFloat("허용 화면 오차", &renderer.settings.lodErrorThreshold, 0.1F, 32.0F, "%.2f px");

            ImGui::Checkbox("Neural LOD", &renderer.settings.useNeuralLod);
            if (renderer.settings.useNeuralLod) {
                ImGui::Checkbox("학습", &renderer.settings.trainLodNetwork);
                ImGui::SameLine();
                if (ImGui::Button("가중치 초기화")) {
                    renderer.lodNetwork.reset();
                }
                ImGui::SliderFloat("삼각형 예산",
                                   &renderer.settings.triangleBudget,
                                   1000.0F,
                                   500000.0F,
                                   "%.0f",
                                   ImGuiSliderFlags_Logarithmic);
                ImGui::SliderFloat(
                    "학습률", &renderer.lodNetwork.learningRate, 0.001F, 0.5F, "%.3f", ImGuiSliderFlags_Logarithmic);
                ImGui::Text("손실 %.5f, 기대 삼각형 %.0f",
                            static_cast<double>(renderer.lodNetwork.lastLoss()),
                            static_cast<double>(renderer.lodNetwork.lastSoftTriangleCount()));
            }
        } else {
            int lodLevel = static_cast<int>(renderer.settings.lodLevel);
            int maxLod = std::max(static_cast<int>(geometryStore != nullptr ? geometryStore->maxLodCount() : 1) - 1, 0);
            if (ImGui::SliderInt("LOD 단계", &lodLevel, 0, std::max(maxLod, 0))) {
                renderer.settings.lodLevel = static_cast<uint32_t>(lodLevel);
            }
        }

        ImGui::EndDisabled();
        if (rasterOnly) {
            ImGui::TextDisabled("Path Tracing 중에는 래스터 패스를 건너뛰므로 적용되지 않는다");
        }
    }

    if (section("해상도와 Upscaling")) {
        if (ImGui::SliderFloat("렌더 배율", &renderer.settings.renderScale, 0.25F, 2.0F, "%.2f")) {
            // 배율은 다음 프레임의 표시 크기 갱신에서 반영된다.
        }
        ImGui::Text("장면 %ux%u -> 표시 %ux%u",
                    renderer.renderExtent().width,
                    renderer.renderExtent().height,
                    renderer.displayExtent().width,
                    renderer.displayExtent().height);

        std::vector<gfx::UpscalerInfo> upscalers = renderer.upscalers();
        bool dlssSelected =
            renderer.settings.upscaler == gfx::Upscaler::DLSS || renderer.settings.upscaler == gfx::Upscaler::DLSS_RR;
        for (const gfx::UpscalerInfo& info : upscalers) {
            // Ray Reconstruction 은 DLSS 의 한 모드다. 목록에 따로 두면 초해상과 무관한 별개 기법처럼
            // 보이고, Path Tracing을 켜야 한다는 조건도 드러나지 않는다. 아래에서 체크박스로 다룬다.
            if (info.kind == gfx::Upscaler::DLSS_RR) {
                continue;
            }
            bool selected = info.kind == gfx::Upscaler::DLSS ? dlssSelected : renderer.settings.upscaler == info.kind;
            ImGui::BeginDisabled(!info.available);
            if (ImGui::RadioButton(info.name, selected)) {
                renderer.settings.upscaler = info.kind;
            }
            ImGui::EndDisabled();
            if (!info.available) {
                ImGui::SameLine();
                ImGui::TextDisabled("(%s)", info.reason);
            }
        }
        if (renderer.settings.upscaler == gfx::Upscaler::SPATIAL) {
            ImGui::SliderFloat("Sharpening", &renderer.settings.upscaleSharpness, 0.0F, 1.0F, "%.2f");
        }
        if (dlssSelected) {
            ImGui::Indent();

            // 사전 설정은 곧 렌더 배율이다. NGX 에 넘기는 값도 배율에서 되돌리므로 둘이 어긋날 수 없다.
            gfx::DlssQuality quality = gfx::dlssQualityForScale(renderer.settings.renderScale);
            if (ImGui::BeginCombo("품질", gfx::dlssQualityName(quality))) {
                for (uint32_t index = 0; index < gfx::DLSS_QUALITY_COUNT; ++index) {
                    auto candidate = static_cast<gfx::DlssQuality>(index);
                    if (ImGui::Selectable(gfx::dlssQualityName(candidate), candidate == quality)) {
                        renderer.settings.renderScale = gfx::dlssQualityScale(candidate);
                    }
                }
                ImGui::EndCombo();
            }

            gfx::UpscalerInfo reconstruction{};
            for (const gfx::UpscalerInfo& info : upscalers) {
                if (info.kind == gfx::Upscaler::DLSS_RR) {
                    reconstruction = info;
                }
            }
            bool useReconstruction = renderer.settings.upscaler == gfx::Upscaler::DLSS_RR;
            ImGui::BeginDisabled(!reconstruction.available);
            if (ImGui::Checkbox("Ray Reconstruction", &useReconstruction)) {
                renderer.settings.upscaler = useReconstruction ? gfx::Upscaler::DLSS_RR : gfx::Upscaler::DLSS;
            }
            ImGui::EndDisabled();
            if (!reconstruction.available) {
                ImGui::SameLine();
                ImGui::TextDisabled("(%s)", reconstruction.reason);
            } else if (useReconstruction && !renderer.settings.usePathTracing) {
                ImGui::TextDisabled("Path Tracing을 켜야 동작한다. 그전까지는 Super Resolution 으로 돌아간다");
            } else if (useReconstruction) {
                ImGui::TextDisabled("Path Tracing 1표본을 Denoise 하면서 확대한다");
            }

            ImGui::Unindent();
        }
    }

    if (section("Path Tracing")) {
        ImGui::BeginDisabled(!renderer.pathTracingAvailable());
        ImGui::Checkbox("Path Tracing", &renderer.settings.usePathTracing);
        ImGui::EndDisabled();
        if (!renderer.pathTracingAvailable()) {
            ImGui::SameLine();
            ImGui::TextDisabled("(미지원)");
            if (!renderer.rayTracingBlocked().empty()) {
                ImGui::TextWrapped("%s", renderer.rayTracingBlocked().c_str());
            }
        } else if (renderer.settings.usePathTracing) {
            gfx::PathTraceOptions& options = renderer.settings.pathTrace;
            int bounces = static_cast<int>(options.maxBounces);
            if (ImGui::SliderInt("반사 횟수", &bounces, 1, 16)) {
                options.maxBounces = static_cast<uint32_t>(bounces);
            }
            int perFrame = static_cast<int>(options.samplesPerFrame);
            if (ImGui::SliderInt("프레임당 표본", &perFrame, 1, 16)) {
                options.samplesPerFrame = static_cast<uint32_t>(perFrame);
            }
            int maxSamples = static_cast<int>(options.maxSamples);
            if (ImGui::SliderInt("표본 상한", &maxSamples, 0, 4096, maxSamples == 0 ? "무제한" : "%d")) {
                options.maxSamples = static_cast<uint32_t>(maxSamples);
            }
            ImGui::Checkbox("다음 사건 추정", &options.nextEventEstimation);
            ImGui::SameLine();
            ImGui::Checkbox("Russian Roulette", &options.russianRoulette);
            ImGui::SliderFloat("복사휘도 상한", &options.radianceClamp, 1.0F, 64.0F, "%.1f");
            ImGui::SliderFloat("하늘 밝기", &options.skyIntensity, 0.0F, 4.0F, "%.2f");
            ImGui::Text("누적 표본 %u", renderer.pathTraceSamples());
            ImGui::SameLine();
            if (ImGui::Button("누적 초기화")) {
                renderer.resetPathAccumulation();
            }
        }
    }

    if (section("파이프라인")) {
        ImGui::Checkbox("Wireframe", &renderer.settings.wireframe);
        ImGui::BeginDisabled(!renderer.meshShaderAvailable() || rasterOnly);
        ImGui::Checkbox("Mesh Shader 경로", &renderer.settings.useMeshShader);
        ImGui::EndDisabled();
        if (!renderer.meshShaderAvailable()) {
            ImGui::SameLine();
            ImGui::TextDisabled("(미지원)");
        } else if (rasterOnly) {
            // 광선 순회는 가속 구조를 타지 mesh 셰이더를 실행하지 않는다. 둘은 아예 다른 파이프라인이다.
            ImGui::TextDisabled("Path Tracing은 래스터 파이프라인을 타지 않는다");
        }

        static constexpr const char* DEBUG_MODE_NAMES[] = {"Shading",
                                                           "Meshlet",
                                                           "Normal",
                                                           "UV",
                                                           "Depth",
                                                           "LOD",
                                                           "Shadow Cascade",
                                                           "Shadow",
                                                           "Motion Vector",
                                                           "Cull Pass",
                                                           "Reflection Raw",
                                                           "Reflection Accumulated"};
        // Path Tracing이나 이 장치가 못 만드는 값은 개별로 잠그고 사유를 보인다.
        if (ImGui::BeginCombo("디버그 뷰", DEBUG_MODE_NAMES[renderer.settings.debugMode])) {
            for (uint32_t mode = 0; mode < IM_ARRAYSIZE(DEBUG_MODE_NAMES); ++mode) {
                const char* blocked = renderer.debugModeBlockedReason(mode);
                ImGui::BeginDisabled(blocked != nullptr);
                if (ImGui::Selectable(DEBUG_MODE_NAMES[mode], renderer.settings.debugMode == mode)) {
                    renderer.settings.debugMode = mode;
                }
                ImGui::EndDisabled();
                if (blocked != nullptr && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                    ImGui::SetTooltip("%s", blocked);
                }
            }
            ImGui::EndCombo();
        }
        if (const char* blocked = renderer.debugModeBlockedReason(renderer.settings.debugMode); blocked != nullptr) {
            ImGui::TextDisabled("%s", blocked);
        }

        bool vsync = renderer.vsyncEnabled();
        if (ImGui::Checkbox("수직 동기화", &vsync)) {
            renderer.setVsync(vsync);
        }

        ImGui::Separator();
        ImGui::TextDisabled("하드웨어 기능");
        const gfx::Capabilities& caps = context.caps;
        ImGui::BeginDisabled();
        bool meshShader = caps.meshShader;
        bool rayTracing = caps.rayTracingPipeline;
        bool drawIndirectCount = caps.drawIndirectCount;
        bool drawIndex = caps.shaderDrawIndex;
        ImGui::Checkbox("mesh shader", &meshShader);
        ImGui::Checkbox("Ray Tracing Pipeline", &rayTracing);
        ImGui::Checkbox("drawIndirectCount", &drawIndirectCount);
        ImGui::Checkbox("gl_DrawID (없으면 meshlet 디버그 뷰가 메쉬 단위)", &drawIndex);
        ImGui::EndDisabled();
    }
    // 플러그인의 절(물리·유체·프로파일러·콜라이더 표시). 렌더러 필드만 만지는 절은 위에 남아 있다.
    if (pluginSettings) {
        pluginSettings();
    }
    ImGui::End();
}

bool Editor::settingsSection(const char* name) {
    if (settingsFilter[0] != '\0') {
        if (std::strstr(name, settingsFilter.data()) == nullptr) {
            return false;
        }
        ImGui::SeparatorText(name);
        return true;
    }
    // 헤더 이름이 안의 체크박스 이름("Ray Traced Reflections", "Path Tracing")과 같으면 ID 가 겹쳐 체크박스가
    // 눌리지 않는다. 숨은 접미사로 헤더 ID 를 따로 둔다.
    std::string header = std::string(name) + "##section";
    return ImGui::CollapsingHeader(header.c_str(), ImGuiTreeNodeFlags_DefaultOpen);
}

void Editor::buildConsole() {
    if (!ImGui::Begin(WINDOW_CONSOLE)) {
        ImGui::End();
        return;
    }

    std::deque<LogEntry> entries = logSink->snapshot();
    for (const LogEntry& entry : entries) {
        ImGui::PushStyleColor(ImGuiCol_Text, levelColor(entry.level));
        ImGui::TextUnformatted(entry.text.c_str());
        ImGui::PopStyleColor();
    }
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 1.0F) {
        ImGui::SetScrollHereY(1.0F);
    }
    ImGui::End();
}

// F 키로 선택한 오브젝트를 궤도 중심으로 삼는다. 텍스트를 입력하는 중에는 받지 않는다.
void Editor::buildLoadOverlay() {
    if (!loadStatus.active) {
        return;
    }
    // 장면 뷰가 아니라 주 뷰포트 아래 가운데에 띄운다. 도킹 배치와 무관하게 늘 같은 자리에 보인다.
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImVec2 position{viewport->WorkPos.x + viewport->WorkSize.x * 0.5F,
                    viewport->WorkPos.y + viewport->WorkSize.y - 24.0F};
    ImGui::SetNextWindowPos(position, ImGuiCond_Always, ImVec2{0.5F, 1.0F});
    ImGui::SetNextWindowBgAlpha(0.9F);
    ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoDocking |
                             ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                             ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs;
    if (ImGui::Begin("##load_overlay", nullptr, flags)) {
        ImGui::Text("모델 적재 중: %s", loadStatus.file.c_str());
        std::string label = loadStatus.stage;
        if (loadStatus.queued > 0) {
            label += " (대기 " + std::to_string(loadStatus.queued) + "개)";
        }
        if (loadStatus.indeterminate) {
            // 음수 분율은 끝을 모르는 단계의 흐르는 막대다. 살아 있음을 보이는 것이 목적이다.
            ImGui::ProgressBar(-1.0F * static_cast<float>(ImGui::GetTime()), ImVec2{360.0F, 0.0F}, label.c_str());
        } else {
            std::string percent = label + " " + std::to_string(static_cast<int>(loadStatus.fraction * 100.0F)) + "%";
            ImGui::ProgressBar(loadStatus.fraction, ImVec2{360.0F, 0.0F}, percent.c_str());
        }
    }
    ImGui::End();
}

void Editor::focusSelected(scene::Scene& active, const gfx::GeometryStore& geometry) {
    int selectedObject = primarySelection();
    if (selectedObject < 0 || static_cast<size_t>(selectedObject) >= active.objects.size()) {
        return;
    }
    if (ImGui::GetIO().WantTextInput || !ImGui::IsKeyPressed(ImGuiKey_F, false)) {
        return;
    }

    auto index = static_cast<uint32_t>(selectedObject);
    glm::mat4 world = active.worldMatrix(index);
    glm::vec3 center = glm::vec3(world[3]);
    float radius = 1.0F;
    if (active.meshOf(index) < geometry.meshCount()) {
        glm::vec4 sphere = geometry.mesh(active.meshOf(index)).boundingSphere;
        center = glm::vec3(world * glm::vec4{glm::vec3(sphere), 1.0F});
        // 비균등 스케일은 가장 긴 축으로 보수적으로 잡는다.
        radius = sphere.w * std::sqrt(std::max({glm::dot(glm::vec3(world[0]), glm::vec3(world[0])),
                                                glm::dot(glm::vec3(world[1]), glm::vec3(world[1])),
                                                glm::dot(glm::vec3(world[2]), glm::vec3(world[2]))}));
    }
    active.camera.focusOn(center, std::max(radius * 3.0F, 0.5F));
}

void Editor::build(scene::SceneManager& scenes, const gfx::GeometryStore& geometry, float deltaSeconds) {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
    ImGuizmo::BeginFrame();

    geometryStore = &geometry;

    // 단축키는 패널을 그리기 전에 처리한다. 패널이 들고 있는 참조가 아직 없을 때라 배열을 바꿔도 안전하다.
    handleShortcuts(scenes, geometry);
    buildDockspace(scenes, geometry);
    buildHierarchy(scenes, geometry);
    buildInspector(scenes.active(), geometry);
    buildSceneView(scenes.active());
    buildRenderSettings(scenes.active(), deltaSeconds);
    buildConsole();
    buildLoadOverlay();
    focusSelected(scenes.active(), geometry);

    // 메뉴와 우클릭에서 고른 편집 동작. 패널이 참조를 다 놓은 뒤에 한 번에 적용한다.
    if (deferred) {
        std::function<void()> action = std::exchange(deferred, nullptr);
        action();
    }

    // 적재와 장면 전환은 지오메트리 버퍼를 다시 만들기 때문에 패널을 다 그린 뒤에 처리한다.
    if (!pendingModel.empty()) {
        std::filesystem::path path = std::exchange(pendingModel, {});
        if (modelLoader) {
            modelLoader(path);
        }
    }
    if (!pendingSceneSave.empty()) {
        std::filesystem::path path = std::exchange(pendingSceneSave, {});
        if (sceneSaver) {
            sceneSaver(path);
        }
    }
    if (!pendingSceneOpen.empty()) {
        std::filesystem::path path = std::exchange(pendingSceneOpen, {});
        if (sceneOpener) {
            sceneOpener(path);
            clearSelection();
        }
    }

    // 콜라이더를 밝게 그릴 오브젝트. 삭제와 장면 열기가 선택을 바꾸므로 그것들을 다 처리한 뒤에
    // 넘긴다. 이 프레임의 렌더는 build 가 끝난 다음이다.
    renderer.selectedObject = primarySelection();

    updateHistory(scenes.active(), scenes.current());

    ImGui::Render();
}

// 되돌리기 기록은 한 번에 하나씩 쌓는다. 기즈모를 끄는 동안이나 슬라이더를 잡고 있는 동안에는
// 담지 않고, 손을 뗀 뒤 한 덩어리로 담는다. 그렇지 않으면 끌기 한 번이 수십 개의 기록이 된다.
bool Editor::isSelected(int index) const {
    return std::find(selection.begin(), selection.end(), index) != selection.end();
}

void Editor::selectOnly(int index) {
    selection.clear();
    if (index >= 0) {
        selection.push_back(index);
    }
}

void Editor::toggleSelect(int index) {
    if (index < 0) {
        return;
    }
    auto found = std::find(selection.begin(), selection.end(), index);
    if (found != selection.end()) {
        selection.erase(found);
    } else {
        // 뒤에 붙여 방금 고른 것이 기준이 되게 한다.
        selection.push_back(index);
    }
}

void Editor::updateHistory(scene::Scene& active, size_t sceneIndex) {
    // 기록이 너무 길어지면 애니메이터 스켈레톤 사본이 쌓여 메모리를 먹는다.
    constexpr size_t MAX_HISTORY = 64;

    if (histories.size() <= sceneIndex) {
        histories.resize(sceneIndex + 1);
    }
    History& history = histories[sceneIndex];
    if (!history.started) {
        history.baseline = active.capture();
        history.started = true;
        return;
    }
    // 재생 중에는 물리가 프레임마다 변환을 바꾼다. 기록하면 스텝마다 항목이 쌓이고, 정지하면 어차피
    // 재생 전으로 돌아간다.
    if (active.simulating) {
        return;
    }

    ImGuiIO& io = ImGui::GetIO();
    bool editing = ImGui::IsAnyItemActive() || gizmoUsing;
    // 글자를 입력하는 중에는 단축키를 받지 않는다. 이름을 고치다 장면이 되돌아가면 곤란하다.
    bool shortcutsAllowed = io.KeyCtrl && !io.WantTextInput;
    bool wantUndo =
        std::exchange(menuUndo, false) || (shortcutsAllowed && !io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_Z, false));
    bool wantRedo = std::exchange(menuRedo, false) ||
                    (shortcutsAllowed && (ImGui::IsKeyPressed(ImGuiKey_Y, false) ||
                                          (io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_Z, false))));

    if (wantUndo && !history.undoStack.empty()) {
        history.redoStack.push_back(active.capture());
        active.restore(history.undoStack.back());
        history.undoStack.pop_back();
        history.baseline = active.capture();
        // 오브젝트 번호가 통째로 달라질 수 있어 선택은 놓는다.
        clearSelection();
        return;
    }
    if (wantRedo && !history.redoStack.empty()) {
        history.undoStack.push_back(active.capture());
        active.restore(history.redoStack.back());
        history.redoStack.pop_back();
        history.baseline = active.capture();
        clearSelection();
        return;
    }

    if (!editing && active.differsFrom(history.baseline)) {
        history.undoStack.push_back(std::move(history.baseline));
        if (history.undoStack.size() > MAX_HISTORY) {
            history.undoStack.erase(history.undoStack.begin());
        }
        // 새로 편집했으므로 앞서 되돌린 것들은 이어 갈 수 없다.
        history.redoStack.clear();
        history.baseline = active.capture();
    }
}

bool Editor::referencesModel(uint32_t meshBase, uint32_t meshCount, int32_t modelIndex) const {
    auto snapshotReferences = [&](const scene::SceneSnapshot& snapshot) {
        for (const scene::MeshRenderer& renderer : snapshot.meshRenderers) {
            if (renderer.mesh >= meshBase && renderer.mesh < meshBase + meshCount) {
                return true;
            }
        }
        for (const scene::Animator& animator : snapshot.animators) {
            if (animator.model == modelIndex) {
                return true;
            }
        }
        return false;
    };
    for (const History& history : histories) {
        if (!history.started) {
            continue;
        }
        if (snapshotReferences(history.baseline)) {
            return true;
        }
        for (const scene::SceneSnapshot& snapshot : history.undoStack) {
            if (snapshotReferences(snapshot)) {
                return true;
            }
        }
        for (const scene::SceneSnapshot& snapshot : history.redoStack) {
            if (snapshotReferences(snapshot)) {
                return true;
            }
        }
    }
    return false;
}

void Editor::startSimulation(scene::SceneManager& scenes) {
    scene::Scene& active = scenes.active();
    if (active.simulating) {
        return;
    }
    playSnapshot = active.capture();
    playSceneIndex = scenes.current();
    // GPU 강체 솔버는 다음 프레임 머리에서 이 변화를 보고 제 상태를 버린다(PhysicsPlugin).
    active.simulating = true;
}

void Editor::stopSimulation(scene::SceneManager& scenes) {
    scene::Scene& active = scenes.active();
    active.simulating = false;
    // 시작할 때 떠 둔 장면으로 되돌린다. 다른 장면으로 옮긴 채 멈췄으면 그 장면은 건드리지 않는다.
    if (playSnapshot && playSceneIndex == scenes.current()) {
        active.restore(*playSnapshot);
        // 되돌린 상태가 곧 기록의 기준이다. 안 그러면 되돌리기 항목이 하나 더 생긴다.
        if (playSceneIndex < histories.size() && histories[playSceneIndex].started) {
            histories[playSceneIndex].baseline = active.capture();
        }
    }
    playSnapshot.reset();
}

void Editor::clearHistories() {
    // 기준(baseline)은 지금 장면과 같으므로 남긴다. 다음 updateHistory 가 새로 잡는다.
    for (History& history : histories) {
        history.undoStack.clear();
        history.redoStack.clear();
        history.started = false;
    }
}

void Editor::record(VkCommandBuffer commandBuffer) {
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
}

} // namespace editor
