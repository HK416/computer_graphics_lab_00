#include "editor/editor.h"

#include <algorithm>
#include <cmath>
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

#include "core/error.h"
#include "editor/log_sink.h"
#include "gfx/context.h"
#include "gfx/geometry.h"
#include "gfx/profiler.h"
#include "gfx/renderer.h"
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
        if (mesh >= geometry.meshCount() || !scene.visibleInTree(index)) {
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
constexpr const char* WINDOW_TARGETS = "렌더 타겟";
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

void Editor::buildDockspace() {
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
        ImGui::DockBuilderDockWindow(WINDOW_SETTINGS, right);
        ImGui::DockBuilderDockWindow(WINDOW_CONSOLE, bottom);
        ImGui::DockBuilderDockWindow(WINDOW_TARGETS, bottom);
        ImGui::DockBuilderDockWindow(WINDOW_SCENE, center);
        ImGui::DockBuilderFinish(dockspaceId);
    }
    ImGui::DockSpace(dockspaceId, ImVec2{0.0F, 0.0F}, ImGuiDockNodeFlags_None);

    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("창")) {
            if (ImGui::MenuItem("배치 초기화")) {
                layoutBuilt = false;
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }
    ImGui::End();
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

void Editor::drawHierarchyNode(scene::Scene& active, int index) {
    scene::Object& object = active.objects[static_cast<size_t>(index)];
    bool hasChildren = std::ranges::any_of(
        active.objects, [index](const scene::Object& candidate) { return candidate.parent == index; });

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

    if (open) {
        for (int child = 0; child < static_cast<int>(active.objects.size()); ++child) {
            if (active.objects[static_cast<size_t>(child)].parent == index) {
                drawHierarchyNode(active, child);
            }
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
    if (ImGui::Button("장면 저장")) {
        std::string suggested = scenes.active().name + ".json";
        std::copy_n(suggested.c_str(), std::min(suggested.size() + 1, sceneNameInput.size()), sceneNameInput.begin());
        sceneNameInput.back() = '\0';
        ImGui::OpenPopup("장면 저장");
    }
    ImGui::SameLine();
    if (ImGui::Button("장면 열기")) {
        sceneFiles.clear();
        std::error_code error;
        for (const auto& entry : std::filesystem::directory_iterator(sceneRoot, error)) {
            if (entry.is_regular_file() && entry.path().extension() == ".json") {
                sceneFiles.push_back(entry.path());
            }
        }
        std::ranges::sort(sceneFiles);
        ImGui::OpenPopup("장면 열기");
    }

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
    ImGui::Separator();

    scene::Scene& active = scenes.active();
    bool anySelected = hasSelection();

    if (ImGui::Button("추가")) {
        ImGui::OpenPopup("메쉬 선택");
    }
    ImGui::SameLine();
    if (ImGui::Button("조명")) {
        ImGui::OpenPopup("조명 선택");
    }
    ImGui::SameLine();
    if (ImGui::Button("모델")) {
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
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(!anySelected);
    if (ImGui::Button("복제")) {
        // 여러 개를 고른 채 복제하면 각각 복제하고 사본들을 새 선택으로 삼는다.
        std::vector<int> copies;
        for (int selected : selection) {
            if (selected >= 0 && selected < static_cast<int>(active.objects.size())) {
                copies.push_back(static_cast<int>(active.duplicateObject(static_cast<uint32_t>(selected))));
            }
        }
        selection = std::move(copies);
    }
    ImGui::SameLine();
    if (ImGui::Button("삭제") || (anySelected && ImGui::IsKeyPressed(ImGuiKey_Delete))) {
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
    ImGui::EndDisabled();

    if (ImGui::BeginPopup("메쉬 선택")) {
        for (uint32_t meshIndex = 0; meshIndex < geometry.meshCount(); ++meshIndex) {
            const asset::Material& material = geometry.material(geometry.mesh(meshIndex).materialIndex);
            std::string label = std::to_string(meshIndex) + ": " + geometry.meshName(meshIndex) + " / " + material.name;
            if (ImGui::Selectable(label.c_str())) {
                // 새 오브젝트는 카메라 앞쪽, 바운딩 반지름을 고려한 거리에 놓는다.
                float radius = std::max(geometry.mesh(meshIndex).boundingSphere.w, 0.1F);
                scene::Object object;
                object.name = geometry.meshName(meshIndex);
                object.transform.position = active.camera.position + active.camera.forward() * (radius * 3.0F);
                active.objects.push_back(std::move(object));
                selectOnly(static_cast<int>(active.objects.size()) - 1);
                active.attachMeshRenderer(static_cast<uint32_t>(primarySelection()), meshIndex);
            }
        }
        ImGui::EndPopup();
    }

    if (ImGui::BeginPopup("조명 선택")) {
        constexpr std::array<const char*, 4> LIGHT_NAMES{"방향광", "점광", "스폿광", "영역광"};
        for (uint32_t type = 0; type < LIGHT_NAMES.size(); ++type) {
            if (!ImGui::Selectable(LIGHT_NAMES[type])) {
                continue;
            }
            scene::Light light;
            light.type = static_cast<scene::LightType>(type);
            active.lights.push_back(light);

            scene::Object object;
            object.name = LIGHT_NAMES[type];
            object.light = static_cast<int32_t>(active.lights.size()) - 1;
            if (light.type == scene::LightType::DIRECTIONAL) {
                object.transform.rotation = glm::quat(glm::radians(glm::vec3{-50.0F, -30.0F, 0.0F}));
            } else {
                // 카메라 앞쪽에 놓고 보고 있는 쪽을 비추게 한다.
                object.transform.position = active.camera.position + active.camera.forward();
                object.transform.rotation = glm::quatLookAt(active.camera.forward(), glm::vec3{0.0F, 1.0F, 0.0F});
            }
            active.objects.push_back(std::move(object));
            selectOnly(static_cast<int>(active.objects.size()) - 1);
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
            }
        }
        ImGui::Separator();
        ImGui::SetNextItemWidth(320.0F);
        ImGui::InputText("경로", modelPathInput.data(), modelPathInput.size());
        ImGui::SameLine();
        if (ImGui::Button("열기") && modelPathInput[0] != '\0') {
            pendingModel = std::filesystem::path{modelPathInput.data()};
        }
        ImGui::EndPopup();
    }
    ImGui::Separator();

    // 여기부터는 오브젝트의 부모-자식 구조만 보여준다.
    for (int i = 0; i < static_cast<int>(active.objects.size()); ++i) {
        if (active.objects[static_cast<size_t>(i)].parent < 0) {
            drawHierarchyNode(active, i);
        }
    }

    // 남는 공간에 놓으면 뿌리로 끌어올린다.
    ImGui::Dummy(ImGui::GetContentRegionAvail());
    if (ImGui::BeginDragDropTarget()) {
        const ImGuiPayload* payload = ImGui::AcceptDragDropPayload(HIERARCHY_PAYLOAD);
        if (payload != nullptr) {
            pendingChild = *static_cast<const int*>(payload->Data);
            pendingParent = -1;
        }
        ImGui::EndDragDropTarget();
    }

    // 부모가 바뀌어도 화면에서의 위치는 그대로 두려고 지역 변환을 다시 계산한다.
    if (pendingChild >= 0 && pendingChild < static_cast<int>(active.objects.size()) &&
        !active.isDescendant(static_cast<uint32_t>(pendingParent < 0 ? pendingChild : pendingParent),
                             static_cast<uint32_t>(pendingChild))) {
        glm::mat4 world = active.worldMatrix(static_cast<uint32_t>(pendingChild));
        glm::mat4 parentWorld =
            pendingParent >= 0 ? active.worldMatrix(static_cast<uint32_t>(pendingParent)) : glm::mat4{1.0F};
        scene::Object& child = active.objects[static_cast<size_t>(pendingChild)];
        child.parent = pendingParent;
        child.transform = scene::Transform::fromMatrix(glm::inverse(parentWorld) * world);
    }
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
    ImGui::TextUnformatted(object.name.c_str());
    ImGui::Separator();

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
        ImGui::CollapsingHeader("조명", ImGuiTreeNodeFlags_DefaultOpen)) {
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

    uint32_t selectedMesh = active.meshOf(static_cast<uint32_t>(selectedObject));
    if (selectedMesh < geometry.meshCount() &&
        ImGui::CollapsingHeader("메쉬와 재질", ImGuiTreeNodeFlags_DefaultOpen)) {
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

    ImGui::Image(ImTextureRef{static_cast<ImTextureID>(reinterpret_cast<uintptr_t>(
                     textureFor(renderer.presentView(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)))},
                 available);
    sceneHovered = ImGui::IsItemHovered();
    bool imageClicked = ImGui::IsItemClicked(ImGuiMouseButton_Left);

    ImVec2 imagePosition = ImGui::GetItemRectMin();
    ImVec2 imageSize = ImGui::GetItemRectSize();

    gizmoUsing = false;
    bool anySelected = hasSelection();
    if (anySelected && imageSize.x > 1.0F && imageSize.y > 1.0F) {
        if (!active.camera.isLooking()) {
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

        // Ctrl 을 누르면 스냅을 건다. 조작 종류마다 단위가 다르다.
        glm::vec3 snapValue{0.25F};
        if (gizmoOperation == ImGuizmo::ROTATE) {
            snapValue = glm::vec3{15.0F};
        } else if (gizmoOperation == ImGuizmo::SCALE) {
            snapValue = glm::vec3{0.1F};
        }
        const float* snap = ImGui::GetIO().KeyCtrl ? glm::value_ptr(snapValue) : nullptr;

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
                if (selected < 0 || selected >= static_cast<int>(active.objects.size()) ||
                    ancestorSelected(selected)) {
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
    if (imageClicked && !gizmoUsing && !ImGuizmo::IsOver() && !active.camera.isLooking() &&
        geometryStore != nullptr && imageSize.x > 1.0F && imageSize.y > 1.0F) {
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

    // 조작 도구 선택은 장면 뷰 위에 겹쳐 둔다.
    ImGui::SetCursorScreenPos(ImVec2{imagePosition.x + 8.0F, imagePosition.y + 8.0F});
    ImGui::BeginGroup();
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

    ImGui::End();
}

void Editor::buildRenderSettings(scene::Scene& active, float deltaSeconds) {
    if (!ImGui::Begin(WINDOW_SETTINGS)) {
        ImGui::End();
        return;
    }

    frameTimeMilliseconds = frameTimeMilliseconds * 0.9F + deltaSeconds * 1000.0F * 0.1F;
    ImGui::Text("프레임 %.2f ms (%.0f FPS)",
                frameTimeMilliseconds,
                frameTimeMilliseconds > 0.0F ? 1000.0F / frameTimeMilliseconds : 0.0F);
    ImGui::Text("렌더 해상도 %ux%u", renderer.renderExtent().width, renderer.renderExtent().height);
    ImGui::Text("작업 워커 %u", workerCount);

    ImGui::SeparatorText("프로파일러");
    gfx::GpuProfiler& profiler = renderer.profiler();
    ImGui::Checkbox("구간 계측", &profiler.enabled);
    if (!profiler.gpuAvailable()) {
        ImGui::SameLine();
        ImGui::TextDisabled("(GPU 타임스탬프 미지원, CPU 만)");
    }
    if (profiler.enabled) {
        ImGui::SliderFloat("평활", &profiler.smoothing, 0.01F, 1.0F, "%.2f");
        const std::vector<gfx::ProfilerZone>& zones = profiler.zones();
        if (zones.empty()) {
            ImGui::TextDisabled("측정 중...");
        } else if (ImGui::BeginTable("구간", 3, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
            ImGui::TableSetupColumn("구간");
            ImGui::TableSetupColumn("CPU");
            ImGui::TableSetupColumn("GPU");
            ImGui::TableHeadersRow();
            for (const gfx::ProfilerZone& zone : zones) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                // 중첩된 구간은 들여써서 상위 구간과 구분한다.
                ImGui::Text("%*s%s", static_cast<int>(zone.depth) * 2, "", zone.name);
                ImGui::TableNextColumn();
                ImGui::Text("%.3f", static_cast<double>(zone.cpuMilliseconds));
                ImGui::TableNextColumn();
                if (zone.hasGpu) {
                    ImGui::Text("%.3f", static_cast<double>(zone.gpuMilliseconds));
                } else {
                    ImGui::TextDisabled("-");
                }
            }
            ImGui::EndTable();
        }
    }
    ImGui::Separator();

    ImGui::SliderFloat("노출", &renderer.exposure, 0.05F, 8.0F, "%.2f");
    ImGui::Checkbox("와이어프레임", &renderer.wireframe);

    ImGui::SeparatorText("조명");
    ImGui::Text("장면 조명 %zu개", active.lights.size());
    ImGui::ColorEdit3("환경광", glm::value_ptr(active.ambientColor));
    ImGui::SliderFloat("환경광 세기", &active.ambientIntensity, 0.0F, 4.0F, "%.2f");
    ImGui::Checkbox("그림자", &renderer.shadowsEnabled);
    ImGui::BeginDisabled(!renderer.shadowsEnabled);
    ImGui::Checkbox("시점 절두체 컬링", &renderer.shadowViewCulling);
    ImGui::SameLine();
    ImGui::Checkbox("캐스터 컬링", &renderer.shadowCasterCulling);
    ImGui::Checkbox("시점 캐싱", &renderer.shadowCaching);
    int cascades = static_cast<int>(renderer.shadowCascades);
    if (ImGui::SliderInt("캐스케이드", &cascades, 1, static_cast<int>(gfx::MAX_SHADOW_CASCADES))) {
        renderer.shadowCascades = static_cast<uint32_t>(cascades);
    }
    ImGui::SliderFloat("분할 혼합", &renderer.shadowSplitLambda, 0.0F, 1.0F, "%.2f");
    ImGui::DragFloat("그림자 거리", &renderer.shadowDistance, 1.0F, 0.0F, 10000.0F, "%.0f (0 이면 자동)");
    ImGui::Text("드로우 %u / %u, 다시 그린 층 %u",
                renderer.shadowDrawCount(),
                renderer.shadowDrawCandidates(),
                renderer.shadowLayersDrawn());
    ImGui::EndDisabled();
    ImGui::TextDisabled("그림자 시점 %u개까지 (방향광/스폿광 1, 점광 6)", gfx::MAX_SHADOW_VIEWS);

    bool rayQueryReady = renderer.rayQueryShadowsAvailable();
    ImGui::BeginDisabled(!rayQueryReady);
    ImGui::Checkbox("광선 그림자 (하이브리드)", &renderer.useRayQueryShadows);
    ImGui::BeginDisabled(!renderer.useRayQueryShadows);
    ImGui::SliderFloat("광선 거리", &renderer.rayShadowDistance, 1.0F, 200.0F, "%.0f");
    ImGui::EndDisabled();
    ImGui::EndDisabled();
    if (!rayQueryReady) {
        ImGui::TextDisabled("이 장치는 광선 질의를 지원하지 않는다");
    } else {
        ImGui::TextDisabled("이 거리 안쪽만 광선으로 판정하고 바깥은 그림자 맵을 쓴다");
    }

    ImGui::SeparatorText("환경 (IBL)");
    scene::Environment& env = active.environment;
    ImGui::Checkbox("IBL 사용", &renderer.useIbl);
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

    ImGui::SeparatorText("SSAO");
    ImGui::Checkbox("사용", &renderer.useSsao);
    ImGui::BeginDisabled(!renderer.useSsao);
    ImGui::SliderFloat("반지름", &renderer.ssaoRadius, 0.005F, 0.3F, "장면의 %.3f배");
    ImGui::SliderFloat("세기", &renderer.ssaoIntensity, 0.0F, 3.0F, "%.2f");
    ImGui::SliderFloat("편향", &renderer.ssaoBias, 0.0F, 0.02F, "%.4f");
    int samples = static_cast<int>(renderer.ssaoSamples);
    if (ImGui::SliderInt("표본", &samples, 4, 64)) {
        renderer.ssaoSamples = static_cast<uint32_t>(samples);
    }
    ImGui::EndDisabled();

    ImGui::SeparatorText("컬링과 LOD");
    ImGui::Checkbox("컴퓨트 컬링", &renderer.useComputeCulling);
    ImGui::BeginDisabled(!renderer.useComputeCulling);
    ImGui::Checkbox("절두체 컬링", &renderer.frustumCulling);
    ImGui::SameLine();
    ImGui::Checkbox("법선 원뿔 컬링", &renderer.coneCulling);
    ImGui::Checkbox("HZB 오클루전 컬링", &renderer.occlusionCulling);
    ImGui::EndDisabled();

    ImGui::Checkbox("자동 LOD 선정", &renderer.automaticLod);
    if (renderer.automaticLod) {
        ImGui::SliderFloat("허용 화면 오차", &renderer.lodErrorThreshold, 0.1F, 32.0F, "%.2f px");

        ImGui::Checkbox("신경망 보정", &renderer.useNeuralLod);
        if (renderer.useNeuralLod) {
            ImGui::Checkbox("학습", &renderer.trainLodNetwork);
            ImGui::SameLine();
            if (ImGui::Button("가중치 초기화")) {
                renderer.lodNetwork.reset();
            }
            ImGui::SliderFloat(
                "삼각형 예산", &renderer.triangleBudget, 1000.0F, 500000.0F, "%.0f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat(
                "학습률", &renderer.lodNetwork.learningRate, 0.001F, 0.5F, "%.3f", ImGuiSliderFlags_Logarithmic);
            ImGui::Text("손실 %.5f, 기대 삼각형 %.0f",
                        static_cast<double>(renderer.lodNetwork.lastLoss()),
                        static_cast<double>(renderer.lodNetwork.lastSoftTriangleCount()));
        }
    } else {
        int lodLevel = static_cast<int>(renderer.lodLevel);
        int maxLod = static_cast<int>(geometryStore != nullptr ? geometryStore->maxLodCount() : 1) - 1;
        if (ImGui::SliderInt("LOD 단계", &lodLevel, 0, std::max(maxLod, 0))) {
            renderer.lodLevel = static_cast<uint32_t>(lodLevel);
        }
    }

    ImGui::SeparatorText("해상도와 업스케일");
    if (ImGui::SliderFloat("렌더 배율", &renderer.renderScale, 0.25F, 2.0F, "%.2f")) {
        // 배율은 다음 프레임의 표시 크기 갱신에서 반영된다.
    }
    ImGui::Text("장면 %ux%u -> 표시 %ux%u",
                renderer.renderExtent().width,
                renderer.renderExtent().height,
                renderer.displayExtent().width,
                renderer.displayExtent().height);

    std::vector<gfx::UpscalerInfo> upscalers = renderer.upscalers();
    for (const gfx::UpscalerInfo& info : upscalers) {
        ImGui::BeginDisabled(!info.available);
        if (ImGui::RadioButton(info.name, renderer.upscaler == info.kind)) {
            renderer.upscaler = info.kind;
        }
        ImGui::EndDisabled();
        if (!info.available) {
            ImGui::SameLine();
            ImGui::TextDisabled("(%s)", info.reason);
        }
    }
    if (renderer.upscaler == gfx::Upscaler::SPATIAL) {
        ImGui::SliderFloat("선명화", &renderer.upscaleSharpness, 0.0F, 1.0F, "%.2f");
    }

    ImGui::SeparatorText("경로 추적");
    ImGui::BeginDisabled(!renderer.pathTracingAvailable());
    ImGui::Checkbox("경로 추적", &renderer.usePathTracing);
    ImGui::EndDisabled();
    if (!renderer.pathTracingAvailable()) {
        ImGui::SameLine();
        ImGui::TextDisabled("(미지원)");
    } else if (renderer.usePathTracing) {
        gfx::PathTraceOptions& options = renderer.pathTrace;
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
        ImGui::Checkbox("러시안 룰렛", &options.russianRoulette);
        ImGui::SliderFloat("복사휘도 상한", &options.radianceClamp, 1.0F, 64.0F, "%.1f");
        ImGui::SliderFloat("하늘 밝기", &options.skyIntensity, 0.0F, 4.0F, "%.2f");
        ImGui::Text("누적 표본 %u", renderer.pathTraceSamples());
        ImGui::SameLine();
        if (ImGui::Button("누적 초기화")) {
            renderer.resetPathAccumulation();
        }
    }

    ImGui::SeparatorText("파이프라인");
    ImGui::BeginDisabled(!renderer.meshShaderAvailable());
    ImGui::Checkbox("mesh shader 경로", &renderer.useMeshShader);
    ImGui::EndDisabled();
    if (!renderer.meshShaderAvailable()) {
        ImGui::SameLine();
        ImGui::TextDisabled("(미지원)");
    }

    static constexpr const char* DEBUG_MODE_NAMES[] = {
        "셰이딩", "meshlet", "노멀", "UV", "깊이", "LOD", "캐스케이드", "그림자", "모션 벡터"};
    // 경로 추적은 래스터에 있는 모드를 다 그리지는 못한다. 못 그리는 것만 개별로 잠근다.
    auto modeUsable = [this](uint32_t mode) {
        return !renderer.usePathTracing || gfx::pathTraceSupportsDebugMode(mode);
    };
    if (ImGui::BeginCombo("디버그 뷰", DEBUG_MODE_NAMES[renderer.debugMode])) {
        for (uint32_t mode = 0; mode < IM_ARRAYSIZE(DEBUG_MODE_NAMES); ++mode) {
            bool usable = modeUsable(mode);
            ImGui::BeginDisabled(!usable);
            if (ImGui::Selectable(DEBUG_MODE_NAMES[mode], renderer.debugMode == mode)) {
                renderer.debugMode = mode;
            }
            ImGui::EndDisabled();
            if (!usable && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled)) {
                ImGui::SetTooltip("경로 추적에는 이 값이 없다");
            }
        }
        ImGui::EndCombo();
    }
    if (!modeUsable(renderer.debugMode)) {
        ImGui::TextDisabled("경로 추적 중에는 셰이딩으로 그린다");
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
    bool minmax = caps.samplerFilterMinmax;
    ImGui::Checkbox("mesh shader", &meshShader);
    ImGui::Checkbox("레이트레이싱 파이프라인", &rayTracing);
    ImGui::Checkbox("drawIndirectCount", &drawIndirectCount);
    ImGui::Checkbox("samplerFilterMinmax", &minmax);
    ImGui::EndDisabled();
    ImGui::End();
}

void Editor::buildRenderTargets() {
    if (!ImGui::Begin(WINDOW_TARGETS)) {
        ImGui::End();
        return;
    }

    std::vector<gfx::Renderer::TargetView> views = renderer.targetViews();
    selectedTarget = std::clamp(selectedTarget, 0, static_cast<int>(views.size()) - 1);

    for (int i = 0; i < static_cast<int>(views.size()); ++i) {
        if (i > 0) {
            ImGui::SameLine();
        }
        if (ImGui::RadioButton(views[static_cast<size_t>(i)].name, selectedTarget == i)) {
            selectedTarget = i;
        }
    }

    ImVec2 available = ImGui::GetContentRegionAvail();
    if (available.y > 8.0F && !views.empty()) {
        VkExtent2D extent = renderer.renderExtent();
        float aspect = static_cast<float>(extent.width) / static_cast<float>(extent.height);
        float height = std::min(available.y - 4.0F, available.x / aspect);
        ImGui::Image(
            ImTextureRef{static_cast<ImTextureID>(reinterpret_cast<uintptr_t>(textureFor(
                views[static_cast<size_t>(selectedTarget)].view, views[static_cast<size_t>(selectedTarget)].layout)))},
            ImVec2{height * aspect, height});
    }
    ImGui::End();
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

    buildDockspace();
    buildHierarchy(scenes, geometry);
    buildInspector(scenes.active(), geometry);
    buildSceneView(scenes.active());
    buildRenderSettings(scenes.active(), deltaSeconds);
    buildRenderTargets();
    buildConsole();
    focusSelected(scenes.active(), geometry);

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

    ImGuiIO& io = ImGui::GetIO();
    bool editing = ImGui::IsAnyItemActive() || gizmoUsing;
    // 글자를 입력하는 중에는 단축키를 받지 않는다. 이름을 고치다 장면이 되돌아가면 곤란하다.
    bool shortcutsAllowed = io.KeyCtrl && !io.WantTextInput;
    bool wantUndo = shortcutsAllowed && !io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_Z, false);
    bool wantRedo = shortcutsAllowed && (ImGui::IsKeyPressed(ImGuiKey_Y, false) ||
                                         (io.KeyShift && ImGui::IsKeyPressed(ImGuiKey_Z, false)));

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

void Editor::record(VkCommandBuffer commandBuffer) {
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
}

} // namespace editor
