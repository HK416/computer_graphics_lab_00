#include "editor/editor.h"

#include <algorithm>
#include <filesystem>
#include <string>

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
#include "gfx/renderer.h"
#include "scene/scene.h"

namespace editor {
namespace {

constexpr float BASE_FONT_SIZE = 16.0F;
constexpr const char* WINDOW_HIERARCHY = "계층";
constexpr const char* WINDOW_INSPECTOR = "인스펙터";
constexpr const char* WINDOW_SCENE = "장면";
constexpr const char* WINDOW_CONSOLE = "콘솔";
constexpr const char* WINDOW_TARGETS = "렌더 타겟";
constexpr const char* WINDOW_SETTINGS = "렌더 설정";

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
                selectedObject = -1;
            }
        }
        ImGui::EndCombo();
    }
    ImGui::Separator();

    scene::Scene& active = scenes.active();

    if (ImGui::Button("추가")) {
        ImGui::OpenPopup("메쉬 선택");
    }
    ImGui::SameLine();
    bool hasSelection = selectedObject >= 0 && selectedObject < static_cast<int>(active.objects.size());
    ImGui::BeginDisabled(!hasSelection);
    if (ImGui::Button("복제")) {
        scene::Object copy = active.objects[static_cast<size_t>(selectedObject)];
        copy.name += " (복사)";
        active.objects.push_back(std::move(copy));
        selectedObject = static_cast<int>(active.objects.size()) - 1;
    }
    ImGui::SameLine();
    if (ImGui::Button("삭제") || (hasSelection && ImGui::IsKeyPressed(ImGuiKey_Delete))) {
        active.objects.erase(active.objects.begin() + selectedObject);
        selectedObject = -1;
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
                object.meshIndex = meshIndex;
                object.transform.position = active.camera.position + active.camera.forward() * (radius * 3.0F);
                active.objects.push_back(std::move(object));
                selectedObject = static_cast<int>(active.objects.size()) - 1;
            }
        }
        ImGui::EndPopup();
    }
    ImGui::Separator();

    for (int i = 0; i < static_cast<int>(active.objects.size()); ++i) {
        scene::Object& object = active.objects[static_cast<size_t>(i)];
        ImGui::PushID(i);
        ImGui::Checkbox("##visible", &object.visible);
        ImGui::SameLine();
        if (ImGui::Selectable(object.name.c_str(), selectedObject == i)) {
            selectedObject = i;
        }
        ImGui::PopID();
    }
    ImGui::End();
}

void Editor::buildInspector(scene::Scene& active, const gfx::GeometryStore& geometry) {
    if (!ImGui::Begin(WINDOW_INSPECTOR)) {
        ImGui::End();
        return;
    }

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
    }

    if (object.meshIndex < geometry.meshCount() &&
        ImGui::CollapsingHeader("메쉬와 재질", ImGuiTreeNodeFlags_DefaultOpen)) {
        const gfx::GpuMesh& mesh = geometry.mesh(object.meshIndex);
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

    ImVec2 imagePosition = ImGui::GetItemRectMin();
    ImVec2 imageSize = ImGui::GetItemRectSize();

    gizmoUsing = false;
    bool hasSelection = selectedObject >= 0 && selectedObject < static_cast<int>(active.objects.size());
    if (hasSelection && imageSize.x > 1.0F && imageSize.y > 1.0F) {
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

        scene::Object& object = active.objects[static_cast<size_t>(selectedObject)];
        glm::mat4 model = object.transform.matrix();

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
            object.transform = scene::Transform::fromMatrix(model);
        }
        gizmoUsing = ImGuizmo::IsUsing();
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
    ImGui::EndGroup();

    ImGui::End();
}

void Editor::buildRenderSettings(float deltaSeconds) {
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
    ImGui::Separator();

    ImGui::SliderFloat("노출", &renderer.exposure, 0.05F, 8.0F, "%.2f");
    ImGui::Checkbox("와이어프레임", &renderer.wireframe);

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
        int bounces = static_cast<int>(renderer.pathTraceBounces);
        if (ImGui::SliderInt("반사 횟수", &bounces, 1, 8)) {
            renderer.pathTraceBounces = static_cast<uint32_t>(bounces);
        }
        ImGui::Text("누적 표본 %u", renderer.pathTraceSamples());
    }

    ImGui::SeparatorText("파이프라인");
    ImGui::BeginDisabled(!renderer.meshShaderAvailable());
    ImGui::Checkbox("mesh shader 경로", &renderer.useMeshShader);
    ImGui::EndDisabled();
    if (!renderer.meshShaderAvailable()) {
        ImGui::SameLine();
        ImGui::TextDisabled("(미지원)");
    }

    static constexpr const char* DEBUG_MODE_NAMES[] = {"셰이딩", "meshlet", "노멀", "UV", "깊이", "LOD"};
    int debugMode = static_cast<int>(renderer.debugMode);
    if (ImGui::Combo("디버그 뷰", &debugMode, DEBUG_MODE_NAMES, IM_ARRAYSIZE(DEBUG_MODE_NAMES))) {
        renderer.debugMode = static_cast<uint32_t>(debugMode);
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
    buildRenderSettings(deltaSeconds);
    buildRenderTargets();
    buildConsole();

    ImGui::Render();
}

void Editor::record(VkCommandBuffer commandBuffer) {
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
}

} // namespace editor
