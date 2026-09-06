#include "editor/editor.h"

#include "editor/editor_internal.h"

namespace editor {

namespace {

constexpr float BASE_FONT_SIZE = 16.0F;

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
        ImGui::DockBuilderDockWindow(WINDOW_PROFILER, bottom);
        ImGui::DockBuilderDockWindow(WINDOW_SCENE, center);
        ImGui::DockBuilderFinish(dockspaceId);
    }
    ImGui::DockSpace(dockspaceId, ImVec2{0.0F, 0.0F}, ImGuiDockNodeFlags_None);

    buildMenuBar(scenes, geometry);
    buildPopups(scenes);
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
    if (pluginWindows) {
        pluginWindows();
    }
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
    selectedObject = primarySelection();

    updateHistory(scenes.active());

    ImGui::Render();
}

void Editor::record(VkCommandBuffer commandBuffer) {
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
}

} // namespace editor
