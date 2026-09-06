// 메뉴 막대·모달 대화 상자·오브젝트 만들기/편집 명령·단축키·재생.
// Editor 의 멤버 함수 정의만 나눠 담은 번역 단위다. 선언은 editor.h 하나에 있다.

#include "editor/editor_internal.h"

namespace editor {

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
        // 마지막 장면과 재생 중인 장면은 닫지 않는다(정지가 먼저).
        if (ImGui::MenuItem("장면 닫기", nullptr, false, scenes.count() > 1 && !scenes.active().simulating)) {
            deferred = [this, &scenes] { closeScene(scenes, scenes.current()); };
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
        auto found = histories.find(scenes.active().id);
        const History* history = found != histories.end() ? &found->second : nullptr;
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
        if (geometry.meshCount() == 0) {
            ImGui::TextDisabled("올라온 메쉬가 없습니다");
        }
        // 메쉬가 수백 개면 메뉴가 화면을 넘는다. 이름·재질로 걸러 보인다.
        // ponytail: 메뉴 안의 입력 칸은 키보드로 메뉴를 오갈 때 어색하다. 마우스로는 문제없다.
        ImGui::SetNextItemWidth(220.0F);
        ImGui::InputTextWithHint("##meshFilter", "이름 검색", meshFilter.data(), meshFilter.size());
        for (uint32_t meshIndex = 0; meshIndex < geometry.meshCount(); ++meshIndex) {
            if (!geometry.meshLive(meshIndex)) {
                continue;
            }
            const asset::Material& material = geometry.material(geometry.mesh(meshIndex).materialIndex);
            std::string label = std::to_string(meshIndex) + ": " + geometry.meshName(meshIndex) + " / " + material.name;
            if (!containsNoCase(label, meshFilter.data())) {
                continue;
            }
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

void Editor::closeScene(scene::SceneManager& scenes, size_t index) {
    if (index >= scenes.count() || scenes.at(index).simulating) {
        return;
    }
    uint64_t closedId = scenes.at(index).id;
    if (!scenes.close(index)) {
        return;
    }
    histories.erase(closedId);
    clearSelection();
    // 닫힌 장면만 쓰던 모델을 내린다. 기록은 방금 지웠으니 붙들지 않는다.
    if (modelCollector) {
        modelCollector();
    }
}

void Editor::startSimulation(scene::SceneManager& scenes) {
    scene::Scene& active = scenes.active();
    if (active.simulating) {
        return;
    }
    playSnapshot = active.capture();
    playSceneId = active.id;
    // GPU 강체 솔버는 다음 프레임 머리에서 이 변화를 보고 제 상태를 버린다(PhysicsPlugin).
    active.simulating = true;
}

void Editor::stopSimulation(scene::SceneManager& scenes) {
    scene::Scene& active = scenes.active();
    active.simulating = false;
    // 시작할 때 떠 둔 장면으로 되돌린다. 다른 장면으로 옮긴 채 멈췄으면 그 장면은 건드리지 않는다.
    if (playSnapshot && playSceneId == active.id) {
        active.restore(*playSnapshot);
        // 되돌린 상태가 곧 기록의 기준이다. 안 그러면 되돌리기 항목이 하나 더 생긴다.
        auto found = histories.find(active.id);
        if (found != histories.end() && found->second.started) {
            found->second.baseline = active.capture();
        }
    }
    playSnapshot.reset();
}

} // namespace editor
