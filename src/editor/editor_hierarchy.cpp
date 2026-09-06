// 계층 패널과 선택.
// Editor 의 멤버 함수 정의만 나눠 담은 번역 단위다. 선언은 editor.h 하나에 있다.

#include "editor/editor_internal.h"

namespace editor {

namespace {

// 계층 패널에서 오브젝트를 끌 때 쓰는 페이로드 이름.
constexpr const char* HIERARCHY_PAYLOAD = "계층 오브젝트";

} // namespace

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

} // namespace editor
