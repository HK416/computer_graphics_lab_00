// 되돌리기 기록과 모델 참조 검사.
// Editor 의 멤버 함수 정의만 나눠 담은 번역 단위다. 선언은 editor.h 하나에 있다.

#include "editor/editor_internal.h"

namespace editor {

void Editor::updateHistory(scene::Scene& active) {
    // 기록이 너무 길어지면 애니메이터 스켈레톤 사본이 쌓여 메모리를 먹는다.
    constexpr size_t MAX_HISTORY = 64;

    History& history = histories[active.id];
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
    for (const auto& [sceneId, history] : histories) {
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

void Editor::clearHistories() {
    // 기준(baseline)은 지금 장면과 같으므로 남긴다. 다음 updateHistory 가 새로 잡는다.
    for (auto& [sceneId, history] : histories) {
        history.undoStack.clear();
        history.redoStack.clear();
        history.started = false;
    }
}

} // namespace editor
