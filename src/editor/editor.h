#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include <SDL3/SDL_events.h>
#include <vulkan/vulkan.h>

// 되돌리기 기록이 SceneSnapshot 을 값으로 들고 있어 전방 선언으로는 부족하다.
#include "scene/scene.h"

struct SDL_Window;

namespace gfx {
struct Context;
class GeometryStore;
class Renderer;
} // namespace gfx

namespace scene {
class SceneManager;
} // namespace scene

namespace editor {

class LogSink;

// Unity 편집기와 비슷한 도킹 구성의 GUI. 장면 뷰는 렌더러의 표시 이미지를 그대로 띄운다.
class Editor {
public:
    Editor(gfx::Context& context, gfx::Renderer& renderer, SDL_Window* window);
    ~Editor();
    Editor(const Editor&) = delete;
    Editor& operator=(const Editor&) = delete;

    void processEvent(const SDL_Event& event);
    // 편집기에서 glTF 를 런타임에 불러올 때 쓸 후보 목록의 뿌리와 실제 적재 함수.
    void setModelLoader(std::filesystem::path root, std::function<void(const std::filesystem::path&)> loader);
    // 장면 파일을 저장하고 여는 함수와 그 파일들이 놓이는 폴더.
    void setSceneIo(std::filesystem::path root,
                    std::function<void(const std::filesystem::path&)> saver,
                    std::function<void(const std::filesystem::path&)> opener);
    void build(scene::SceneManager& scenes, const gfx::GeometryStore& geometry, float deltaSeconds);
    void record(VkCommandBuffer commandBuffer);

    VkExtent2D desiredRenderExtent() const { return viewportExtent; }
    bool viewportHovered() const { return sceneHovered && !gizmoUsing; }

private:
    void buildDockspace();
    void buildHierarchy(scene::SceneManager& scenes, const gfx::GeometryStore& geometry);
    // F 키로 선택한 오브젝트를 궤도 중심으로 옮긴다.
    void focusSelected(scene::Scene& active, const gfx::GeometryStore& geometry);
    // 계층 패널의 노드 하나와 그 자식들을 그린다.
    // children 은 buildHierarchy 가 프레임마다 한 번 만든 부모별 자식 목록이다. 노드마다 전체를 훑으면 O(n²)다.
    void drawHierarchyNode(scene::Scene& active, const std::vector<std::vector<int>>& children, int index);
    void buildInspector(scene::Scene& scene, const gfx::GeometryStore& geometry);
    void buildSceneView(scene::Scene& active);
    void buildRenderSettings(scene::Scene& active, float deltaSeconds);
    void buildConsole();
    // 장면이 바뀌었으면 되돌리기 기록에 담고, Ctrl+Z / Ctrl+Y 를 처리한다.
    void updateHistory(scene::Scene& active, size_t sceneIndex);
    VkDescriptorSet textureFor(VkImageView view, VkImageLayout layout);

    gfx::Context& context;
    gfx::Renderer& renderer;
    // build() 호출마다 갱신된다. 렌더 설정 패널에서 LOD 범위를 알기 위해 쓴다.
    const gfx::GeometryStore* geometryStore = nullptr;
    std::shared_ptr<LogSink> logSink;

    std::unordered_map<VkImageView, VkDescriptorSet> textures;
    uint64_t cachedGeneration = 0;

    VkExtent2D viewportExtent{1280, 720};
    bool sceneHovered = false;
    bool layoutBuilt = false;
    // 장면마다의 되돌리기 기록. 장면을 오갈 수 있으므로 하나로 묶어 두면 섞인다.
    struct History {
        std::vector<scene::SceneSnapshot> undoStack;
        std::vector<scene::SceneSnapshot> redoStack;
        // 마지막으로 기록에 담은 상태. 다음 변경이 생기면 이것이 되돌릴 자리가 된다.
        scene::SceneSnapshot baseline;
        bool started = false;
    };
    std::vector<History> histories;

    // 선택된 오브젝트들. 마지막 항목이 기준(primary)이라 인스펙터와 기즈모가 그것을 쓴다.
    std::vector<int> selection;

    bool hasSelection() const { return !selection.empty(); }
    int primarySelection() const { return selection.empty() ? -1 : selection.back(); }
    bool isSelected(int index) const;
    // index 가 음수면 선택을 비운다.
    void selectOnly(int index);
    void toggleSelect(int index);
    // 지우거나 다시 열어 인덱스가 통째로 달라졌을 때.
    void clearSelection() { selection.clear(); }
    // 계층 순회 도중 부모를 바꾸면 그리기가 어긋나므로 순회가 끝난 뒤에 적용한다.
    int pendingChild = -1;
    int pendingParent = -1;
    std::function<void(const std::filesystem::path&)> modelLoader;
    std::filesystem::path modelRoot;
    std::vector<std::filesystem::path> modelFiles;
    std::filesystem::path pendingModel;
    std::array<char, 512> modelPathInput{};
    std::vector<std::filesystem::path> hdrFiles;
    std::array<char, 512> hdrPathInput{};
    std::function<void(const std::filesystem::path&)> sceneSaver;
    std::function<void(const std::filesystem::path&)> sceneOpener;
    std::filesystem::path sceneRoot;
    std::vector<std::filesystem::path> sceneFiles;
    std::filesystem::path pendingSceneSave;
    std::filesystem::path pendingSceneOpen;
    std::array<char, 256> sceneNameInput{};
    // 장면 뷰에 보여줄 렌더 타깃. 음수면 표시 이미지(기본)다. 층이나 밉이 있는 대상은 slice 로 고른다.
    int selectedTarget = -1;
    int selectedSlice = 0;
    // ImGuizmo::OPERATION 과 MODE 값. 헤더에서 ImGuizmo 를 끌어오지 않으려고 정수로 둔다.
    int gizmoOperation = 7;
    int gizmoMode = 1;
    bool gizmoUsing = false;
    // 기즈모 스냅. Ctrl 을 누르는 동안 걸리고, snapAlways 면 늘 건다. 이동은 단위, 회전은 도, 크기는 배율.
    float snapTranslate = 0.25F;
    float snapRotate = 15.0F;
    float snapScale = 0.1F;
    bool snapAlways = false;
    // 렌더 설정 패널의 그룹 검색어. 비어 있으면 모든 그룹을 접었다 펼 수 있게 보여 준다.
    std::array<char, 64> settingsFilter{};
    float frameTimeMilliseconds = 0.0F;

public:
    // 통계 표시용. 애플리케이션이 채운다.
    unsigned workerCount = 0;
};

} // namespace editor
