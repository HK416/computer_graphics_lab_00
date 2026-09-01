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

struct SDL_Window;

namespace gfx {
struct Context;
class GeometryStore;
class Renderer;
} // namespace gfx

namespace scene {
class SceneManager;
struct Scene;
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
    // 계층 패널의 노드 하나와 그 자식들을 그린다.
    void drawHierarchyNode(scene::Scene& active, int index);
    void buildInspector(scene::Scene& scene, const gfx::GeometryStore& geometry);
    void buildSceneView(scene::Scene& active);
    void buildRenderSettings(scene::Scene& active, float deltaSeconds);
    void buildRenderTargets();
    void buildConsole();
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
    int selectedObject = -1;
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
    int selectedTarget = 0;
    // ImGuizmo::OPERATION 과 MODE 값. 헤더에서 ImGuizmo 를 끌어오지 않으려고 정수로 둔다.
    int gizmoOperation = 7;
    int gizmoMode = 1;
    bool gizmoUsing = false;
    float frameTimeMilliseconds = 0.0F;

public:
    // 통계 표시용. 애플리케이션이 채운다.
    unsigned workerCount = 0;
};

} // namespace editor
