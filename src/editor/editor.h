#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>

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
    void build(scene::SceneManager& scenes, const gfx::GeometryStore& geometry, float deltaSeconds);
    void record(VkCommandBuffer commandBuffer);

    VkExtent2D desiredRenderExtent() const { return viewportExtent; }
    bool viewportHovered() const { return sceneHovered; }

private:
    void buildDockspace();
    void buildHierarchy(scene::SceneManager& scenes);
    void buildInspector(scene::Scene& scene, const gfx::GeometryStore& geometry);
    void buildSceneView();
    void buildRenderSettings(float deltaSeconds);
    void buildRenderTargets();
    void buildConsole();
    VkDescriptorSet textureFor(VkImageView view);

    gfx::Context& context;
    gfx::Renderer& renderer;
    std::shared_ptr<LogSink> logSink;

    std::unordered_map<VkImageView, VkDescriptorSet> textures;
    uint64_t cachedGeneration = 0;

    VkExtent2D viewportExtent{1280, 720};
    bool sceneHovered = false;
    bool layoutBuilt = false;
    int selectedObject = -1;
    int selectedTarget = 0;
    float frameTimeMilliseconds = 0.0F;
};

} // namespace editor
