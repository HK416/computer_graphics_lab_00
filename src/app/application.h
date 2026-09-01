#pragma once

#include <memory>

#include "gfx/context.h"
#include "gfx/renderer.h"

struct SDL_Window;

namespace app {

class Application {
public:
    Application();
    ~Application();
    Application(const Application&) = delete;
    Application& operator=(const Application&) = delete;

    void run();

private:
    SDL_Window* window = nullptr;
    std::unique_ptr<gfx::Context> context;
    std::unique_ptr<gfx::Renderer> renderer;
};

} // namespace app
