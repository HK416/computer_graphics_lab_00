#pragma once

#include <memory>

#include "gfx/context.h"

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
};

} // namespace app
