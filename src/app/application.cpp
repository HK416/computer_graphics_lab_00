#include "app/application.h"

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>

#include "core/error.h"

namespace app {
namespace {
constexpr int DEFAULT_WINDOW_WIDTH = 1600;
constexpr int DEFAULT_WINDOW_HEIGHT = 900;
} // namespace

Application::Application() {
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        core::fatal("SDL 초기화에 실패했습니다: {}", SDL_GetError());
    }

    window = SDL_CreateWindow("Computer Graphics Lab",
                              DEFAULT_WINDOW_WIDTH,
                              DEFAULT_WINDOW_HEIGHT,
                              SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (window == nullptr) {
        core::fatal("윈도우 생성에 실패했습니다: {}", SDL_GetError());
    }

    spdlog::info("윈도우 생성 완료 ({}x{})", DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);

    context = std::make_unique<gfx::Context>(window);
    renderer = std::make_unique<gfx::Renderer>(*context, window);
}

Application::~Application() {
    // 렌더러가 쓰는 서피스는 윈도우보다 먼저 파괴되어야 한다.
    renderer.reset();
    context.reset();
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void Application::run() {
    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            switch (event.type) {
            case SDL_EVENT_QUIT:
                running = false;
                break;
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
                renderer->requestResize();
                break;
            default:
                break;
            }
        }

        if ((SDL_GetWindowFlags(window) & SDL_WINDOW_MINIMIZED) != 0) {
            SDL_Delay(10);
            continue;
        }
        renderer->drawFrame();
    }
    renderer->waitIdle();
}

} // namespace app
