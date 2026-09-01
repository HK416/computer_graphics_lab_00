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
}

Application::~Application() {
    // 서피스가 윈도우보다 먼저 파괴되어야 한다.
    context.reset();
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void Application::run() {
    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            }
        }
        SDL_Delay(1);
    }
}

} // namespace app
