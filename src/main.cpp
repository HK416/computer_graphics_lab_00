#include <SDL3/SDL_main.h>
#include <spdlog/spdlog.h>

#include "app/application.h"

int main(int /*argc*/, char* /*argv*/[]) {
    spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
#ifndef NDEBUG
    spdlog::set_level(spdlog::level::debug);
#endif

    app::Application application;
    application.run();
    return 0;
}
