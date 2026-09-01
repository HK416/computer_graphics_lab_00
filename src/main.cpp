#include <cstdlib>
#include <string_view>

#include <SDL3/SDL_main.h>
#include <spdlog/spdlog.h>

#include "app/application.h"

int main(int argc, char* argv[]) {
    spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
#ifndef NDEBUG
    spdlog::set_level(spdlog::level::debug);
#endif

    app::Options options;
    for (int i = 1; i < argc; ++i) {
        std::string_view argument = argv[i];
        if (argument == "--screenshot" && i + 1 < argc) {
            options.screenshotPath = argv[++i];
        } else if (argument == "--scene" && i + 1 < argc) {
            options.initialScene = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (argument == "--debug" && i + 1 < argc) {
            options.debugMode = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (argument == "--lod" && i + 1 < argc) {
            options.lodLevel = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (argument == "--lod-error" && i + 1 < argc) {
            options.lodErrorThreshold = static_cast<float>(std::atof(argv[++i]));
        } else if (argument == "--neural-lod") {
            options.neuralLod = true;
        } else if (argument == "--triangle-budget" && i + 1 < argc) {
            options.triangleBudget = static_cast<float>(std::atof(argv[++i]));
        }
    }

    app::Application application(options);
    application.run();
    return 0;
}
