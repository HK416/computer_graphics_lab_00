#include <cstdlib>
#include <string_view>

#if defined(_WIN32)
#include <windows.h>
#endif

#include <SDL3/SDL_main.h>
#include <spdlog/spdlog.h>

#include "app/application.h"

int main(int argc, char* argv[]) {
#if defined(_WIN32)
    // 매니페스트의 activeCodePage 는 콘솔 출력 코드 페이지까지 바꾸지는 않는다. 로그가 한글이라
    // 이걸 안 맞추면 cmd.exe 에 깨져 나온다.
    SetConsoleOutputCP(CP_UTF8);
#endif
    spdlog::set_pattern("[%H:%M:%S.%e] [%^%l%$] %v");
#ifndef NDEBUG
    spdlog::set_level(spdlog::level::debug);
#endif

    app::Options options;
    for (int i = 1; i < argc; ++i) {
        std::string_view argument = argv[i];
        if (argument == "--screenshot" && i + 1 < argc) {
            options.screenshotPath = argv[++i];
        } else if (argument == "--screenshot-frame" && i + 1 < argc) {
            options.screenshotFrame = static_cast<uint64_t>(std::atoll(argv[++i]));
        } else if (argument == "--scene" && i + 1 < argc) {
            options.initialScene = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (argument == "--open" && i + 1 < argc) {
            options.scenePath = argv[++i];
        } else if (argument == "--model" && i + 1 < argc) {
            options.modelPaths.emplace_back(argv[++i]);
        } else if (argument == "--debug" && i + 1 < argc) {
            options.debugMode = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (argument == "--lod" && i + 1 < argc) {
            options.lodLevel = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (argument == "--lod-error" && i + 1 < argc) {
            options.lodErrorThreshold = static_cast<float>(std::atof(argv[++i]));
        } else if (argument == "--neural-lod") {
            options.neuralLod = true;
        } else if (argument == "--pathtrace") {
            options.pathTracing = true;
        } else if (argument == "--empty") {
            options.emptyScene = true;
        } else if (argument == "--reflections") {
            options.reflections = true;
        } else if (argument == "--no-occlusion") {
            options.occlusionCulling = false;
        } else if (argument == "--no-mesh-shader") {
            options.meshShader = false;
        } else if (argument == "--orbit" && i + 1 < argc) {
            options.orbitDegreesPerFrame = static_cast<float>(std::atof(argv[++i]));
        } else if (argument == "--profile") {
            options.profile = true;
        } else if (argument == "--triangle-budget" && i + 1 < argc) {
            options.triangleBudget = static_cast<float>(std::atof(argv[++i]));
        } else if (argument == "--render-scale" && i + 1 < argc) {
            options.renderScale = static_cast<float>(std::atof(argv[++i]));
        } else if (argument == "--upscaler" && i + 1 < argc) {
            options.upscaler = static_cast<uint32_t>(std::atoi(argv[++i]));
        } else if (argument == "--threads" && i + 1 < argc) {
            options.threadCount = static_cast<unsigned>(std::atoi(argv[++i]));
        } else if (argument == "--gpu-budget" && i + 1 < argc) {
            options.gpuBudgetMegabytes = static_cast<uint64_t>(std::atoll(argv[++i]));
        } else if (argument == "--weld-angle" && i + 1 < argc) {
            options.weldAngleDegrees = static_cast<float>(std::atof(argv[++i]));
        }
    }

    app::Application application(options);
    application.run();
    return 0;
}
