#include "core/error.h"

#include <cstdlib>

#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>

namespace core::detail {

void fatalImpl(const std::string& message) {
    spdlog::critical("{}", message);
    SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "치명적 오류", message.c_str(), nullptr);
    std::exit(EXIT_FAILURE);
}

} // namespace core::detail
