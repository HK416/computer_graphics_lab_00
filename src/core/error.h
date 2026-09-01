#pragma once

#include <format>
#include <string>
#include <utility>

namespace core {

namespace detail {
[[noreturn]] void fatalImpl(const std::string& message);
} // namespace detail

// 복구 불가능한 오류를 메시지 박스로 알린 뒤 프로세스를 종료한다. 폴백 경로는 두지 않는다.
template <typename... Args> [[noreturn]] void fatal(std::format_string<Args...> format, Args&&... args) {
    detail::fatalImpl(std::format(format, std::forward<Args>(args)...));
}

} // namespace core
