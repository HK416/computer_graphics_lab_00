#pragma once

#include <deque>
#include <mutex>
#include <string>

#include <spdlog/sinks/base_sink.h>

namespace editor {

struct LogEntry {
    spdlog::level::level_enum level;
    std::string text;
};

// 콘솔 패널이 보여줄 최근 로그를 담아 두는 싱크.
class LogSink : public spdlog::sinks::base_sink<std::mutex> {
public:
    std::deque<LogEntry> snapshot() {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries;
    }

protected:
    void sink_it_(const spdlog::details::log_msg& message) override {
        spdlog::memory_buf_t formatted;
        base_sink<std::mutex>::formatter_->format(message, formatted);
        std::string text = fmt::to_string(formatted);
        if (!text.empty() && text.back() == '\n') {
            text.pop_back();
        }
        entries.push_back({message.level, std::move(text)});
        if (entries.size() > MAX_ENTRIES) {
            entries.pop_front();
        }
    }

    void flush_() override {}

private:
    static constexpr size_t MAX_ENTRIES = 512;
    std::deque<LogEntry> entries;
};

} // namespace editor
