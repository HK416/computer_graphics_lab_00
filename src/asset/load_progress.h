#pragma once

#include <atomic>
#include <cstdint>

namespace asset {

// 모델 적재의 진행 상황. 적재 스레드가 쓰고 메인 스레드가 프레임마다 읽으므로 원자값만 둔다.
struct LoadProgress {
    enum class Stage : uint32_t { IDLE, PARSE, CONVERT, TEXTURES, LOD, UPLOAD, DONE };

    std::atomic<Stage> stage{Stage::IDLE};
    std::atomic<uint64_t> done{0};
    std::atomic<uint64_t> total{0};

    // work 가 0 이면 끝을 모르는 단계다. 편집기는 이때 불확정 표시를 한다.
    void begin(Stage next, uint64_t work = 0) {
        done.store(0, std::memory_order_relaxed);
        total.store(work, std::memory_order_relaxed);
        stage.store(next, std::memory_order_release);
    }

    void advance(uint64_t amount = 1) { done.fetch_add(amount, std::memory_order_relaxed); }

    // 0 과 1 사이. 끝을 모르는 단계면 0 이다.
    float fraction() const {
        uint64_t work = total.load(std::memory_order_relaxed);
        if (work == 0) {
            return 0.0F;
        }
        uint64_t finished = done.load(std::memory_order_relaxed);
        return finished >= work ? 1.0F : static_cast<float>(static_cast<double>(finished) / static_cast<double>(work));
    }
};

} // namespace asset
