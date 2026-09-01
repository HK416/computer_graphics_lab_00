#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace core {

// Vyukov 의 유계 MPMC 큐. 칸마다 순번을 두어 잠금 없이 여러 생산자와 소비자가 오갈 수 있다.
// 용량은 2의 거듭제곱이어야 색인을 비트 마스크로 처리할 수 있다.
template <typename T> class LockFreeQueue {
public:
    explicit LockFreeQueue(size_t capacity) : cells(capacity), mask(capacity - 1) {
        for (size_t i = 0; i < capacity; ++i) {
            cells[i].sequence.store(i, std::memory_order_relaxed);
        }
        enqueuePosition.store(0, std::memory_order_relaxed);
        dequeuePosition.store(0, std::memory_order_relaxed);
    }

    bool push(const T& value) {
        Cell* cell = nullptr;
        size_t position = enqueuePosition.load(std::memory_order_relaxed);
        while (true) {
            cell = &cells[position & mask];
            size_t sequence = cell->sequence.load(std::memory_order_acquire);
            auto difference = static_cast<intptr_t>(sequence) - static_cast<intptr_t>(position);
            if (difference == 0) {
                if (enqueuePosition.compare_exchange_weak(position, position + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (difference < 0) {
                return false; // 가득 찼다.
            } else {
                position = enqueuePosition.load(std::memory_order_relaxed);
            }
        }
        cell->data = value;
        cell->sequence.store(position + 1, std::memory_order_release);
        return true;
    }

    bool pop(T& value) {
        Cell* cell = nullptr;
        size_t position = dequeuePosition.load(std::memory_order_relaxed);
        while (true) {
            cell = &cells[position & mask];
            size_t sequence = cell->sequence.load(std::memory_order_acquire);
            auto difference = static_cast<intptr_t>(sequence) - static_cast<intptr_t>(position + 1);
            if (difference == 0) {
                if (dequeuePosition.compare_exchange_weak(position, position + 1, std::memory_order_relaxed)) {
                    break;
                }
            } else if (difference < 0) {
                return false; // 비었다.
            } else {
                position = dequeuePosition.load(std::memory_order_relaxed);
            }
        }
        value = cell->data;
        cell->sequence.store(position + mask + 1, std::memory_order_release);
        return true;
    }

private:
    struct Cell {
        std::atomic<size_t> sequence;
        T data{};
    };

    // 생산자와 소비자 위치가 같은 캐시 줄에 놓이면 서로를 무효화하므로 떼어 둔다.
    static constexpr size_t CACHE_LINE = 64;

    std::vector<Cell> cells;
    size_t mask;
    alignas(CACHE_LINE) std::atomic<size_t> enqueuePosition{0};
    alignas(CACHE_LINE) std::atomic<size_t> dequeuePosition{0};
};

} // namespace core
