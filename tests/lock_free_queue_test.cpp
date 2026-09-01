#include <atomic>
#include <cassert>
#include <cstdio>
#include <thread>
#include <vector>

#include "core/job_system.h"
#include "core/lock_free_queue.h"

namespace {

// 여러 생산자와 소비자가 동시에 오갈 때 모든 항목이 정확히 한 번씩만 나오는지 본다.
void testQueue() {
    constexpr size_t CAPACITY = 1024;
    constexpr int PRODUCERS = 4;
    constexpr int CONSUMERS = 4;
    constexpr int PER_PRODUCER = 20000;

    core::LockFreeQueue<int> queue(CAPACITY);
    std::vector<std::atomic<int>> seen(PRODUCERS * PER_PRODUCER);
    std::atomic<int> consumed{0};
    std::atomic<bool> producing{true};

    std::vector<std::thread> threads;
    for (int producer = 0; producer < PRODUCERS; ++producer) {
        threads.emplace_back([&queue, producer]() {
            for (int i = 0; i < PER_PRODUCER; ++i) {
                int value = producer * PER_PRODUCER + i;
                while (!queue.push(value)) {
                    std::this_thread::yield();
                }
            }
        });
    }
    for (int consumer = 0; consumer < CONSUMERS; ++consumer) {
        threads.emplace_back([&queue, &seen, &consumed, &producing]() {
            int value = 0;
            while (producing.load() || consumed.load() < PRODUCERS * PER_PRODUCER) {
                if (queue.pop(value)) {
                    seen[static_cast<size_t>(value)].fetch_add(1);
                    consumed.fetch_add(1);
                } else {
                    std::this_thread::yield();
                }
            }
        });
    }

    for (int i = 0; i < PRODUCERS; ++i) {
        threads[static_cast<size_t>(i)].join();
    }
    producing.store(false);
    for (size_t i = PRODUCERS; i < threads.size(); ++i) {
        threads[i].join();
    }

    assert(consumed.load() == PRODUCERS * PER_PRODUCER);
    for (const std::atomic<int>& count : seen) {
        assert(count.load() == 1);
    }
    std::printf("큐: 생산자 %d, 소비자 %d, 항목 %d개 모두 정확히 한 번씩 소비됨\n",
                PRODUCERS,
                CONSUMERS,
                PRODUCERS * PER_PRODUCER);
}

// 분배기가 범위를 빠짐없이 겹치지 않게 나누는지 본다.
void testJobSystem() {
    core::JobSystem jobs(4);
    constexpr uint32_t COUNT = 100000;
    std::vector<std::atomic<int>> visits(COUNT);

    jobs.parallelFor(COUNT, 64, [&visits](uint32_t begin, uint32_t end) {
        for (uint32_t i = begin; i < end; ++i) {
            visits[i].fetch_add(1);
        }
    });

    for (const std::atomic<int>& count : visits) {
        assert(count.load() == 1);
    }
    std::printf("분배기: 워커 %u, 항목 %u개 모두 한 번씩 처리됨\n", jobs.workerCount(), COUNT);
}

} // namespace

int main() {
    testQueue();
    testJobSystem();
    std::puts("잠금 없는 큐와 작업 분배기 자체 점검 통과");
    return 0;
}
