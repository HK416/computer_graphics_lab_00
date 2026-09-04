#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "core/lock_free_queue.h"

namespace core {

// 잠금 없는 큐로 돌아가는 작업 분배기. 범위를 잘게 나눠 워커에게 흘려보내고, 호출한 스레드도
// 함께 일해서 큐가 가득 차도 교착에 빠지지 않는다.
class JobSystem {
public:
    // threadCount 가 0 이면 하드웨어 동시성에서 호출 스레드 몫을 뺀 만큼 만든다.
    explicit JobSystem(unsigned threadCount = 0);
    ~JobSystem();
    JobSystem(const JobSystem&) = delete;
    JobSystem& operator=(const JobSystem&) = delete;

    void parallelFor(uint32_t count, uint32_t granularity, const std::function<void(uint32_t, uint32_t)>& body);
    unsigned workerCount() const { return static_cast<unsigned>(workers.size()); }

private:
    struct Job {
        const std::function<void(uint32_t, uint32_t)>* body = nullptr;
        uint32_t begin = 0;
        uint32_t end = 0;
        std::atomic<uint32_t>* remaining = nullptr;
    };

    bool runOne();
    void workerLoop();

    LockFreeQueue<Job> jobs;
    std::vector<std::thread> workers;
    std::atomic<bool> running{true};
    std::mutex wakeMutex;
    std::condition_variable wakeCondition;
    std::atomic<uint32_t> pendingJobs{0};
    // 지금 조건 변수에서 자고 있는 워커 수. 0 이면 깨울 것이 없어 notify 를 건너뛴다.
    std::atomic<uint32_t> sleepers{0};
};

} // namespace core
