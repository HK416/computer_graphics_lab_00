#include "core/job_system.h"

#include <algorithm>
#include <chrono>

namespace core {
namespace {
constexpr size_t JOB_QUEUE_CAPACITY = 4096;
} // namespace

JobSystem::JobSystem(unsigned threadCount) : jobs(JOB_QUEUE_CAPACITY) {
    unsigned hardware = std::max(std::thread::hardware_concurrency(), 2U);
    unsigned count = threadCount > 0 ? threadCount : hardware - 1;
    workers.reserve(count);
    for (unsigned i = 0; i < count; ++i) {
        workers.emplace_back([this]() { workerLoop(); });
    }
}

JobSystem::~JobSystem() {
    running.store(false, std::memory_order_release);
    wakeCondition.notify_all();
    for (std::thread& worker : workers) {
        worker.join();
    }
}

bool JobSystem::runOne() {
    Job job;
    if (!jobs.pop(job)) {
        return false;
    }
    (*job.body)(job.begin, job.end);
    job.remaining->fetch_sub(1, std::memory_order_release);
    return true;
}

void JobSystem::workerLoop() {
    while (running.load(std::memory_order_acquire)) {
        if (runOne()) {
            continue;
        }
        // 큐가 비면 잠들었다가 새 작업이 들어올 때 깨어난다. 계속 도는 것보다 전력이 덜 든다.
        std::unique_lock<std::mutex> lock(wakeMutex);
        wakeCondition.wait_for(lock, std::chrono::milliseconds(1), [this]() {
            return !running.load(std::memory_order_acquire) || pendingJobs.load(std::memory_order_acquire) > 0;
        });
    }
}

void JobSystem::parallelFor(uint32_t count, uint32_t granularity, const std::function<void(uint32_t, uint32_t)>& body) {
    if (count == 0) {
        return;
    }
    if (workers.empty() || count <= granularity) {
        body(0, count);
        return;
    }

    uint32_t chunkSize = std::max(granularity, 1U);
    uint32_t chunkCount = (count + chunkSize - 1) / chunkSize;
    std::atomic<uint32_t> remaining{chunkCount};

    for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        Job job;
        job.body = &body;
        job.begin = chunk * chunkSize;
        job.end = std::min(job.begin + chunkSize, count);
        job.remaining = &remaining;
        if (!jobs.push(job)) {
            // 큐가 가득 차면 호출 스레드가 바로 처리한다.
            body(job.begin, job.end);
            remaining.fetch_sub(1, std::memory_order_release);
        }
    }

    pendingJobs.fetch_add(1, std::memory_order_release);
    wakeCondition.notify_all();

    while (remaining.load(std::memory_order_acquire) > 0) {
        if (!runOne()) {
            std::this_thread::yield();
        }
    }
    pendingJobs.fetch_sub(1, std::memory_order_release);
}

} // namespace core
