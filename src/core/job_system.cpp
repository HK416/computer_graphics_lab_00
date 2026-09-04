#include "core/job_system.h"

#include <algorithm>
#include <chrono>
#include <thread>

namespace core {
namespace {
constexpr size_t JOB_QUEUE_CAPACITY = 4096;
// 큐가 비었을 때 잠들기 전에 다시 살펴보는 횟수. 한 프레임에 parallelFor 를 수십 번 부르고 그 사이
// 간격이 마이크로초 단위라, 매번 조건 변수로 재우고 깨우면 그 값이 작업 자체보다 비싸다. 잠깐 돌며
// 기다리면 그 비용이 사라지고, 정말로 할 일이 없을 때만 잠든다.
constexpr uint32_t IDLE_SPINS = 512;
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
    {
        // 자고 있는 워커가 조건을 다시 보게 하려면 잠금을 한 번 잡았다 놓아야 한다.
        std::lock_guard<std::mutex> lock(wakeMutex);
    }
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
    uint32_t idle = 0;
    while (running.load(std::memory_order_acquire)) {
        if (runOne()) {
            idle = 0;
            continue;
        }
        if (idle < IDLE_SPINS) {
            ++idle;
            std::this_thread::yield();
            continue;
        }
        // 한참 비어 있으면 잠들었다가 새 작업이 들어올 때 깨어난다. 계속 도는 것보다 전력이 덜 든다.
        //
        // sleepers 는 seq_cst 다. 아래 «자는 워커가 없으면 깨우지 않는다» 와 짝을 이루는데, 이 증가와
        // 저쪽의 pendingJobs 증가가 서로의 옛 값을 볼 수 있으면(store buffering) 깨우기를 놓친다.
        // 놓쳐도 부르는 쪽이 큐를 직접 비우고 아래 1 ms 시간 제한이 덮지만, 그 한 번이 혼자 도느라
        // 느려지므로 x86 이 아닌 곳에서도 닫아 둔다.
        sleepers.fetch_add(1, std::memory_order_seq_cst);
        bool woken = false;
        {
            std::unique_lock<std::mutex> lock(wakeMutex);
            woken = wakeCondition.wait_for(lock, std::chrono::milliseconds(1), [this]() {
                return !running.load(std::memory_order_acquire) || pendingJobs.load(std::memory_order_acquire) > 0;
            });
        }
        sleepers.fetch_sub(1, std::memory_order_seq_cst);
        // 시간 제한으로 그냥 돌아온 것이면 여전히 할 일이 없다는 뜻이다. 다시 512 번 돌면 쉬는 동안
        // 워커마다 코어를 태운다. 실제로 깨워진 때만 도는 몫을 되돌린다.
        idle = woken ? 0 : IDLE_SPINS;
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

    pendingJobs.fetch_add(1, std::memory_order_seq_cst);
    // 자고 있는 워커가 없으면 깨울 것도 없다. 조건 변수를 건드리는 값이 짧은 작업 하나보다 비싸다.
    // 위 증가와 이 읽기가 seq_cst 라 워커 쪽 sleepers 증가와 서로를 반드시 본다.
    if (sleepers.load(std::memory_order_seq_cst) > 0) {
        wakeCondition.notify_all();
    }

    while (remaining.load(std::memory_order_acquire) > 0) {
        if (!runOne()) {
            std::this_thread::yield();
        }
    }
    pendingJobs.fetch_sub(1, std::memory_order_release);
}

} // namespace core
