#pragma once

// 저차원 진자로 DDPG/TD3 를 돌리는 루프. **픽셀 이전에 «알고리즘이 실제로 학습하는가» 를 가리는
// 자리다.** 나중에 픽셀 학습이 안 붙을 때 «GPU 커널 버그» / «RL 알고리즘 버그» / «픽셀 관측이 나쁨» 을
// 갈라 보려면 이 셋 중 둘째를 여기서 미리 지워 두어야 한다.
//
// 테스트는 짧게(씨앗 몇 개, 몇 초) 돌리고 같은 함수를 길게 돌리면 진화 전략의 0.768 과 견줄 수 있다.
// 그래서 길이를 인자로 뺐다.
//
// **점수를 재는 눈금이 진화 전략과 같아야 한다**(evaluateFrames). 그러지 않으면 두 수를 나란히 적어 놓고
// 견주는 것이 뜻을 잃는다.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "gfx/rl_agent.h"
#include "pendulum_scene.h"
#include "physics/robot.h"
#include "scene/scene.h"

namespace {

struct PendulumSettings {
    // 한 에피소드의 프레임 수. 프레임마다 정책을 한 번 부르고 물리를 두 번 진행한다(재생 루프와 같다).
    uint32_t episodeFrames = 150;
    // 점수를 재는 프레임 수. **진화 전략과 같은 눈금이어야 한다** — physics::RolloutSettings 의 기본이
    // 300 이고, 같은 정책도 60 프레임에서 0.16, 300 프레임에서 0.61 이 나온다(진자가 올라가는 데 걸리는
    // 시간이 앞쪽 프레임을 통째로 깎아 먹는다). 학습 길이와 따로 두는 이유가 그것이다.
    uint32_t evaluateFrames = 300;
    // 전부 몇 프레임을 살아 볼지.
    uint32_t totalFrames = 9000;
    // 이만큼은 무작위로 움직이며 리플레이만 채운다. 학습은 그 뒤에 시작한다.
    uint32_t seedFrames = 500;
    // 몇 프레임마다 갱신 한 번인지.
    uint32_t framesPerUpdate = 1;
    uint32_t replayCapacity = 20000;

    uint32_t featureCount = 16;
    uint32_t hidden = 32;
    uint32_t batch = 32;
    float discount = 0.99F;
    // 탐험 잡음. 처음에는 크게, 학습이 진행되며 줄인다.
    float explorationBegin = 0.6F;
    float explorationEnd = 0.1F;

    uint64_t seed = 1;
    gfx::AgentUpdateSettings update;
};

struct PendulumResult {
    // 학습 전 정책(초기 가중치)의 점수.
    float baseline = 0.0F;
    float trained = 0.0F;
    // 마지막 갱신의 통계.
    float criticLoss = 0.0F;
    float value = 0.0F;
    uint32_t updates = 0;
};

// 정책을 그대로 돌려 프레임당 평균 보상을 낸다(physics::rollout 과 같은 눈금이라 진화 전략의 점수와
// 그대로 견줄 수 있다).
float evaluatePendulum(scene::Scene scene,
                       gfx::AgentTrainer& trainer,
                       const physics::RobotLayout& layout,
                       uint32_t frames) {
    // 재생 중이어야 부품이 스냅샷으로 되돌아가지 않는다(physics::rollout 과 같은 방어다).
    scene.simulating = true;
    std::vector<float> observation(layout.observationCount(), 0.0F);
    std::vector<float> action(layout.actionCount(), 0.0F);
    float reward = 0.0F;
    for (uint32_t frame = 0; frame < frames; ++frame) {
        physics::observe(scene, layout, observation.data());
        trainer.act(observation.data(), action.data());
        physics::act(scene, layout, action.data());
        physics::stepRigidBodies(scene, physics::STEP_SECONDS, nullptr);
        physics::stepRigidBodies(scene, physics::STEP_SECONDS, nullptr);
        reward += physics::stepReward(scene, layout);
    }
    return reward / static_cast<float>(std::max(frames, 1U));
}

// 표본 하나를 고르는 균등 난수. neuralGaussian 은 정규분포라 첨자를 고르는 데 쓰면 한쪽으로 쏠린다.
uint64_t uniformDraw(uint64_t stream, uint64_t index) {
    uint64_t value = stream * 0x9E3779B97F4A7C15ULL + index;
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31);
}

// 리플레이 링. 전이가 «다음 관측» 을 통째로 들고 있어 에피소드 경계를 따로 다루지 않아도 된다
// (프레임 스택을 번호로 겹쳐 읽는 GPU 판은 그럴 수 없고, 그래서 11단계에서 따로 만든다).
struct PendulumReplay {
    uint32_t observationSize = 0;
    uint32_t actionCount = 0;
    uint32_t capacity = 0;
    uint32_t stored = 0;
    uint32_t cursor = 0;
    std::vector<float> observations;
    std::vector<float> actions;
    std::vector<float> rewards;
    std::vector<float> discounts;
    std::vector<float> nextObservations;

    void reset(uint32_t size, uint32_t width, uint32_t slots) {
        observationSize = size;
        actionCount = width;
        capacity = slots;
        stored = 0;
        cursor = 0;
        observations.assign(static_cast<size_t>(capacity) * observationSize, 0.0F);
        nextObservations.assign(static_cast<size_t>(capacity) * observationSize, 0.0F);
        actions.assign(static_cast<size_t>(capacity) * actionCount, 0.0F);
        rewards.assign(capacity, 0.0F);
        discounts.assign(capacity, 0.0F);
    }

    void store(const float* observation, const float* action, float reward, float discount, const float* next) {
        size_t o = static_cast<size_t>(cursor) * observationSize;
        size_t a = static_cast<size_t>(cursor) * actionCount;
        std::copy(observation, observation + observationSize, observations.begin() + static_cast<ptrdiff_t>(o));
        std::copy(next, next + observationSize, nextObservations.begin() + static_cast<ptrdiff_t>(o));
        std::copy(action, action + actionCount, actions.begin() + static_cast<ptrdiff_t>(a));
        rewards[cursor] = reward;
        discounts[cursor] = discount;
        cursor = (cursor + 1) % capacity;
        stored = std::min(stored + 1, capacity);
    }
};

// 학습을 돌리고 점수를 돌려준다. 씨앗이 같으면 결과도 같다.
PendulumResult trainPendulum(const PendulumSettings& settings) {
    PendulumResult result;
    scene::Scene start = makePendulum();
    physics::RobotLayout layout = physics::buildRobotLayout(start);

    gfx::AgentConfig config;
    config.imageSize = 0;
    config.observationCount = layout.observationCount();
    config.actionCount = layout.actionCount();
    config.batch = settings.batch;
    config.featureCount = settings.featureCount;
    config.hidden = settings.hidden;

    gfx::AgentTrainer trainer;
    if (!trainer.build(config, settings.seed)) {
        return result;
    }
    result.baseline = evaluatePendulum(start, trainer, layout, settings.evaluateFrames);

    PendulumReplay replay;
    replay.reset(config.observationSize(), config.actionCount, settings.replayCapacity);

    std::vector<float> observation(layout.observationCount(), 0.0F);
    std::vector<float> next(layout.observationCount(), 0.0F);
    std::vector<float> action(layout.actionCount(), 0.0F);
    // 배치 하나를 미리 잡아 두고 되쓴다.
    std::vector<float> batchObservations(static_cast<size_t>(config.batch) * config.observationSize(), 0.0F);
    std::vector<float> batchNext(batchObservations.size(), 0.0F);
    std::vector<float> batchActions(static_cast<size_t>(config.batch) * config.actionCount, 0.0F);
    std::vector<float> batchRewards(config.batch, 0.0F);
    std::vector<float> batchDiscounts(config.batch, 0.0F);

    scene::Scene world = start;
    uint32_t episodeFrame = 0;
    uint64_t noiseStream = settings.seed * 7919 + 13;
    for (uint32_t frame = 0; frame < settings.totalFrames; ++frame) {
        if (episodeFrame >= settings.episodeFrames) {
            world = start;
            episodeFrame = 0;
        }
        physics::observe(world, layout, observation.data());

        // 리플레이를 채우는 동안은 순수 무작위로 움직인다. 학습 초반의 정책은 «아무 것도 하지 않음» 에
        // 가까워, 그대로 두면 진자가 아래에만 머물러 리플레이에 위쪽 상태가 아예 담기지 않는다.
        float progress = static_cast<float>(frame) / static_cast<float>(std::max(settings.totalFrames, 1U));
        float exploration =
            settings.explorationBegin + (settings.explorationEnd - settings.explorationBegin) * progress;
        if (frame < settings.seedFrames) {
            for (uint32_t i = 0; i < config.actionCount; ++i) {
                action[i] = std::clamp(gfx::neuralGaussian(noiseStream, frame * config.actionCount + i), -1.0F, 1.0F);
            }
        } else {
            trainer.act(observation.data(), action.data());
            for (uint32_t i = 0; i < config.actionCount; ++i) {
                float sample = exploration * gfx::neuralGaussian(noiseStream, frame * config.actionCount + i);
                action[i] = std::clamp(action[i] + sample, -1.0F, 1.0F);
            }
        }

        physics::act(world, layout, action.data());
        physics::stepRigidBodies(world, physics::STEP_SECONDS, nullptr);
        physics::stepRigidBodies(world, physics::STEP_SECONDS, nullptr);
        float reward = physics::stepReward(world, layout);
        physics::observe(world, layout, next.data());
        // 진자는 «끝나는» 상태가 없다. 에피소드가 시간으로 끊길 뿐이라 감가를 늘 그대로 둔다 — 시간
        // 제한을 종료로 다루면 «시간이 다 됐으니 앞으로 받을 것이 없다» 는 거짓을 배운다.
        replay.store(observation.data(), action.data(), reward, settings.discount, next.data());
        ++episodeFrame;

        if (frame < settings.seedFrames || replay.stored < config.batch) {
            continue;
        }
        if ((frame - settings.seedFrames) % std::max(settings.framesPerUpdate, 1U) != 0) {
            continue;
        }
        for (uint32_t i = 0; i < config.batch; ++i) {
            uint32_t slot = static_cast<uint32_t>(
                uniformDraw(noiseStream + 977, static_cast<uint64_t>(result.updates) * config.batch + i) %
                replay.stored);
            std::copy_n(replay.observations.begin() + static_cast<ptrdiff_t>(slot) * config.observationSize(),
                        config.observationSize(),
                        batchObservations.begin() + static_cast<ptrdiff_t>(i) * config.observationSize());
            std::copy_n(replay.nextObservations.begin() + static_cast<ptrdiff_t>(slot) * config.observationSize(),
                        config.observationSize(),
                        batchNext.begin() + static_cast<ptrdiff_t>(i) * config.observationSize());
            std::copy_n(replay.actions.begin() + static_cast<ptrdiff_t>(slot) * config.actionCount,
                        config.actionCount,
                        batchActions.begin() + static_cast<ptrdiff_t>(i) * config.actionCount);
            batchRewards[i] = replay.rewards[slot];
            batchDiscounts[i] = replay.discounts[slot];
        }
        gfx::AgentBatch batch;
        batch.observations = batchObservations.data();
        batch.nextObservations = batchNext.data();
        batch.actions = batchActions.data();
        batch.rewards = batchRewards.data();
        batch.discounts = batchDiscounts.data();
        gfx::AgentUpdateStats stats = trainer.update(batch, settings.update);
        result.criticLoss = stats.criticLoss;
        result.value = stats.value;
        ++result.updates;
    }

    result.trained = evaluatePendulum(start, trainer, layout, settings.evaluateFrames);
    return result;
}

} // namespace
