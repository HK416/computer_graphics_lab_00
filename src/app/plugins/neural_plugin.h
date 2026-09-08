#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "app/plugin.h"
#include "gfx/headless_compute.h"
#include "gfx/neural.h"
#include "gfx/observation.h"
#include "gfx/replay.h"
#include "gfx/rl_agent.h"
#include "physics/robot.h"
#include "scene/scene.h"

namespace app {

// 픽셀만 보고 관절을 모는 정책. 편집기 안에서도, 헤드리스에서도 **같은 구조**로 돈다 — 정책 스텝
// 하나가 명령 버퍼 하나이고, 그 안에 [관측 렌더 → 인코드 → 링에 담기 → 행동 순전파 → (학습 중이면)
// 갱신 K 회 → 되읽기] 가 순서대로 들어간다. 되읽는 것은 행동 A 개와 통계 몇 개뿐이다.
//
// **행동과 보상은 관측보다 늦게 정해진다.** 행동은 같은 제출의 순전파가 내놓고, 보상은 그 행동으로
// 물리를 한 걸음 밟은 다음 프레임에야 안다. 그래서 링에는 관측만 먼저 담고 둘은 뒤에 고쳐 넣는다.
// 순서가 어긋나지 않는 근거는 방금 담은 칸이 **뒤가 없어 다음 프레임까지 표본이 되지 못한다**는 것이다.
class NeuralPlugin : public Plugin {
public:
    // **소멸자가 있어야 한다.** 잡음 버퍼는 unique_ptr 이 아닌 raw Buffer 라 스스로 놓지 못한다.
    // Application 이 플러그인을 컨텍스트보다 먼저 지우므로 여기서 놓는 것이 안전하다.
    ~NeuralPlugin() override;
    const char* name() const override { return "픽셀 학습"; }
    void build(Services& services) override;
    void update(Services& services, float deltaSeconds) override;
    void ui(Services& services) override;

private:
    // 장면에 맞춰 에이전트·실행기·링을 짓는다. 이미 같은 장면으로 지었으면 아무 것도 하지 않는다.
    bool ensure(Services& services);
    // 정책 스텝 하나. 참을 돌려주면 action 에 이번 행동이 들어 있다.
    bool step(Services& services);
    void resetEpisode(Services& services);

    bool enabled = false;
    bool training = false;
    // 한 번 실패하면 다시 시도하지 않는다. 프레임마다 만들었다 지우면 경고만 쏟아진다.
    bool disabled = false;

    std::unique_ptr<gfx::ObservationRenderer> observation;
    std::unique_ptr<gfx::NeuralExecutor> executor;
    std::unique_ptr<gfx::ReplayBuffer> replay;
    std::unique_ptr<gfx::HeadlessCompute> submit;
    // 잡음 버퍼를 놓을 때 쓴다. Services 는 프레임마다 새로 만들어져 붙잡아 둘 수 없다.
    gfx::Context* context = nullptr;

    gfx::Agent agent;
    gfx::MergedGraph merged;
    // merged 를 만든 표 셋. mergedTensor 가 이것을 본다.
    std::vector<const gfx::Graph*> sources;
    gfx::AgentUpdateSettings settings;
    physics::RobotLayout robot;
    // 이 장면으로 지었는지. 장면이 바뀌면 다시 짓는다.
    uint64_t builtScene = 0;
    uint64_t builtTopology = 0;
    // **부품 변경도 본다.** 카메라나 관절 부품을 붙였다 떼면 오브젝트 수는 그대로인데 관측 표와
    // 액추에이터 첨자가 어긋난다 — RobotPlugin 이 componentRevision 을 보는 것과 같은 이유다.
    uint64_t builtComponents = 0;

    // 합친 표에서의 텐서 번호. 매 프레임 다시 풀지 않으려고 붙잡아 둔다.
    uint32_t actObservation = gfx::NO_TENSOR;
    uint32_t actAction = gfx::NO_TENSOR;
    uint32_t criticLoss = gfx::NO_TENSOR;
    uint32_t actorLoss = gfx::NO_TENSOR;
    uint32_t criticNoise = gfx::NO_TENSOR;
    gfx::ReplayBatchTargets criticTargets;
    gfx::ReplayBatchTargets actorTargets;

    uint32_t episode = 0;
    uint32_t stepInEpisode = 0;
    uint64_t totalSteps = 0;
    uint32_t pendingSlot = 0;
    bool hasPending = false;
    // 실제로 밟은 갱신 횟수. Adam 의 편향 보정이 이 번호를 본다 — 프레임 수로 세면 첫 갱신이 이미
    // 수천 번째가 되어 보정이 1 에 가까워지고, 초반 몇 걸음이 의도한 학습률의 세 배로 뛴다.
    uint32_t updateCount = 0;
    float episodeReward = 0.0F;
    float lastEpisodeReward = 0.0F;
    float bestEpisodeReward = 0.0F;

    std::vector<float> action;
    // 저작 자세. 에피소드마다 여기로 되돌린다. 진화 전략의 rollout 이 장면을 값으로 받아 매번 저작
    // 자세에서 출발하므로, 점수를 견주려면 이쪽도 같아야 한다.
    std::vector<scene::Transform> authoredTransforms;
    // 타깃 정책 평활화 잡음. 갱신마다 다른 값을 써야 해서 프레임당 갱신 수만큼 담아 한 번에 올린다.
    gfx::Buffer noiseBuffer;
    float* noiseStaging = nullptr;
    uint64_t sampleStream = 0;
    // --policy-net 가 준 경로. 학습 중이면 에피소드마다 여기에 가중치를 쓴다.
    std::string savePath;

    // 프레임 예산 가드. 지난 프레임의 학습 구간이 길면 갱신 수를 내린다.
    // 지금의 탐험 잡음 세기. 걸음 수에 따라 줄어든다.
    float explorationNoise = 0.0F;
    uint32_t updatesPerFrame = 1;
    float lastStepMilliseconds = 0.0F;
};

} // namespace app
