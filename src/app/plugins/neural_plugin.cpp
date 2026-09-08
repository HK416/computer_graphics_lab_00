#include "app/plugins/neural_plugin.h"

#include <algorithm>
#include <chrono>
#include <cmath>

#include <imgui.h>
#include <spdlog/spdlog.h>

#include "app/application.h"
#include "editor/editor.h"

namespace app {
namespace {

// 학습 중 행동에 섞는 잡음의 세기. **상수로 두면 안 된다** — 정책이 그 잡음에 기대어 학습되어, 잡음을
// 끈 평가에서 점수가 무너진다(실측: 상수 0.2 로 학습하면 학습 중 0.68 인데 잡음 없는 평가는 0.28 이다).
// DrQ-v2 가 스케줄을 쓰는 이유가 이것이다. 처음에는 넓게 훑고 뒤로 갈수록 결정적 정책에 가까워진다.
constexpr float EXPLORATION_NOISE_BEGIN = 0.5F;
constexpr float EXPLORATION_NOISE_END = 0.05F;
// 이 걸음 수에 걸쳐 선형으로 줄인다. 논문은 50 만 걸음이지만 우리 태스크는 2 만 대에 붙는다.
constexpr float EXPLORATION_NOISE_STEPS = 15000.0F;
// 감가. 저차원 진화 전략과 견주려면 에피소드 길이가 같아야 하고(300 프레임), 그 길이에서 0.99 면
// 지평선이 100 프레임쯤이라 진자를 세우기에 넉넉하다.
constexpr float DISCOUNT = 0.99F;
// 에피소드 하나의 프레임 수. physics::RolloutSettings::frames 와 같은 값이라야 점수를 견줄 수 있다.
constexpr uint32_t EPISODE_FRAMES = 300;
// 링을 이만큼 채우기 전에는 갱신하지 않는다. 배치보다 조금만 커도 같은 전이를 되씹어 발산한다.
constexpr uint32_t WARMUP_TRANSITIONS = 512;
// 「신경망 학습」 구간이 이보다 길면 갱신 수를 내린다. 편집기에서 프레임이 끊기지 않게 하는 가드다.
constexpr float FRAME_BUDGET_MILLISECONDS = 10.0F;
constexpr uint32_t MAX_UPDATES_PER_FRAME = 4;

} // namespace

NeuralPlugin::~NeuralPlugin() {
    if (context != nullptr) {
        destroyBuffer(*context, noiseBuffer);
    }
}

void NeuralPlugin::build(Services& services) {
    enabled = services.options.trainPixels || !services.options.policyNetPath.empty();
    training = services.options.trainPixels;
    settings.critic.learningRate = 1.0e-4F;
    settings.actor.learningRate = 1.0e-4F;
}

bool NeuralPlugin::ensure(Services& services) {
    if (disabled || services.context == nullptr || services.bindless == nullptr || services.geometry == nullptr) {
        return false;
    }
    scene::Scene& scene = services.scenes.active();
    if (executor != nullptr && builtScene == scene.id && builtTopology == scene.topologyRevision() &&
        builtComponents == scene.componentRevision()) {
        return true;
    }

    gfx::ObservationLayout layout = gfx::buildObservationLayout(scene);
    // **저작 값으로 되돌린 뒤 표를 만든다.** buildRobotLayout 은 joint.targetSpeed 를 «상한» 으로 읽는데
    // physics::act 가 같은 필드를 «이번 목표» 로 덮어쓴다. 되돌리지 않으면 상한이 마지막 행동 배로 줄고,
    // 행동이 0 근처였으면 액추에이터가 통째로 버려진다(RobotPlugin 도 같은 이유로 되돌린다).
    physics::restoreAuthoredMotors(scene, robot);
    robot = physics::buildRobotLayout(scene);
    if (layout.empty() || robot.actuators.empty()) {
        spdlog::warn(
            "픽셀 학습: 관측 카메라 {} 개(부품의 «관측 카메라» 를 켠 것), 액추에이터 {} 개 — 둘 다 있어야 합니다",
            layout.count(),
            robot.actuators.size());
        disabled = true;
        return false;
    }

    gfx::AgentConfig config;
    config.imageSize = gfx::OBSERVATION_SIZE;
    // **뷰마다 스택 하나가 아니라, 뷰 전부를 채널로 잇는다.** 12단계는 단일 뷰라 이 둘이 같고,
    // 14단계의 MAD 가 뷰마다 인코더를 따로 태울 때 갈린다.
    config.frameStack = gfx::OBSERVATION_STACK * layout.count();
    config.actionCount = static_cast<uint32_t>(robot.actuators.size());
    config.batch = 32;
    if (!gfx::buildAgent(config, agent)) {
        spdlog::warn("픽셀 학습: 에이전트 표를 짓지 못했습니다");
        disabled = true;
        return false;
    }

    sources = {&agent.act.graph, &agent.critic.graph, &agent.actor.graph};
    if (!gfx::mergeGraphs(sources, merged)) {
        spdlog::warn("픽셀 학습: 표 셋을 잇지 못했습니다");
        disabled = true;
        return false;
    }

    observation = std::make_unique<gfx::ObservationRenderer>(*services.context, *services.bindless, *services.geometry);
    executor = std::make_unique<gfx::NeuralExecutor>(*services.context);
    replay = std::make_unique<gfx::ReplayBuffer>(*services.context, layout.count(), config.actionCount);
    // 은퇴 자원을 거두지 않는다. 창이 있는 실행에서는 렌더러가 아직 도는 프레임에 맡긴 것이 섞여 있다.
    submit = std::make_unique<gfx::HeadlessCompute>(*services.context, false);
    if (!observation->available() || !executor->available() || !replay->available() || !observation->reserve(layout) ||
        !executor->build(merged.graph)) {
        spdlog::warn("픽셀 학습: GPU 자원을 만들지 못했습니다");
        disabled = true;
        return false;
    }
    if (executor->unsupportedOps() != 0) {
        spdlog::warn("픽셀 학습: GPU 커널이 없는 연산이 {} 개입니다", executor->unsupportedOps());
        disabled = true;
        return false;
    }
    replay->reserve(50000, config.batch);

    // 최적화기 둘. 크리틱 구간과 액터 구간을 따로 밟는다.
    if (!executor->reserveMoments(0, gfx::adamMomentCount(agent.parameters.criticUpdate.count())) ||
        !executor->reserveMoments(1, gfx::adamMomentCount(agent.parameters.actorUpdate.count()))) {
        disabled = true;
        return false;
    }

    actObservation = gfx::mergedTensor(merged, 0, agent.act.observation);
    actAction = gfx::mergedTensor(merged, 0, agent.act.action);
    criticLoss = gfx::mergedTensor(merged, 1, agent.critic.loss);
    criticNoise = gfx::mergedTensor(merged, 1, agent.critic.noise);
    actorLoss = gfx::mergedTensor(merged, 2, agent.actor.loss);

    auto address = [&](size_t which, uint32_t tensor) {
        uint32_t index = gfx::mergedTensor(merged, which, tensor);
        return executor->activationAddress() +
               static_cast<VkDeviceSize>(merged.graph.tensors[index].offset) * sizeof(float);
    };
    criticTargets.state = address(1, agent.critic.observation);
    criticTargets.nextState = address(1, agent.critic.nextObservation);
    criticTargets.action = address(1, agent.critic.action);
    criticTargets.reward = address(1, agent.critic.reward);
    criticTargets.discount = address(1, agent.critic.discount);
    // 액터 표는 관측만 쓴다. 나머지는 크리틱 쪽에 채운 것을 그대로 두면 되므로 상태만 가리킨다.
    actorTargets = criticTargets;
    actorTargets.state = address(2, agent.actor.observation);

    // 가중치를 초기화해 올린다. 파일이 있으면 그것으로 덮는다.
    std::vector<float> parameters(merged.graph.parameterCount, 0.0F);
    gfx::initializeAgent(agent, 1234, parameters.data());
    savePath = services.options.policyNetPath.generic_string();
    if (!savePath.empty() && gfx::loadAgent(agent, parameters.data(), savePath)) {
        spdlog::info("픽셀 정책을 읽었습니다: {}", savePath);
    }
    std::copy(parameters.begin(), parameters.end(), executor->parameterStaging());
    std::fill_n(executor->activationStaging(), merged.graph.activationCount, 0.0F);
    std::fill_n(executor->activationGradientStaging(), merged.graph.activationCount, 0.0F);
    submit->submit([&](VkCommandBuffer commandBuffer, uint64_t) {
        executor->recordUpload(commandBuffer);
        executor->recordClearMoments(commandBuffer, 0);
        executor->recordClearMoments(commandBuffer, 1);
    });

    action.assign(config.actionCount, 0.0F);
    // 갱신마다 다른 잡음을 써야 하므로 프레임당 갱신 수만큼 담아 한 번에 올린다.
    context = services.context;
    destroyBuffer(*context, noiseBuffer);
    noiseBuffer = createBuffer(*services.context,
                               static_cast<VkDeviceSize>(MAX_UPDATES_PER_FRAME) * config.batch * config.actionCount *
                                   sizeof(float),
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                               gfx::MemoryLocation::HOST_WRITE,
                               "타깃 평활화 잡음");
    noiseStaging = static_cast<float*>(noiseBuffer.mapped);
    builtScene = scene.id;
    builtTopology = scene.topologyRevision();
    builtComponents = scene.componentRevision();
    // 저작 자세를 붙잡아 둔다. 에피소드마다 여기로 되돌린다.
    authoredTransforms.clear();
    authoredTransforms.reserve(scene.objects.size());
    for (const scene::Object& object : scene.objects) {
        authoredTransforms.push_back(object.transform);
    }
    episode = 0;
    stepInEpisode = 0;
    totalSteps = 0;
    hasPending = false;
    episodeReward = 0.0F;
    spdlog::info("픽셀 학습 준비: 뷰 {}, 행동 {}, 가중치 {} 개, 배치 {}",
                 layout.count(),
                 config.actionCount,
                 merged.graph.parameterCount,
                 config.batch);
    return true;
}

void NeuralPlugin::resetEpisode(Services& services) {
    scene::Scene& scene = services.scenes.active();
    // **자세까지 되돌린다.** 속도만 지우면 지난 에피소드가 끝난 각도에서 이어 시작해, 세워 둔 자세를
    // 물려받은 에피소드가 거저 높은 점수를 받는다. 비교 기준인 physics::rollout 은 장면을 값으로 받아
    // 매번 저작 자세에서 출발하므로, 그것과 견주려면 여기도 같아야 한다.
    //
    // 자세는 Object::transform 에 있고 RigidBody 에는 없다 — 되돌릴 것이 둘로 나뉘어 있다.
    if (authoredTransforms.size() == scene.objects.size()) {
        for (size_t i = 0; i < scene.objects.size(); ++i) {
            if (scene.component<scene::RigidBody>(static_cast<uint32_t>(i)) != nullptr) {
                scene.objects[i].transform = authoredTransforms[i];
            }
        }
    }
    for (scene::RigidBody& body : scene.rigidBodies) {
        body.velocity = glm::vec3{0.0F};
        body.angularVelocity = glm::vec3{0.0F};
    }
    // **구조가 바뀐 것이 아니다.** markStructureDirty 를 부르면 개정 번호가 올라 ensure() 가 에피소드마다
    // 통째로 다시 짓고, 그러면 링이 비워져 워밍업에 영영 닿지 못한다(학습이 도는 것처럼 보이면서 갱신이
    // 한 번도 일어나지 않는다). 세계 변환 캐시는 이 프레임 끝의 scene.refresh 가 어차피 다시 만든다.
    // **마지막 걸음의 보상을 먼저 채운다.** 여기서 hasPending 을 끄면 그 칸에 결과가 영영 오지 않고,
    // 점수도 보상 299 개를 300 으로 나눈 것이 된다.
    if (hasPending) {
        float reward = physics::stepReward(scene, robot);
        episodeReward += reward;
        replay->patchOutcome(pendingSlot, reward, 0.0F);
    }
    ++episode;
    stepInEpisode = 0;
    lastEpisodeReward = episodeReward;
    bestEpisodeReward = std::max(bestEpisodeReward, episodeReward);
    // **점수의 눈금을 저차원 학습과 맞춰 둔다** — 걸음마다의 보상 합을 프레임 수로 나눈다. 같은 장면·
    // 같은 프레임 수라야 견줄 수 있어 EPISODE_FRAMES 를 physics::RolloutSettings::frames 와 같은 300 으로
    // 두었다. 진자에서 같은 눈금의 값들은: 진화 전략 0.768, CPU DDPG(저차원 관측) 0.685(rl_agent 테스트),
    // 이 픽셀 경로 0.62. 픽셀의 비용은 «같은 알고리즘의 저차원 판» 과 견주는 쪽이 뜻이 있다.
    spdlog::info("픽셀 학습 에피소드 {}: 점수 {:.4f} (최고 {:.4f}), 걸음 {}, 리플레이 {}, 잡음 {:.3f}, 갱신 {}",
                 episode - 1,
                 static_cast<double>(lastEpisodeReward / static_cast<float>(EPISODE_FRAMES)),
                 static_cast<double>(bestEpisodeReward / static_cast<float>(EPISODE_FRAMES)),
                 totalSteps,
                 replay != nullptr ? replay->count() : 0,
                 static_cast<double>(explorationNoise),
                 updatesPerFrame);
    episodeReward = 0.0F;
    hasPending = false;

    // **에피소드마다 저장한다.** 학습이 몇 시간짜리라 중간에 끊겨도 남는 것이 있어야 하고, 저장 비용은
    // 300 걸음에 한 번이라 값이 없다. 가중치를 통째로 되읽는 유일한 자리다.
    if (training && !savePath.empty() && executor != nullptr) {
        submit->submit([&](VkCommandBuffer commandBuffer, uint64_t) { executor->recordDownload(commandBuffer); });
        executor->invalidateReadback();
        std::vector<float> parameters(merged.graph.parameterCount, 0.0F);
        std::copy(
            executor->parameterResult(), executor->parameterResult() + merged.graph.parameterCount, parameters.begin());
        if (!gfx::saveAgent(agent, parameters.data(), savePath)) {
            spdlog::warn("픽셀 정책을 쓰지 못했습니다: {}", savePath);
        }
    }
}

bool NeuralPlugin::step(Services& services) {
    scene::Scene& scene = services.scenes.active();
    gfx::ObservationLayout layout = gfx::buildObservationLayout(scene);
    if (layout.count() != observation->viewCount()) {
        return false;
    }
    auto begin = std::chrono::steady_clock::now();

    // **지난 프레임의 결과를 먼저 채운다.** 그 행동으로 물리가 한 걸음 밟았으니 이제 보상을 안다.
    if (hasPending) {
        float reward = physics::stepReward(scene, robot);
        episodeReward += reward;
        bool terminal = stepInEpisode + 1 >= EPISODE_FRAMES;
        replay->patchOutcome(pendingSlot, reward, terminal ? 0.0F : DISCOUNT);
    }

    // 이번 칸의 메타. 행동과 보상은 아직 모르므로 뒤에 고친다.
    replay->beginSlot(episode, stepInEpisode, 0.0F, DISCOUNT, nullptr);

    uint32_t updates = training && replay->count() >= WARMUP_TRANSITIONS ? updatesPerFrame : 0;
    // 타깃 평활화 잡음을 채워 둔다. 자르는 폭까지 CPU 기준과 같은 규약이다.
    if (updates > 0 && noiseStaging != nullptr) {
        uint32_t perUpdate = merged.graph.tensors[criticNoise].count();
        for (uint32_t k = 0; k < updates; ++k) {
            for (uint32_t i = 0; i < perUpdate; ++i) {
                float value = settings.targetNoise * gfx::neuralGaussian(updateCount + k + 1, i);
                noiseStaging[k * perUpdate + i] = std::clamp(value, -settings.noiseClip, settings.noiseClip);
            }
        }
        vmaFlushAllocation(services.context->allocator, noiseBuffer.allocation, 0, VK_WHOLE_SIZE);
    }
    uint32_t stored = gfx::ReplayBuffer::NO_SLOT;
    submit->submit([&](VkCommandBuffer commandBuffer, uint64_t) {
        observation->record(commandBuffer, scene, layout, 0, stepInEpisode == 0);
        stored = replay->recordStore(commandBuffer, observation->featureAddress());
        // 관측을 행동 표의 입력 자리에 밀어 넣는다.
        executor->recordUploadActivationRange(commandBuffer,
                                              observation->featureBuffer(),
                                              0,
                                              merged.graph.tensors[actObservation].offset,
                                              merged.graph.tensors[actObservation].count());
        executor->recordForward(commandBuffer, merged.opBegin[0], merged.opCount[0]);

        for (uint32_t k = 0; k < updates; ++k) {
            // 크리틱. 표집이 관측·행동·보상·감가를 표의 입력 자리에 바로 채운다.
            if (!replay->recordSample(commandBuffer, criticTargets, 20260908, sampleStream++)) {
                break;
            }
            // **타깃 정책 평활화 잡음.** 표집이 채우는 다섯에는 없어서 따로 올린다 — 빠뜨리면 잡음이
            // 늘 0 이라 알고리즘이 조용히 TD3 가 아니게 되고, 연산 단위만 보는 자기 검사는 못 잡는다.
            const gfx::Tensor& noiseTensor = merged.graph.tensors[criticNoise];
            executor->recordUploadActivationRange(commandBuffer,
                                                  noiseBuffer.handle,
                                                  static_cast<VkDeviceSize>(k) * noiseTensor.count() * sizeof(float),
                                                  noiseTensor.offset,
                                                  noiseTensor.count());
            executor->recordClearGradients(commandBuffer);
            executor->recordForward(commandBuffer, merged.opBegin[1], merged.opCount[1]);
            executor->recordSeedLossGradient(commandBuffer, criticLoss);
            executor->recordBackward(commandBuffer, merged.opBegin[1], merged.opCount[1]);
            executor->recordAdam(commandBuffer,
                                 0,
                                 static_cast<uint32_t>(agent.parameters.criticUpdate.begin),
                                 static_cast<uint32_t>(agent.parameters.criticUpdate.count()),
                                 settings.critic,
                                 updateCount + k + 1);
            // 액터. 같은 표본으로 관측을 액터 표의 입력 자리에도 채운다.
            //   ponytail: 셰이더에 «이 대상은 건너뛰라» 가 없어 다음 관측·행동·보상·감가도 다시 쓴다.
            //   값이 같아 해롭지는 않지만 갱신마다 2.7 MB 를 헛되이 쓴다.
            replay->recordSampleWith(commandBuffer, actorTargets, replay->lastSamples());
            executor->recordClearGradients(commandBuffer);
            executor->recordForward(commandBuffer, merged.opBegin[2], merged.opCount[2]);
            executor->recordSeedLossGradient(commandBuffer, actorLoss);
            executor->recordBackward(commandBuffer, merged.opBegin[2], merged.opCount[2]);
            executor->recordAdam(commandBuffer,
                                 1,
                                 static_cast<uint32_t>(agent.parameters.actorUpdate.begin),
                                 static_cast<uint32_t>(agent.parameters.actorUpdate.count()),
                                 settings.actor,
                                 updateCount + k + 1);
            executor->recordPolyak(commandBuffer,
                                   static_cast<uint32_t>(agent.parameters.online.begin),
                                   static_cast<uint32_t>(agent.parameters.target.begin),
                                   static_cast<uint32_t>(agent.parameters.online.count()),
                                   settings.tau);
        }
        // **행동만 되읽는다.** 전체 되읽기는 가중치 4.3M 개를 매 걸음 가져와 그 복사가 걸음의 값을
        // 통째로 먹는다.
        executor->recordDownloadTensor(commandBuffer, actAction);
    });
    executor->invalidateReadback();

    // 행동을 읽어 잡음을 섞는다. **링에는 실제로 한 행동을 담는다.**
    const float* result = executor->activationResult() + merged.graph.tensors[actAction].offset;
    for (size_t i = 0; i < action.size(); ++i) {
        float value = result[i];
        if (training) {
            float progress = std::min(static_cast<float>(totalSteps) / EXPLORATION_NOISE_STEPS, 1.0F);
            explorationNoise = EXPLORATION_NOISE_BEGIN + (EXPLORATION_NOISE_END - EXPLORATION_NOISE_BEGIN) * progress;
            value += explorationNoise * gfx::neuralGaussian(totalSteps * 31 + 7, i);
        }
        action[i] = std::clamp(value, -1.0F, 1.0F);
    }
    physics::act(scene, robot, action.data());

    // 담지 못했으면 결과를 고쳐 넣을 자리도 없다.
    if (stored != gfx::ReplayBuffer::NO_SLOT) {
        replay->patchAction(stored, action.data());
        pendingSlot = stored;
        hasPending = true;
    }
    ++stepInEpisode;
    ++totalSteps;
    updateCount += updates;

    auto end = std::chrono::steady_clock::now();
    lastStepMilliseconds = std::chrono::duration<float, std::milli>(end - begin).count();
    // 프레임 예산 가드. 길면 내리고 넉넉하면 다시 올린다.
    //
    // **편집기가 있을 때만 건다.** 헤드리스 학습에는 부드럽게 유지할 프레임이 없고, 걸음 하나의 바닥
    // 비용(관측 렌더 + 순전파)이 이미 예산을 넘으면 갱신이 0 으로 내려가 영영 안 올라온다 — 학습이
    // 도는 것처럼 보이면서 아무 것도 배우지 않는다.
    if (training && services.editor != nullptr) {
        if (lastStepMilliseconds > FRAME_BUDGET_MILLISECONDS && updatesPerFrame > 0) {
            --updatesPerFrame;
        } else if (lastStepMilliseconds < FRAME_BUDGET_MILLISECONDS * 0.5F && updatesPerFrame < MAX_UPDATES_PER_FRAME) {
            ++updatesPerFrame;
        }
    }
    if (stepInEpisode >= EPISODE_FRAMES) {
        resetEpisode(services);
    }
    return true;
}

void NeuralPlugin::update(Services& services, float deltaSeconds) {
    (void)deltaSeconds;
    if (!enabled || !services.scenes.active().simulating) {
        return;
    }
    if (!ensure(services)) {
        return;
    }
    step(services);
}

void NeuralPlugin::ui(Services& services) {
    // 인자를 안 준 실행에는 절 자체를 열지 않는다. 열어 두면 체크박스가 아무 일도 하지 않는다.
    if (!enabled || services.editor == nullptr || !services.editor->settingsSection("픽셀 학습")) {
        return;
    }
    if (disabled) {
        ImGui::TextDisabled("이 장면에서는 돌지 않습니다 (관측 카메라와 모터 관절이 필요합니다)");
        return;
    }
    ImGui::Checkbox("학습", &training);
    ImGui::Text("에피소드 %u, 걸음 %u, 총 %llu", episode, stepInEpisode, static_cast<unsigned long long>(totalSteps));
    ImGui::Text(
        // **로그와 같은 눈금이다** — 보상 합을 프레임 수로 나눈다. 합을 그대로 찍으면 진화 전략의
        // 0.768 과 300 배 어긋난 숫자가 보인다.
        "점수: 지난 %.3f, 최고 %.3f",
        static_cast<double>(lastEpisodeReward / static_cast<float>(EPISODE_FRAMES)),
        static_cast<double>(bestEpisodeReward / static_cast<float>(EPISODE_FRAMES)));
    if (replay != nullptr) {
        ImGui::Text("리플레이 %u / %u 전이", replay->count(), replay->capacity());
    }
    ImGui::Text("스텝 %.2f ms, 프레임당 갱신 %u", static_cast<double>(lastStepMilliseconds), updatesPerFrame);
    if (observation != nullptr && observation->viewCount() > 0) {
        // ponytail: 「정책이 보는 그림」을 여기 이미지로 띄우려면 bindless 슬롯을 ImGui 의 디스크립터로
        // 한 번 더 묶어야 한다(ImGui_ImplVulkan_AddTexture). 지금은 --observation-dump 가 그 자리를 맡는다.
        ImGui::TextDisabled(
            "관측 %u x %u, 뷰 %u", gfx::OBSERVATION_SIZE, gfx::OBSERVATION_SIZE, observation->viewCount());
    }
}

} // namespace app
