#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "physics/policy.h"
#include "physics/rigid_body.h"
#include "scene/scene.h"

namespace core {
class JobSystem;
} // namespace core

namespace physics {

// 액추에이터 하나가 내는 관측 수(sin θ, cos θ, 정규화한 각속도).
inline constexpr uint32_t OBSERVATIONS_PER_ACTUATOR = 3;
// 각속도를 관측에 담을 때 나누는 값(라디안/초). tanh 망은 입력이 1 근처여야 잘 돈다.
inline constexpr float OBSERVATION_SPEED_SCALE = 10.0F;

// 정책이 모는 관절 하나. 장면에서 **모터가 달린 경첩**을 오브젝트 번호 순으로 모은 것이라, 장면
// 파일이 «어디에 모터를 달지» 를 정하면 관측과 행동의 자리는 여기서 따라온다.
struct Actuator {
    // 관절 부품이 붙은 오브젝트.
    uint32_t object = 0;
    // scene.joints 첨자.
    uint32_t joint = 0;
    // 상대 오브젝트. 없으면(-1) 세계에 붙은 고정점이라 기준 자세가 항등이다.
    int32_t other = -1;
    // 정규화한 A 지역 경첩 축.
    glm::vec3 axis{0.0F, 0.0F, 1.0F};
    // 장면에 적힌 값을 여기 떠 둔다. act 가 모터 목표를 덮어쓰므로 장면만 보면 첫 스텝 뒤에 잃는다.
    scene::JointMotor motor = scene::JointMotor::VELOCITY;
    // 목표 각속도의 상한(라디안/초). VELOCITY 모터가 쓴다.
    float speedLimit = 0.0F;
    // 목표 각도가 오갈 범위(라디안). POSITION 모터가 쓴다. 한계각이 없으면 ±pi 다.
    float angleCenter = 0.0F;
    float angleSpan = 0.0F;
    // 보상이 목표로 삼는 자세(라디안). 장면의 targetAngle 을 기동 시에 떠 둔 것이다 — POSITION
    // 모터에서는 act 가 그 값을 덮어쓴다.
    float rewardAngle = 0.0F;
};

// 정책이 보는 장면의 모습.
struct RobotLayout {
    std::vector<Actuator> actuators;

    uint32_t observationCount() const { return static_cast<uint32_t>(actuators.size()) * OBSERVATIONS_PER_ACTUATOR; }
    uint32_t actionCount() const { return static_cast<uint32_t>(actuators.size()); }
    bool empty() const { return actuators.empty(); }
};

// 모터가 달린 경첩 관절을 오브젝트 번호 순으로 모은다. 순서가 번호 순이라 같은 장면이면 언제나 같은
// 자리를 쓴다 — 학습한 정책을 다시 읽어 쓸 수 있는 근거다. 솔버가 버리는 관절(자기 자신을 가리키는
// 것, 강체가 없는 오브젝트, 토크나 목표 각속도가 0 인 모터)은 여기서도 버린다.
//
// **정책이 몰기 전에** 불러야 한다. act 가 모터 목표를 덮어쓰므로, 몰던 장면에서 다시 만들면 저작한
// 상한 대신 «마지막 행동» 을 상한으로 읽는다(RobotPlugin 이 저작 값을 되돌린 뒤 다시 만든다).
//
// ponytail: 관측은 액추에이터의 각과 각속도뿐이다. 뿌리 몸통의 높이·기울기가 필요한 태스크(걷기 같은
// 것)는 관측을 늘려야 한다.
RobotLayout buildRobotLayout(const scene::Scene& scene);

// 레이아웃이 떠 둔 저작 값을 장면의 관절에 되돌린다. act 가 덮어쓴 모터 목표를 지우는 것이라, 자리를
// 다시 만들기 전이나 정책을 끌 때 부른다. 상한의 부호는 뜻이 없으므로 양수로 돌아간다.
void restoreAuthoredMotors(scene::Scene& scene, const RobotLayout& layout);

// 장면에서 관측을 읽는다. out 은 observationCount() 개다.
void observe(const scene::Scene& scene, const RobotLayout& layout, float* out);

// 행동을 관절의 모터 목표로 쓴다. action 은 actionCount() 개이고 정책이 tanh 로 낸 [-1, 1] 이다.
void act(scene::Scene& scene, const RobotLayout& layout, const float* action);

// 이번 스텝의 보상. 액추에이터마다 목표 자세에 얼마나 가까운지(cos 오차)를 더하고 각속도를 조금 뺀다.
// 최대는 액추에이터 수와 같고, 목표에서 정확히 반대면 그만큼 음수다.
float stepReward(const scene::Scene& scene, const RobotLayout& layout);

struct RolloutSettings {
    // 몇 «프레임» 을 밟을지. 프레임마다 정책을 한 번 부르고 물리를 stepsPerFrame 번 진행한다 —
    // Application::run 의 재생 루프(1/60 프레임 안에서 1/120 스텝 두 번)와 같은 구조다. 그래야 학습한
    // 정책이 재생에서도 같은 간격으로 몬다.
    //
    // ponytail: 재생의 프레임 간격이 1/60 이 아니면 스텝 수가 달라져 정책이 학습할 때와 다른 간격으로
    // 몰게 된다. 헤드리스(기본 간격이 정확히 1/60)만 언제나 프레임당 두 스텝이고, 창을 띄우면 화면
    // 주사율과 `--fixed-dt` 의 반올림에 따라 이따금 한 스텝인 프레임이 섞인다.
    uint32_t frames = 300;
    uint32_t stepsPerFrame = 2;
    float stepSeconds = STEP_SECONDS;
};

// 장면 사본 하나를 정책으로 몰아 보상 합을 낸다. 장면을 **값으로** 받아 부르는 쪽의 것을 건드리지
// 않으므로, 표본마다 하나씩 워커에 그냥 나눌 수 있다. 잠금도 공유 상태도 없다.
//
// ponytail: CPU 강체 솔버만 부른다. 액추에이터가 GPU 백엔드면 장면이 한 걸음도 움직이지 않는 채로
// 롤아웃이 돌아 점수가 전부 같아진다(RobotPlugin 이 학습 전에 경고한다).
// ponytail: 애니메이션(scene.update)을 밟지 않는다. 액추에이터가 애니메이터에 물려 있으면 학습과
// 재생의 궤적이 갈린다.
// ponytail: 표본마다 장면을 통째로 깊은 복사한다. 리깅된 캐릭터가 있는 장면은 애니메이터가 스켈레톤을
// 값으로 담고 있어 그 비용이 크다.
float rollout(scene::Scene scene, const Policy& policy, const RobotLayout& layout, const RolloutSettings& settings);

struct TrainingSettings {
    uint32_t generations = 200;
    uint32_t population = 32;
    uint32_t hidden = 16;
    float sigma = 0.05F;
    float learningRate = 0.03F;
    uint64_t seed = 1;
    RolloutSettings rollout;
};

struct TrainingProgress {
    uint32_t generation = 0;
    // 이번 세대 표본의 가장 좋은 점수와 평균.
    float best = 0.0F;
    float mean = 0.0F;
    // 평균 가중치를 그대로 돌린 점수. 학습이 실제로 나아지는지는 표본의 최고·평균이 아니라 이 값으로
    // 본다. **이 세대의 걸음을 밟기 전** 의 가중치이고(표본과 같은 병렬 구간에서 재려면 그래야 한다),
    // 그래서 마지막 보고는 돌려주는 정책보다 한 걸음 앞선 값이다.
    float center = 0.0F;
};

// 진화 전략으로 정책을 학습한다. 세대마다 표본을 워커에 나눠 롤아웃하고, report 가 있으면 세대마다
// 부른다. 롤아웃이 결정적이고 잡음도 씨앗에서 다시 만드므로, 같은 설정이면 워커 수와 무관하게 같은
// 정책이 나온다.
Policy trainRobot(const scene::Scene& scene,
                  const RobotLayout& layout,
                  const TrainingSettings& settings,
                  core::JobSystem* jobs,
                  const std::function<void(const TrainingProgress&)>& report = {});

// 정책을 JSON 으로 저장하고 읽는다. 모양(inputs/hidden/outputs)이 맞지 않으면 거짓을 돌려주고
// policy 를 건드리지 않는다.
bool savePolicy(const Policy& policy, const std::string& path);
bool loadPolicy(Policy& policy, const std::string& path);

} // namespace physics
