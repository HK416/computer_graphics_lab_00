// 로봇 관측·행동과 진화 전략 학습. Vulkan 없이 CPU 강체 솔버만 탄다.
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>

#include "core/job_system.h"
#include "pendulum_scene.h"
#include "physics/robot.h"
#include "scene/scene.h"

namespace {

bool nearly(float a, float b, float tolerance = 1.0e-4F) {
    return std::abs(a - b) <= tolerance;
}

void testLayout() {
    scene::Scene scene = makePendulum();
    // 모터가 없는 관절과 관절이 없는 오브젝트는 액추에이터가 아니다.
    addBar(scene, glm::vec3{3.0F, 2.0F, 0.0F}, 0.5F, scene::JointMotor::NONE);
    scene::Object plain;
    plain.name = "장식";
    scene.objects.push_back(plain);
    uint32_t second = addBar(scene, glm::vec3{6.0F, 2.0F, 0.0F}, 0.5F, scene::JointMotor::POSITION);
    scene.refresh();

    physics::RobotLayout layout = physics::buildRobotLayout(scene);
    assert(layout.actuators.size() == 2);
    assert(layout.actuators[0].object == 0);
    assert(layout.actuators[1].object == second);
    assert(layout.observationCount() == 2 * physics::OBSERVATIONS_PER_ACTUATOR);
    assert(layout.actionCount() == 2);
    // 장면에 적힌 상한과 목표를 라디안으로 떠 둔다.
    assert(nearly(layout.actuators[0].speedLimit, glm::radians(360.0F)));
    assert(nearly(layout.actuators[0].rewardAngle, glm::pi<float>()));
    assert(layout.actuators[0].motor == scene::JointMotor::VELOCITY);
    assert(layout.actuators[1].motor == scene::JointMotor::POSITION);

    // 액추에이터가 없는 장면은 빈 레이아웃이다.
    scene::Scene bare;
    assert(physics::buildRobotLayout(bare).empty());
}

// 솔버가 버리는 관절은 액추에이터가 아니어야 한다. 남겨 두면 아무 데도 가지 않는 관측·행동 차원이 된다.
void testLayoutGates() {
    // 자기 자신을 가리키는 관절.
    scene::Scene self = makePendulum();
    self.joints[0].other = 0;
    assert(physics::buildRobotLayout(self).empty());

    // 강체가 없는 오브젝트의 관절.
    scene::Scene bodiless = makePendulum();
    bodiless.detachComponent(0, &scene::Object::rigidBody);
    assert(physics::buildRobotLayout(bodiless).empty());

    // 토크가 0 인 모터는 솔버가 아예 돌리지 않는다.
    scene::Scene torqueless = makePendulum();
    torqueless.joints[0].maxTorque = 0.0F;
    assert(physics::buildRobotLayout(torqueless).empty());

    // 목표 각속도가 0 이면 어떤 행동도 0 을 명령한다.
    scene::Scene still = makePendulum();
    still.joints[0].targetSpeed = 0.0F;
    assert(physics::buildRobotLayout(still).empty());

    // 상대 오브젝트가 있어도 강체가 아니면 솔버는 항등 자세의 고정점으로 본다. 관측도 같게 봐야 한다.
    scene::Scene anchored = makePendulum();
    scene::Object anchor;
    anchor.name = "앵커";
    float half = glm::radians(90.0F) * 0.5F;
    anchor.transform.rotation = glm::quat{std::cos(half), 0.0F, 0.0F, std::sin(half)};
    anchored.objects.push_back(anchor);
    anchored.joints[0].other = static_cast<int32_t>(anchored.objects.size() - 1);
    anchored.refresh();
    physics::RobotLayout anchoredLayout = physics::buildRobotLayout(anchored);
    assert(anchoredLayout.actuators.size() == 1);
    assert(anchoredLayout.actuators[0].other == -1);
    std::vector<float> observation(anchoredLayout.observationCount(), 0.0F);
    physics::observe(anchored, anchoredLayout, observation.data());
    // 앵커의 90 도 회전이 각에 섞이면 sin 이 -1 이 된다. 솔버가 보는 각은 0 이다.
    assert(nearly(observation[0], 0.0F));
    assert(nearly(observation[1], 1.0F));
}

// 정책이 모터 목표를 덮어쓴 뒤에도 저작 값을 되돌리면 자리를 그대로 다시 만들 수 있어야 한다. 그러지
// 않으면 상한이 «마지막 행동» 으로 줄어드는 래칫이 걸린다.
void testRestoreAuthored() {
    scene::Scene scene = makePendulum();
    physics::RobotLayout layout = physics::buildRobotLayout(scene);
    float authoredSpeed = layout.actuators[0].speedLimit;
    float authoredAngle = layout.actuators[0].rewardAngle;

    std::vector<float> action{0.3F};
    for (int round = 0; round < 3; ++round) {
        physics::act(scene, layout, action.data());
        // 되돌리지 않고 다시 만들면 상한이 방금 명령한 값으로 줄어든다.
        physics::RobotLayout ratcheted = physics::buildRobotLayout(scene);
        assert(!ratcheted.empty());
        assert(ratcheted.actuators[0].speedLimit < authoredSpeed);
        // 되돌린 뒤에 만들면 그대로다.
        physics::restoreAuthoredMotors(scene, layout);
        physics::RobotLayout restored = physics::buildRobotLayout(scene);
        assert(restored.actuators.size() == 1);
        assert(nearly(restored.actuators[0].speedLimit, authoredSpeed));
        assert(nearly(restored.actuators[0].rewardAngle, authoredAngle));
        layout = restored;
    }
}

void testObserve() {
    scene::Scene scene = makePendulum();
    physics::RobotLayout layout = physics::buildRobotLayout(scene);
    std::vector<float> observation(layout.observationCount(), 9.0F);

    // 정지 자세는 각 0 이다.
    physics::observe(scene, layout, observation.data());
    assert(nearly(observation[0], 0.0F));
    assert(nearly(observation[1], 1.0F));
    assert(nearly(observation[2], 0.0F));

    // 축 둘레로 90 도 돌리면 sin·cos 이 따라 돈다. 각의 부호는 축의 오른손 방향이 양이다.
    float half = glm::radians(90.0F) * 0.5F;
    scene.objects[0].transform.rotation = glm::quat{std::cos(half), 0.0F, 0.0F, std::sin(half)};
    scene.refresh();
    physics::observe(scene, layout, observation.data());
    assert(nearly(observation[0], 1.0F));
    assert(nearly(observation[1], 0.0F));

    // 각속도는 상한으로 잘려 들어간다. 축 방향 성분만 본다.
    scene.rigidBodies[0].angularVelocity = glm::vec3{0.0F, 0.0F, physics::OBSERVATION_SPEED_SCALE * 0.5F};
    physics::observe(scene, layout, observation.data());
    assert(nearly(observation[2], 0.5F));
    scene.rigidBodies[0].angularVelocity = glm::vec3{0.0F, 0.0F, physics::OBSERVATION_SPEED_SCALE * 100.0F};
    physics::observe(scene, layout, observation.data());
    assert(nearly(observation[2], 1.0F));
    // 축에 수직한 회전은 이 관절의 관측이 아니다.
    scene.rigidBodies[0].angularVelocity = glm::vec3{5.0F, 0.0F, 0.0F};
    physics::observe(scene, layout, observation.data());
    assert(nearly(observation[2], 0.0F));
}

void testAct() {
    scene::Scene scene = makePendulum();
    physics::RobotLayout layout = physics::buildRobotLayout(scene);

    // VELOCITY 모터는 장면에 적힌 상한 안에서 목표 각속도를 받는다.
    std::vector<float> action{1.0F};
    physics::act(scene, layout, action.data());
    assert(nearly(scene.joints[0].targetSpeed, 360.0F, 1.0e-2F));
    action[0] = -0.5F;
    physics::act(scene, layout, action.data());
    assert(nearly(scene.joints[0].targetSpeed, -180.0F, 1.0e-2F));
    // 범위를 넘겨 줘도 잘린다.
    action[0] = 7.0F;
    physics::act(scene, layout, action.data());
    assert(nearly(scene.joints[0].targetSpeed, 360.0F, 1.0e-2F));

    // POSITION 모터는 한계각 안으로 목표 각도를 받는다.
    scene::Scene servo = makePendulum(scene::JointMotor::POSITION);
    servo.joints[0].useLimit = true;
    servo.joints[0].lowerAngle = -30.0F;
    servo.joints[0].upperAngle = 90.0F;
    physics::RobotLayout servoLayout = physics::buildRobotLayout(servo);
    action[0] = 1.0F;
    physics::act(servo, servoLayout, action.data());
    assert(nearly(servo.joints[0].targetAngle, 90.0F, 1.0e-2F));
    action[0] = -1.0F;
    physics::act(servo, servoLayout, action.data());
    assert(nearly(servo.joints[0].targetAngle, -30.0F, 1.0e-2F));
    action[0] = 0.0F;
    physics::act(servo, servoLayout, action.data());
    assert(nearly(servo.joints[0].targetAngle, 30.0F, 1.0e-2F));
}

void testReward() {
    scene::Scene scene = makePendulum();
    physics::RobotLayout layout = physics::buildRobotLayout(scene);
    // 목표는 180 도다. 아래로 누운 자세(0 도)는 가장 나쁜 -1 이다.
    assert(nearly(physics::stepReward(scene, layout), -1.0F));

    float half = glm::radians(180.0F) * 0.5F;
    scene.objects[0].transform.rotation = glm::quat{std::cos(half), 0.0F, 0.0F, std::sin(half)};
    scene.refresh();
    assert(nearly(physics::stepReward(scene, layout), 1.0F));

    // 목표에 있어도 흔들리면 조금 깎인다.
    scene.rigidBodies[0].angularVelocity = glm::vec3{0.0F, 0.0F, 4.0F};
    assert(physics::stepReward(scene, layout) < 1.0F);
}

void testRollout() {
    scene::Scene scene = makePendulum();
    physics::RobotLayout layout = physics::buildRobotLayout(scene);
    physics::Policy policy(layout.observationCount(), 8, layout.actionCount());
    physics::RolloutSettings settings;
    settings.frames = 60;

    glm::vec3 before = scene.objects[0].transform.position;
    float first = physics::rollout(scene, policy, layout, settings);
    float again = physics::rollout(scene, policy, layout, settings);
    // 같은 시작 상태와 같은 정책이면 같은 점수다.
    assert(nearly(first, again));
    // 장면을 값으로 받으므로 부르는 쪽의 것은 그대로다. 그래서 표본을 워커에 그냥 나눌 수 있다.
    assert(scene.objects[0].transform.position == before);
    assert(!scene.simulating);

    // 아무 것도 하지 않는 정책은 막대를 아래로 떨어뜨린다. 목표가 180 도이므로 점수가 음수다.
    assert(first < 0.0F);

    // 모양이 맞지 않는 정책은 0 이다.
    physics::Policy wrong(layout.observationCount() + 1, 8, layout.actionCount());
    assert(nearly(physics::rollout(scene, wrong, layout, settings), 0.0F));
}

// 학습이 실제로 나아지고, 워커 수와 무관하게 같은 정책이 나오는지. 표본끼리 아무 것도 공유하지 않는
// 설계가 이 성질에 기댄다.
void testTraining() {
    scene::Scene scene = makePendulum();
    physics::RobotLayout layout = physics::buildRobotLayout(scene);
    physics::TrainingSettings settings;
    settings.generations = 12;
    settings.population = 16;
    settings.hidden = 8;
    settings.sigma = 0.2F;
    settings.learningRate = 0.3F;
    settings.rollout.frames = 90;

    physics::Policy untrained(layout.observationCount(), settings.hidden, layout.actionCount());
    float baseline = physics::rollout(scene, untrained, layout, settings.rollout);

    physics::TrainingProgress last;
    physics::Policy trained = physics::trainRobot(
        scene, layout, settings, nullptr, [&](const physics::TrainingProgress& progress) { last = progress; });
    assert(last.generation == settings.generations);
    float score = physics::rollout(scene, trained, layout, settings.rollout);
    // 아무 것도 하지 않는 정책보다는 나아야 한다.
    assert(score > baseline);

    // 보고의 center 는 «그 세대의 걸음을 밟기 전» 의 평균 가중치 점수다. 한 세대 덜 돌린 정책의 점수와
    // 같아야 한다.
    physics::TrainingSettings shorter = settings;
    shorter.generations = settings.generations - 1;
    physics::Policy previous = physics::trainRobot(scene, layout, shorter, nullptr);
    assert(nearly(physics::rollout(scene, previous, layout, settings.rollout), last.center));

    // 워커에 나눠도 같은 정책이 나온다.
    core::JobSystem jobs(4);
    physics::Policy parallel = physics::trainRobot(scene, layout, settings, &jobs);
    assert(trained.parameters().size() == parallel.parameters().size());
    for (size_t i = 0; i < trained.parameters().size(); ++i) {
        assert(nearly(trained.parameters()[i], parallel.parameters()[i], 1.0e-6F));
    }

    // 액추에이터가 없으면 학습할 것도 없다.
    scene::Scene bare;
    physics::RobotLayout empty;
    physics::Policy none = physics::trainRobot(bare, empty, settings, nullptr);
    assert(none.outputCount() == 0);
}

void testPolicyFile() {
    scene::Scene scene = makePendulum();
    physics::RobotLayout layout = physics::buildRobotLayout(scene);
    physics::Policy policy(layout.observationCount(), 8, layout.actionCount());
    for (size_t i = 0; i < policy.parameters().size(); ++i) {
        policy.parameters()[i] = 0.25F * static_cast<float>(i % 7) - 0.5F;
    }
    const std::string path = "robot_policy_test.json";
    assert(physics::savePolicy(policy, path));

    physics::Policy loaded(layout.observationCount(), 8, layout.actionCount());
    assert(physics::loadPolicy(loaded, path));
    for (size_t i = 0; i < policy.parameters().size(); ++i) {
        assert(nearly(loaded.parameters()[i], policy.parameters()[i]));
    }

    // 모양이 다르면 거절하고 가중치를 건드리지 않는다.
    physics::Policy other(layout.observationCount(), 16, layout.actionCount());
    std::vector<float> untouched = other.parameters();
    assert(!physics::loadPolicy(other, path));
    assert(other.parameters() == untouched);
    // 없는 파일도 거짓이다.
    assert(!physics::loadPolicy(loaded, "없는파일.json"));
}

} // namespace

int main() {
    testLayout();
    testLayoutGates();
    testRestoreAuthored();
    testObserve();
    testAct();
    testReward();
    testRollout();
    testTraining();
    testPolicyFile();
    std::printf("로봇 관측·행동과 학습 테스트 통과\n");
    return 0;
}
