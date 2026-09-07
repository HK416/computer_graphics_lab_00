#include "physics/robot.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <numeric>

#include <glm/gtc/quaternion.hpp>
#include <glm/trigonometric.hpp>
#include <nlohmann/json.hpp>

#include "core/job_system.h"
#include "physics/rigid_body.h"

namespace physics {

using nlohmann::json;

namespace {

// 각속도가 관측을 포화시키지 않게 나눈 뒤 자른다. 넘어가는 값은 어차피 «아주 빠름» 하나로 보면 된다.
float scaledSpeed(float radiansPerSecond) {
    return std::clamp(radiansPerSecond / OBSERVATION_SPEED_SCALE, -1.0F, 1.0F);
}

// 오브젝트의 세계 회전. 배율이 섞인 세계 행렬에서 회전만 걷어낸다.
glm::quat worldRotation(const scene::Scene& scene, uint32_t object) {
    glm::mat3 basis{scene.worldMatrix(object)};
    for (int column = 0; column < 3; ++column) {
        float length = glm::length(basis[column]);
        if (length < 1.0e-6F) {
            return glm::quat{1.0F, 0.0F, 0.0F, 0.0F};
        }
        basis[column] /= length;
    }
    return glm::normalize(glm::quat_cast(basis));
}

glm::vec3 angularVelocityOf(const scene::Scene& scene, int32_t object) {
    if (object < 0 || static_cast<size_t>(object) >= scene.objects.size()) {
        return glm::vec3{0.0F};
    }
    const scene::RigidBody* body = scene.component<scene::RigidBody>(static_cast<uint32_t>(object));
    return body != nullptr ? body->angularVelocity : glm::vec3{0.0F};
}

// 액추에이터의 지금 각(라디안)과 축 둘레 각속도(라디안/초).
void readActuator(const scene::Scene& scene, const Actuator& actuator, float& angle, float& speed) {
    if (actuator.object >= scene.objects.size()) {
        angle = 0.0F;
        speed = 0.0F;
        return;
    }
    glm::quat rotationA = worldRotation(scene, actuator.object);
    glm::quat rotationB = actuator.other >= 0 && static_cast<size_t>(actuator.other) < scene.objects.size()
                              ? worldRotation(scene, static_cast<uint32_t>(actuator.other))
                              : glm::quat{1.0F, 0.0F, 0.0F, 0.0F};
    angle = hingeAngle(rotationA, rotationB, actuator.axis);
    // 축은 A 지역이므로 세계로 돌려 각속도 차를 재야 솔버가 보는 것과 같다.
    glm::vec3 worldAxis = rotationA * actuator.axis;
    glm::vec3 relative =
        angularVelocityOf(scene, static_cast<int32_t>(actuator.object)) - angularVelocityOf(scene, actuator.other);
    speed = glm::dot(relative, worldAxis);
}

} // namespace

RobotLayout buildRobotLayout(const scene::Scene& scene) {
    RobotLayout layout;
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        int32_t slot = scene.objects[index].joint;
        if (slot < 0 || static_cast<size_t>(slot) >= scene.joints.size()) {
            continue;
        }
        const scene::Joint& joint = scene.joints[static_cast<size_t>(slot)];
        if (joint.type != scene::JointType::HINGE || joint.motor == scene::JointMotor::NONE) {
            continue;
        }
        float axisLength = glm::length(joint.axis);
        if (axisLength < 1.0e-6F) {
            continue;
        }
        // 아래는 솔버(collectJoints·solveJoint)가 버리는 관절을 여기서도 버리는 것이다. 남겨 두면 정책에
        // 아무 데도 가지 않는 관측·행동 차원이 생겨 학습만 느려진다.
        if (joint.other == static_cast<int32_t>(index) || scene.component<scene::RigidBody>(index) == nullptr) {
            continue;
        }
        if (joint.maxTorque <= 0.0F) {
            continue;
        }
        if (joint.motor == scene::JointMotor::VELOCITY && std::abs(joint.targetSpeed) < 1.0e-6F) {
            continue;
        }
        Actuator actuator;
        actuator.object = index;
        actuator.joint = static_cast<uint32_t>(slot);
        // 상대 오브젝트가 있어도 강체가 아니면 솔버는 «항등 자세의 고정점» 으로 본다(collectJoints 의
        // bodyOf 가 -1 이다). 관측이 그 오브젝트의 실제 회전을 보면 각이 통째로 어긋난다.
        bool otherIsBody = joint.other >= 0 && static_cast<size_t>(joint.other) < scene.objects.size() &&
                           scene.component<scene::RigidBody>(static_cast<uint32_t>(joint.other)) != nullptr;
        actuator.other = otherIsBody ? joint.other : -1;
        actuator.axis = joint.axis / axisLength;
        actuator.motor = joint.motor;
        actuator.speedLimit = glm::radians(std::abs(joint.targetSpeed));
        float lower = glm::radians(std::min(joint.lowerAngle, joint.upperAngle));
        float upper = glm::radians(std::max(joint.lowerAngle, joint.upperAngle));
        if (!joint.useLimit) {
            lower = -3.14159265F;
            upper = 3.14159265F;
        }
        actuator.angleCenter = 0.5F * (lower + upper);
        actuator.angleSpan = 0.5F * (upper - lower);
        actuator.rewardAngle = glm::radians(joint.targetAngle);
        layout.actuators.push_back(actuator);
    }
    return layout;
}

void restoreAuthoredMotors(scene::Scene& scene, const RobotLayout& layout) {
    for (const Actuator& actuator : layout.actuators) {
        if (actuator.joint >= scene.joints.size()) {
            continue;
        }
        scene::Joint& joint = scene.joints[actuator.joint];
        joint.targetSpeed = glm::degrees(actuator.speedLimit);
        joint.targetAngle = glm::degrees(actuator.rewardAngle);
    }
}

void observe(const scene::Scene& scene, const RobotLayout& layout, float* out) {
    for (size_t i = 0; i < layout.actuators.size(); ++i) {
        float angle = 0.0F;
        float speed = 0.0F;
        readActuator(scene, layout.actuators[i], angle, speed);
        // 각을 그대로 주면 ±pi 경계에서 값이 튄다. sin·cos 짝으로 주면 이어진다.
        float* slot = out + i * OBSERVATIONS_PER_ACTUATOR;
        slot[0] = std::sin(angle);
        slot[1] = std::cos(angle);
        slot[2] = scaledSpeed(speed);
    }
}

void act(scene::Scene& scene, const RobotLayout& layout, const float* action) {
    for (size_t i = 0; i < layout.actuators.size(); ++i) {
        const Actuator& actuator = layout.actuators[i];
        if (actuator.joint >= scene.joints.size()) {
            continue;
        }
        scene::Joint& joint = scene.joints[actuator.joint];
        float value = std::clamp(action[i], -1.0F, 1.0F);
        if (actuator.motor == scene::JointMotor::POSITION) {
            joint.targetAngle = glm::degrees(actuator.angleCenter + value * actuator.angleSpan);
        } else {
            joint.targetSpeed = glm::degrees(value * actuator.speedLimit);
        }
    }
}

float stepReward(const scene::Scene& scene, const RobotLayout& layout) {
    float reward = 0.0F;
    for (const Actuator& actuator : layout.actuators) {
        float angle = 0.0F;
        float speed = 0.0F;
        readActuator(scene, actuator, angle, speed);
        // 각 오차의 cos 이라 목표에서 pi 만큼 떨어지면 -1 이고, 각이 접히는 경계에서도 이어진다.
        reward += std::cos(angle - actuator.rewardAngle);
        // 목표에 서 있기만 하고 떨리는 해를 막는다. 계수가 크면 아예 움직이지 않는 쪽이 이긴다.
        reward -= 0.005F * speed * speed;
    }
    return reward;
}

float rollout(scene::Scene scene, const Policy& policy, const RobotLayout& layout, const RolloutSettings& settings) {
    if (layout.empty() || policy.inputCount() != layout.observationCount() ||
        policy.outputCount() != layout.actionCount()) {
        return 0.0F;
    }
    // 재생 중이어야 부품이 스냅샷으로 되돌아가지 않는다. 사본이라 부르는 쪽 장면은 그대로다.
    scene.simulating = true;
    std::vector<float> observation(layout.observationCount(), 0.0F);
    std::vector<float> action(layout.actionCount(), 0.0F);
    uint32_t steps = std::max(settings.stepsPerFrame, 1U);
    float reward = 0.0F;
    for (uint32_t frame = 0; frame < settings.frames; ++frame) {
        observe(scene, layout, observation.data());
        policy.evaluate(observation.data(), action.data());
        act(scene, layout, action.data());
        for (uint32_t step = 0; step < steps; ++step) {
            // 워커 하나가 롤아웃 하나를 통째로 맡으므로 솔버 안에서 다시 나누지 않는다(jobs 는 null).
            stepRigidBodies(scene, settings.stepSeconds, nullptr);
        }
        reward += stepReward(scene, layout);
    }
    return reward / static_cast<float>(std::max(settings.frames, 1U));
}

Policy trainRobot(const scene::Scene& scene,
                  const RobotLayout& layout,
                  const TrainingSettings& settings,
                  core::JobSystem* jobs,
                  const std::function<void(const TrainingProgress&)>& report) {
    Policy policy(layout.observationCount(), settings.hidden, layout.actionCount());
    if (layout.empty()) {
        return policy;
    }
    EvolutionStrategy strategy(policy.parameters().size(), settings.population, settings.seed);
    strategy.sigma = settings.sigma;
    strategy.learningRate = settings.learningRate;

    uint32_t population = strategy.population();
    // 표본마다 정책 사본 하나. 미리 잡아 두고 세대마다 가중치만 갈아 끼운다.
    std::vector<Policy> candidates(population, policy);
    std::vector<float> scores(population, 0.0F);

    for (uint32_t generation = 0; generation < settings.generations; ++generation) {
        for (uint32_t i = 0; i < population; ++i) {
            strategy.sample(i, policy.parameters(), candidates[i].parameters());
        }
        // 보고할 것이 있으면 «지금 평균 가중치» 의 롤아웃도 같은 병렬 구간에 넣는다. 따로 돌리면
        // 세대마다 직렬 롤아웃이 한 번 더 붙어 워커가 많을수록 손해가 커진다.
        uint32_t total = report ? population + 1 : population;
        float center = 0.0F;
        auto evaluate = [&](uint32_t begin, uint32_t end) {
            for (uint32_t i = begin; i < end; ++i) {
                const Policy& sampled = i < population ? candidates[i] : policy;
                float score = rollout(scene, sampled, layout, settings.rollout);
                (i < population ? scores[i] : center) = score;
            }
        };
        if (jobs != nullptr) {
            // 표본 하나가 작업 하나다. 롤아웃끼리 아무 것도 공유하지 않으므로 잠금이 없다.
            jobs->parallelFor(total, 1, evaluate);
        } else {
            evaluate(0, total);
        }
        strategy.step(scores, policy.parameters());

        if (report) {
            TrainingProgress progress;
            progress.generation = generation + 1;
            progress.best = *std::max_element(scores.begin(), scores.end());
            progress.mean = std::accumulate(scores.begin(), scores.end(), 0.0F) / static_cast<float>(population);
            // 이 세대의 걸음을 밟기 «전» 의 평균 가중치 점수다. 표본과 같은 조건에서 잰 것이다.
            progress.center = center;
            report(progress);
        }
    }
    return policy;
}

bool savePolicy(const Policy& policy, const std::string& path) {
    json document;
    document["inputs"] = policy.inputCount();
    document["hidden"] = policy.hiddenCount();
    document["outputs"] = policy.outputCount();
    document["weights"] = policy.parameters();
    // 정책을 처음 저장할 때 폴더가 없을 수 있다. ofstream 은 만들어 주지 않는다.
    std::error_code error;
    std::filesystem::path parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, error);
    }
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    file << document.dump(1, ' ') << "\n";
    return static_cast<bool>(file);
}

bool loadPolicy(Policy& policy, const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    json document = json::parse(file, nullptr, false);
    // 구문이 깨진 것은 is_discarded 로 걸리지만, 구문은 맞고 «키의 타입이 다른» 것은 value() 가 예외를
    // 던진다(문서가 객체가 아닌 경우도 마찬가지다). 손으로 주는 경로라 그대로 두면 크래시가 된다.
    if (document.is_discarded() || !document.is_object()) {
        return false;
    }
    std::vector<float> weights;
    try {
        if (document.value("inputs", 0U) != policy.inputCount() ||
            document.value("hidden", 0U) != policy.hiddenCount() ||
            document.value("outputs", 0U) != policy.outputCount()) {
            return false;
        }
        weights = document.value("weights", std::vector<float>{});
    } catch (const json::exception&) {
        return false;
    }
    if (weights.size() != policy.parameters().size()) {
        return false;
    }
    policy.parameters() = weights;
    return true;
}

} // namespace physics
