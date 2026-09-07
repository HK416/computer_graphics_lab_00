#pragma once

#include <memory>

#include "app/plugin.h"
#include "physics/robot.h"

namespace app {

// 학습된 정책이 장면의 액추에이터(모터가 달린 경첩)를 모는 플러그인.
//
// 기동할 때 `--train` 이면 진화 전략으로 정책을 학습하고 `--policy` 자리에 저장한다. 학습은 프레임
// 루프가 아니라 여기서 한 번에 끝난다 — 롤아웃마다 장면을 처음 상태로 되돌려야 하는데 그것은 «지금
// 도는 장면» 과 맞지 않기 때문이다. 대신 시작 장면의 사본을 떠 두고 그 위에서 표본을 돌린다.
//
// 재생 중에는 update 가 관측을 읽어 정책을 부르고 행동을 관절 목표로 쓴다. 물리보다 **앞**에서
// 등록되어야 이번 스텝이 그 목표를 보고 풀린다(Application::registerPlugins 의 순서).
class RobotPlugin : public Plugin {
public:
    const char* name() const override { return "로봇"; }
    void build(Services& services) override;
    void update(Services& services, float deltaSeconds) override;
    void ui(Services& services) override;

private:
    // 저작 값을 되돌린 뒤 자리를 다시 만든다.
    void rebuildLayout(scene::Scene& scene);
    void applyPolicy(scene::Scene& scene);

    std::unique_ptr<physics::Policy> policy;
    physics::RobotLayout layout;
    // 정책이 이 번호의 장면에 맞춰 만들어졌다. 장면이 바뀌면 다시 만든다.
    uint64_t layoutSceneId = 0;
    uint64_t layoutComponents = 0;
    // 정책이 몰지 여부. 편집기에서 끄면 장면에 적힌 모터 목표가 그대로 쓰인다.
    bool driving = true;
    // 정책과 자리의 모양이 어긋나 몰지 못하는 상태. 로그를 한 번만 적는 데 쓴다.
    bool shapeMismatch = false;
    // 마지막 학습의 결과. 편집기에 보여 준다.
    float lastScore = 0.0F;
    uint32_t lastGenerations = 0;
    // 관측·행동 자리. 프레임마다 잡지 않도록 들고 있는다.
    std::vector<float> observation;
    std::vector<float> action;
};

} // namespace app
