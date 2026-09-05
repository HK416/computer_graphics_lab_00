#pragma once

#include <cstdint>
#include <memory>

#include "app/plugin.h"
#include "gfx/rigid_body_gpu.h"

namespace app {

// 강체 물리. 재생 중 고정 간격(1/120 초)으로 CPU 솔버를 돌리고, GPU 백엔드 강체는 자기가 소유한
// gfx::RigidBodySimulator 를 렌더 그래프의 «스킨» 뒤에 끼워 같은 스텝 수만큼 푼다. 프레임이 길어도 정해진
// 스텝 수까지만 따라잡아 나선형으로 느려지지 않는다.
class PhysicsPlugin : public Plugin {
public:
    static constexpr float STEP_SECONDS = 1.0F / 120.0F;
    static constexpr uint32_t MAX_STEPS_PER_FRAME = 8;

    const char* name() const override { return "물리"; }
    void build(Services& services) override;
    void update(Services& services, float deltaSeconds) override;
    void ui(Services& services) override;

private:
    std::unique_ptr<gfx::RigidBodySimulator> rigid;
    // 고정 간격 누적기와 이번 프레임에 푼 스텝 수. 렌더 그래프 노드가 GPU 솔버에 같은 수를 넘긴다.
    float accumulator = 0.0F;
    uint32_t steps = 0;
    // 재생을 켜고 끈 변화를 다음 프레임 머리에서 알아채 GPU 상태를 버린다.
    bool wasSimulating = false;
};

} // namespace app
