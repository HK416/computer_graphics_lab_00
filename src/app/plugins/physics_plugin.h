#pragma once

#include <cstdint>
#include <memory>

#include "app/plugin.h"
#include "gfx/headless_compute.h"
#include "gfx/rigid_body_gpu.h"

namespace app {

// 강체 물리. 재생 중 고정 간격(1/120 초)으로 CPU 솔버를 돌리고, GPU 백엔드 강체는 자기가 소유한
// gfx::RigidBodySimulator 를 렌더 그래프의 «스킨» 뒤에 끼워 같은 스텝 수만큼 푼다. 프레임이 길어도 정해진
// 스텝 수까지만 따라잡아 나선형으로 느려지지 않는다.
//
// 헤드리스(--headless)에는 렌더 그래프가 없다. 장면에 GPU 백엔드 강체가 있으면 Application 이 창 없는
// 장치를 만들어 두므로, 여기서 gfx::HeadlessCompute 로 그 프레임의 스텝 전부를 명령 버퍼 하나에 담아
// 직접 제출하고 기다린다. 기다리는 만큼 되읽기가 늦지 않아 그 프레임 안에서 결과가 장면에 들어온다.
class PhysicsPlugin : public Plugin {
public:
    static constexpr float STEP_SECONDS = 1.0F / 120.0F;
    static constexpr uint32_t MAX_STEPS_PER_FRAME = 8;

    const char* name() const override { return "물리"; }
    void build(Services& services) override;
    void update(Services& services, float deltaSeconds) override;
    void ui(Services& services) override;

private:
    // 헤드리스에서 이번 프레임의 스텝을 제출하고 기다린 뒤 결과를 장면에 되쓴다.
    void stepHeadless(scene::Scene& scene);

    std::unique_ptr<gfx::RigidBodySimulator> rigid;
    // 렌더러가 없을 때만 만든다. 있으면 렌더 그래프가 같은 일을 한다.
    std::unique_ptr<gfx::HeadlessCompute> headless;
    // 고정 간격 누적기와 이번 프레임에 푼 스텝 수. 렌더 그래프 노드가 GPU 솔버에 같은 수를 넘긴다.
    float accumulator = 0.0F;
    uint32_t steps = 0;
    // 재생을 켜고 끈 변화를 다음 프레임 머리에서 알아채 GPU 상태를 버린다.
    bool wasSimulating = false;
};

} // namespace app
