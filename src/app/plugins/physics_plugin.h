#pragma once

#include <cstdint>

#include "app/plugin.h"

namespace app {

// 강체 물리. 재생 중 고정 간격(1/120 초)으로 CPU 솔버를 돌리고, GPU 백엔드 강체는 같은 간격으로 렌더러가
// 풀도록 스텝 수를 넘긴다. 프레임이 길어도 정해진 스텝 수까지만 따라잡아 나선형으로 느려지지 않는다.
class PhysicsPlugin : public Plugin {
public:
    static constexpr float STEP_SECONDS = 1.0F / 120.0F;
    static constexpr uint32_t MAX_STEPS_PER_FRAME = 8;

    const char* name() const override { return "물리"; }
    void update(Services& services, float deltaSeconds) override;

private:
    // 고정 간격 누적기.
    float accumulator = 0.0F;
};

} // namespace app
