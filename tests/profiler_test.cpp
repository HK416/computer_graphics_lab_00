#include <cassert>
#include <cmath>
#include <cstdio>

#include "gfx/profiler_math.h"

int main() {
    // 첫 표본은 그대로 받는다. 0 에서 서서히 올라오면 한동안 거짓말을 한다.
    assert(gfx::smoothMilliseconds(0.0F, 4.0F, 0.1F) == 4.0F);
    assert(gfx::smoothMilliseconds(-1.0F, 4.0F, 0.1F) == 4.0F);

    // 이후에는 지수 평활. alpha 만큼만 새 표본 쪽으로 움직인다.
    assert(std::abs(gfx::smoothMilliseconds(10.0F, 20.0F, 0.5F) - 15.0F) < 1e-5F);
    assert(std::abs(gfx::smoothMilliseconds(10.0F, 20.0F, 0.0F) - 10.0F) < 1e-5F);

    // 충분히 반복하면 표본값으로 수렴한다.
    float smoothed = 100.0F;
    for (int i = 0; i < 200; ++i) {
        smoothed = gfx::smoothMilliseconds(smoothed, 5.0F, 0.2F);
    }
    assert(std::abs(smoothed - 5.0F) < 1e-3F);

    // 타임스탬프 차이를 밀리초로. 주기 1ns 이면 1,000,000 틱이 1ms 다.
    assert(std::abs(gfx::timestampMilliseconds(1000, 1'001'000, 1.0F, 64) - 1.0F) < 1e-4F);
    // 주기가 2ns 면 같은 틱 수가 두 배 시간이다.
    assert(std::abs(gfx::timestampMilliseconds(0, 1'000'000, 2.0F, 64) - 2.0F) < 1e-4F);

    // 유효 비트 밖의 상위 비트는 정의되지 않은 값이라 잘라 내야 한다.
    uint64_t garbage = 0xFFFF'0000'0000'0000ULL;
    assert(std::abs(gfx::timestampMilliseconds(garbage, garbage + 1'000'000, 1.0F, 32) - 1.0F) < 1e-4F);

    // 카운터가 한 바퀴 돌아도 마스크 덕에 차이가 맞는다.
    uint64_t nearWrap = (1ULL << 32) - 500'000;
    assert(std::abs(gfx::timestampMilliseconds(nearWrap, 500'000, 1.0F, 32) - 1.0F) < 1e-4F);

    // 지원하지 않는 장치는 0 이다.
    assert(gfx::timestampMilliseconds(0, 1'000'000, 0.0F, 64) == 0.0F);
    assert(gfx::timestampMilliseconds(0, 1'000'000, 1.0F, 0) == 0.0F);

    std::printf("프로파일러 자체 점검 통과\n");
    return 0;
}
