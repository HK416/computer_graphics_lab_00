#pragma once

#include <cstdint>

namespace gfx {

// 프로파일러의 순수 계산부. Vulkan 을 끌어오지 않아 테스트가 그대로 링크한다.

// 지수 평활. 구간 시간은 프레임마다 크게 튀어 그대로 보면 읽을 수 없다.
// 이전 값이 없으면(0 이하) 첫 표본을 그대로 받는다. 0 에서 서서히 올라오면 한동안 거짓말을 한다.
float smoothMilliseconds(float previous, float sample, float alpha);

// 타임스탬프 두 개의 차이를 밀리초로 바꾼다. 유효 비트 밖의 상위 비트는 정의되지 않은 값이므로
// 잘라 내고, 그 덕에 카운터가 한 바퀴 돈 경우도 자연히 처리된다.
float timestampMilliseconds(uint64_t begin, uint64_t end, float period, uint32_t validBits);

} // namespace gfx
