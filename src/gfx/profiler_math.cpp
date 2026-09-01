#include "gfx/profiler_math.h"

namespace gfx {

float smoothMilliseconds(float previous, float sample, float alpha) {
    if (previous <= 0.0F) {
        return sample;
    }
    return previous + (sample - previous) * alpha;
}

float timestampMilliseconds(uint64_t begin, uint64_t end, float period, uint32_t validBits) {
    if (validBits == 0 || period <= 0.0F) {
        return 0.0F;
    }
    uint64_t mask = validBits >= 64 ? ~0ULL : ((1ULL << validBits) - 1ULL);
    uint64_t delta = (end - begin) & mask;
    return static_cast<float>(static_cast<double>(delta) * static_cast<double>(period) / 1.0e6);
}

} // namespace gfx
