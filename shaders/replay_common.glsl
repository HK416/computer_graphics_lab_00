#ifndef REPLAY_COMMON_GLSL
#define REPLAY_COMMON_GLSL

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

// 리플레이 링. **전이마다 관측 한 판만 담는다** — 프레임 스택(최근 세 판)은 저장하지 않고 첨자를 거슬러
// 읽어 만든다. 스택째로 담으면 같은 그림이 세 번 들어가 메모리가 세 배가 된다.
//
// 값은 uint8 회색이고 uint32 하나에 넷씩 묶는다. 8비트 저장 기능을 요구하지 않으려는 것이다(FSR 과
// 달리 이 경로는 그 기능이 없는 기기에서도 돌아야 한다).
//
// **첨자 규칙은 gfx::replayStackIndex / replaySampleValid 와 같은 식이다.** 링이 되감기고 에피소드가
// 경계를 긋고 창이 앞에서 잘려 나가는 셋이 한 식에서 만나므로, C++ 쪽에 순수 함수로 떼어 테스트한다.
// 한쪽을 고치면 다른 쪽도 고친다.

// gfx::GpuReplaySlot 과 배치가 같아야 한다.
struct ReplaySlot {
    uint episode;
    uint stepInEpisode;
    float reward;
    // 종료 전이면 0 이다. y = r + discount * minQ' 가 그대로 돈다.
    float discount;
};

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer ReplayFrameBuffer {
    uint items[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer ReplaySlotBuffer {
    ReplaySlot items[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer ReplayFloatBuffer {
    float items[];
};

// gfx::GpuReplaySample 과 배치가 같아야 한다. 표본 첨자와 증강 변위를 **호스트가 고른다** — 그래야
// 자기 검사가 같은 표본을 CPU 로도 만들어 바이트로 견줄 수 있다.
struct ReplaySample {
    uint index;
    int shiftX;
    int shiftY;
    // 다음 관측은 변위를 따로 뽑는다. DrQ-v2 가 그렇게 한다.
    int nextShiftX;
    int nextShiftY;
    uint pad0;
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer ReplaySampleBuffer {
    ReplaySample items[];
};

// 링에서 (슬롯, 뷰, y, x) 의 회색값을 읽는다. 가장자리 밖은 **가장자리를 복제**한다 — DrQ-v2 의 이동
// 증강이 그렇게 덧대고, 0 으로 채우면 없던 검은 테두리가 학습 신호가 된다.
float replayFetch(ReplayFrameBuffer frames, uint slot, uint view, uint views, uint size, int x, int y) {
    int clampedX = clamp(x, 0, int(size) - 1);
    int clampedY = clamp(y, 0, int(size) - 1);
    uint plane = size * size;
    uint pixel = (slot * views + view) * plane + uint(clampedY) * size + uint(clampedX);
    uint word = frames.items[pixel >> 2];
    uint byteIndex = pixel & 3u;
    return float((word >> (byteIndex * 8u)) & 0xFFu) * (1.0 / 255.0);
}

// gfx::replayStackIndex 와 같은 식이다.
uint replayStackIndex(uint capacity, uint count, uint cursor, uint index, uint stepInEpisode, uint age) {
    if (capacity == 0u) {
        return index;
    }
    uint oldest = count < capacity ? 0u : cursor;
    uint depth = (index + capacity - oldest) % capacity;
    uint back = min(min(age, stepInEpisode), depth);
    return (index + capacity - back) % capacity;
}

#endif
