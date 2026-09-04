#ifndef DEBUG_LINE_COMMON_GLSL
#define DEBUG_LINE_COMMON_GLSL

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

// 편집기의 콜라이더·유체 경계 표시. 정점 셰이더와 프래그먼트 셰이더가 같은 푸시 상수를 본다.
// src/gfx/debug_lines.h 의 DebugLineVertex 와 renderer.cpp 의 DebugLinePushConstants 와 배치가 같아야 한다.
struct DebugLineVertex {
    vec3 position;
    uint color;
};

layout(buffer_reference, scalar) buffer DebugLineBuffer {
    DebugLineVertex items[];
};

// scalar 를 붙이지 않으면 std430 이 vec2 를 8 바이트 경계로 밀어 C++ 배치와 어긋난다.
layout(push_constant, scalar) uniform DebugLinePushConstants {
    mat4 viewProjection;
    DebugLineBuffer vertices;
    // 렌더 해상도의 깊이 버퍼. 표시 해상도와 크기가 달라도 uv 로 읽으므로 상관없다.
    uint depthTexture;
    // 0 이 아니면 장면에 가려진 선을 지운다. Unity 의 기즈모처럼 물체 뒤로 숨는다.
    uint occlude;
    vec2 viewportSize;
} push;

#endif
