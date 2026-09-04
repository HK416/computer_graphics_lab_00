#version 460

#include "fluid_draw_common.glsl"

layout(location = 0) out vec3 outWorldPosition;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec4 outCurrentClip;
layout(location = 3) out vec4 outPreviousClip;

// 깊이 프리패스와 표면 패스가 «같은» 깊이를 내야 VK_COMPARE_OP_EQUAL 이 맞는다. 파이프라인이 달라도
// 같은 값이 나오도록 못 박는다.
invariant gl_Position;

// 마칭 큐브 정점을 인덱스 없이 그대로 그린다. 삼각형 순서는 프레임마다 달라질 수 있다.
void main() {
    FluidSurfaceVertex vertex = push.vertices.items[gl_VertexIndex];
    Camera camera = push.camera.item;
    outWorldPosition = vertex.position;
    outNormal = decodeUnitVector(vertex.normal);
    outCurrentClip = camera.viewProjection * vec4(vertex.position, 1.0);
    // 표면은 프레임마다 새로 만들어져 정점이 이어지지 않는다. 지난 위치를 알 수 없으므로 카메라
    // 움직임만 담는다.
    //
    // ponytail: 물이 빠르게 흐르면 시간축 업스케일이 그만큼 뭉갠다. 입자에서 화면 흐름을 따로
    // 뽑아 넣으면 나아진다.
    outPreviousClip = camera.previousViewProjection * vec4(vertex.position, 1.0);
    gl_Position = outCurrentClip;
}
