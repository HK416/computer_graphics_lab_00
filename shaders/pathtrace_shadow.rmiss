#version 460
#extension GL_EXT_ray_tracing : require

// 그림자 광선 전용. 아무것도 맞지 않았다는 사실만 알리면 된다.
layout(location = 1) rayPayloadInEXT float shadowVisibility;

void main() {
    shadowVisibility = 1.0;
}
