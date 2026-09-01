#version 460
#include "scene_types.glsl"

// 그림자 아틀라스와 깊이 선행 패스가 함께 쓴다. 시점 행렬만 바꿔 끼운다.
// mat4 는 16 바이트 정렬을 요구하므로 맨 앞에 두어야 C++ 쪽 배치와 어긋나지 않는다.
layout(push_constant) uniform DepthPushConstants {
    mat4 viewProjection;
    VertexBuffer vertices;
    InstanceBuffer instances;
    JointBuffer joints;
}
depthPush;

void main() {
    Instance instance = depthPush.instances.items[gl_InstanceIndex];
    Vertex vertex = depthPush.vertices.items[gl_VertexIndex];

    vec3 position = vertex.position;
    if (instance.jointOffset != NO_JOINTS) {
        mat4 skin = skinMatrix(depthPush.joints, instance.jointOffset, vertex.joints, vertex.weights);
        position = (skin * vec4(position, 1.0)).xyz;
    }
    gl_Position = depthPush.viewProjection * instance.model * vec4(position, 1.0);
}
