#version 460
#include "scene_types.glsl"

// 그림자 아틀라스와 깊이 선행 패스가 함께 쓴다. 시점 행렬만 바꿔 끼운다.
// mat4 는 16 바이트 정렬을 요구하므로 맨 앞에 두어야 C++ 쪽 배치와 어긋나지 않는다.
layout(push_constant) uniform DepthPushConstants {
    mat4 viewProjection;
    VertexBuffer vertices;
    InstanceBuffer instances;
    JointBuffer joints;
    MeshBuffer meshes;
    MaterialBuffer materials;
}
depthPush;

// 컷오프 파이프라인만 읽는다. 불투명 파이프라인에는 프래그먼트 셰이더가 없어 그냥 버려진다.
layout(location = 0) out vec2 outUv;
layout(location = 1) flat out uint outMaterialIndex;

void main() {
    Instance instance = depthPush.instances.items[gl_InstanceIndex];
    Vertex vertex = depthPush.vertices.items[gl_VertexIndex];
    outUv = vertex.uv;
    outMaterialIndex = depthPush.meshes.items[instance.meshIndex].materialIndex;

    vec3 position = vertex.position;
    if (instance.jointOffset != NO_JOINTS) {
        mat4 skin = skinMatrix(depthPush.joints, instance.jointOffset, vertex.joints, vertex.weights);
        position = (skin * vec4(position, 1.0)).xyz;
    }
    gl_Position = depthPush.viewProjection * instance.model * vec4(position, 1.0);
}
