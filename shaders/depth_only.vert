#version 460
#include "scene_types.glsl"

// 그림자 아틀라스가 쓴다. 시점 행렬만 바꿔 끼운다.
// mat4 는 16 바이트 정렬을 요구하므로 맨 앞에 두어야 C++ 쪽 배치와 어긋나지 않는다.
layout(push_constant) uniform DepthPushConstants {
    mat4 viewProjection;
    VertexBuffer vertices;
    InstanceBuffer instances;
    // 스킨 컴퓨트가 뽑아 둔 변형 정점. 장면 패스와 같은 버퍼다.
    VertexBuffer skinnedVertices;
    MeshBuffer meshes;
    MaterialBuffer materials;
}
depthPush;

// 컷오프 파이프라인만 읽는다. 불투명 파이프라인에는 프래그먼트 셰이더가 없어 그냥 버려진다.
layout(location = 0) out vec2 outUv;
layout(location = 1) flat out uint outMaterialIndex;

void main() {
    Instance instance = depthPush.instances.items[gl_InstanceIndex];
    Mesh mesh = depthPush.meshes.items[instance.meshIndex];
    Vertex vertex = instance.skinnedVertexOffset == NO_SKINNED_VERTICES
                        ? depthPush.vertices.items[gl_VertexIndex]
                        : depthPush.skinnedVertices
                              .items[instance.skinnedVertexOffset + (uint(gl_VertexIndex) - uint(mesh.vertexOffset))];
    outUv = vertex.uv;
    outMaterialIndex = mesh.materialIndex;
    gl_Position = depthPush.viewProjection * instance.model * vec4(vertex.position, 1.0);
}
