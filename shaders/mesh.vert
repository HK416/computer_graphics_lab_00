#version 460
#include "scene_data.glsl"

layout(location = 0) out vec3 outWorldPosition;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec4 outTangent;
layout(location = 3) out vec2 outUv;
layout(location = 4) flat out uint outMaterialIndex;
layout(location = 5) flat out uint outMeshletIndex;
layout(location = 6) out vec4 outCurrentClip;
layout(location = 7) out vec4 outPreviousClip;

void main() {
    Instance instance = pushConstants.instances.items[gl_InstanceIndex];
    Mesh mesh = pushConstants.meshes.items[instance.meshIndex];
    // 스킨이 있으면 이미 포즈 공간으로 옮겨진 정점이 온다. 정점 셰이더는 스키닝을 하지 않는다.
    Vertex vertex = fetchVertex(instance, mesh, gl_VertexIndex);

    vec4 worldPosition = instance.model * vec4(vertex.position, 1.0);
    gl_Position = pushConstants.camera.item.viewProjection * worldPosition;
    outCurrentClip = gl_Position;
    outPreviousClip = previousClipPosition(instance, mesh, gl_VertexIndex, vertex.position);

    mat3 normalMatrix = mat3(instance.normalMatrix);
    outWorldPosition = worldPosition.xyz;
    outNormal = normalMatrix * vertex.normal;
    outTangent = vec4(normalMatrix * vertex.tangent.xyz, vertex.tangent.w);
    outUv = vertex.uv;
    outMaterialIndex = mesh.materialIndex;
    outMeshletIndex = pushConstants.vertexMeshlets.items[gl_VertexIndex];
}
