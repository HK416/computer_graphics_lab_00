#version 460

#include "observation_common.glsl"

layout(location = 0) out vec3 outWorldPosition;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec4 outTangent;
layout(location = 3) out vec2 outUv;
layout(location = 4) flat out uint outMaterialIndex;

void main() {
    Instance instance = observationPush.instances.items[gl_InstanceIndex];
    Mesh mesh = observationPush.meshes.items[instance.meshIndex];
    // 스킨이 있으면 스킨 컴퓨트가 이미 포즈 공간으로 옮겨 둔 정점을 읽는다. depth_only.vert 와 같은 규칙이다.
    Vertex vertex =
        instance.skinnedVertexOffset == NO_SKINNED_VERTICES
            ? observationPush.vertices.items[gl_VertexIndex]
            : observationPush.skinnedVertices
                  .items[instance.skinnedVertexOffset + (uint(gl_VertexIndex) - uint(mesh.vertexOffset))];

    vec4 worldPosition = instance.model * vec4(vertex.position, 1.0);
    gl_Position = observationPush.viewProjection * worldPosition;

    mat3 normalMatrix = mat3(instance.normalMatrix);
    vec4 tangent = decodeTangent(vertex.tangent);
    outWorldPosition = worldPosition.xyz;
    outNormal = normalMatrix * decodeUnitVector(vertex.normal);
    outTangent = vec4(normalMatrix * tangent.xyz, tangent.w);
    outUv = vertex.uv;
    outMaterialIndex = mesh.materialIndex;
}
