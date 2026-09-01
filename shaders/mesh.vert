#version 460
#include "scene_data.glsl"

layout(location = 0) out vec3 outWorldPosition;
layout(location = 1) out vec3 outNormal;
layout(location = 2) out vec4 outTangent;
layout(location = 3) out vec2 outUv;
layout(location = 4) flat out uint outMaterialIndex;
layout(location = 5) flat out uint outMeshletIndex;

void main() {
    Instance instance = pushConstants.instances.items[gl_InstanceIndex];
    Mesh mesh = pushConstants.meshes.items[instance.meshIndex];
    Vertex vertex = pushConstants.vertices.items[gl_VertexIndex];

    vec3 position = vertex.position;
    vec3 normal = vertex.normal;
    vec3 tangent = vertex.tangent.xyz;
    skinVertex(instance, vertex, position, normal, tangent);

    vec4 worldPosition = instance.model * vec4(position, 1.0);
    gl_Position = pushConstants.camera.item.viewProjection * worldPosition;

    mat3 normalMatrix = mat3(instance.normalMatrix);
    outWorldPosition = worldPosition.xyz;
    outNormal = normalMatrix * normal;
    outTangent = vec4(normalMatrix * tangent, vertex.tangent.w);
    outUv = vertex.uv;
    outMaterialIndex = mesh.materialIndex;
    outMeshletIndex = pushConstants.vertexMeshlets.items[gl_VertexIndex];
}
