#version 460
#include "scene_data.glsl"

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec2 outUv;
layout(location = 2) flat out uint outMaterialIndex;

void main() {
    Instance instance = pushConstants.instances.items[gl_InstanceIndex];
    Mesh mesh = pushConstants.meshes.items[instance.meshIndex];
    Vertex vertex = pushConstants.vertices.items[gl_VertexIndex];

    vec4 worldPosition = instance.model * vec4(vertex.position, 1.0);
    gl_Position = pushConstants.camera.item.viewProjection * worldPosition;
    outNormal = mat3(instance.normalMatrix) * vertex.normal;
    outUv = vertex.uv;
    outMaterialIndex = mesh.materialIndex;
}
