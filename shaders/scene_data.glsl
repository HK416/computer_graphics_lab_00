#ifndef SCENE_DATA_GLSL
#define SCENE_DATA_GLSL

#include "scene_types.glsl"

layout(push_constant) uniform PushConstants {
    VertexBuffer vertices;
    MeshBuffer meshes;
    InstanceBuffer instances;
    MaterialBuffer materials;
    CameraBuffer camera;
    MeshletBuffer meshlets;
    MeshletTriangleBuffer meshletTriangles;
    VertexMeshletBuffer vertexMeshlets;
    MeshletGroupBuffer meshletGroups;
    JointBuffer joints;
    // 재질 경로마다 meshlet 그룹 구간이 달라 디스패치 직전에 갱신한다.
    uint meshletGroupBase;
    uint debugMode;
}
pushConstants;

// 스킨이 있으면 정점을 포즈 공간으로 옮긴다. 조인트 변환은 강체에 가까워 노멀도 같은 행렬로 돌린다.
void skinVertex(Instance instance, Vertex vertex, inout vec3 position, inout vec3 normal, inout vec3 tangent) {
    if (instance.jointOffset == NO_JOINTS) {
        return;
    }
    mat4 skin = skinMatrix(pushConstants.joints, instance.jointOffset, vertex.joints, vertex.weights);
    position = (skin * vec4(position, 1.0)).xyz;
    mat3 rotation = mat3(skin);
    normal = rotation * normal;
    tangent = rotation * tangent;
}

#endif
