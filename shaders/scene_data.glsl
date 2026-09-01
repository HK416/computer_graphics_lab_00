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
    // 지난 프레임의 조인트 행렬. 스킨 인스턴스의 이전 위치를 구하는 데만 쓴다.
    JointBuffer previousJoints;
    LightBuffer lights;
    ShadowMatrixBuffer shadowMatrices;
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

// 모션 벡터에 쓸 이전 프레임 클립 좌표. 스킨은 지난 프레임 포즈로 다시 옮긴다.
vec4 previousClipPosition(Instance instance, Vertex vertex) {
    vec3 position = vertex.position;
    if (instance.jointOffset != NO_JOINTS) {
        mat4 skin = skinMatrix(pushConstants.previousJoints, instance.jointOffset, vertex.joints, vertex.weights);
        position = (skin * vec4(position, 1.0)).xyz;
    }
    return pushConstants.camera.item.previousViewProjection * (instance.previousModel * vec4(position, 1.0));
}

#endif
