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
    // 재질 경로마다 meshlet 그룹 구간이 달라 디스패치 직전에 갱신한다.
    uint meshletGroupBase;
    uint debugMode;
}
pushConstants;

#endif
