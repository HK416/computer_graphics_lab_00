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
    MeshletVertexBuffer meshletVertices;
    DrawMeshletBuffer drawMeshlets;
    MeshletGroupBuffer meshletGroups;
    // 스킨 컴퓨트가 뽑아 둔 변형 정점. 이번 포즈와 지난 포즈가 반쪽씩 들어 있고, 인스턴스의
    // 오프셋이 각각을 가리킨다. 광선 경로도 같은 버퍼를 읽는다.
    VertexBuffer skinnedVertices;
    // 변형 정점에서 다시 잰 스킨 인스턴스의 meshlet 경계 구. 태스크 셰이더 컬링이 읽는다.
    SkinnedBoundsBuffer skinnedBounds;
    LightBuffer lights;
    ShadowMatrixBuffer shadowMatrices;
    // 재질 경로마다 meshlet 그룹(태스크 셰이더) 또는 명령(고전 경로) 구간이 달라 그리기 직전에 갱신한다.
    uint meshletGroupBase;
    // 두 패스 컬링 단계(culling.glsl 의 CULL_PHASE_*).
    uint cullPhase;
    // meshlet 가시성 비트. 여기까지 꼭 128 바이트라 규격이 보장하는 한도에 닿아 있다. 더 넣을 수
    // 없으니 디버그 뷰처럼 프레임에 한 번 정해지는 값은 Camera 로 간다.
    VisibilityBuffer meshletVisibility;
}
pushConstants;

uint sceneDebugMode() {
    return pushConstants.camera.item.flags.x;
}

// 전역 정점 번호로 정점을 읽는다. 스킨 인스턴스는 변형 정점 구간에서 같은 지역 번호를 읽는다.
Vertex fetchVertex(Instance instance, Mesh mesh, uint globalIndex) {
    if (instance.skinnedVertexOffset == NO_SKINNED_VERTICES) {
        return pushConstants.vertices.items[globalIndex];
    }
    return pushConstants.skinnedVertices.items[instance.skinnedVertexOffset + (globalIndex - uint(mesh.vertexOffset))];
}

// 모션 벡터에 쓸 이전 프레임 클립 좌표. 스킨 인스턴스는 지난 포즈의 변형 정점을 읽는다.
vec4 previousClipPosition(Instance instance, Mesh mesh, uint globalIndex, vec3 localPosition) {
    if (instance.skinnedVertexOffset != NO_SKINNED_VERTICES) {
        localPosition = pushConstants.skinnedVertices
                            .items[instance.previousSkinnedVertexOffset + (globalIndex - uint(mesh.vertexOffset))]
                            .position;
    }
    return pushConstants.camera.item.previousViewProjection * (instance.previousModel * vec4(localPosition, 1.0));
}

#endif
