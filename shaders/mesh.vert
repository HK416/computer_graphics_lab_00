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
    vec4 tangent = decodeTangent(vertex.tangent);
    outNormal = normalMatrix * decodeUnitVector(vertex.normal);
    outTangent = vec4(normalMatrix * tangent.xyz, tangent.w);
    outUv = vertex.uv;
    outMaterialIndex = mesh.materialIndex;
    // 정점은 meshlet 끼리 공유하므로 정점으로는 meshlet 을 모른다. 명령 하나가 meshlet 하나이니
    // 명령 번호로 찾는다. gl_DrawID 는 호출마다 0 부터라 버킷 구간의 시작을 더한다.
#ifdef NO_DRAW_ID
    // MoltenVK 변종. gl_DrawID 가 없어 명령별 meshlet 을 못 찾으니 메쉬의 첫 meshlet 으로 대신한다.
    // ponytail: meshlet·LOD 디버그 뷰가 메쉬 단위로 뭉개진다. 제대로 하려면 firstInstance 상위
    // 비트에 명령 번호를 실어 gl_BaseInstance 로 읽어야 한다.
    outMeshletIndex = mesh.meshletOffset;
#else
    outMeshletIndex = pushConstants.drawMeshlets.items[pushConstants.meshletGroupBase + gl_DrawID];
#endif
}
