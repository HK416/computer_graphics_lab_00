#version 460
#include "scene_types.glsl"

// depth_only.vert 과 같은 블록이어야 한다. mat4 는 16 바이트 정렬이라 맨 앞에 둔다.
layout(push_constant) uniform DepthPushConstants {
    mat4 viewProjection;
    VertexBuffer vertices;
    InstanceBuffer instances;
    JointBuffer joints;
    MeshBuffer meshes;
    MaterialBuffer materials;
}
depthPush;

layout(location = 0) in vec2 inUv;
layout(location = 1) flat in uint inMaterialIndex;

// 컷오프 재질은 뚫린 곳으로 빛이 지나가야 한다. 불투명 캐스터는 이 셰이더 없이 그린다.
void main() {
    Material material = depthPush.materials.items[inMaterialIndex];
    float alpha = material.baseColorFactor.a;
    if (material.baseColorTexture != INVALID_TEXTURE) {
        alpha *= sampleBindless(material.baseColorTexture, inUv).a;
    }
    if (alpha < material.emissiveAndCutoff.w) {
        discard;
    }
}
