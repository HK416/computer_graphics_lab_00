#version 460
#extension GL_EXT_nonuniform_qualifier : require

#include "bindless.glsl"
#include "pathtrace_common.glsl"

layout(location = 0) rayPayloadInEXT PathPayload payload;
hitAttributeEXT vec2 barycentrics;

void main() {
    Instance instance = pathTrace.instances.items[gl_InstanceCustomIndexEXT];
    Mesh mesh = pathTrace.meshes.items[instance.meshIndex];
    // 가속 구조는 0단계 LOD 로만 만든다.
    MeshLod lod = pathTrace.lods.items[mesh.lodOffset];

    uint base = lod.indexOffset + uint(gl_PrimitiveID) * 3;
    // 스킨 인스턴스는 가속 구조도 변형 정점으로 세웠으므로 셰이딩도 같은 정점을 읽어야 한다.
    // 인덱스는 메쉬 지역 번호라 두 경우 모두 그대로 쓴다.
    bool skinned = instance.skinnedVertexOffset != NO_SKINNED_VERTICES;
    VertexBuffer source = skinned ? pathTrace.skinnedVertices : pathTrace.vertices;
    uint vertexBase = skinned ? instance.skinnedVertexOffset : uint(mesh.vertexOffset);
    Vertex v0 = source.items[vertexBase + pathTrace.indices.items[base]];
    Vertex v1 = source.items[vertexBase + pathTrace.indices.items[base + 1]];
    Vertex v2 = source.items[vertexBase + pathTrace.indices.items[base + 2]];

    vec3 weights = vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
    vec3 localPosition = v0.position * weights.x + v1.position * weights.y + v2.position * weights.z;
    vec3 localNormal = v0.normal * weights.x + v1.normal * weights.y + v2.normal * weights.z;
    vec4 localTangent = v0.tangent * weights.x + v1.tangent * weights.y + v2.tangent * weights.z;
    vec2 uv = v0.uv * weights.x + v1.uv * weights.y + v2.uv * weights.z;

    vec3 worldPosition = (instance.model * vec4(localPosition, 1.0)).xyz;
    // ponytail: 스킨 인스턴스의 변형 정점은 이번 프레임 포즈만 있어, 지난 포즈의 변형은 반영하지
    // 못하고 강체 이동만 담는다. 지난 포즈 정점을 따로 뽑아 두면 정확해진다.
    vec3 previousWorldPosition = (instance.previousModel * vec4(localPosition, 1.0)).xyz;
    mat3 normalMatrix = mat3(instance.normalMatrix);
    vec3 worldNormal = normalize(normalMatrix * localNormal);

    Material material = pathTrace.materials.items[mesh.materialIndex];

    // 접선 공간 노멀 맵. mesh_shading.glsl 의 shadingNormal 과 같은 식이어야 두 경로가 같은
    // 굴곡을 낸다.
    if (material.normalTexture != INVALID_TEXTURE) {
        vec3 worldTangent = normalize(normalMatrix * localTangent.xyz);
        worldTangent = normalize(worldTangent - worldNormal * dot(worldNormal, worldTangent));
        vec3 bitangent = cross(worldNormal, worldTangent) * localTangent.w;
        vec3 sampled = sampleBindless(material.normalTexture, uv).xyz * 2.0 - 1.0;
        sampled.xy *= material.normalScale;
        worldNormal = normalize(mat3(worldTangent, bitangent, worldNormal) * sampled);
    }
    // 양면 재질의 뒷면을 맞으면 노멀을 뒤집는다. 단면 재질은 후면을 아예 컬링하므로 걸리지 않는다.
    if (dot(worldNormal, gl_WorldRayDirectionEXT) > 0.0) {
        worldNormal = -worldNormal;
    }

    vec3 albedo = material.baseColorFactor.rgb;
    if (material.baseColorTexture != INVALID_TEXTURE) {
        albedo *= sampleBindless(material.baseColorTexture, uv).rgb;
    }

    float metallic = material.metallicFactor;
    float roughness = material.roughnessFactor;
    if (material.metallicRoughnessTexture != INVALID_TEXTURE) {
        vec4 sampled = sampleBindless(material.metallicRoughnessTexture, uv);
        roughness *= sampled.g;
        metallic *= sampled.b;
    }

    vec3 emissive = material.emissiveAndCutoff.rgb;
    if (material.emissiveTexture != INVALID_TEXTURE) {
        emissive *= sampleBindless(material.emissiveTexture, uv).rgb;
    }

    payload.albedo = albedo;
    payload.emissive = emissive;
    payload.normal = worldNormal;
    payload.position = worldPosition;
    payload.metallic = clamp(metallic, 0.0, 1.0);
    // 완전한 거울은 GGX 표본의 분모를 0 으로 만든다. 래스터 경로와 같은 하한을 쓴다.
    payload.roughness = clamp(roughness, 0.03, 1.0);
    payload.previousPosition = previousWorldPosition;
    payload.uv = uv;
    payload.hitDistance = gl_HitTEXT;
    payload.missed = false;
}
