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
    uint vertexBase = uint(mesh.vertexOffset);
    Vertex v0 = pathTrace.vertices.items[vertexBase + pathTrace.indices.items[base]];
    Vertex v1 = pathTrace.vertices.items[vertexBase + pathTrace.indices.items[base + 1]];
    Vertex v2 = pathTrace.vertices.items[vertexBase + pathTrace.indices.items[base + 2]];

    vec3 weights = vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
    vec3 localPosition = v0.position * weights.x + v1.position * weights.y + v2.position * weights.z;
    vec3 localNormal = v0.normal * weights.x + v1.normal * weights.y + v2.normal * weights.z;
    vec2 uv = v0.uv * weights.x + v1.uv * weights.y + v2.uv * weights.z;

    vec3 worldPosition = (instance.model * vec4(localPosition, 1.0)).xyz;
    vec3 worldNormal = normalize(mat3(instance.normalMatrix) * localNormal);
    if (dot(worldNormal, gl_WorldRayDirectionEXT) > 0.0) {
        worldNormal = -worldNormal;
    }

    Material material = pathTrace.materials.items[mesh.materialIndex];
    vec3 albedo = material.baseColorFactor.rgb;
    if (material.baseColorTexture != INVALID_TEXTURE) {
        albedo *= sampleBindless(material.baseColorTexture, uv).rgb;
    }
    vec3 emissive = material.emissiveAndCutoff.rgb;
    if (material.emissiveTexture != INVALID_TEXTURE) {
        emissive *= sampleBindless(material.emissiveTexture, uv).rgb;
    }

    payload.albedo = albedo;
    payload.emissive = emissive;
    payload.normal = worldNormal;
    payload.position = worldPosition;
    payload.hitDistance = gl_HitTEXT;
    payload.missed = false;
}
