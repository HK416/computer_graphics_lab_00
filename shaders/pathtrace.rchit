#version 460
#extension GL_EXT_nonuniform_qualifier : require

#include "bindless.glsl"
#include "material.glsl"
#include "pathtrace_common.glsl"

layout(location = 0) rayPayloadInEXT PathPayload payload;
hitAttributeEXT vec2 barycentrics;

void main() {
    // 정점 보간과 재질 읽기는 반사 컴퓨트와 같은 함수를 쓴다.
    HitSurface hit = interpolateHit(pathTrace.instances,
                                    pathTrace.meshes,
                                    pathTrace.lods,
                                    pathTrace.vertices,
                                    pathTrace.skinnedVertices,
                                    pathTrace.indices,
                                    uint(gl_InstanceCustomIndexEXT),
                                    uint(gl_PrimitiveID),
                                    barycentrics);
    Material material = pathTrace.materials.items[hit.materialIndex];

    vec3 normal = perturbNormal(material, hit.normal, hit.tangent, hit.tangentSign, hit.uv);
    // 양면 재질의 뒷면을 맞으면 노멀을 뒤집는다. 단면 재질은 후면을 아예 컬링하므로 걸리지 않는다.
    if (dot(normal, gl_WorldRayDirectionEXT) > 0.0) {
        normal = -normal;
    }
    MaterialSample surface = sampleMaterial(material, hit.uv);

    payload.albedo = surface.albedo;
    payload.emissive = surface.emissive;
    payload.normal = normal;
    payload.position = hit.position;
    payload.metallic = surface.metallic;
    payload.roughness = surface.roughness;
    payload.previousPosition = hit.previousPosition;
    payload.uv = hit.uv;
    payload.hitDistance = gl_HitTEXT;
    payload.missed = false;
}
