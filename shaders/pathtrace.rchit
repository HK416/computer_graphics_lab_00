#version 460
#extension GL_EXT_nonuniform_qualifier : require

#include "bindless.glsl"
#include "material.glsl"
#include "pathtrace_common.glsl"

layout(location = 0) rayPayloadInEXT PathPayload payload;
hitAttributeEXT vec2 barycentrics;

void main() {
    payload.water = false;
    payload.absorption = vec3(0.0);
    // 물 표면. 마칭 큐브가 만든 세계 공간 삼각형이라 인스턴스·메쉬·재질 표를 거치지 않고 정점을 바로 읽는다.
    if ((uint(gl_InstanceCustomIndexEXT) & FLUID_SURFACE_CUSTOM_INDEX) != 0u) {
        FluidSurfaceInfo info = pathTrace.fluidSurfaces.items[uint(gl_InstanceCustomIndexEXT) & 0xFFFFu];
        uint base = uint(gl_PrimitiveID) * 3u;
        FluidSurfaceVertex v0 = info.vertices.items[base];
        FluidSurfaceVertex v1 = info.vertices.items[base + 1u];
        FluidSurfaceVertex v2 = info.vertices.items[base + 2u];
        vec3 weights = vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
        vec3 normal = normalize(decodeUnitVector(v0.normal) * weights.x + decodeUnitVector(v1.normal) * weights.y +
                                decodeUnitVector(v2.normal) * weights.z);
        vec3 position = v0.position * weights.x + v1.position * weights.y + v2.position * weights.z;
        payload.albedo = info.waterColor.rgb;
        payload.emissive = vec3(0.0);
        // 바깥 법선 그대로 둔다. 안에서 나가는 광선인지는 광선 생성 셰이더가 진행 방향으로 가른다.
        payload.normal = normal;
        payload.position = position;
        payload.metallic = 0.0;
        payload.roughness = clamp(info.waterColor.w, 0.02, 1.0);
        payload.previousPosition = position;
        payload.uv = vec2(0.0);
        payload.hitDistance = gl_HitTEXT;
        payload.missed = false;
        payload.water = true;
        payload.absorption = info.absorption.rgb * info.absorption.w;
        return;
    }
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
