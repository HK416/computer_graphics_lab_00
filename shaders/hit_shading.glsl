#ifndef HIT_SHADING_GLSL
#define HIT_SHADING_GLSL

// 광선 질의 히트 셰이딩. 광선 반사(reflect.comp)와 DDGI 프로브 추적(ddgi.comp)이 함께 쓴다. 두 셰이더의 푸시
// 상수 블록이 같은 멤버 이름(vertices, skinnedVertices, indices, meshes, instances, materials, lods, lights,
// fluidSurfaces)을 가지고, 집합 1 에 topLevel 가속 구조를 묶고, 직접광 후보 수를 돌려주는 hitLightCandidates() 를
// include 앞에 정의해야 한다. 히트 표면의 재질과 조명은 경로 추적 적중 셰이더와 같은 함수를 쓴다.

// 직접광은 광원 후보를 재추출(RIS, restir.glsl)해 하나를 고르고 그림자 광선 하나로 판정한다. 경로 추적의 다음
// 사건 추정과 같은 식이라 시간축 누적이 평균을 맞춘다.
vec3 sampleDirectLight(Camera camera, Surface surface, inout uint seed) {
    uint lightCount = camera.shading.x;
    if (lightCount == 0u) {
        return vec3(0.0);
    }
    Reservoir reservoir = risPickLight(push.lights, lightCount, surface, hitLightCandidates(), seed);
    if (reservoir.light == RESTIR_NO_LIGHT || reservoir.w <= 0.0) {
        return vec3(0.0);
    }
    Light light = push.lights.items[reservoir.light];
    vec3 lightDirection;
    float distanceToLight;
    vec3 radiance = sampleLightRadiance(light, surface.position, lightDirection, distanceToLight);
    if (radiance == vec3(0.0) || dot(surface.normal, lightDirection) <= 0.0) {
        return vec3(0.0);
    }
    float visibility =
        rayQueryVisibility(topLevel, surface.position, surface.normal, lightDirection, min(distanceToLight - 1e-3, 1e4));
    return evaluateBrdf(surface, lightDirection, radiance) * visibility * reservoir.w;
}

// 물 표면 히트. 마칭 큐브 정점을 바로 읽고, 래스터 물 표면과 같은 식(water_shading.glsl)으로 셰이딩한다. 두께
// 텍스처가 없으니 기본 두께를 쓴다.
//
// ponytail: 물 뒤에 있는 것을 굴절로 보지 않는다(재귀 없음). 반사 안의 물은 «반사 + 물빛» 만 있고 그 뒤는
// 물빛으로 가려진다.
vec3 shadeWaterHit(Camera camera, vec3 origin, vec3 direction, float hitDistance, uint customIndex, uint primitive, vec2 barycentrics, inout uint seed) {
    FluidSurfaceInfo info = push.fluidSurfaces.items[customIndex & 0xFFFFu];
    uint base = primitive * 3u;
    FluidSurfaceVertex v0 = info.vertices.items[base];
    FluidSurfaceVertex v1 = info.vertices.items[base + 1u];
    FluidSurfaceVertex v2 = info.vertices.items[base + 2u];
    vec3 weights = vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
    vec3 normal = normalize(decodeUnitVector(v0.normal) * weights.x + decodeUnitVector(v1.normal) * weights.y +
                            decodeUnitVector(v2.normal) * weights.z);
    if (dot(normal, direction) > 0.0) {
        normal = -normal;
    }
    Surface surface;
    surface.position = v0.position * weights.x + v1.position * weights.y + v2.position * weights.z;
    surface.normal = normal;
    surface.view = -direction;
    surface.albedo = vec3(0.0);
    surface.metallic = 0.0;
    surface.roughness = clamp(info.waterColor.w, 0.02, 1.0);

    vec3 reflection = environmentLight(camera, surface, 1.0, true) + sampleDirectLight(camera, surface, seed);
    WaterShade shade = shadeWater(reflection,
                                  info.waterColor.rgb,
                                  info.absorption.rgb * info.absorption.w,
                                  WATER_DEFAULT_THICKNESS,
                                  dot(normal, surface.view));
    return applyFog(camera, shade.color, origin, direction, hitDistance);
}

// 히트 표면을 직접광 한 표본 + 환경광으로 셰이딩한다. 재귀는 없다(환경광이 프로브를 읽으면 다중 반사가 한 프레임씩
// 번져 간다).
vec3 shadeHit(Camera camera, vec3 origin, vec3 direction, float hitDistance, uint instanceIndex, uint primitive, vec2 barycentrics, inout uint seed) {
    HitSurface hit = interpolateHit(push.instances,
                                    push.meshes,
                                    push.lods,
                                    push.vertices,
                                    push.skinnedVertices,
                                    push.indices,
                                    instanceIndex,
                                    primitive,
                                    barycentrics);
    Material material = push.materials.items[hit.materialIndex];
    vec3 normal = perturbNormal(material, hit.normal, hit.tangent, hit.tangentSign, hit.uv);
    if (dot(normal, direction) > 0.0) {
        normal = -normal;
    }
    MaterialSample sampled = sampleMaterial(material, hit.uv);

    Surface surface;
    surface.position = hit.position;
    surface.normal = normal;
    surface.view = -direction;
    surface.albedo = sampled.albedo;
    surface.metallic = sampled.metallic;
    surface.roughness = sampled.roughness;

    vec3 color = sampled.emissive + sampleDirectLight(camera, surface, seed);
    color += environmentLight(camera, surface, sampled.occlusion, true);
    return applyFog(camera, color, origin, direction, hitDistance);
}

// 미스는 이 거리로 친다. 시차 재투영과 프로브 가시성이 하늘을 아주 먼 점으로 다루게 된다.
const float MISS_DISTANCE = 1.0e4;

// 컷오프 재질의 알파를 보고 후보를 확정한다. 반투명은 그냥 통과시킨다(ponytail: 유리 뒤의 반사가 유리를 무시한다).
// rayQueryEXT 는 함수 인자로 넘길 수 없어 매크로다.
bool cutoffCandidateOpaque(uint instanceIndex, uint primitive, vec2 barycentrics) {
    Instance instance = push.instances.items[instanceIndex];
    Mesh mesh = push.meshes.items[instance.meshIndex];
    Material material = push.materials.items[mesh.materialIndex];
    if (material.alphaMode != ALPHA_MODE_CUTOFF) {
        return false;
    }
    vec2 uv = interpolateHitUv(push.instances,
                               push.meshes,
                               push.lods,
                               push.vertices,
                               push.skinnedVertices,
                               push.indices,
                               instanceIndex,
                               primitive,
                               barycentrics);
    return materialAlpha(material, uv) >= material.emissiveAndCutoff.w;
}
#define CONFIRM_CUTOFF_CANDIDATES(query) \
    while (rayQueryProceedEXT(query)) { \
        if (rayQueryGetIntersectionTypeEXT(query, false) == gl_RayQueryCandidateIntersectionTriangleEXT && \
            cutoffCandidateOpaque(uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query, false)), \
                                  uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, false)), \
                                  rayQueryGetIntersectionBarycentricsEXT(query, false))) { \
            rayQueryConfirmIntersectionEXT(query); \
        } \
    }

// 확정된 히트를 셰이딩한다. 물 표면 인스턴스는 표식으로 가른다.
vec3 shadeCommittedHit(Camera camera, vec3 origin, vec3 direction, float hitDistance, uint customIndex, uint primitive, vec2 barycentrics, inout uint seed) {
    if ((customIndex & FLUID_SURFACE_CUSTOM_INDEX) != 0u) {
        return shadeWaterHit(camera, origin, direction, hitDistance, customIndex, primitive, barycentrics, seed);
    }
    return shadeHit(camera, origin, direction, hitDistance, customIndex, primitive, barycentrics, seed);
}

// 반사 광선. 단면 재질은 후면을 컬링해 래스터와 같은 면만 보이게 한다. 컷오프·반투명 재질은 가속 구조에
// 불투명으로 올라가지 않아 후보로 온다. 물 표면 인스턴스(마스크 0x01)도 맞힌다. 미스는 프리필터 큐브맵을 거칠기에
// 맞춰 읽어 스페큘러 IBL 이 보던 것과 같은 하늘이다.
vec3 traceReflection(Camera camera, vec3 origin, vec3 normal, vec3 direction, float roughness, inout uint seed, out float hitDistance) {
    rayQueryEXT query;
    rayQueryInitializeEXT(
        query, topLevel, gl_RayFlagsCullBackFacingTrianglesEXT, 0xFFu, origin + normal * 1.0e-3, 1.0e-4, direction, 1.0e4);
    CONFIRM_CUTOFF_CANDIDATES(query)
    if (rayQueryGetIntersectionTypeEXT(query, true) == gl_RayQueryCommittedIntersectionNoneEXT) {
        hitDistance = MISS_DISTANCE;
        uvec4 environment = camera.environment;
        vec3 sky = sampleBindlessCubeLod(environment.y, direction, roughness * float(environment.w - 1u)).rgb;
        return applyFog(camera, sky, origin, direction, 1.0e30);
    }
    hitDistance = rayQueryGetIntersectionTEXT(query, true);
    return shadeCommittedHit(camera,
                             origin,
                             direction,
                             hitDistance,
                             uint(rayQueryGetIntersectionInstanceCustomIndexEXT(query, true)),
                             uint(rayQueryGetIntersectionPrimitiveIndexEXT(query, true)),
                             rayQueryGetIntersectionBarycentricsEXT(query, true),
                             seed);
}

#endif
