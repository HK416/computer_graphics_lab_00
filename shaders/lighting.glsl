#ifndef LIGHTING_GLSL
#define LIGHTING_GLSL

#include "bindless.glsl"
#include "ibl.glsl"
#include "pbr.glsl"
#include "scene_types.glsl"

// 한 점의 재질과 시선. 조명 종류가 달라도 BRDF 계산은 이 값들만 쓴다.
struct Surface {
    vec3 position;
    vec3 normal;
    vec3 view;
    vec3 albedo;
    float metallic;
    float roughness;
};

vec3 evaluateBrdf(Surface surface, vec3 lightDirection, vec3 radiance) {
    vec3 halfway = normalize(surface.view + lightDirection);
    float nDotL = max(dot(surface.normal, lightDirection), 0.0);
    if (nDotL <= 0.0) {
        return vec3(0.0);
    }
    float nDotV = max(dot(surface.normal, surface.view), 1e-4);
    float nDotH = max(dot(surface.normal, halfway), 0.0);
    float vDotH = max(dot(surface.view, halfway), 0.0);

    vec3 f0 = mix(vec3(0.04), surface.albedo, surface.metallic);
    vec3 fresnel = fresnelSchlick(vDotH, f0);
    float distribution = distributionGgx(nDotH, surface.roughness);
    float geometry = geometrySmith(nDotV, nDotL, surface.roughness);

    vec3 specular = (distribution * geometry * fresnel) / max(4.0 * nDotV * nDotL, 1e-4);
    vec3 diffuse = (vec3(1.0) - fresnel) * (1.0 - surface.metallic) * surface.albedo / PI;
    return (diffuse + specular) * radiance * nDotL;
}

// 도달 거리에서 부드럽게 0 이 되는 역제곱 감쇠 (UE4 의 창 함수).
float distanceAttenuation(float distanceToLight, float range) {
    float windowed = clamp(1.0 - pow(distanceToLight / max(range, 1e-3), 4.0), 0.0, 1.0);
    return (windowed * windowed) / max(distanceToLight * distanceToLight, 1e-4);
}

vec3 closestPointOnRectangle(vec3 point, vec3 center, vec3 right, vec3 up, vec2 halfSize) {
    vec3 offset = point - center;
    float x = clamp(dot(offset, right), -halfSize.x, halfSize.x);
    float y = clamp(dot(offset, up), -halfSize.y, halfSize.y);
    return center + right * x + up * y;
}

// 직사각형 영역광의 대표점. 반사 벡터가 광원 평면과 만나는 점을 사각형 안으로 자른다.
//
// ponytail: Karis 의 대표점 근사라 넓은 광원의 반사가 조금 좁게 나온다. 더 정확히 하려면
// 선형 변환 코사인(LTC) 표를 텍스처로 올려야 한다.
vec3 rectangleRepresentativePoint(Surface surface, Light light, vec3 center, vec3 right, vec3 up, vec2 halfSize) {
    vec3 planeNormal = light.directionIntensity.xyz;
    vec3 reflection = reflect(-surface.view, surface.normal);
    float denominator = dot(reflection, planeNormal);
    if (abs(denominator) > 1e-4) {
        float travel = dot(center - surface.position, planeNormal) / denominator;
        if (travel > 0.0) {
            vec3 hit = surface.position + reflection * travel;
            return closestPointOnRectangle(hit, center, right, up, halfSize);
        }
    }
    return closestPointOnRectangle(surface.position, center, right, up, halfSize);
}

// 광원 하나에서 표면으로 오는 방향과 거리, 도달 복사휘도를 구한다. 확산만 쓰는 경로 추적의
// 다음 사건 추정이 쓴다. 영역광은 가장 가까운 점으로 근사한다.
vec3 sampleLightRadiance(Light light, vec3 position, out vec3 lightDirection, out float distanceToLight) {
    uint type = uint(light.colorType.w);
    vec3 radiance = light.colorType.rgb * light.directionIntensity.w;

    if (type == LIGHT_TYPE_DIRECTIONAL) {
        lightDirection = -light.directionIntensity.xyz;
        distanceToLight = 1.0e30;
        return radiance;
    }

    vec3 lightPosition = light.positionRange.xyz;
    if (type == LIGHT_TYPE_AREA) {
        float facing = dot(normalize(position - lightPosition), light.directionIntensity.xyz);
        if (facing <= 0.0) {
            lightDirection = vec3(0.0, 1.0, 0.0);
            distanceToLight = 0.0;
            return vec3(0.0);
        }
        lightPosition = closestPointOnRectangle(
            position, lightPosition, light.rightShadow.xyz, light.up.xyz, light.coneSize.zw);
        radiance *= facing;
    }

    vec3 offset = lightPosition - position;
    distanceToLight = length(offset);
    lightDirection = offset / max(distanceToLight, 1e-4);
    radiance *= distanceAttenuation(distanceToLight, light.positionRange.w);

    if (type == LIGHT_TYPE_SPOT) {
        float aligned = dot(-lightDirection, light.directionIntensity.xyz);
        float falloff = clamp((aligned - light.coneSize.y) / max(light.coneSize.x - light.coneSize.y, 1e-4), 0.0, 1.0);
        radiance *= falloff * falloff;
    }
    return radiance;
}

// split-sum 의 스페큘러 가중 F·A + B. 프리필터 큐브맵이나 추적한 반사 색에 곱한다.
vec3 specularAlbedo(Camera camera, Surface surface) {
    float nDotV = max(dot(surface.normal, surface.view), 1e-4);
    vec3 f0 = mix(vec3(0.04), surface.albedo, surface.metallic);
    vec3 fresnel = fresnelSchlickRoughness(nDotV, f0, surface.roughness);
    vec2 integrated = sampleBindlessArray(camera.environment.z, vec2(nDotV, surface.roughness), 0.0).rg;
    return fresnel * integrated.x + integrated.y;
}

// 환경광. 프리필터 밉 수가 0 이면 IBL 이 꺼진 것이라 균일 환경광만 남긴다. ambient 는 IBL 에
// 곱하는 색조 겸 세기로 계속 쓰인다. includeSpecular 를 끄면 확산만 돌려주고, 스페큘러는 부르는
// 쪽이 추적한 반사에 specularAlbedo 를 곱해 대신 넣는다.
vec3 environmentLight(Camera camera, Surface surface, float occlusion, bool includeSpecular) {
    vec3 tint = camera.ambient.rgb;
    uvec4 environment = camera.environment;
    if (environment.w == 0u) {
        return tint * surface.albedo * occlusion;
    }

    float nDotV = max(dot(surface.normal, surface.view), 1e-4);
    vec3 f0 = mix(vec3(0.04), surface.albedo, surface.metallic);
    vec3 fresnel = fresnelSchlickRoughness(nDotV, f0, surface.roughness);
    vec3 diffuseWeight = (vec3(1.0) - fresnel) * (1.0 - surface.metallic);

    vec3 irradiance = sampleBindlessCube(environment.x, surface.normal).rgb;
    vec3 diffuse = diffuseWeight * irradiance * surface.albedo;

    vec3 specular = vec3(0.0);
    if (includeSpecular) {
        vec3 reflection = reflect(-surface.view, surface.normal);
        vec3 prefiltered =
            sampleBindlessCubeLod(environment.y, reflection, surface.roughness * float(environment.w - 1u)).rgb;
        vec2 integrated = sampleBindlessArray(environment.z, vec2(nDotV, surface.roughness), 0.0).rg;
        specular = prefiltered * (fresnel * integrated.x + integrated.y);
    }
    return (diffuse + specular) * occlusion * tint;
}

// 그림자를 뺀 조명 하나의 기여. attenuation 은 그림자 계산에도 쓰라고 따로 돌려준다.
vec3 lightContribution(Light light, Surface surface, out vec3 lightDirection) {
    uint type = uint(light.colorType.w);
    vec3 radiance = light.colorType.rgb * light.directionIntensity.w;

    if (type == LIGHT_TYPE_DIRECTIONAL) {
        lightDirection = -light.directionIntensity.xyz;
        return evaluateBrdf(surface, lightDirection, radiance);
    }

    vec3 lightPosition = light.positionRange.xyz;
    if (type == LIGHT_TYPE_AREA) {
        // 사각형 뒷면은 빛을 내지 않는다.
        vec3 toSurface = normalize(surface.position - lightPosition);
        float facing = dot(toSurface, light.directionIntensity.xyz);
        if (facing <= 0.0) {
            lightDirection = vec3(0.0, 1.0, 0.0);
            return vec3(0.0);
        }
        lightPosition = rectangleRepresentativePoint(
            surface, light, lightPosition, light.rightShadow.xyz, light.up.xyz, light.coneSize.zw);
        radiance *= facing;
    }

    vec3 offset = lightPosition - surface.position;
    float distanceToLight = length(offset);
    lightDirection = offset / max(distanceToLight, 1e-4);
    radiance *= distanceAttenuation(distanceToLight, light.positionRange.w);

    if (type == LIGHT_TYPE_SPOT) {
        float aligned = dot(-lightDirection, light.directionIntensity.xyz);
        float falloff = clamp((aligned - light.coneSize.y) / max(light.coneSize.x - light.coneSize.y, 1e-4), 0.0, 1.0);
        radiance *= falloff * falloff;
    }
    return evaluateBrdf(surface, lightDirection, radiance);
}

#endif
