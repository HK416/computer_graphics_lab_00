#ifndef CULLING_GLSL
#define CULLING_GLSL

#include "lod_network.glsl"
#include "scene_types.glsl"

// 뷰 프로젝션 행렬에서 절두체 평면을 뽑는다 (Gribb-Hartmann).
// 무한 원거리 reverse-Z 에서는 원거리 평면이 존재하지 않고, row3 - row2 가 근평면이 된다.
void extractFrustumPlanes(mat4 viewProjection, out vec4 planes[5]) {
    vec4 row0 = vec4(viewProjection[0][0], viewProjection[1][0], viewProjection[2][0], viewProjection[3][0]);
    vec4 row1 = vec4(viewProjection[0][1], viewProjection[1][1], viewProjection[2][1], viewProjection[3][1]);
    vec4 row2 = vec4(viewProjection[0][2], viewProjection[1][2], viewProjection[2][2], viewProjection[3][2]);
    vec4 row3 = vec4(viewProjection[0][3], viewProjection[1][3], viewProjection[2][3], viewProjection[3][3]);

    planes[0] = row3 + row0; // 좌
    planes[1] = row3 - row0; // 우
    planes[2] = row3 + row1; // 하
    planes[3] = row3 - row1; // 상
    planes[4] = row3 - row2; // 근

    for (int i = 0; i < 5; ++i) {
        planes[i] /= max(length(planes[i].xyz), 1e-8);
    }
}

bool sphereInFrustum(vec4 planes[5], vec3 center, float radius) {
    for (int i = 0; i < 5; ++i) {
        if (dot(planes[i].xyz, center) + planes[i].w < -radius) {
            return false;
        }
    }
    return true;
}

// meshoptimizer 가 만든 법선 원뿔로 후면을 통째로 버린다.
// 원뿔 컷오프가 1 이상이면 방향이 흩어진 meshlet 이라 판정하지 않는다.
bool coneVisible(vec3 center, float radius, vec3 coneAxis, float coneCutoff, vec3 cameraPosition) {
    if (coneCutoff >= 1.0) {
        return true;
    }
    vec3 toMeshlet = center - cameraPosition;
    return dot(toMeshlet, coneAxis) < coneCutoff * length(toMeshlet) + radius;
}

// 인스턴스 변환을 반영한 월드 공간 바운딩 스피어. 비균등 스케일은 최대 성분으로 보수적으로 잡는다.
vec4 transformBoundingSphere(mat4 model, vec4 sphere) {
    vec3 center = (model * vec4(sphere.xyz, 1.0)).xyz;
    float scale = sqrt(
        max(max(dot(model[0].xyz, model[0].xyz), dot(model[1].xyz, model[1].xyz)), dot(model[2].xyz, model[2].xyz)));
    return vec4(center, sphere.w * scale);
}

// 화면 공간 오차. 부모가 없는 meshlet 은 오차가 무한대로 들어와 항상 임계값을 넘는다.
float projectedError(vec3 cameraPosition, float projectionScale, vec4 sphere, float error) {
    if (isinf(error) || error > 3.0e37) {
        return 1.0e38;
    }
    float viewDistance = max(length(sphere.xyz - cameraPosition) - sphere.w, 1e-3);
    return error * projectionScale / viewDistance;
}

// 자신의 오차는 허용되고 부모의 오차는 허용되지 않는 meshlet 만 그린다. 같은 그룹은 같은 판정을
// 받으므로 서로 다른 단계가 맞닿아도 틈이 생기지 않는다.
// 신경망 보정을 넣어도 입력이 (경계 구, 오차) 쌍뿐이므로, 자식의 부모 판정과 부모의 자기 판정이
// 같은 값을 내놓아 LOD 경계에 틈이 생기지 않는다.
float biasedProjectedError(vec3 cameraPosition,
                           float projectionScale,
                           vec4 sphere,
                           float error,
                           bool useNetwork,
                           LodNetwork network) {
    float projected = projectedError(cameraPosition, projectionScale, sphere, error);
    if (!useNetwork || projected > 1.0e37) {
        return projected;
    }
    float viewDistance = max(length(sphere.xyz - cameraPosition) - sphere.w, 1e-3);
    float features[LOD_NETWORK_INPUTS];
    lodNetworkFeatures(viewDistance, sphere.w, error, projected, features);
    return projected * exp(lodNetworkBias(network, features));
}

bool selectLod(Meshlet meshlet,
               mat4 model,
               vec3 cameraPosition,
               float projectionScale,
               float threshold,
               float forcedLevel,
               bool useNetwork,
               LodNetwork network) {
    if (forcedLevel >= 0.0) {
        return float(meshlet.level) == forcedLevel;
    }
    vec4 ownSphere = transformBoundingSphere(model, meshlet.errorSphere);
    vec4 parentSphere = transformBoundingSphere(model, meshlet.parentSphere);
    float own = biasedProjectedError(cameraPosition, projectionScale, ownSphere, meshlet.error, useNetwork, network);
    float parent =
        biasedProjectedError(cameraPosition, projectionScale, parentSphere, meshlet.parentError, useNetwork, network);
    return own <= threshold && parent > threshold;
}

// 이전 프레임 HZB 로 가림 여부를 판정한다. 바운딩 상자의 여덟 꼭짓점을 투영해 화면 사각형을 잡고,
// 그 크기에 맞는 밉 단계에서 가장 먼 깊이를 읽어 비교한다.
//
// ponytail: 이전 프레임 깊이만 쓰는 한 패스 방식이라 카메라가 빠르게 움직이면 한 프레임 늦게 나타난다.
// 완전히 없애려면 이 프레임 깊이로 다시 판정하는 두 패스 방식으로 올려야 한다.
bool occludedByHzb(mat4 viewProjection, vec4 sphere, uint hzbTexture, vec2 hzbSize, float hzbMaxLevel) {
    vec2 minUv = vec2(1.0e9);
    vec2 maxUv = vec2(-1.0e9);
    float nearestDepth = 0.0;

    for (int i = 0; i < 8; ++i) {
        vec3 offset = vec3((i & 1) != 0 ? 1.0 : -1.0, (i & 2) != 0 ? 1.0 : -1.0, (i & 4) != 0 ? 1.0 : -1.0);
        vec4 clip = viewProjection * vec4(sphere.xyz + offset * sphere.w, 1.0);
        if (clip.w <= 0.0) {
            // 근평면을 물고 있으면 화면 사각형을 신뢰할 수 없으므로 가림 판정을 건너뛴다.
            return false;
        }
        vec3 ndc = clip.xyz / clip.w;
        vec2 uv = ndc.xy * 0.5 + 0.5;
        minUv = min(minUv, uv);
        maxUv = max(maxUv, uv);
        nearestDepth = max(nearestDepth, ndc.z);
    }

    minUv = clamp(minUv, vec2(0.0), vec2(1.0));
    maxUv = clamp(maxUv, vec2(0.0), vec2(1.0));

    vec2 sizeInTexels = (maxUv - minUv) * hzbSize;
    float level = clamp(ceil(log2(max(max(sizeInTexels.x, sizeInTexels.y), 1.0))), 0.0, hzbMaxLevel);

    float farthest = fetchBindlessLod(hzbTexture, vec2(minUv.x, minUv.y), level).r;
    farthest = min(farthest, fetchBindlessLod(hzbTexture, vec2(maxUv.x, minUv.y), level).r);
    farthest = min(farthest, fetchBindlessLod(hzbTexture, vec2(minUv.x, maxUv.y), level).r);
    farthest = min(farthest, fetchBindlessLod(hzbTexture, vec2(maxUv.x, maxUv.y), level).r);

    return nearestDepth < farthest;
}

#endif
