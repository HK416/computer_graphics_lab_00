#version 460
#include "mesh_shading.glsl"

layout(location = 0) out vec4 outColor;
// 화면 UV 단위 모션 벡터. 업스케일러가 해상도에 맞춰 다시 잰다.
layout(location = 1) out vec2 outVelocity;

// 디버그 모드는 셰이딩 대신 중간 값을 그대로 보여준다.
vec4 debugColor() {
    switch (pushConstants.debugMode) {
    case DEBUG_MODE_MESHLET:
        return vec4(debugPalette(inMeshletIndex), 1.0);
    case DEBUG_MODE_NORMAL:
        return vec4(normalize(inNormal) * 0.5 + 0.5, 1.0);
    case DEBUG_MODE_LOD:
        return vec4(debugPalette(pushConstants.meshlets.items[inMeshletIndex].level * 977u + 13u), 1.0);
    case DEBUG_MODE_CASCADE: {
        // 첫 방향광 기준으로 이 픽셀이 어느 캐스케이드를 쓰는지 보여준다.
        uint lightCount = pushConstants.camera.item.shading.x;
        for (uint i = 0; i < lightCount; ++i) {
            Light light = pushConstants.lights.items[i];
            if (uint(light.colorType.w) == LIGHT_TYPE_DIRECTIONAL) {
                return vec4(debugPalette(shadowCascadeIndex(light, inWorldPosition) * 613u + 7u), 1.0);
            }
        }
        return vec4(0.0);
    }
    case DEBUG_MODE_SHADOW: {
        // 첫 그림자 조명의 가시성만 회색조로 보여준다.
        uint lightCount = pushConstants.camera.item.shading.x;
        for (uint i = 0; i < lightCount; ++i) {
            Light light = pushConstants.lights.items[i];
            if (light.rightShadow.w >= 0.0) {
                vec3 normal = normalize(inNormal);
                vec3 toLight = uint(light.colorType.w) == LIGHT_TYPE_DIRECTIONAL
                                   ? -light.directionIntensity.xyz
                                   : normalize(light.positionRange.xyz - inWorldPosition);
                return vec4(vec3(shadowFactor(light, inWorldPosition, normal, toLight)), 1.0);
            }
        }
        return vec4(1.0, 0.0, 1.0, 1.0);
    }
    case DEBUG_MODE_VELOCITY: {
        // 픽셀 단위 변위를 8 픽셀에서 포화시켜 본다. 정지 화면은 회색이다.
        vec2 pixels = motionVector() * pushConstants.camera.item.viewport.xy / 8.0;
        return vec4(clamp(pixels * 0.5 + 0.5, 0.0, 1.0), 0.5, 1.0);
    }
    case DEBUG_MODE_UV:
        return vec4(fract(inUv), 0.0, 1.0);
    case DEBUG_MODE_DEPTH: {
        // reverse-Z 라서 깊이 값이 근평면 근처에 몰린다. 시야 공간 거리로 되돌려 보여준다.
        float viewDepth = pushConstants.camera.item.parameters.x / max(gl_FragCoord.z, 1e-6);
        return vec4(vec3(viewDepth / (viewDepth + 10.0)), 1.0);
    }
    default:
        return vec4(0.0);
    }
}

void main() {
    outVelocity = motionVector();
    if (pushConstants.debugMode != DEBUG_MODE_SHADED) {
        outColor = debugColor();
        return;
    }
    outColor = shadeSurface();
}
