#version 460
#include "mesh_shading.glsl"

layout(location = 0) out vec4 outColor;

// 디버그 모드는 셰이딩 대신 중간 값을 그대로 보여준다.
vec4 debugColor() {
    switch (pushConstants.debugMode) {
    case DEBUG_MODE_MESHLET:
        return vec4(debugPalette(inMeshletIndex), 1.0);
    case DEBUG_MODE_NORMAL:
        return vec4(normalize(inNormal) * 0.5 + 0.5, 1.0);
    case DEBUG_MODE_LOD:
        return vec4(debugPalette(pushConstants.meshlets.items[inMeshletIndex].level * 977u + 13u), 1.0);
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
    if (pushConstants.debugMode != DEBUG_MODE_SHADED) {
        outColor = debugColor();
        return;
    }
    outColor = shadeSurface();
}
