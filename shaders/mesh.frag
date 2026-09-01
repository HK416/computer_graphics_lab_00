#version 460
#include "mesh_shading.glsl"

layout(location = 0) out vec4 outColor;

void main() {
    outColor = shadeSurface();
}
