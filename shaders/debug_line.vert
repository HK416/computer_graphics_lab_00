#version 460

#include "debug_line_common.glsl"

layout(location = 0) out vec4 outColor;

void main() {
    DebugLineVertex vertex = push.vertices.items[gl_VertexIndex];
    gl_Position = push.viewProjection * vec4(vertex.position, 1.0);
    outColor = unpackUnorm4x8(vertex.color);
}
