#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

layout(location = 0) out vec3 outColor;

struct Vertex {
    vec2 position;
    vec3 color;
};

layout(buffer_reference, scalar) readonly buffer VertexBuffer {
    Vertex vertices[];
};

layout(push_constant) uniform PushConstants {
    VertexBuffer vertexBuffer;
} pushConstants;

void main() {
    Vertex vertex = pushConstants.vertexBuffer.vertices[gl_VertexIndex];
    gl_Position = vec4(vertex.position, 0.0, 1.0);
    outColor = vertex.color;
}
