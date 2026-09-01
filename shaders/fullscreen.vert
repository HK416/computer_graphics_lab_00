#version 460

layout(location = 0) out vec2 outUv;

// 정점 버퍼 없이 화면을 덮는 삼각형 하나를 만든다.
void main() {
    outUv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUv * 2.0 - 1.0, 0.0, 1.0);
}
