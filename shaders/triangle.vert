#version 460

layout(location = 0) out vec3 outColor;

// 정점 버퍼 없이 정점 인덱스만으로 삼각형을 만든다. 파이프라인 점검용이다.
const vec2 POSITIONS[3] = vec2[](vec2(0.0, -0.6), vec2(0.6, 0.6), vec2(-0.6, 0.6));
const vec3 COLORS[3] = vec3[](vec3(1.0, 0.2, 0.2), vec3(0.2, 1.0, 0.2), vec3(0.2, 0.4, 1.0));

void main() {
    gl_Position = vec4(POSITIONS[gl_VertexIndex], 0.0, 1.0);
    outColor = COLORS[gl_VertexIndex];
}
