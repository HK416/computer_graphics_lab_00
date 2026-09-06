#version 460

#include "particle_common.glsl"

layout(location = 0) out vec2 outUv;
layout(location = 1) out vec3 outWorldPosition;
layout(location = 2) out vec4 outColor;
layout(location = 3) out float outSize;

// 입자마다 카메라를 향한 사각형 하나. 인스턴스 번호가 입자, 정점 번호 0..5 가 모서리다. 죽은 입자는 퇴화
// 삼각형으로 내보내 프래그먼트가 나오지 않게 한다.
void main() {
    Particle particle = push.particles.items[gl_InstanceIndex];
    if (particle.age >= particle.lifetime) {
        gl_Position = vec4(0.0);
        outUv = vec2(0.0);
        outWorldPosition = vec3(0.0);
        outColor = vec4(0.0);
        outSize = 0.0;
        return;
    }
    ParticleParams params = push.params.item;
    Camera camera = push.camera.item;

    const vec2 CORNERS[6] = vec2[6](vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0));
    vec2 corner = CORNERS[gl_VertexIndex];
    float t = clamp(particle.age / particle.lifetime, 0.0, 1.0);
    float size = mix(params.sizeSpeedSpread.x, params.sizeSpeedSpread.y, t);

    vec3 toCamera = camera.position.xyz - particle.position;
    vec3 viewDirection = normalize(toCamera);
    vec3 up = abs(viewDirection.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 right = normalize(cross(up, viewDirection));
    vec3 billboardUp = cross(viewDirection, right);
    vec3 world = particle.position + (right * corner.x + billboardUp * corner.y) * (0.5 * size);

    outUv = corner;
    outWorldPosition = world;
    // 알파는 나이에 따라 0 으로 줄어든다. 색은 미리 곱하지 않은 채 넘기고 프래그먼트가 곱한다.
    outColor = vec4(params.color.rgb + params.emissive.rgb, params.color.a * (1.0 - t));
    outSize = size;
    gl_Position = camera.viewProjection * vec4(world, 1.0);
}
