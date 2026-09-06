#ifndef PARTICLE_COMMON_GLSL
#define PARTICLE_COMMON_GLSL

#include "scene_types.glsl"

// 아래 셋은 src/gfx/particles.h 의 GpuParticle / GpuParticleParams / ParticlePushConstants 와 배치가 같아야
// 한다(scalar). 진행 컴퓨트와 스프라이트 정점·프래그먼트가 같은 블록을 쓴다.

#define PARTICLE_GROUP_SIZE 128u
#define PARTICLE_FLAG_COLLIDE 1u

struct Particle {
    vec3 position;
    // 살아온 시간. lifetime 이상이면 죽은 것이다.
    float age;
    vec3 velocity;
    float lifetime;
};

struct ParticleParams {
    mat4 emitterWorld;
    vec4 color;           // rgb 색, w 알파
    vec4 emissive;        // rgb 발광, w 반발 계수
    vec4 sizeSpeedSpread; // x 시작 지름, y 끝 지름, z 초기 속력, w 원뿔 반각의 코사인
    vec4 gravityDrag;     // xyz 중력, w 항력
    uint particleCount;
    uint spawnFirst;
    uint spawnCount;
    uint frameIndex;
    float dt;
    float lifetime;
    uint flags;
    uint pad0;
    VertexBuffer vertices;
    VertexBuffer skinnedVertices;
    IndexBuffer indices;
    MeshBuffer meshes;
    MeshLodBuffer lods;
    InstanceBuffer instances;
};

layout(buffer_reference, scalar) buffer ParticleBuffer {
    Particle items[];
};
layout(buffer_reference, scalar) readonly buffer ParticleParamsBuffer {
    ParticleParams item;
};

layout(push_constant, scalar) uniform ParticlePushConstants {
    ParticleBuffer particles;
    ParticleParamsBuffer params;
    CameraBuffer camera;
    uint depthTexture;
    uint particleCount;
}
push;

#endif
