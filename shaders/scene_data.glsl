#ifndef SCENE_DATA_GLSL
#define SCENE_DATA_GLSL

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require

// combined image sampler 는 샘플러 한도에도 함께 걸리므로 이미지와 샘플러 배열을 따로 둔다.
// 재질이 들고 있는 슬롯은 하위 24비트가 이미지, 상위 8비트가 샘플러 인덱스다.
layout(set = 0, binding = 0) uniform texture2D bindlessImages[];
layout(set = 0, binding = 1) uniform sampler bindlessSamplers[];

vec4 sampleBindless(uint slot, vec2 uv) {
    uint imageIndex = slot & 0x00FFFFFFu;
    uint samplerIndex = slot >> 24;
    return texture(sampler2D(bindlessImages[nonuniformEXT(imageIndex)], bindlessSamplers[nonuniformEXT(samplerIndex)]),
                   uv);
}

// 아래 구조체는 src/gfx/geometry.h 및 src/asset/model.h 의 정의와 배치가 일치해야 한다.
struct Vertex {
    vec3 position;
    vec3 normal;
    vec4 tangent;
    vec2 uv;
};

struct Mesh {
    vec4 boundingSphere;
    uint indexOffset;
    uint indexCount;
    int vertexOffset;
    uint materialIndex;
};

#define INVALID_TEXTURE 0xFFFFFFFFu

#define ALPHA_MODE_SOLID 0u
#define ALPHA_MODE_CUTOFF 1u
#define ALPHA_MODE_TRANSLUCENT 2u

struct Material {
    vec4 baseColorFactor;
    vec4 emissiveAndCutoff;
    float metallicFactor;
    float roughnessFactor;
    float normalScale;
    float occlusionStrength;
    uint baseColorTexture;
    uint metallicRoughnessTexture;
    uint normalTexture;
    uint occlusionTexture;
    uint emissiveTexture;
    uint alphaMode;
    uint flags;
    uint padding;
};

struct Instance {
    mat4 model;
    mat4 normalMatrix;
    uint meshIndex;
    uint padding0;
    uint padding1;
    uint padding2;
};

struct Camera {
    mat4 viewProjection;
    vec4 position;
};

layout(buffer_reference, scalar) readonly buffer VertexBuffer { Vertex items[]; };
layout(buffer_reference, scalar) readonly buffer MeshBuffer { Mesh items[]; };
layout(buffer_reference, scalar) readonly buffer InstanceBuffer { Instance items[]; };
layout(buffer_reference, scalar) readonly buffer MaterialBuffer { Material items[]; };
layout(buffer_reference, scalar) readonly buffer CameraBuffer { Camera item; };

layout(push_constant) uniform PushConstants {
    VertexBuffer vertices;
    MeshBuffer meshes;
    InstanceBuffer instances;
    MaterialBuffer materials;
    CameraBuffer camera;
} pushConstants;

#endif
