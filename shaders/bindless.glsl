#ifndef BINDLESS_GLSL
#define BINDLESS_GLSL

#extension GL_EXT_nonuniform_qualifier : require

// combined image sampler 는 샘플러 한도에도 함께 걸리므로 이미지와 샘플러 배열을 따로 둔다.
// 슬롯은 하위 24비트가 이미지, 상위 8비트가 샘플러 인덱스다.
layout(set = 0, binding = 0) uniform texture2D bindlessImages[];
layout(set = 0, binding = 1) uniform sampler bindlessSamplers[];
layout(set = 0, binding = 2, r32f) uniform image2D bindlessStorageImages[];
layout(set = 0, binding = 3, rgba32f) uniform image2D bindlessStorageImagesRgba[];

vec4 sampleBindless(uint slot, vec2 uv) {
    uint imageIndex = slot & 0x00FFFFFFu;
    uint samplerIndex = slot >> 24;
    return texture(sampler2D(bindlessImages[nonuniformEXT(imageIndex)], bindlessSamplers[nonuniformEXT(samplerIndex)]),
                   uv);
}

// 명시적 밉 단계로 읽는다. HZB 처럼 단계를 직접 고르는 곳에서 쓴다.
vec4 fetchBindlessLod(uint slot, vec2 uv, float level) {
    uint imageIndex = slot & 0x00FFFFFFu;
    uint samplerIndex = slot >> 24;
    return textureLod(
        sampler2D(bindlessImages[nonuniformEXT(imageIndex)], bindlessSamplers[nonuniformEXT(samplerIndex)]), uv, level);
}

#endif
