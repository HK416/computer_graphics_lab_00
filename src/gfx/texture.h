#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

#include "asset/model.h"
#include "gfx/resources.h"

namespace gfx {

struct Context;
class BindlessTextures;
class Uploader;

// 텍스처를 GPU 로 올리고 bindless 슬롯에 등록한다. 샘플러는 설정이 같으면 재사용한다.
class TextureCache {
public:
    TextureCache(Context& context, BindlessTextures& bindless);
    ~TextureCache();
    TextureCache(const TextureCache&) = delete;
    TextureCache& operator=(const TextureCache&) = delete;

    // 등록에 성공하면 bindless 슬롯 번호를, 픽셀이 비어 있으면 INVALID_TEXTURE 를 돌려준다.
    uint32_t add(Uploader& uploader, const asset::Texture& texture);

private:
    VkSampler samplerFor(const asset::SamplerDesc& desc);

    Context& context;
    BindlessTextures& bindless;
    std::vector<Image> images;
    std::unordered_map<uint64_t, VkSampler> samplers;
};

} // namespace gfx
