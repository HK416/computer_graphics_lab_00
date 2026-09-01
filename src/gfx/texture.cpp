#include "gfx/texture.h"

#include <algorithm>
#include <bit>
#include <cmath>

#include "core/error.h"
#include "gfx/bindless.h"
#include "gfx/context.h"
#include "gfx/uploader.h"
#include "gfx/vk_check.h"

namespace gfx {
namespace {

VkFilter toFilter(uint32_t gltfFilter) {
    // glTF 의 NEAREST 계열은 9728, 9984, 9986 이다.
    switch (gltfFilter) {
    case 9728:
    case 9984:
    case 9986:
        return VK_FILTER_NEAREST;
    default:
        return VK_FILTER_LINEAR;
    }
}

VkSamplerMipmapMode toMipmapMode(uint32_t gltfMinFilter) {
    // *_MIPMAP_NEAREST 는 9984, 9985 이다.
    switch (gltfMinFilter) {
    case 9984:
    case 9985:
        return VK_SAMPLER_MIPMAP_MODE_NEAREST;
    default:
        return VK_SAMPLER_MIPMAP_MODE_LINEAR;
    }
}

VkSamplerAddressMode toAddressMode(uint32_t gltfWrap) {
    switch (gltfWrap) {
    case 33071:
        return VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    case 33648:
        return VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT;
    default:
        return VK_SAMPLER_ADDRESS_MODE_REPEAT;
    }
}

uint64_t samplerKey(const asset::SamplerDesc& desc) {
    return (static_cast<uint64_t>(desc.magFilter) << 48) | (static_cast<uint64_t>(desc.minFilter) << 32) |
           (static_cast<uint64_t>(desc.wrapS) << 16) | static_cast<uint64_t>(desc.wrapT);
}

uint32_t mipLevelsFor(uint32_t width, uint32_t height) {
    return 1U + static_cast<uint32_t>(std::bit_width(std::max(width, height))) - 1U;
}

} // namespace

TextureCache::TextureCache(Context& context, BindlessTextures& bindless) : context(context), bindless(bindless) {}

TextureCache::~TextureCache() {
    for (auto& entry : samplers) {
        vkDestroySampler(context.device, entry.second, nullptr);
    }
    for (Image& image : images) {
        destroyImage(context, image);
    }
}

VkSampler TextureCache::samplerFor(const asset::SamplerDesc& desc) {
    uint64_t lookupKey = samplerKey(desc);
    auto found = samplers.find(lookupKey);
    if (found != samplers.end()) {
        return found->second;
    }

    VkSamplerCreateInfo info{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    info.magFilter = toFilter(desc.magFilter);
    info.minFilter = toFilter(desc.minFilter);
    info.mipmapMode = toMipmapMode(desc.minFilter);
    info.addressModeU = toAddressMode(desc.wrapS);
    info.addressModeV = toAddressMode(desc.wrapT);
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    info.anisotropyEnable = VK_TRUE;
    info.maxAnisotropy = std::min(16.0F, context.properties.limits.maxSamplerAnisotropy);
    info.maxLod = VK_LOD_CLAMP_NONE;

    VkSampler sampler = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSampler(context.device, &info, nullptr, &sampler));
    samplers.emplace(lookupKey, sampler);
    return sampler;
}

uint32_t TextureCache::add(Uploader& uploader, const asset::Texture& texture) {
    if (texture.pixels.empty() || texture.width == 0 || texture.height == 0) {
        return asset::INVALID_TEXTURE;
    }

    ImageDesc desc;
    desc.extent = {texture.width, texture.height, 1};
    desc.format = texture.srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    desc.mipLevels = mipLevelsFor(texture.width, texture.height);
    desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    Image image = createImage(context, desc, texture.name.c_str());
    uploader.uploadImage(image, texture.pixels.data(), texture.pixels.size(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    uint32_t slot = bindless.add(image.view, samplerFor(texture.sampler));
    images.push_back(image);
    return slot;
}

} // namespace gfx
