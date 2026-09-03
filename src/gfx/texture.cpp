#include "gfx/texture.h"

#include <algorithm>
#include <bit>
#include <cmath>

#include <spdlog/spdlog.h>

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

// 색 공간은 재질 슬롯이 정한 srgb 로 붙인다. 데이터는 같고 해석만 다르다. BC4/BC5 는 선형뿐이다.
VkFormat toVkFormat(asset::TextureFormat format, bool srgb) {
    switch (format) {
    case asset::TextureFormat::BC1:
        // RGB 와 RGBA 변종은 인코딩이 같다. RGBA 로 두면 1비트 알파가 있는 파일도 컷오프에 쓸 수 있다.
        return srgb ? VK_FORMAT_BC1_RGBA_SRGB_BLOCK : VK_FORMAT_BC1_RGBA_UNORM_BLOCK;
    case asset::TextureFormat::BC3:
        return srgb ? VK_FORMAT_BC3_SRGB_BLOCK : VK_FORMAT_BC3_UNORM_BLOCK;
    case asset::TextureFormat::BC4:
        return VK_FORMAT_BC4_UNORM_BLOCK;
    case asset::TextureFormat::BC5:
        return VK_FORMAT_BC5_UNORM_BLOCK;
    case asset::TextureFormat::BC7:
        return srgb ? VK_FORMAT_BC7_SRGB_BLOCK : VK_FORMAT_BC7_UNORM_BLOCK;
    default:
        return srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    }
}

} // namespace

TextureCache::TextureCache(Context& context, BindlessTextures& bindless) : context(context), bindless(bindless) {}

TextureCache::~TextureCache() {
    for (auto& entry : samplers) {
        vkDestroySampler(context.device, entry.second, nullptr);
    }
    for (auto& entry : images) {
        destroyImage(context, entry.second);
    }
}

void TextureCache::remove(uint32_t slot) {
    auto found = images.find(slot);
    if (found == images.end()) {
        return;
    }
    destroyImage(context, found->second);
    images.erase(found);
    bindless.freeImage(slot);
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

VkDeviceSize TextureCache::estimateBytes(const asset::Texture& texture) {
    VkDeviceSize bytes = texture.pixels.size();
    bool prebuiltMips = texture.mipLevels > 1 || asset::isBlockCompressed(texture.format);
    return prebuiltMips ? bytes : bytes + bytes / 3;
}

uint32_t TextureCache::add(Uploader& uploader, const asset::Texture& texture) {
    if (texture.pixels.empty() || texture.width == 0 || texture.height == 0) {
        return asset::INVALID_TEXTURE;
    }

    bool compressed = asset::isBlockCompressed(texture.format);
    if (compressed && !context.caps.textureCompressionBc) {
        // 폴백은 두지 않는다. 장치가 못 읽는 포맷이면 텍스처를 빼고 사유를 남긴다.
        spdlog::warn("이 장치는 BC 텍스처를 지원하지 않아 텍스처를 뺍니다: {}", texture.name);
        return asset::INVALID_TEXTURE;
    }

    ImageDesc desc;
    desc.extent = {texture.width, texture.height, 1};
    desc.format = toVkFormat(texture.format, texture.srgb);
    // 밉이 담겨 오면 그대로 쓴다. 압축 포맷은 블릿을 못 하므로 담긴 만큼이 전부다.
    bool prebuiltMips = texture.mipLevels > 1 || compressed;
    desc.mipLevels = prebuiltMips ? texture.mipLevels : mipLevelsFor(texture.width, texture.height);
    desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    Image image = createImage(context, desc, texture.name.c_str());
    if (prebuiltMips) {
        std::vector<VkDeviceSize> levelBytes(texture.mipLevels);
        for (uint32_t level = 0; level < texture.mipLevels; ++level) {
            levelBytes[level] = asset::textureLevelBytes(
                texture.format, std::max(texture.width >> level, 1U), std::max(texture.height >> level, 1U));
        }
        uploader.uploadImageLevels(image, texture.pixels.data(), levelBytes, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    } else {
        uploader.uploadImage(
            image, texture.pixels.data(), texture.pixels.size(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }

    uint32_t slot = bindless.add(image.view, samplerFor(texture.sampler));
    images.emplace(slot, image);
    return slot;
}

} // namespace gfx
