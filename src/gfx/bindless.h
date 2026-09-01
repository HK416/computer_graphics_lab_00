#pragma once

#include <cstdint>

#include <vulkan/vulkan.h>

namespace gfx {

struct Context;

// 텍스처 슬롯은 이미지 인덱스와 샘플러 인덱스를 한 uint 에 담는다. combined image sampler 는
// 샘플러 한도(보통 수천)에 함께 걸리므로 이미지 배열과 샘플러 배열을 따로 둔다.
inline constexpr uint32_t TEXTURE_IMAGE_MASK = 0x00FFFFFFU;
inline constexpr uint32_t TEXTURE_SAMPLER_SHIFT = 24;

class BindlessTextures {
public:
    explicit BindlessTextures(Context& context);
    ~BindlessTextures();
    BindlessTextures(const BindlessTextures&) = delete;
    BindlessTextures& operator=(const BindlessTextures&) = delete;

    // 이미지와 샘플러를 등록하고 셰이더가 쓰는 묶음 슬롯을 돌려준다.
    uint32_t add(VkImageView view, VkSampler sampler);
    // 이미 발급된 슬롯의 이미지를 바꾼다. 크기 변경으로 렌더 타겟을 다시 만들 때 쓴다.
    void update(uint32_t slot, VkImageView view, VkSampler sampler);

    VkDescriptorSetLayout layout() const { return descriptorSetLayout; }
    VkDescriptorSet set() const { return descriptorSet; }

private:
    uint32_t addSampler(VkSampler sampler);

    Context& context;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    uint32_t imageCapacity = 0;
    uint32_t samplerCapacity = 0;
    uint32_t imageCount = 0;
    uint32_t samplerCount = 0;
    VkSampler registeredSamplers[64]{};
};

} // namespace gfx
