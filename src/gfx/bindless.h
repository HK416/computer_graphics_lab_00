#pragma once

#include <cstdint>
#include <vector>

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

    // 이미지와 샘플러를 등록하고 셰이더가 쓰는 묶음 슬롯을 돌려준다. 해제된 번호가 있으면 그것부터 쓴다.
    uint32_t add(VkImageView view, VkSampler sampler, VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    // 이미지 번호를 돌려준다. 샘플러 번호는 설정별로 하나뿐이라 그대로 둔다.
    void freeImage(uint32_t slot);
    // 이미 발급된 슬롯의 이미지를 바꾼다. 크기 변경으로 렌더 타겟을 다시 만들 때 쓴다.
    void update(uint32_t slot,
                VkImageView view,
                VkSampler sampler,
                VkImageLayout layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    // 컴퓨트가 쓰는 스토리지 이미지. 슬롯 번호는 이미지 배열과 별개로 센다.
    uint32_t addStorageImage(VkImageView view);
    void updateStorageImage(uint32_t slot, VkImageView view);
    // rgba32f 로 쓰는 스토리지 이미지는 포맷 한정자가 달라 배열을 따로 둔다.
    uint32_t addStorageImageRgba(VkImageView view);
    void updateStorageImageRgba(uint32_t slot, VkImageView view);
    // 시간축 업스케일의 히스토리와 출력처럼 rgba16f 로 쓰는 스토리지 이미지.
    uint32_t addStorageImageRgba16(VkImageView view);
    void updateStorageImageRgba16(uint32_t slot, VkImageView view);
    // 경로 추적이 모션 벡터를 쓰는 rg16f 스토리지 이미지.
    uint32_t addStorageImageRg16(VkImageView view);
    void updateStorageImageRg16(uint32_t slot, VkImageView view);
    // 큐브맵을 굽는 컴퓨트가 쓰는 rgba16f 2D 배열 스토리지. 면을 층으로 다룬다.
    uint32_t addStorageArray(VkImageView view);
    void updateStorageArray(uint32_t slot, VkImageView view);
    // 2D 배열과 큐브맵은 GLSL 타입이 달라 배열을 각각 따로 둔다. 슬롯 인코딩은 2D 와 같지만
    // 번호 공간이 별개이므로 재질 슬롯과 섞어 쓰면 안 된다.
    uint32_t addArray(VkImageView view, VkSampler sampler);
    void updateArray(uint32_t slot, VkImageView view, VkSampler sampler);
    uint32_t addCube(VkImageView view, VkSampler sampler);
    void updateCube(uint32_t slot, VkImageView view, VkSampler sampler);

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
    uint32_t storageCapacity = 0;
    uint32_t storageCount = 0;
    uint32_t storageRgbaCount = 0;
    uint32_t storageRgba16Count = 0;
    uint32_t storageRg16Count = 0;
    uint32_t storageArrayCount = 0;
    uint32_t arrayCapacity = 0;
    uint32_t arrayCount = 0;
    uint32_t cubeCount = 0;
    uint32_t samplerCount = 0;
    VkSampler registeredSamplers[64]{};
    // freeImage 가 돌려준 이미지 번호. add 가 먼저 꺼내 쓴다.
    std::vector<uint32_t> freeImages;
};

} // namespace gfx
