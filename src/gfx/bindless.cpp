#include "gfx/bindless.h"

#include <algorithm>
#include <array>

#include <spdlog/spdlog.h>

#include "core/error.h"
#include "gfx/context.h"
#include "gfx/vk_check.h"

namespace gfx {
namespace {
// 한계가 아무리 커도 실제로는 이 이상 쓰지 않는다.
constexpr uint32_t MAX_BINDLESS_IMAGES = 16384;
constexpr uint32_t MAX_BINDLESS_SAMPLERS = 64;

constexpr uint32_t IMAGE_BINDING = 0;
constexpr uint32_t SAMPLER_BINDING = 1;
constexpr uint32_t STORAGE_IMAGE_BINDING = 2;
constexpr uint32_t STORAGE_IMAGE_RGBA_BINDING = 3;
constexpr uint32_t ARRAY_BINDING = 4;
constexpr uint32_t CUBE_BINDING = 5;
constexpr uint32_t STORAGE_ARRAY_BINDING = 6;
constexpr uint32_t STORAGE_IMAGE_RGBA16_BINDING = 7;
constexpr uint32_t STORAGE_IMAGE_RG16_BINDING = 8;
constexpr uint32_t MAX_BINDLESS_STORAGE_IMAGES = 256;
// 배열 텍스처와 큐브맵은 몇 개면 충분하다. 그림자 아틀라스와 환경 맵 정도만 쓴다.
constexpr uint32_t MAX_BINDLESS_ARRAYS = 16;
} // namespace

BindlessTextures::BindlessTextures(Context& context) : context(context) {
    VkPhysicalDeviceVulkan12Properties properties12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES};
    VkPhysicalDeviceProperties2 properties2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    properties2.pNext = &properties12;
    vkGetPhysicalDeviceProperties2(context.physicalDevice, &properties2);

    imageCapacity = std::min({properties12.maxDescriptorSetUpdateAfterBindSampledImages,
                              properties12.maxPerStageDescriptorUpdateAfterBindSampledImages,
                              MAX_BINDLESS_IMAGES});
    samplerCapacity = std::min({properties12.maxDescriptorSetUpdateAfterBindSamplers,
                                properties12.maxPerStageDescriptorUpdateAfterBindSamplers,
                                MAX_BINDLESS_SAMPLERS});
    if (imageCapacity == 0 || samplerCapacity == 0) {
        core::fatal("bindless 텍스처 배열을 만들 수 없습니다");
    }

    storageCapacity = std::min({properties12.maxDescriptorSetUpdateAfterBindStorageImages,
                                properties12.maxPerStageDescriptorUpdateAfterBindStorageImages,
                                MAX_BINDLESS_STORAGE_IMAGES});

    arrayCapacity = std::min(imageCapacity, MAX_BINDLESS_ARRAYS);

    std::array<VkDescriptorSetLayoutBinding, 9> bindings{};
    bindings[0].binding = IMAGE_BINDING;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[0].descriptorCount = imageCapacity;
    bindings[0].stageFlags = VK_SHADER_STAGE_ALL;
    bindings[1].binding = SAMPLER_BINDING;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    bindings[1].descriptorCount = samplerCapacity;
    bindings[1].stageFlags = VK_SHADER_STAGE_ALL;
    bindings[2].binding = STORAGE_IMAGE_BINDING;
    bindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[2].descriptorCount = storageCapacity;
    bindings[2].stageFlags = VK_SHADER_STAGE_ALL;
    bindings[3].binding = STORAGE_IMAGE_RGBA_BINDING;
    bindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[3].descriptorCount = storageCapacity;
    bindings[3].stageFlags = VK_SHADER_STAGE_ALL;
    bindings[4].binding = ARRAY_BINDING;
    bindings[4].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[4].descriptorCount = arrayCapacity;
    bindings[4].stageFlags = VK_SHADER_STAGE_ALL;
    bindings[5].binding = CUBE_BINDING;
    bindings[5].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    bindings[5].descriptorCount = arrayCapacity;
    bindings[5].stageFlags = VK_SHADER_STAGE_ALL;
    // 환경 맵을 굽는 컴퓨트가 큐브 면을 층으로 보고 쓴다.
    bindings[6].binding = STORAGE_ARRAY_BINDING;
    bindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[6].descriptorCount = storageCapacity;
    bindings[6].stageFlags = VK_SHADER_STAGE_ALL;
    // 시간축 업스케일이 히스토리와 출력을 rgba16f 로 쓴다.
    bindings[7].binding = STORAGE_IMAGE_RGBA16_BINDING;
    bindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[7].descriptorCount = storageCapacity;
    bindings[7].stageFlags = VK_SHADER_STAGE_ALL;

    bindings[8].binding = STORAGE_IMAGE_RG16_BINDING;
    bindings[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    bindings[8].descriptorCount = storageCapacity;
    bindings[8].stageFlags = VK_SHADER_STAGE_ALL;

    VkDescriptorBindingFlags commonFlags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                                           VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                                           VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
    std::array<VkDescriptorBindingFlags, 8> bindingFlags{
        commonFlags, commonFlags, commonFlags, commonFlags, commonFlags, commonFlags, commonFlags, commonFlags};
    VkDescriptorSetLayoutBindingFlagsCreateInfo flagsInfo{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO};
    flagsInfo.bindingCount = static_cast<uint32_t>(bindingFlags.size());
    flagsInfo.pBindingFlags = bindingFlags.data();

    VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layoutInfo.pNext = &flagsInfo;
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(context.device, &layoutInfo, nullptr, &descriptorSetLayout));

    std::array<VkDescriptorPoolSize, 3> poolSizes{
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, imageCapacity + arrayCapacity * 2},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, samplerCapacity},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storageCapacity * 5}};
    VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    VK_CHECK(vkCreateDescriptorPool(context.device, &poolInfo, nullptr, &descriptorPool));

    VkDescriptorSetAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocateInfo.descriptorPool = descriptorPool;
    allocateInfo.descriptorSetCount = 1;
    allocateInfo.pSetLayouts = &descriptorSetLayout;
    VK_CHECK(vkAllocateDescriptorSets(context.device, &allocateInfo, &descriptorSet));

    spdlog::info("bindless 슬롯: 이미지 {}, 샘플러 {}, 스토리지 이미지 {}, 배열/큐브 {}",
                 imageCapacity,
                 samplerCapacity,
                 storageCapacity,
                 arrayCapacity);
}

BindlessTextures::~BindlessTextures() {
    vkDestroyDescriptorPool(context.device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(context.device, descriptorSetLayout, nullptr);
}

uint32_t BindlessTextures::addSampler(VkSampler sampler) {
    for (uint32_t i = 0; i < samplerCount; ++i) {
        if (registeredSamplers[i] == sampler) {
            return i;
        }
    }
    if (samplerCount >= samplerCapacity) {
        core::fatal("bindless 샘플러 슬롯이 부족합니다 (한계 {})", samplerCapacity);
    }
    uint32_t index = samplerCount++;
    registeredSamplers[index] = sampler;

    VkDescriptorImageInfo samplerInfo{};
    samplerInfo.sampler = sampler;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptorSet;
    write.dstBinding = SAMPLER_BINDING;
    write.dstArrayElement = index;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    write.pImageInfo = &samplerInfo;
    vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
    return index;
}

void BindlessTextures::update(uint32_t slot, VkImageView view, VkSampler sampler, VkImageLayout layout) {
    uint32_t imageIndex = slot & TEXTURE_IMAGE_MASK;
    if (imageIndex >= imageCount) {
        core::fatal("등록되지 않은 bindless 슬롯을 갱신할 수 없습니다: {}", slot);
    }

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = view;
    imageInfo.imageLayout = layout;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptorSet;
    write.dstBinding = IMAGE_BINDING;
    write.dstArrayElement = imageIndex;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
    addSampler(sampler);
}

void BindlessTextures::updateStorageImage(uint32_t slot, VkImageView view) {
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptorSet;
    write.dstBinding = STORAGE_IMAGE_BINDING;
    write.dstArrayElement = slot;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
}

void BindlessTextures::updateStorageImageRgba(uint32_t slot, VkImageView view) {
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptorSet;
    write.dstBinding = STORAGE_IMAGE_RGBA_BINDING;
    write.dstArrayElement = slot;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
}

uint32_t BindlessTextures::addStorageImageRgba(VkImageView view) {
    if (storageRgbaCount >= storageCapacity) {
        core::fatal("bindless rgba 스토리지 이미지 슬롯이 부족합니다 (한계 {})", storageCapacity);
    }
    uint32_t slot = storageRgbaCount++;
    updateStorageImageRgba(slot, view);
    return slot;
}

void BindlessTextures::updateStorageImageRgba16(uint32_t slot, VkImageView view) {
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptorSet;
    write.dstBinding = STORAGE_IMAGE_RGBA16_BINDING;
    write.dstArrayElement = slot;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
}

uint32_t BindlessTextures::addStorageImageRgba16(VkImageView view) {
    if (storageRgba16Count >= storageCapacity) {
        core::fatal("bindless rgba16f 스토리지 이미지 슬롯이 부족합니다 (한계 {})", storageCapacity);
    }
    uint32_t slot = storageRgba16Count++;
    updateStorageImageRgba16(slot, view);
    return slot;
}

void BindlessTextures::updateStorageImageRg16(uint32_t slot, VkImageView view) {
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptorSet;
    write.dstBinding = STORAGE_IMAGE_RG16_BINDING;
    write.dstArrayElement = slot;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
}

uint32_t BindlessTextures::addStorageImageRg16(VkImageView view) {
    if (storageRg16Count >= storageCapacity) {
        core::fatal("bindless rg16f 스토리지 이미지 슬롯이 부족합니다 (한계 {})", storageCapacity);
    }
    uint32_t slot = storageRg16Count++;
    updateStorageImageRg16(slot, view);
    return slot;
}

void BindlessTextures::updateStorageArray(uint32_t slot, VkImageView view) {
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptorSet;
    write.dstBinding = STORAGE_ARRAY_BINDING;
    write.dstArrayElement = slot;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
}

uint32_t BindlessTextures::addStorageArray(VkImageView view) {
    if (storageArrayCount >= storageCapacity) {
        core::fatal("bindless 스토리지 배열 슬롯이 부족합니다 (한계 {})", storageCapacity);
    }
    uint32_t slot = storageArrayCount++;
    updateStorageArray(slot, view);
    return slot;
}

uint32_t BindlessTextures::addStorageImage(VkImageView view) {
    if (storageCount >= storageCapacity) {
        core::fatal("bindless 스토리지 이미지 슬롯이 부족합니다 (한계 {})", storageCapacity);
    }
    uint32_t slot = storageCount++;
    updateStorageImage(slot, view);
    return slot;
}

void BindlessTextures::updateArray(uint32_t slot, VkImageView view, VkSampler sampler) {
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptorSet;
    write.dstBinding = ARRAY_BINDING;
    write.dstArrayElement = slot & TEXTURE_IMAGE_MASK;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
    addSampler(sampler);
}

uint32_t BindlessTextures::addArray(VkImageView view, VkSampler sampler) {
    if (arrayCount >= arrayCapacity) {
        core::fatal("bindless 배열 텍스처 슬롯이 부족합니다 (한계 {})", arrayCapacity);
    }
    uint32_t index = arrayCount++;
    updateArray(index, view, sampler);
    return index | (addSampler(sampler) << TEXTURE_SAMPLER_SHIFT);
}

void BindlessTextures::updateCube(uint32_t slot, VkImageView view, VkSampler sampler) {
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = view;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptorSet;
    write.dstBinding = CUBE_BINDING;
    write.dstArrayElement = slot & TEXTURE_IMAGE_MASK;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);
    addSampler(sampler);
}

uint32_t BindlessTextures::addCube(VkImageView view, VkSampler sampler) {
    if (cubeCount >= arrayCapacity) {
        core::fatal("bindless 큐브맵 슬롯이 부족합니다 (한계 {})", arrayCapacity);
    }
    uint32_t index = cubeCount++;
    updateCube(index, view, sampler);
    return index | (addSampler(sampler) << TEXTURE_SAMPLER_SHIFT);
}

uint32_t BindlessTextures::add(VkImageView view, VkSampler sampler, VkImageLayout layout) {
    if (imageCount >= imageCapacity) {
        core::fatal("bindless 이미지 슬롯이 부족합니다 (한계 {})", imageCapacity);
    }
    uint32_t imageIndex = imageCount++;

    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = view;
    imageInfo.imageLayout = layout;

    VkWriteDescriptorSet write{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet = descriptorSet;
    write.dstBinding = IMAGE_BINDING;
    write.dstArrayElement = imageIndex;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    write.pImageInfo = &imageInfo;
    vkUpdateDescriptorSets(context.device, 1, &write, 0, nullptr);

    return imageIndex | (addSampler(sampler) << TEXTURE_SAMPLER_SHIFT);
}

} // namespace gfx
