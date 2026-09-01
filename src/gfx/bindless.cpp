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
constexpr uint32_t MAX_BINDLESS_STORAGE_IMAGES = 256;
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

    std::array<VkDescriptorSetLayoutBinding, 4> bindings{};
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

    VkDescriptorBindingFlags commonFlags = VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT |
                                           VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
                                           VK_DESCRIPTOR_BINDING_UPDATE_UNUSED_WHILE_PENDING_BIT;
    std::array<VkDescriptorBindingFlags, 4> bindingFlags{commonFlags, commonFlags, commonFlags, commonFlags};
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
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, imageCapacity},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, samplerCapacity},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storageCapacity * 2}};
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

    spdlog::info(
        "bindless 슬롯: 이미지 {}, 샘플러 {}, 스토리지 이미지 {}", imageCapacity, samplerCapacity, storageCapacity);
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

uint32_t BindlessTextures::addStorageImage(VkImageView view) {
    if (storageCount >= storageCapacity) {
        core::fatal("bindless 스토리지 이미지 슬롯이 부족합니다 (한계 {})", storageCapacity);
    }
    uint32_t slot = storageCount++;
    updateStorageImage(slot, view);
    return slot;
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
