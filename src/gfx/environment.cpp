#include "gfx/environment.h"

#include <array>
#include <cmath>

#include <spdlog/spdlog.h>
#include <stb_image.h>

#include "core/error.h"
#include "gfx/bindless.h"
#include "gfx/context.h"
#include "gfx/uploader.h"
#include "gfx/vk_check.h"

namespace gfx {
namespace {

// 하늘은 이 정도면 충분하다. 프리필터가 밉을 타고 읽으므로 너무 키워도 이득이 없다.
constexpr uint32_t ENVIRONMENT_SIZE = 512;
constexpr uint32_t IRRADIANCE_SIZE = 32;
constexpr uint32_t PREFILTER_SIZE = 128;
constexpr uint32_t PREFILTER_MIPS = 5;
constexpr uint32_t BRDF_SIZE = 256;
constexpr VkFormat ENVIRONMENT_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr uint32_t INVALID_SLOT = 0xFFFFFFFFU;

// shaders/sky.comp 의 푸시 상수와 배치가 같아야 한다. 가장 큰 구조체라 이 크기로 범위를 잡는다.
struct SkyPushConstants {
    glm::vec4 sunDirection;
    glm::vec4 sunColor;
    glm::vec4 zenith;
    glm::vec4 horizon;
    glm::vec4 ground;
    uint32_t targetStorage;
    uint32_t equirectTexture;
    int32_t size;
};

struct IrradiancePushConstants {
    uint32_t environmentCube;
    uint32_t targetStorage;
    int32_t size;
    float sampleDelta;
};

struct PrefilterPushConstants {
    uint32_t environmentCube;
    uint32_t targetStorage;
    int32_t size;
    float roughness;
    float environmentSize;
    float environmentMips;
    uint32_t sampleCount;
};

struct BrdfLutPushConstants {
    uint32_t targetStorage;
    int32_t size;
    uint32_t sampleCount;
};

uint32_t mipCountFor(uint32_t size) {
    uint32_t levels = 1;
    while (size > 1) {
        size /= 2;
        ++levels;
    }
    return levels;
}

VkImageView createStorageView(Context& context, const Image& image, uint32_t mipLevel) {
    VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    info.image = image.handle;
    info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    info.format = image.format;
    info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    info.subresourceRange.baseMipLevel = mipLevel;
    info.subresourceRange.levelCount = 1;
    info.subresourceRange.layerCount = image.arrayLayers;
    VkImageView view = VK_NULL_HANDLE;
    VK_CHECK(vkCreateImageView(context.device, &info, nullptr, &view));
    return view;
}

// 컴퓨트가 쓰기 전후로 레이아웃을 바꾼다. 굽기는 드물어 배리어를 세밀하게 나눌 이유가 없다.
void toGeneral(VkCommandBuffer commandBuffer, const Image& image) {
    imageBarrier(commandBuffer,
                 image.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 0,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_WRITE_BIT);
}

void toSampled(VkCommandBuffer commandBuffer, const Image& image) {
    imageBarrier(commandBuffer,
                 image.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_SHADER_READ_BIT);
}

uint32_t groupsOf(uint32_t size) {
    return (size + 7) / 8;
}

} // namespace

EnvironmentMap::EnvironmentMap(Context& context, BindlessTextures& bindless) : context(context), bindless(bindless) {
    createImages();
    createPipelines();
}

EnvironmentMap::~EnvironmentMap() {
    vkDestroyPipeline(context.device, skyPipeline, nullptr);
    vkDestroyPipeline(context.device, irradiancePipeline, nullptr);
    vkDestroyPipeline(context.device, prefilterPipeline, nullptr);
    vkDestroyPipeline(context.device, brdfPipeline, nullptr);
    vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
    vkDestroySampler(context.device, sampler, nullptr);
    vkDestroyImageView(context.device, environmentStorageView, nullptr);
    vkDestroyImageView(context.device, irradianceStorageView, nullptr);
    for (VkImageView view : prefilterStorageViews) {
        vkDestroyImageView(context.device, view, nullptr);
    }
    vkDestroyImageView(context.device, brdfStorageView, nullptr);
    destroyImage(context, environmentCube);
    destroyImage(context, irradianceCube);
    destroyImage(context, prefilterCube);
    destroyImage(context, brdfLut);
    destroyImage(context, equirect);
}

uint32_t EnvironmentMap::prefilterMipCount() const {
    return PREFILTER_MIPS;
}

void EnvironmentMap::createImages() {
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
    VK_CHECK(vkCreateSampler(context.device, &samplerInfo, nullptr, &sampler));

    VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    ImageDesc cubeDesc{};
    cubeDesc.format = ENVIRONMENT_FORMAT;
    cubeDesc.usage = usage;
    cubeDesc.arrayLayers = 6;
    cubeDesc.viewType = VK_IMAGE_VIEW_TYPE_CUBE;

    cubeDesc.extent = {ENVIRONMENT_SIZE, ENVIRONMENT_SIZE, 1};
    cubeDesc.mipLevels = mipCountFor(ENVIRONMENT_SIZE);
    environmentCube = createImage(context, cubeDesc, "환경 큐브맵");

    cubeDesc.extent = {IRRADIANCE_SIZE, IRRADIANCE_SIZE, 1};
    cubeDesc.mipLevels = 1;
    irradianceCube = createImage(context, cubeDesc, "조도 큐브맵");

    cubeDesc.extent = {PREFILTER_SIZE, PREFILTER_SIZE, 1};
    cubeDesc.mipLevels = PREFILTER_MIPS;
    prefilterCube = createImage(context, cubeDesc, "프리필터 큐브맵");

    ImageDesc lutDesc{};
    lutDesc.extent = {BRDF_SIZE, BRDF_SIZE, 1};
    lutDesc.format = ENVIRONMENT_FORMAT;
    lutDesc.usage = usage;
    lutDesc.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    brdfLut = createImage(context, lutDesc, "BRDF 표");

    environmentStorageView = createStorageView(context, environmentCube, 0);
    irradianceStorageView = createStorageView(context, irradianceCube, 0);
    prefilterStorageViews.resize(PREFILTER_MIPS);
    prefilterStorageSlots.resize(PREFILTER_MIPS);
    for (uint32_t level = 0; level < PREFILTER_MIPS; ++level) {
        prefilterStorageViews[level] = createStorageView(context, prefilterCube, level);
        prefilterStorageSlots[level] = bindless.addStorageArray(prefilterStorageViews[level]);
    }
    brdfStorageView = createStorageView(context, brdfLut, 0);

    environmentStorageSlot = bindless.addStorageArray(environmentStorageView);
    irradianceStorageSlot = bindless.addStorageArray(irradianceStorageView);
    brdfStorageSlot = bindless.addStorageArray(brdfStorageView);

    environmentCubeSlot = bindless.addCube(environmentCube.view, sampler);
    irradianceCubeSlot = bindless.addCube(irradianceCube.view, sampler);
    prefilterCubeSlot = bindless.addCube(prefilterCube.view, sampler);
    brdfArraySlot = bindless.addArray(brdfLut.view, sampler);
}

void EnvironmentMap::createPipelines() {
    VkDescriptorSetLayout bindlessLayout = bindless.layout();
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.size = sizeof(SkyPushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &bindlessLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &range;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &pipelineLayout));

    struct Entry {
        const char* shader;
        VkPipeline* target;
    };
    std::array<Entry, 4> entries{Entry{"sky.comp.spv", &skyPipeline},
                                 Entry{"irradiance.comp.spv", &irradiancePipeline},
                                 Entry{"prefilter.comp.spv", &prefilterPipeline},
                                 Entry{"brdf_lut.comp.spv", &brdfPipeline}};
    for (const Entry& entry : entries) {
        VkShaderModule module = createShaderModule(context.device, entry.shader);
        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module;
        stage.pName = "main";

        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = stage;
        info.layout = pipelineLayout;
        VK_CHECK(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, entry.target));
        vkDestroyShaderModule(context.device, module, nullptr);
    }
}

bool EnvironmentMap::loadHdr(const std::filesystem::path& path) {
    if (path == loadedHdr && equirectSlot != INVALID_SLOT) {
        return true;
    }

    int width = 0;
    int height = 0;
    int channels = 0;
    float* pixels = stbi_loadf(path.string().c_str(), &width, &height, &channels, 4);
    if (pixels == nullptr || width <= 0 || height <= 0) {
        spdlog::error("HDR 환경 맵을 읽지 못했습니다: {}", path.string());
        return false;
    }

    // 파일마다 크기가 달라 이미지를 새로 만든다. 굽기는 드물어 재사용할 이유가 없다.
    destroyImage(context, equirect);
    ImageDesc desc{};
    desc.extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
    desc.format = VK_FORMAT_R32G32B32A32_SFLOAT;
    desc.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    equirect = createImage(context, desc, "환경 등정방형 원본");

    VkDeviceSize size = static_cast<VkDeviceSize>(width) * height * 4 * sizeof(float);
    {
        Uploader uploader(context);
        uploader.uploadImage(equirect, pixels, size, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        uploader.flush();
    }
    stbi_image_free(pixels);

    if (equirectSlot == INVALID_SLOT) {
        equirectSlot = bindless.add(equirect.view, sampler);
    } else {
        bindless.update(equirectSlot, equirect.view, sampler);
    }
    loadedHdr = path;
    spdlog::info("HDR 환경 맵 적재: {} ({}x{})", path.filename().string(), width, height);
    return true;
}

void EnvironmentMap::bakeBrdf(VkCommandBuffer commandBuffer) {
    toGeneral(commandBuffer, brdfLut);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, brdfPipeline);
    BrdfLutPushConstants push{brdfStorageSlot, static_cast<int32_t>(BRDF_SIZE), 512};
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(commandBuffer, groupsOf(BRDF_SIZE), groupsOf(BRDF_SIZE), 1);
    toSampled(commandBuffer, brdfLut);
    brdfBaked = true;
}

bool EnvironmentMap::update(VkCommandBuffer commandBuffer,
                            const scene::Environment& desired,
                            const glm::vec3& sunDirection) {
    if (baked && desired == current && sunDirection == currentSun) {
        return false;
    }

    scene::Environment settings = desired;
    if (settings.useHdr && (settings.hdrPath.empty() || !loadHdr(settings.hdrPath))) {
        settings.useHdr = false;
    }
    current = settings;
    currentSun = sunDirection;
    baked = true;
    spdlog::debug("환경 굽기");

    VkDescriptorSet bindlessSet = bindless.set();
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &bindlessSet, 0, nullptr);

    if (!brdfBaked) {
        bakeBrdf(commandBuffer);
    }

    // ---- 환경 큐브맵 ----
    toGeneral(commandBuffer, environmentCube);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, skyPipeline);
    SkyPushConstants sky{};
    sky.sunDirection = glm::vec4{-glm::normalize(sunDirection), settings.sunIntensity};
    sky.sunColor = glm::vec4{settings.sunColor, 0.0F};
    sky.zenith = glm::vec4{settings.zenithColor, settings.intensity};
    sky.horizon = glm::vec4{settings.horizonColor, glm::radians(settings.yawDegrees)};
    sky.ground = glm::vec4{settings.groundColor, 0.0F};
    sky.targetStorage = environmentStorageSlot;
    sky.equirectTexture = settings.useHdr ? equirectSlot : INVALID_SLOT;
    sky.size = static_cast<int32_t>(ENVIRONMENT_SIZE);
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(sky), &sky);
    vkCmdDispatch(commandBuffer, groupsOf(ENVIRONMENT_SIZE), groupsOf(ENVIRONMENT_SIZE), 6);

    // 밉 사슬은 blit 로 내린다. 프리필터가 표본 확률에 맞는 밉을 읽어 반딧불이를 줄인다.
    imageBarrier(commandBuffer,
                 environmentCube.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_BLIT_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT,
                 VK_QUEUE_FAMILY_IGNORED,
                 VK_QUEUE_FAMILY_IGNORED,
                 0,
                 1);
    for (uint32_t level = 1; level < environmentCube.mipLevels; ++level) {
        imageBarrier(commandBuffer,
                     environmentCube.handle,
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_BLIT_BIT,
                     0,
                     VK_PIPELINE_STAGE_2_BLIT_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_QUEUE_FAMILY_IGNORED,
                     VK_QUEUE_FAMILY_IGNORED,
                     level,
                     1);

        int32_t sourceSize = static_cast<int32_t>(ENVIRONMENT_SIZE >> (level - 1));
        int32_t targetSize = static_cast<int32_t>(ENVIRONMENT_SIZE >> level);
        VkImageBlit blit{};
        blit.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level - 1, 0, 6};
        blit.srcOffsets[1] = {sourceSize, sourceSize, 1};
        blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 6};
        blit.dstOffsets[1] = {targetSize, targetSize, 1};
        vkCmdBlitImage(commandBuffer,
                       environmentCube.handle,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       environmentCube.handle,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1,
                       &blit,
                       VK_FILTER_LINEAR);

        imageBarrier(commandBuffer,
                     environmentCube.handle,
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_PIPELINE_STAGE_2_BLIT_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_BLIT_BIT,
                     VK_ACCESS_2_TRANSFER_READ_BIT,
                     VK_QUEUE_FAMILY_IGNORED,
                     VK_QUEUE_FAMILY_IGNORED,
                     level,
                     1);
    }
    imageBarrier(commandBuffer,
                 environmentCube.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_2_BLIT_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT,
                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_SHADER_READ_BIT);

    // ---- 조도 ----
    toGeneral(commandBuffer, irradianceCube);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, irradiancePipeline);
    IrradiancePushConstants irradiance{
        environmentCubeSlot, irradianceStorageSlot, static_cast<int32_t>(IRRADIANCE_SIZE), 0.05F};
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(irradiance), &irradiance);
    vkCmdDispatch(commandBuffer, groupsOf(IRRADIANCE_SIZE), groupsOf(IRRADIANCE_SIZE), 6);
    toSampled(commandBuffer, irradianceCube);

    // ---- 프리필터 ----
    toGeneral(commandBuffer, prefilterCube);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, prefilterPipeline);
    for (uint32_t level = 0; level < PREFILTER_MIPS; ++level) {
        uint32_t size = PREFILTER_SIZE >> level;
        PrefilterPushConstants prefilter{};
        prefilter.environmentCube = environmentCubeSlot;
        prefilter.targetStorage = prefilterStorageSlots[level];
        prefilter.size = static_cast<int32_t>(size);
        prefilter.roughness = static_cast<float>(level) / static_cast<float>(PREFILTER_MIPS - 1);
        prefilter.environmentSize = static_cast<float>(ENVIRONMENT_SIZE);
        prefilter.environmentMips = static_cast<float>(environmentCube.mipLevels);
        prefilter.sampleCount = 128;
        vkCmdPushConstants(
            commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(prefilter), &prefilter);
        vkCmdDispatch(commandBuffer, groupsOf(size), groupsOf(size), 6);
    }
    toSampled(commandBuffer, prefilterCube);
    return true;
}

} // namespace gfx
