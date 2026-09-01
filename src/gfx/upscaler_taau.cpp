#include <algorithm>
#include <array>

#include <glm/vec4.hpp>

#include "gfx/bindless.h"
#include "gfx/context.h"
#include "gfx/resources.h"
#include "gfx/upscaler.h"
#include "gfx/vk_check.h"

namespace gfx {
namespace {

// shaders/taau.comp 의 local_size 와 같아야 한다.
constexpr uint32_t TAAU_GROUP_SIZE = 8;
constexpr VkFormat HISTORY_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
// 쌓을 표본 수. 배율이 1 이면 8 프레임이면 충분하고, 확대할수록 한 프레임이 채우는 몫이 줄어
// 더 오래 쌓아야 한다. 너무 크게 잡으면 새 표본이 묻혀 화면이 멈춘 것처럼 보인다.
constexpr float BASE_SAMPLES = 8.0F;
constexpr float MAXIMUM_SAMPLES = 64.0F;

// shaders/taau.comp 의 푸시 상수와 배치가 같아야 한다.
struct TaauPushConstants {
    uint32_t colorTexture;
    uint32_t depthTexture;
    uint32_t velocityTexture;
    uint32_t historyTexture;
    uint32_t outputStorage;
    uint32_t historyStorage;
    uint32_t reset;
    uint32_t padding;
    glm::vec4 renderSize;
    glm::vec4 displaySize;
    glm::vec4 jitterSamples;
};

// 이번 저장소의 시간축 업스케일. 표시 해상도 히스토리를 재투영해 쌓고, 이웃 분산으로 가둬
// 잔상을 막는다. 상용 SDK 처럼 잠금이나 반응 마스크는 두지 않는다.
class TaauUpscaler final : public TemporalUpscaler {
public:
    TaauUpscaler(Context& context, BindlessTextures& bindless) : context(context), bindless(bindless) {
        // 히스토리는 재투영된 임의의 소수 좌표에서 읽으므로 선형 보간이 필요하다. 렌더러의
        // 후처리 샘플러는 최근접이라 여기서만 쓸 샘플러를 따로 만든다.
        VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        VK_CHECK(vkCreateSampler(context.device, &samplerInfo, nullptr, &sampler));

        VkPushConstantRange pushConstantRange{};
        pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstantRange.size = sizeof(TaauPushConstants);

        VkDescriptorSetLayout bindlessLayout = bindless.layout();
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &bindlessLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pushConstantRange;
        VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &pipelineLayout));

        VkShaderModule module = createShaderModule(context.device, "taau.comp.spv");
        VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = module;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = pipelineLayout;
        VK_CHECK(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));
        vkDestroyShaderModule(context.device, module, nullptr);
    }

    ~TaauUpscaler() override {
        for (Image& image : history) {
            destroyImage(context, image);
        }
        vkDestroyPipeline(context.device, pipeline, nullptr);
        vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
        vkDestroySampler(context.device, sampler, nullptr);
    }

    void resize(VkExtent2D render, VkExtent2D display) override {
        renderExtent = render;
        displayExtent = display;
        for (Image& image : history) {
            destroyImage(context, image);
        }

        ImageDesc desc;
        desc.extent = {display.width, display.height, 1};
        desc.format = HISTORY_FORMAT;
        // 컴퓨트가 쓰고 다음 프레임이 읽는다. 둘 다 GENERAL 로 두고 레이아웃을 오가지 않는다.
        desc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        for (size_t i = 0; i < history.size(); ++i) {
            history[i] = createImage(context, desc, "시간축 히스토리");
            if (slotsAllocated) {
                bindless.update(sampledSlot[i], history[i].view, sampler, VK_IMAGE_LAYOUT_GENERAL);
                bindless.updateStorageImageRgba16(storageSlot[i], history[i].view);
            } else {
                sampledSlot[i] = bindless.add(history[i].view, sampler, VK_IMAGE_LAYOUT_GENERAL);
                storageSlot[i] = bindless.addStorageImageRgba16(history[i].view);
            }
        }
        slotsAllocated = true;
        // 새 이미지의 내용은 정의되지 않았다. 첫 프레임은 히스토리 없이 시작한다.
        historyValid = false;
        needsInitialLayout = true;
    }

    void evaluate(VkCommandBuffer commandBuffer, const UpscaleInputs& inputs) override {
        if (history[0].handle == VK_NULL_HANDLE) {
            return;
        }
        if (needsInitialLayout) {
            for (Image& image : history) {
                imageBarrier(commandBuffer,
                             image.handle,
                             VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                             0,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            }
            needsInitialLayout = false;
        }

        uint32_t current = historyIndex;
        uint32_t previous = 1 - historyIndex;

        float renderArea = static_cast<float>(renderExtent.width) * static_cast<float>(renderExtent.height);
        float displayArea = static_cast<float>(displayExtent.width) * static_cast<float>(displayExtent.height);
        float samples =
            std::clamp(BASE_SAMPLES * displayArea / std::max(renderArea, 1.0F), BASE_SAMPLES, MAXIMUM_SAMPLES);

        TaauPushConstants pushConstants{};
        pushConstants.colorTexture = inputs.colorTexture;
        pushConstants.depthTexture = inputs.depthTexture;
        pushConstants.velocityTexture = inputs.velocityTexture;
        pushConstants.historyTexture = sampledSlot[previous];
        pushConstants.outputStorage = inputs.outputStorage;
        pushConstants.historyStorage = storageSlot[current];
        pushConstants.reset = inputs.reset || !historyValid ? 1U : 0U;
        pushConstants.renderSize = glm::vec4{static_cast<float>(renderExtent.width),
                                             static_cast<float>(renderExtent.height),
                                             1.0F / static_cast<float>(renderExtent.width),
                                             1.0F / static_cast<float>(renderExtent.height)};
        pushConstants.displaySize = glm::vec4{static_cast<float>(displayExtent.width),
                                              static_cast<float>(displayExtent.height),
                                              1.0F / static_cast<float>(displayExtent.width),
                                              1.0F / static_cast<float>(displayExtent.height)};
        pushConstants.jitterSamples = glm::vec4{inputs.jitter.x, inputs.jitter.y, samples, 0.0F};

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &inputs.bindlessSet, 0, nullptr);
        vkCmdPushConstants(
            commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(commandBuffer,
                      (displayExtent.width + TAAU_GROUP_SIZE - 1) / TAAU_GROUP_SIZE,
                      (displayExtent.height + TAAU_GROUP_SIZE - 1) / TAAU_GROUP_SIZE,
                      1);

        // 이번에 쓴 히스토리를 다음 프레임이 읽는다. 프레임 경계를 넘는 의존이라 여기서 걸어 둔다.
        VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependency);

        historyIndex = previous;
        historyValid = true;
    }

private:
    Context& context;
    BindlessTextures& bindless;
    VkSampler sampler = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    // 한 장을 읽는 동안 다른 한 장에 쓴다. 같은 이미지에 하면 재투영이 자기가 쓰는 화소를 읽는다.
    std::array<Image, 2> history{};
    std::array<uint32_t, 2> sampledSlot{};
    std::array<uint32_t, 2> storageSlot{};
    bool slotsAllocated = false;
    bool needsInitialLayout = false;
    bool historyValid = false;
    uint32_t historyIndex = 0;
    VkExtent2D renderExtent{};
    VkExtent2D displayExtent{};
};

} // namespace

std::unique_ptr<TemporalUpscaler> createTaauUpscaler(Context& context, BindlessTextures& bindless) {
    return std::make_unique<TaauUpscaler>(context, bindless);
}

} // namespace gfx
