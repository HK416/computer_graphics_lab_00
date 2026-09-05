// SSAO.
// Renderer 의 멤버 함수 정의만 나눠 담은 번역 단위다. 선언은 renderer.h 하나에 있다.

#include "gfx/renderer_internal.h"

namespace gfx {

void Renderer::createSsaoPipelines() {
    VkDescriptorSetLayout bindlessLayout = bindless.layout();

    VkPushConstantRange ssaoRange{};
    ssaoRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    ssaoRange.size = sizeof(SsaoPushConstants);
    VkPipelineLayoutCreateInfo ssaoLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    ssaoLayoutInfo.setLayoutCount = 1;
    ssaoLayoutInfo.pSetLayouts = &bindlessLayout;
    ssaoLayoutInfo.pushConstantRangeCount = 1;
    ssaoLayoutInfo.pPushConstantRanges = &ssaoRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &ssaoLayoutInfo, nullptr, &ssaoPipelineLayout));

    VkShaderModule ssaoModule = createShaderModule(context.device, "ssao.comp.spv");
    VkComputePipelineCreateInfo ssaoInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ssaoInfo.stage = shaderStage(VK_SHADER_STAGE_COMPUTE_BIT, ssaoModule);
    ssaoInfo.layout = ssaoPipelineLayout;
    VK_CHECK(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &ssaoInfo, nullptr, &ssaoPipeline));
    vkDestroyShaderModule(context.device, ssaoModule, nullptr);

    VkPushConstantRange blurRange{};
    blurRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    blurRange.size = sizeof(SsaoBlurPushConstants);
    VkPipelineLayoutCreateInfo blurLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    blurLayoutInfo.setLayoutCount = 1;
    blurLayoutInfo.pSetLayouts = &bindlessLayout;
    blurLayoutInfo.pushConstantRangeCount = 1;
    blurLayoutInfo.pPushConstantRanges = &blurRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &blurLayoutInfo, nullptr, &ssaoBlurPipelineLayout));

    VkShaderModule blurModule = createShaderModule(context.device, "ssao_blur.comp.spv");
    VkComputePipelineCreateInfo blurInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    blurInfo.stage = shaderStage(VK_SHADER_STAGE_COMPUTE_BIT, blurModule);
    blurInfo.layout = ssaoBlurPipelineLayout;
    VK_CHECK(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &blurInfo, nullptr, &ssaoBlurPipeline));
    vkDestroyShaderModule(context.device, blurModule, nullptr);
}

void Renderer::recordSsaoPass(VkCommandBuffer commandBuffer, const Frame& frame) {
    VkDescriptorSet bindlessSet = bindless.set();
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ssaoPipeline);
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ssaoPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);

    SsaoPushConstants pushConstants{};
    pushConstants.camera = frame.cameraBuffer.address;
    pushConstants.depthTexture = targets.depthSlot;
    pushConstants.occlusionStorage = targets.ssaoRawStorageSlot;
    pushConstants.size[0] = static_cast<int32_t>(targets.ssaoExtent.width);
    pushConstants.size[1] = static_cast<int32_t>(targets.ssaoExtent.height);
    // 반지름은 장면 크기에 대한 비율이라 여우든 헬멧이든 비슷하게 보인다.
    pushConstants.radius = settings.ssaoRadius * sceneRadius;
    pushConstants.intensity = settings.ssaoIntensity;
    pushConstants.bias = settings.ssaoBias;
    pushConstants.sampleCount = settings.ssaoSamples;
    vkCmdPushConstants(
        commandBuffer, ssaoPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, (targets.ssaoExtent.width + 7) / 8, (targets.ssaoExtent.height + 7) / 8, 1);

    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ssaoBlurPipeline);
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ssaoBlurPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    SsaoBlurPushConstants blurConstants{};
    blurConstants.sourceTexture = targets.ssaoRawSlot;
    blurConstants.destinationStorage = targets.ssaoStorageSlot;
    blurConstants.size[0] = pushConstants.size[0];
    blurConstants.size[1] = pushConstants.size[1];
    vkCmdPushConstants(
        commandBuffer, ssaoBlurPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(blurConstants), &blurConstants);
    vkCmdDispatch(commandBuffer, (targets.ssaoExtent.width + 7) / 8, (targets.ssaoExtent.height + 7) / 8, 1);
}

} // namespace gfx
