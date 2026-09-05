// 광선 반사 컴퓨트.
// Renderer 의 멤버 함수 정의만 나눠 담은 번역 단위다. 선언은 renderer.h 하나에 있다.

#include "gfx/renderer_internal.h"

namespace gfx {

void Renderer::createReflectionPipelines() {
    if (!rayQueryShadowsAvailable()) {
        return;
    }
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.size = sizeof(ReflectPushConstants);
    std::array<VkDescriptorSetLayout, 2> sets{bindless.layout(), rayTracer->accelerationLayout()};
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = static_cast<uint32_t>(sets.size());
    layoutInfo.pSetLayouts = sets.data();
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &range;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &reflectionPipelineLayout));

    // 추적과 해결은 한 셰이더의 특수화 상수로 갈린다.
    VkShaderModule module = createShaderModule(context.device, "reflect.comp.spv");
    uint32_t stage = 0;
    VkSpecializationMapEntry entry{0, 0, sizeof(uint32_t)};
    VkSpecializationInfo specialization{};
    specialization.mapEntryCount = 1;
    specialization.pMapEntries = &entry;
    specialization.dataSize = sizeof(stage);
    specialization.pData = &stage;
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = shaderStage(VK_SHADER_STAGE_COMPUTE_BIT, module);
    info.stage.pSpecializationInfo = &specialization;
    info.layout = reflectionPipelineLayout;
    VK_CHECK(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &reflectionTracePipeline));
    stage = 1;
    VK_CHECK(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &reflectionResolvePipeline));
    vkDestroyShaderModule(context.device, module, nullptr);
}

void Renderer::recordReflectionPass(VkCommandBuffer commandBuffer, const Frame& frame) {
    uint32_t zone = frameProfiler.begin("반사", commandBuffer);

    // 깊이와 모션 벡터는 읽기 전용으로, 색상은 컴퓨트가 더할 수 있게 스토리지 레이아웃으로 옮긴다.
    imageBarrier(commandBuffer,
                 targets.depth.handle,
                 VK_IMAGE_ASPECT_DEPTH_BIT,
                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    imageBarrier(commandBuffer,
                 targets.velocity.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    imageBarrier(commandBuffer,
                 targets.color.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    auto memoryBarrier = [&](VkAccessFlags2 sourceAccess, VkAccessFlags2 destinationAccess) {
        VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = sourceAccess;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = destinationAccess;
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependency);
    };
    // 지난 프레임 해결이 쓴 히스토리를 이번에 읽고, 지난 프레임이 읽던 원본을 이번에 덮어쓴다.
    memoryBarrier(VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    size_t write = frameIndex & 1;
    size_t read = write ^ 1;
    ReflectPushConstants pushConstants{};
    pushConstants.vertices = geometry.vertexBuffer.address;
    pushConstants.skinnedVertices = skinnedVertexBuffer.address;
    pushConstants.indices = geometry.indexBuffer.address;
    pushConstants.meshes = geometry.meshBuffer.address;
    pushConstants.instances = frame.instanceBuffer.address;
    pushConstants.materials = geometry.materialBuffer.address;
    pushConstants.lods = geometry.lodBuffer.address;
    pushConstants.camera = frame.cameraBuffer.address;
    pushConstants.lights = frame.lightBuffer.address;
    pushConstants.normalRoughnessTexture = targets.guideNormalSlot;
    pushConstants.weightTexture = targets.guideSpecularAlbedoSlot;
    pushConstants.depthTexture = targets.depthSlot;
    pushConstants.velocityTexture = targets.velocitySlot;
    pushConstants.rawTexture = targets.reflectionRawSlot;
    pushConstants.rawStorage = targets.reflectionRawStorageSlot;
    pushConstants.historyTexture = targets.reflectionHistorySlots[read];
    pushConstants.historyStorage = targets.reflectionHistoryStorageSlots[write];
    pushConstants.colorStorage = targets.colorStorageSlot;
    pushConstants.frameIndex = static_cast<uint32_t>(frameIndex);
    pushConstants.maxSamples = std::max(reflectionMaxSamples, 1U);
    pushConstants.reset = reflectionHistoryValid && !temporalResetThisFrame ? 0U : 1U;
    pushConstants.debugMode = debugMode;

    std::array<VkDescriptorSet, 2> sets{bindless.set(), rayTracer->accelerationSet()};
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            reflectionPipelineLayout,
                            0,
                            static_cast<uint32_t>(sets.size()),
                            sets.data(),
                            0,
                            nullptr);
    vkCmdPushConstants(
        commandBuffer, reflectionPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    uint32_t groupsX = (currentRenderExtent.width + 7) / 8;
    uint32_t groupsY = (currentRenderExtent.height + 7) / 8;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, reflectionTracePipeline);
    vkCmdDispatch(commandBuffer, groupsX, groupsY, 1);
    memoryBarrier(VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, reflectionResolvePipeline);
    vkCmdDispatch(commandBuffer, groupsX, groupsY, 1);
    reflectionHistoryValid = true;

    // 색상은 다시 첨부물로, 깊이도 다시 첨부물로. 반투명 패스가 둘 다 이어 쓴다.
    imageBarrier(commandBuffer,
                 targets.color.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    imageBarrier(commandBuffer,
                 targets.depth.handle,
                 VK_IMAGE_ASPECT_DEPTH_BIT,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                 VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    frameProfiler.end(zone, commandBuffer);
}

} // namespace gfx
