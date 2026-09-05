// 스킨 컴퓨트, GPU 컬링, HZB.
// Renderer 의 멤버 함수 정의만 나눠 담은 번역 단위다. 선언은 renderer.h 하나에 있다.

#include "gfx/renderer_internal.h"

namespace gfx {

void Renderer::createSkinPipeline() {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.size = sizeof(SkinPushConstants);

    VkDescriptorSetLayout skinBindlessLayout = bindless.layout();
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &skinBindlessLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &skinPipelineLayout));

    VkShaderModule module = createShaderModule(context.device, "skin.comp.spv");
    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = shaderStage(VK_SHADER_STAGE_COMPUTE_BIT, module);
    pipelineInfo.layout = skinPipelineLayout;
    VK_CHECK(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &skinPipeline));
    vkDestroyShaderModule(context.device, module, nullptr);

    pushConstantRange.size = sizeof(SkinBoundsPushConstants);
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &skinBoundsPipelineLayout));
    VkShaderModule boundsModule = createShaderModule(context.device, "skin_bounds.comp.spv");
    pipelineInfo.stage = shaderStage(VK_SHADER_STAGE_COMPUTE_BIT, boundsModule);
    pipelineInfo.layout = skinBoundsPipelineLayout;
    VK_CHECK(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &skinBoundsPipeline));
    vkDestroyShaderModule(context.device, boundsModule, nullptr);
}

void Renderer::createCullPipeline() {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.size = sizeof(CullPushConstants);

    VkDescriptorSetLayout cullBindlessLayout = bindless.layout();
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &cullBindlessLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &cullPipelineLayout));

    VkShaderModule module = createShaderModule(context.device, "cull_meshlets.comp.spv");
    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = shaderStage(VK_SHADER_STAGE_COMPUTE_BIT, module);
    pipelineInfo.layout = cullPipelineLayout;
    VK_CHECK(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &cullPipeline));
    vkDestroyShaderModule(context.device, module, nullptr);

    if (context.caps.drawIndirectCount) {
        drawIndexedIndirectCount = reinterpret_cast<PFN_vkCmdDrawIndexedIndirectCount>(
            vkGetDeviceProcAddr(context.device, "vkCmdDrawIndexedIndirectCount"));
    }
    VkPushConstantRange hzbRange{};
    hzbRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    hzbRange.size = sizeof(HzbPushConstants);

    VkDescriptorSetLayout bindlessLayout = bindless.layout();
    VkPipelineLayoutCreateInfo hzbLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    hzbLayoutInfo.setLayoutCount = 1;
    hzbLayoutInfo.pSetLayouts = &bindlessLayout;
    hzbLayoutInfo.pushConstantRangeCount = 1;
    hzbLayoutInfo.pPushConstantRanges = &hzbRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &hzbLayoutInfo, nullptr, &hzbPipelineLayout));

    VkShaderModule hzbModule = createShaderModule(context.device, "hzb_reduce.comp.spv");
    VkComputePipelineCreateInfo hzbPipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    hzbPipelineInfo.stage = shaderStage(VK_SHADER_STAGE_COMPUTE_BIT, hzbModule);
    hzbPipelineInfo.layout = hzbPipelineLayout;
    VK_CHECK(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &hzbPipelineInfo, nullptr, &hzbPipeline));
    vkDestroyShaderModule(context.device, hzbModule, nullptr);

    spdlog::info("컴퓨트 컬링 준비 완료 (압축 간접 그리기: {})", drawIndexedIndirectCount != nullptr);
}

void Renderer::recordHzbPass(VkCommandBuffer commandBuffer) {
    VkDescriptorSet bindlessSet = bindless.set();
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, hzbPipeline);
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, hzbPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);

    // 다음 단계 리덕션과, 마지막에는 2차 패스의 컬 컴퓨트·태스크 셰이더가 읽는다.
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;

    uint32_t sourceWidth = currentRenderExtent.width;
    uint32_t sourceHeight = currentRenderExtent.height;
    for (size_t level = 0; level < targets.hzbStorageSlots.size(); ++level) {
        uint32_t destinationWidth = std::max(targets.hzbExtent.width >> level, 1U);
        uint32_t destinationHeight = std::max(targets.hzbExtent.height >> level, 1U);

        HzbPushConstants pushConstants{};
        // 0단계는 깊이 버퍼에서, 나머지는 바로 위 단계에서 줄인다.
        pushConstants.sourceTexture = level == 0 ? targets.depthSlot : targets.hzbSampledSlot;
        pushConstants.sourceLevel = level == 0 ? 0.0F : static_cast<float>(level - 1);
        pushConstants.destinationStorage = targets.hzbStorageSlots[level];
        pushConstants.sourceSize[0] = static_cast<int32_t>(sourceWidth);
        pushConstants.sourceSize[1] = static_cast<int32_t>(sourceHeight);
        pushConstants.destinationSize[0] = static_cast<int32_t>(destinationWidth);
        pushConstants.destinationSize[1] = static_cast<int32_t>(destinationHeight);

        vkCmdPushConstants(
            commandBuffer, hzbPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(commandBuffer, (destinationWidth + 7) / 8, (destinationHeight + 7) / 8, 1);
        vkCmdPipelineBarrier2(commandBuffer, &dependency);

        sourceWidth = destinationWidth;
        sourceHeight = destinationHeight;
    }
}

void Renderer::recordCullPass(VkCommandBuffer commandBuffer, const FrameBatches& batches, uint32_t phase) {
    Frame& frame = frames[frameIndex % FRAMES_IN_FLIGHT];

    // 1차 패스의 간접 드로우가 아직 명령 버퍼를 읽는 중일 수 있고, 지난 컬 컴퓨트의 비트 쓰기도
    // 이번 읽기에 앞서야 한다. 개수 버퍼를 지우기 전에 둘 다 끝낸다.
    VkMemoryBarrier2 drawBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    drawBarrier.srcStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    drawBarrier.srcAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    drawBarrier.dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    drawBarrier.dstAccessMask =
        VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo drawDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    drawDependency.memoryBarrierCount = 1;
    drawDependency.pMemoryBarriers = &drawBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &drawDependency);

    vkCmdFillBuffer(commandBuffer, frame.drawCountBuffer.handle, 0, VK_WHOLE_SIZE, 0);
    if (drawIndexedIndirectCount == nullptr) {
        // 압축 간접 그리기가 없으면 상한만큼 그리므로, 남은 자리는 0 으로 채워 무효 명령으로 만든다.
        vkCmdFillBuffer(commandBuffer, frame.meshletDrawBuffer.handle, 0, VK_WHOLE_SIZE, 0);
    }

    VkMemoryBarrier2 clearBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    clearBarrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    clearBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    clearBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    clearBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo clearDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    clearDependency.memoryBarrierCount = 1;
    clearDependency.pMemoryBarriers = &clearBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &clearDependency);

    CullPushConstants pushConstants{};
    pushConstants.instances = frame.instanceBuffer.address;
    pushConstants.meshes = geometry.meshBuffer.address;
    pushConstants.meshlets = geometry.meshletBuffer.address;
    pushConstants.camera = frame.cameraBuffer.address;
    pushConstants.drawCommands = frame.meshletDrawBuffer.address;
    pushConstants.drawCounts = frame.drawCountBuffer.address;
    pushConstants.instanceCount = batches.instanceCount;
    pushConstants.flags =
        (settings.frustumCulling ? CULL_FLAG_FRUSTUM : 0U) | (settings.coneCulling ? CULL_FLAG_CONE : 0U);
    pushConstants.phase = phase;
    pushConstants.network = frame.lodNetworkBuffer.address;
    pushConstants.skinnedBounds = skinnedBoundsBuffer.address;
    pushConstants.visibility = meshletVisibilityBuffer.address;
    pushConstants.drawMeshlets = frame.meshletDrawMeshletBuffer.address;
    if (settings.useNeuralLod) {
        pushConstants.flags |= CULL_FLAG_NEURAL_LOD;
    }

    VkDescriptorSet cullBindlessSet = bindless.set();
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cullPipelineLayout, 0, 1, &cullBindlessSet, 0, nullptr);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cullPipeline);
    vkCmdPushConstants(
        commandBuffer, cullPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, std::max(batches.instanceCount, 1U), 1, 1);

    VkMemoryBarrier2 cullBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    cullBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    cullBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    // 명령은 간접 그리기가, 명령별 meshlet 번호는 정점 셰이더가 읽는다.
    cullBarrier.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT;
    cullBarrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    VkDependencyInfo cullDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    cullDependency.memoryBarrierCount = 1;
    cullDependency.pMemoryBarriers = &cullBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &cullDependency);
}

void Renderer::recordSkinPass(VkCommandBuffer commandBuffer, const Frame& frame) {
    if (skinDispatches.empty()) {
        return;
    }
    uint32_t zone = frameProfiler.begin("스킨", commandBuffer);
    VkDescriptorSet bindlessSet = bindless.set();

    auto memoryBarrier = [&](VkPipelineStageFlags2 sourceStage,
                             VkAccessFlags2 sourceAccess,
                             VkPipelineStageFlags2 destinationStage,
                             VkAccessFlags2 destinationAccess) {
        VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        barrier.srcStageMask = sourceStage;
        barrier.srcAccessMask = sourceAccess;
        barrier.dstStageMask = destinationStage;
        barrier.dstAccessMask = destinationAccess;
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependency);
    };
    // 이 버퍼를 읽는 단계 전부. 지난 프레임의 읽기가 끝나기를 기다리고, 이번 프레임의 읽기에 앞선다.
    constexpr VkPipelineStageFlags2 READER_STAGES =
        VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
        VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
        VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
        VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    memoryBarrier(READER_STAGES,
                  VK_ACCESS_2_SHADER_READ_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    uint32_t currentBase = skinnedHalf * skinnedVertexCapacity;
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, skinPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, skinPipeline);
    for (const SkinDispatch& dispatch : skinDispatches) {
        SkinPushConstants pushConstants{geometry.vertexBuffer.address,
                                        skinnedVertexBuffer.address,
                                        frame.jointBuffer.address,
                                        geometry.skinWeightBuffer.address,
                                        dispatch.sourceOffset,
                                        currentBase + dispatch.destinationOffset,
                                        dispatch.jointOffset,
                                        dispatch.vertexCount,
                                        dispatch.skinWeightOffset};
        vkCmdPushConstants(
            commandBuffer, skinPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(commandBuffer, (dispatch.vertexCount + SKIN_GROUP_SIZE - 1) / SKIN_GROUP_SIZE, 1, 1);
    }

    // 경계 구 컴퓨트가 변형 정점을 읽는다.
    memoryBarrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, skinBoundsPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, skinBoundsPipeline);
    for (const SkinDispatch& dispatch : skinDispatches) {
        SkinBoundsPushConstants pushConstants{skinnedVertexBuffer.address,
                                              geometry.meshletBuffer.address,
                                              skinnedBoundsBuffer.address,
                                              dispatch.meshletOffset,
                                              dispatch.meshletCount,
                                              dispatch.sourceOffset,
                                              currentBase + dispatch.destinationOffset,
                                              dispatch.boundsOffset,
                                              geometry.meshletVertexBuffer.address};
        vkCmdPushConstants(commandBuffer,
                           skinBoundsPipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT,
                           0,
                           sizeof(pushConstants),
                           &pushConstants);
        vkCmdDispatch(commandBuffer, dispatch.meshletCount, 1, 1);
    }

    // 정점은 그림자·장면 패스와 가속 구조 구축이, 경계 구는 컬 컴퓨트와 태스크 셰이더가 읽는다.
    memoryBarrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  READER_STAGES,
                  VK_ACCESS_2_SHADER_READ_BIT);
    frameProfiler.end(zone, commandBuffer);
}

} // namespace gfx
