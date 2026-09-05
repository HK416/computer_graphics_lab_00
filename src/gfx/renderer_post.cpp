// Bloom·자동 노출·톤 매핑·공간 업스케일 파이프라인.
// Renderer 의 멤버 함수 정의만 나눠 담은 번역 단위다. 선언은 renderer.h 하나에 있다.

#include "gfx/renderer_internal.h"

namespace gfx {

void Renderer::createBloomPipelines() {
    VkDescriptorSetLayout bindlessLayout = bindless.layout();
    auto createLayout = [&](uint32_t size) {
        VkPushConstantRange range{};
        range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        range.size = size;
        VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &bindlessLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &range;
        VkPipelineLayout layout = VK_NULL_HANDLE;
        VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &layout));
        return layout;
    };
    auto createPipeline = [&](const char* shader, VkPipelineLayout layout) {
        VkShaderModule module = createShaderModule(context.device, shader);
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = shaderStage(VK_SHADER_STAGE_COMPUTE_BIT, module);
        info.layout = layout;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VK_CHECK(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
        vkDestroyShaderModule(context.device, module, nullptr);
        return pipeline;
    };

    bloomPipelineLayout = createLayout(sizeof(BloomPushConstants));
    bloomDownsamplePipeline = createPipeline("bloom_downsample.comp.spv", bloomPipelineLayout);
    bloomUpsamplePipeline = createPipeline("bloom_upsample.comp.spv", bloomPipelineLayout);
    histogramPipelineLayout = createLayout(sizeof(HistogramPushConstants));
    histogramPipeline = createPipeline("exposure_histogram.comp.spv", histogramPipelineLayout);
    exposurePipelineLayout = createLayout(sizeof(ExposurePushConstants));
    exposurePipeline = createPipeline("exposure_average.comp.spv", exposurePipelineLayout);
}

void Renderer::recordPostEffects(VkCommandBuffer commandBuffer,
                                 const scene::PostProcess& post,
                                 uint32_t sourceSlot,
                                 VkExtent2D sourceExtent,
                                 uint32_t sampleCount) {
    bool useBloom = post.bloomIntensity > 0.0F;
    bloomActive = useBloom;
    if (!useBloom && !post.autoExposure) {
        // 다음에 켜면 옛 값에서 느리게 옮겨 가지 않고 바로 맞춘다.
        exposureNeedsReset = true;
        return;
    }
    uint32_t zone = frameProfiler.begin("후처리", commandBuffer);
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
    auto extentOf = [&](size_t level) {
        return VkExtent2D{std::max(targets.bloomExtent.width >> level, 1U),
                          std::max(targets.bloomExtent.height >> level, 1U)};
    };

    // 지난 프레임 톤 매핑이 Bloom 과 노출 버퍼를 아직 읽고 있을 수 있다.
    memoryBarrier(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    if (post.autoExposure) {
        vkCmdFillBuffer(commandBuffer, histogramBuffer.handle, 0, VK_WHOLE_SIZE, 0);
    }

    // 1) 아래로 내려가며 밉 사슬을 만든다. 첫 단계에서 임계값 아래를 깎아 내므로, 이 사슬에는
    // «밝은 곳»만 남는다. 자동 노출이 이 밉을 장면 휘도로 쓸 수 없는 이유다(아래 2번).
    size_t levels = targets.bloomStorageSlots.size();
    if (useBloom) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bloomDownsamplePipeline);
        vkCmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bloomPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
        for (size_t level = 0; level < levels; ++level) {
            VkExtent2D destination = extentOf(level);
            BloomPushConstants pushConstants{};
            pushConstants.sourceTexture = level == 0 ? sourceSlot : targets.bloomSampledSlots[level - 1];
            pushConstants.destinationStorage = targets.bloomStorageSlots[level];
            // 13탭 커널의 탭은 «대상» 화소의 ±0.5, ±1.0 자리에 놓여야 한다. 원본 텍셀 기준으로 잡으면
            // 렌더 배율 때문에 축소가 2 배가 아닐 때 발자국이 대상 화소를 덮지 못한다. 축소가 정확히
            // 2 배면 이 값은 예전의 1/원본 과 같다.
            pushConstants.sourceTexelSize[0] = 0.5F / static_cast<float>(destination.width);
            pushConstants.sourceTexelSize[1] = 0.5F / static_cast<float>(destination.height);
            pushConstants.destinationSize[0] = static_cast<int32_t>(destination.width);
            pushConstants.destinationSize[1] = static_cast<int32_t>(destination.height);
            pushConstants.sampleCount = level == 0 ? sampleCount : 0U;
            pushConstants.firstLevel = level == 0 ? 1U : 0U;
            // 무릎이 임계값보다 크면 곡선이 뒤집혀 어두운 화소를 오히려 증폭한다. Unity 가 무릎을
            // 임계값의 비율로 유도해 이 관계를 강제하는 것과 같은 이유로 여기서 자른다.
            pushConstants.threshold = std::max(post.bloomThreshold, 0.0F);
            pushConstants.knee = std::clamp(post.bloomKnee, 1e-3F, std::max(pushConstants.threshold, 1e-3F));
            vkCmdPushConstants(commandBuffer,
                               bloomPipelineLayout,
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0,
                               sizeof(pushConstants),
                               &pushConstants);
            vkCmdDispatch(commandBuffer, (destination.width + 7) / 8, (destination.height + 7) / 8, 1);
            memoryBarrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        }
    }

    // 2) 자동 노출. HDR 원본을 1/8 격자로 성기게 훑는다. Bloom 밉을 쓰던 때와 화소 수는 같지만,
    // 임계값 아래가 깎이지 않은 «장면 그대로»의 휘도를 본다.
    if (post.autoExposure) {
        VkExtent2D histogramExtent{std::max(sourceExtent.width / 8, 1U), std::max(sourceExtent.height / 8, 1U)};
        memoryBarrier(VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                      VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, histogramPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, histogramPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
        HistogramPushConstants histogram{};
        histogram.histogram = histogramBuffer.address;
        histogram.sourceTexture = sourceSlot;
        histogram.sourceSize[0] = static_cast<int32_t>(histogramExtent.width);
        histogram.sourceSize[1] = static_cast<int32_t>(histogramExtent.height);
        histogram.minLog = EXPOSURE_MIN_LOG;
        histogram.inverseLogRange = 1.0F / (EXPOSURE_MAX_LOG - EXPOSURE_MIN_LOG);
        histogram.sampleCount = sampleCount;
        vkCmdPushConstants(
            commandBuffer, histogramPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(histogram), &histogram);
        vkCmdDispatch(commandBuffer, (histogramExtent.width + 15) / 16, (histogramExtent.height + 15) / 16, 1);
        memoryBarrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, exposurePipeline);
        vkCmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, exposurePipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
        ExposurePushConstants exposurePush{};
        exposurePush.histogram = histogramBuffer.address;
        exposurePush.exposure = exposureBuffer.address;
        exposurePush.minLog = EXPOSURE_MIN_LOG;
        exposurePush.logRange = EXPOSURE_MAX_LOG - EXPOSURE_MIN_LOG;
        exposurePush.deltaSeconds = frameDeltaSeconds;
        exposurePush.adaptationSpeed = post.adaptationSpeed;
        exposurePush.minEv = post.exposureMinEv;
        exposurePush.maxEv = std::max(post.exposureMaxEv, post.exposureMinEv);
        exposurePush.reset = exposureNeedsReset ? 1U : 0U;
        vkCmdPushConstants(
            commandBuffer, exposurePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(exposurePush), &exposurePush);
        vkCmdDispatch(commandBuffer, 1, 1, 1);
        exposureNeedsReset = false;
    } else {
        exposureNeedsReset = true;
    }

    // 3) 위로 올라오며 텐트 필터로 더한다. 0단계에 모든 단계가 겹쳐 쌓인다.
    if (useBloom) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bloomUpsamplePipeline);
        vkCmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, bloomPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
        for (size_t level = levels - 1; level > 0; --level) {
            VkExtent2D from = extentOf(level);
            VkExtent2D destination = extentOf(level - 1);
            BloomPushConstants pushConstants{};
            pushConstants.sourceTexture = targets.bloomSampledSlots[level];
            pushConstants.destinationStorage = targets.bloomStorageSlots[level - 1];
            pushConstants.sourceTexelSize[0] = 1.0F / static_cast<float>(from.width);
            pushConstants.sourceTexelSize[1] = 1.0F / static_cast<float>(from.height);
            pushConstants.destinationSize[0] = static_cast<int32_t>(destination.width);
            pushConstants.destinationSize[1] = static_cast<int32_t>(destination.height);
            pushConstants.scatter = std::clamp(post.bloomScatter, 0.0F, 1.0F);
            vkCmdPushConstants(commandBuffer,
                               bloomPipelineLayout,
                               VK_SHADER_STAGE_COMPUTE_BIT,
                               0,
                               sizeof(pushConstants),
                               &pushConstants);
            vkCmdDispatch(commandBuffer, (destination.width + 7) / 8, (destination.height + 7) / 8, 1);
            memoryBarrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        }
    }

    // 톤 매핑(프래그먼트)이 Bloom 0단계와 노출 버퍼를 읽는다.
    memoryBarrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    frameProfiler.end(zone, commandBuffer);
}

void Renderer::createPostPipelines() {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.size =
        std::max({sizeof(TonemapPushConstants), sizeof(UpscalePushConstants), sizeof(SkyPushConstants)});

    VkDescriptorSetLayout bindlessLayout = bindless.layout();
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &bindlessLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &postPipelineLayout));

    VkShaderModule vertexModule = createShaderModule(context.device, "fullscreen.vert.spv");
    VkShaderModule compositeFragment = createShaderModule(context.device, "oit_composite.frag.spv");
    VkShaderModule skyFragment = createShaderModule(context.device, "sky.frag.spv");
    VkShaderModule tonemapFragment = createShaderModule(context.device, "tonemap.frag.spv");

    std::array<VkPipelineShaderStageCreateInfo, 2> stages{shaderStage(VK_SHADER_STAGE_VERTEX_BIT, vertexModule),
                                                          shaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, compositeFragment)};

    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterization{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.lineWidth = 1.0F;

    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};

    constexpr VkColorComponentFlags ALL_CHANNELS =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    std::array<VkPipelineColorBlendAttachmentState, 2> blendAttachments{};
    blendAttachments[0].colorWriteMask = ALL_CHANNELS;
    blendAttachments[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
    // 합성은 알파에 담긴 잔여 투과율로 src*(1-a) + dst*a 를 만든다.
    blendAttachments[0].blendEnable = VK_TRUE;
    blendAttachments[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachments[0].dstColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachments[0].colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachments[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    blendAttachments[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachments[0].alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = blendAttachments.data();

    VkDynamicState dynamicStates[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    std::array<VkFormat, 2> colorFormats{COLOR_FORMAT, VELOCITY_FORMAT};
    VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = colorFormats.data();

    VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.pNext = &renderingInfo;
    pipelineInfo.stageCount = static_cast<uint32_t>(stages.size());
    pipelineInfo.pStages = stages.data();
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterization;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = postPipelineLayout;
    VK_CHECK(vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &compositePipeline));

    // 하늘은 색상과 모션 벡터 둘에 쓴다. 깊이가 정확히 0(reverse-Z 의 무한 원거리)인 화소만
    // 통과시켜 셰이더에서 다시 거르지 않는다.
    stages[1].module = skyFragment;
    blendAttachments[0].blendEnable = VK_FALSE;
    renderingInfo.colorAttachmentCount = 2;
    renderingInfo.depthAttachmentFormat = DEPTH_FORMAT;
    colorBlend.attachmentCount = 2;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_EQUAL;
    VK_CHECK(vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &skyPipeline));

    stages[1].module = tonemapFragment;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.depthAttachmentFormat = VK_FORMAT_UNDEFINED;
    colorBlend.attachmentCount = 1;
    depthStencil = {VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    colorFormats[0] = PRESENT_FORMAT;
    VK_CHECK(vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &tonemapPipeline));

    // 업스케일은 통과와 공간 확대 두 벌을 특수화 상수로 나눈다.
    VkShaderModule upscaleFragment = createShaderModule(context.device, "upscale.frag.spv");
    uint32_t upscaleMode = 0;
    VkSpecializationMapEntry upscaleEntry{0, 0, sizeof(uint32_t)};
    VkSpecializationInfo upscaleSpecialization{};
    upscaleSpecialization.mapEntryCount = 1;
    upscaleSpecialization.pMapEntries = &upscaleEntry;
    upscaleSpecialization.dataSize = sizeof(upscaleMode);
    upscaleSpecialization.pData = &upscaleMode;

    stages[1].module = upscaleFragment;
    stages[1].pSpecializationInfo = &upscaleSpecialization;
    for (uint32_t variant = 0; variant < upscalePipelines.size(); ++variant) {
        upscaleMode = variant;
        VK_CHECK(vkCreateGraphicsPipelines(
            context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &upscalePipelines[variant]));
    }

    vkDestroyShaderModule(context.device, upscaleFragment, nullptr);
    vkDestroyShaderModule(context.device, tonemapFragment, nullptr);
    vkDestroyShaderModule(context.device, skyFragment, nullptr);
    vkDestroyShaderModule(context.device, compositeFragment, nullptr);
    vkDestroyShaderModule(context.device, vertexModule, nullptr);
}

} // namespace gfx
