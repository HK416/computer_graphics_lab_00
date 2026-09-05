// 콜라이더·유체 경계 선 표시.
// Renderer 의 멤버 함수 정의만 나눠 담은 번역 단위다. 선언은 renderer.h 하나에 있다.

#include "gfx/renderer_internal.h"

namespace gfx {

void Renderer::createDebugLinePipeline() {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.size = sizeof(DebugLinePushConstants);

    VkDescriptorSetLayout bindlessLayout = bindless.layout();
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &bindlessLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &debugLinePipelineLayout));

    VkShaderModule vertexModule = createShaderModule(context.device, "debug_line.vert.spv");
    VkShaderModule fragmentModule = createShaderModule(context.device, "debug_line.frag.spv");
    std::array<VkPipelineShaderStageCreateInfo, 2> stages{shaderStage(VK_SHADER_STAGE_VERTEX_BIT, vertexModule),
                                                          shaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, fragmentModule)};

    // 정점은 buffer device address 로 직접 읽는다. 이 저장소의 다른 파이프라인과 같은 방식이다.
    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterization{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    // 굵은 선은 선택 기능이다. 없으면 1 화소로 그린다.
    rasterization.lineWidth = context.caps.wideLines ? 2.0F : 1.0F;

    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // 깊이 첨부물이 없다. 표시 해상도에 그리는데 깊이 버퍼는 렌더 해상도라 크기가 다르기 때문이다.
    // 대신 프래그먼트가 깊이를 텍스처로 읽어 가림을 판정한다.
    VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.blendEnable = VK_TRUE;
    blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo colorBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &blendAttachment;

    VkDynamicState dynamicStates[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkFormat colorFormat = PRESENT_FORMAT;
    VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat;

    VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.pNext = &renderingInfo;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages.data();
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterization;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = debugLinePipelineLayout;
    VK_CHECK(vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &debugLinePipeline));

    vkDestroyShaderModule(context.device, fragmentModule, nullptr);
    vkDestroyShaderModule(context.device, vertexModule, nullptr);
}

void Renderer::reserveDebugLines(Frame& frame, uint32_t vertexCount) {
    uint32_t needed = std::max(vertexCount, 1U);
    if (needed <= frame.debugLineCapacity) {
        return;
    }
    uint32_t capacity = std::max(needed, frame.debugLineCapacity * 2);
    // 프레임 슬롯마다 하나라 이 시점에는 아무도 읽지 않는다. 타임라인 대기를 이미 지났다.
    destroyBuffer(context, frame.debugLineBuffer);
    frame.debugLineBuffer = createBuffer(context,
                                         static_cast<VkDeviceSize>(capacity) * sizeof(DebugLineVertex),
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                         MemoryLocation::HOST_WRITE,
                                         "디버그 선");
    frame.debugLineCapacity = capacity;
}

void Renderer::recordDebugLines(VkCommandBuffer commandBuffer,
                                Frame& frame,
                                const scene::Scene& scene,
                                VkExtent2D extent) {
    DebugLineOptions options;
    options.colliders = showColliders;
    options.fluidBounds = showColliders;
    options.selected = selectedObject;
    buildDebugLines(scene, options, debugLineVertices);
    if (debugLineVertices.empty()) {
        return;
    }

    uint32_t zone = frameProfiler.begin("디버그 선", commandBuffer);
    auto vertexCount = static_cast<uint32_t>(debugLineVertices.size());
    reserveDebugLines(frame, vertexCount);
    std::copy_n(debugLineVertices.data(), vertexCount, static_cast<DebugLineVertex*>(frame.debugLineBuffer.mapped));

    DebugLinePushConstants pushConstants{};
    // 종횡비는 장면을 그린 렌더 해상도로 잡는다. 표시 해상도로 잡으면 배율 반올림만큼 선이 어긋난다.
    float aspect = static_cast<float>(currentRenderExtent.width) / static_cast<float>(currentRenderExtent.height);
    // 지터는 넣지 않는다. 톤 매핑과 업스케일이 끝난 뒤에 그리므로 흔들 이유가 없다.
    //
    // ponytail: 깊이 버퍼는 지터가 들어간 투영으로 쓰였다. 시간축 업스케일을 켜면 실루엣 경계에서
    // 가림 판정이 프레임마다 한 화소쯤 흔들린다. 정확히 하려면 깊이를 읽을 uv 에서 지터를 빼야 한다.
    pushConstants.viewProjection = scene.camera.projectionMatrix(aspect) * scene.camera.viewMatrix();
    pushConstants.vertices = frame.debugLineBuffer.address;
    pushConstants.depthTexture = targets.depthSlot;
    // 경로 추적은 깊이 버퍼에 아무것도 쓰지 않는다(레이아웃만 맞춰 둔다). 그 내용으로 가림을
    // 판정하면 미정의 값이나 직전 래스터 프레임의 깊이를 읽으므로 그 모드에서는 늘 그린다.
    bool depthAvailable = !(usePathTracing && rayTracer != nullptr);
    pushConstants.occlude = colliderOcclusion && depthAvailable ? 1U : 0U;
    pushConstants.viewportSize[0] = static_cast<float>(extent.width);
    pushConstants.viewportSize[1] = static_cast<float>(extent.height);

    // 바로 앞 패스(톤 매핑 또는 공간 업스케일)가 같은 이미지를 색상 첨부물로 썼다. 렌더 패스
    // 인스턴스 사이에는 자동 순서가 없으므로 하늘·OIT 합성과 같은 배리어를 넣는다.
    imageBarrier(commandBuffer,
                 targets.present.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo lineColor = colorAttachment(targets.present.view, VK_ATTACHMENT_LOAD_OP_LOAD, {});
    VkRenderingInfo linePass{VK_STRUCTURE_TYPE_RENDERING_INFO};
    linePass.renderArea.extent = extent;
    linePass.layerCount = 1;
    linePass.colorAttachmentCount = 1;
    linePass.pColorAttachments = &lineColor;

    VkDescriptorSet bindlessSet = bindless.set();
    vkCmdBeginRendering(commandBuffer, &linePass);
    setFullViewport(commandBuffer, extent);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, debugLinePipeline);
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, debugLinePipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer,
                       debugLinePipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       sizeof(pushConstants),
                       &pushConstants);
    vkCmdDraw(commandBuffer, vertexCount, 1, 0, 0);
    vkCmdEndRendering(commandBuffer);
    frameProfiler.end(zone, commandBuffer);
}

} // namespace gfx
