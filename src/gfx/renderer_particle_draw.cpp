// GPU 입자 진행과 스프라이트 그리기(시뮬레이션은 gfx::ParticleSimulator).
// Renderer 의 멤버 함수 정의만 나눠 담은 번역 단위다. 선언은 renderer.h 하나에 있다.

#include "gfx/renderer_internal.h"

namespace gfx {

void Renderer::createParticlePipelines() {
    if (!particles->gpuAvailable()) {
        return;
    }
    // 컴퓨트와 같은 푸시 상수 범위·집합 0 이라 시뮬레이터의 블록을 그대로 쓴다.
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    range.size = sizeof(ParticlePushConstants);
    VkDescriptorSetLayout bindlessLayout = bindless.layout();
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &bindlessLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &range;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &particlePipelineLayout));

    VkShaderModule vertexModule = createShaderModule(context.device, "particle_sprite.vert.spv");
    VkShaderModule fragmentModule = createShaderModule(context.device, "particle_sprite.frag.spv");
    std::array<VkPipelineShaderStageCreateInfo, 2> stages{shaderStage(VK_SHADER_STAGE_VERTEX_BIT, vertexModule),
                                                          shaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, fragmentModule)};

    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkDynamicState dynamicStates[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;
    // 카메라를 향한 사각형이라 컬링할 면이 없다.
    VkPipelineRasterizationStateCreateInfo rasterization{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.lineWidth = 1.0F;
    // 깊이 첨부물이 없다. 장면 깊이는 프래그먼트가 텍스처로 읽어 손으로 판정한다(소프트 파티클).
    VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    // 미리 곱해진 알파. 하늘과 불투명이 이미 색상 대상에 들어 있다.
    VkPipelineColorBlendAttachmentState blend{};
    blend.blendEnable = VK_TRUE;
    blend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.colorBlendOp = VK_BLEND_OP_ADD;
    blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend.alphaBlendOp = VK_BLEND_OP_ADD;
    blend.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo colorBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &blend;
    VkFormat format = COLOR_FORMAT;
    VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &format;

    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    info.pNext = &renderingInfo;
    info.stageCount = static_cast<uint32_t>(stages.size());
    info.pStages = stages.data();
    info.pVertexInputState = &vertexInput;
    info.pInputAssemblyState = &inputAssembly;
    info.pViewportState = &viewportState;
    info.pRasterizationState = &rasterization;
    info.pMultisampleState = &multisample;
    info.pDepthStencilState = &depthStencil;
    info.pColorBlendState = &colorBlend;
    info.pDynamicState = &dynamicState;
    info.layout = particlePipelineLayout;
    VK_CHECK(vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &particleSpritePipeline));

    vkDestroyShaderModule(context.device, fragmentModule, nullptr);
    vkDestroyShaderModule(context.device, vertexModule, nullptr);
}

void Renderer::recordParticlePass(VkCommandBuffer commandBuffer, const Frame& frame, const scene::Scene& scene) {
    // 충돌을 원하는 시스템이 있고 광선 질의가 되면 이번 프레임의 가속 구조를 먼저 세운다. 스킨 노드가 이미
    // 돌았으므로 변형 정점까지 최신이다. updateAccelerationStructures 는 두 번 불러도 early-out 이라 DDGI·불투명
    // 노드가 다시 불러도 무해하다.
    bool rayQuery = particles->wantsCollision() && particles->collisionAvailable() && rayQueryShadowsAvailable();
    if (rayQuery) {
        updateAccelerationStructures(commandBuffer, scene);
        rayQuery = rayTracer->ready();
    }
    ParticleSimulator::SceneBuffers buffers;
    buffers.vertices = geometry.vertexBuffer.address;
    buffers.skinnedVertices = skinnedVertexBuffer.address;
    buffers.indices = geometry.indexBuffer.address;
    buffers.meshes = geometry.meshBuffer.address;
    buffers.lods = geometry.lodBuffer.address;
    buffers.instances = frame.instanceBuffer.address;
    auto slot = static_cast<uint32_t>(frameIndex % FRAMES_IN_FLIGHT);
    VkDescriptorSet accelerationSet = rayQuery ? rayTracer->accelerationSet() : VK_NULL_HANDLE;
    for (uint32_t index = 0; index < particles->systemCount(); ++index) {
        particles->record(commandBuffer,
                          slot,
                          index,
                          scene,
                          frameIndex,
                          rayQuery,
                          accelerationSet,
                          buffers,
                          frame.cameraBuffer.address);
    }
}

void Renderer::recordParticleSpritePass(VkCommandBuffer commandBuffer, const Frame& frame) {
    if (particleSpritePipeline == VK_NULL_HANDLE) {
        return;
    }
    VkRenderingAttachmentInfo color = colorAttachment(targets.color.view, VK_ATTACHMENT_LOAD_OP_LOAD, {});
    VkRenderingInfo pass{VK_STRUCTURE_TYPE_RENDERING_INFO};
    pass.renderArea.extent = currentRenderExtent;
    pass.layerCount = 1;
    pass.colorAttachmentCount = 1;
    pass.pColorAttachments = &color;
    vkCmdBeginRendering(commandBuffer, &pass);
    setFullViewport(commandBuffer, currentRenderExtent);
    VkDescriptorSet bindlessSet = bindless.set();
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, particlePipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, particleSpritePipeline);
    auto slot = static_cast<uint32_t>(frameIndex % FRAMES_IN_FLIGHT);
    for (uint32_t index = 0; index < particles->systemCount(); ++index) {
        uint32_t count = particles->count(index);
        if (count == 0) {
            continue;
        }
        ParticlePushConstants push;
        push.particles = particles->particleAddress(index);
        push.params = particles->paramsAddress(slot, index);
        push.camera = frame.cameraBuffer.address;
        push.depthTexture = targets.depthSlot;
        push.particleCount = count;
        push.sorted = particles->sortedAddress(index);
        vkCmdPushConstants(commandBuffer,
                           particlePipelineLayout,
                           VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           sizeof(push),
                           &push);
        // 진행 컴퓨트가 카메라 거리로 정렬해 둔 순서(먼 것 먼저)로 그린다.
        vkCmdDraw(commandBuffer, 6, count, 0, 0);
    }
    vkCmdEndRendering(commandBuffer);
}

} // namespace gfx
