// 유체 입자 인스턴스·물 표면 그리기(시뮬레이션은 gfx::FluidSimulator).
// Renderer 의 멤버 함수 정의만 나눠 담은 번역 단위다. 선언은 renderer.h 하나에 있다.

#include "gfx/renderer_internal.h"

namespace gfx {

void Renderer::createFluidSurfacePipelines() {
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    range.size = sizeof(FluidDrawPushConstants);
    VkDescriptorSetLayout bindlessLayout = bindless.layout();
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &bindlessLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &range;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &fluidSurfaceLayout));

    VkShaderModule vertexModule = createShaderModule(context.device, "fluid_surface.vert.spv");
    VkShaderModule thicknessModule = createShaderModule(context.device, "fluid_thickness.frag.spv");
    VkShaderModule surfaceModule = createShaderModule(context.device, "fluid_surface.frag.spv");

    // 정점은 buffer device address 로 직접 읽는다. 이 저장소의 다른 파이프라인과 같은 방식이다.
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
    // 앞뒤 면을 모두 그린다. 두께 패스는 양쪽이 다 필요하고, 표면 패스는 카메라가 물 안에 들어가도
    // 표면이 사라지지 않아야 한다. 프래그먼트가 법선을 시선 쪽으로 뒤집는다.
    VkPipelineRasterizationStateCreateInfo rasterization{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.lineWidth = 1.0F;

    // 1) 두께. 뒷면은 더하고 앞면은 빼는 것을 가산 혼합으로 쌓으면 겹친 층까지 모두 더해진다.
    //    장면보다 뒤에 있는 조각은 깊이 검사로 걸러지지만 깊이를 쓰지는 않는다.
    {
        std::array<VkPipelineShaderStageCreateInfo, 2> stages{
            shaderStage(VK_SHADER_STAGE_VERTEX_BIT, vertexModule),
            shaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, thicknessModule)};
        VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_FALSE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
        VkPipelineColorBlendAttachmentState blend{};
        blend.blendEnable = VK_TRUE;
        blend.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.colorBlendOp = VK_BLEND_OP_ADD;
        blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.alphaBlendOp = VK_BLEND_OP_ADD;
        blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
        VkPipelineColorBlendStateCreateInfo colorBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        colorBlend.attachmentCount = 1;
        colorBlend.pAttachments = &blend;
        VkFormat format = thicknessFormat;
        VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachmentFormats = &format;
        renderingInfo.depthAttachmentFormat = DEPTH_FORMAT;
        VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        info.pNext = &renderingInfo;
        info.stageCount = 2;
        info.pStages = stages.data();
        info.pVertexInputState = &vertexInput;
        info.pInputAssemblyState = &inputAssembly;
        info.pViewportState = &viewportState;
        info.pRasterizationState = &rasterization;
        info.pMultisampleState = &multisample;
        info.pDepthStencilState = &depthStencil;
        info.pColorBlendState = &colorBlend;
        info.pDynamicState = &dynamicState;
        info.layout = fluidSurfaceLayout;
        VK_CHECK(vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &fluidThicknessPipeline));
    }

    // 2) 깊이 프리패스. 앞면만 그려 가장 가까운 물 표면의 깊이를 박는다. 프래그먼트 셰이더가 없다.
    {
        VkPipelineShaderStageCreateInfo stage = shaderStage(VK_SHADER_STAGE_VERTEX_BIT, vertexModule);
        VkPipelineRasterizationStateCreateInfo frontOnly = rasterization;
        frontOnly.cullMode = VK_CULL_MODE_BACK_BIT;
        VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
        VkPipelineColorBlendStateCreateInfo colorBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        renderingInfo.depthAttachmentFormat = DEPTH_FORMAT;
        VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        info.pNext = &renderingInfo;
        info.stageCount = 1;
        info.pStages = &stage;
        info.pVertexInputState = &vertexInput;
        info.pInputAssemblyState = &inputAssembly;
        info.pViewportState = &viewportState;
        info.pRasterizationState = &frontOnly;
        info.pMultisampleState = &multisample;
        info.pDepthStencilState = &depthStencil;
        info.pColorBlendState = &colorBlend;
        info.pDynamicState = &dynamicState;
        info.layout = fluidSurfaceLayout;
        VK_CHECK(vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &fluidDepthPipeline));
    }

    // 3) 표면. 프리패스가 박은 깊이와 «같은» 조각만 남긴다. 그래야 화소마다 한 번만 섞인다. 마칭
    //    큐브의 삼각형 순서는 원자 덧셈으로 정해져 프레임마다 달라지므로, 깊이 쓰기로 앞뒤를 가리게
    //    두면 어느 화소가 두 번 섞이는지가 매 프레임 바뀌어 물이 깜빡인다.
    {
        std::array<VkPipelineShaderStageCreateInfo, 2> stages{shaderStage(VK_SHADER_STAGE_VERTEX_BIT, vertexModule),
                                                              shaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, surfaceModule)};
        VkPipelineRasterizationStateCreateInfo frontOnly = rasterization;
        frontOnly.cullMode = VK_CULL_MODE_BACK_BIT;
        VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_FALSE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_EQUAL;
        std::array<VkPipelineColorBlendAttachmentState, 3> blends{};
        // 미리 곱해진 알파. 뒤에 있는 배경은 하늘 패스까지 끝나 이미 색상 대상에 들어 있다.
        blends[0].blendEnable = VK_TRUE;
        blends[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blends[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blends[0].colorBlendOp = VK_BLEND_OP_ADD;
        blends[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blends[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blends[0].alphaBlendOp = VK_BLEND_OP_ADD;
        blends[0].colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        // 모션 벡터는 섞으면 안 된다. 물이 덮은 화소는 물의 변위로 바꿔 쓴다.
        blends[1].blendEnable = VK_FALSE;
        blends[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
        // 반사 가중치도 덮어쓴다. 0 이면 광선 반사 컴퓨트가 그 화소를 건너뛰므로, 물에 가려진 표면의
        // 법선으로 광선을 쏘는 일이 없다.
        blends[2].blendEnable = VK_FALSE;
        blends[2].colorWriteMask =
            VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo colorBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        colorBlend.attachmentCount = static_cast<uint32_t>(blends.size());
        colorBlend.pAttachments = blends.data();
        std::array<VkFormat, 3> formats{COLOR_FORMAT, VELOCITY_FORMAT, COLOR_FORMAT};
        VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
        renderingInfo.colorAttachmentCount = static_cast<uint32_t>(formats.size());
        renderingInfo.pColorAttachmentFormats = formats.data();
        renderingInfo.depthAttachmentFormat = DEPTH_FORMAT;
        VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        info.pNext = &renderingInfo;
        info.stageCount = 2;
        info.pStages = stages.data();
        info.pVertexInputState = &vertexInput;
        info.pInputAssemblyState = &inputAssembly;
        info.pViewportState = &viewportState;
        info.pRasterizationState = &rasterization;
        info.pMultisampleState = &multisample;
        info.pDepthStencilState = &depthStencil;
        info.pColorBlendState = &colorBlend;
        info.pDynamicState = &dynamicState;
        info.layout = fluidSurfaceLayout;
        VK_CHECK(vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &fluidSurfacePipeline));
    }

    vkDestroyShaderModule(context.device, surfaceModule, nullptr);
    vkDestroyShaderModule(context.device, thicknessModule, nullptr);
    vkDestroyShaderModule(context.device, vertexModule, nullptr);
}

void Renderer::recordFluidSurfacePass(VkCommandBuffer commandBuffer,
                                      const Frame& frame,
                                      const FrameBatches& batches,
                                      const scene::Scene& scene) {
    // 표면으로 그릴 유체를 먼저 모은다. 하나도 없으면 두께 대상을 지울 이유도 없다.
    std::vector<uint32_t> surfaces;
    for (uint32_t index = 0; index < batches.fluidDraws.count; ++index) {
        if (fluid->surfaceActive(index)) {
            surfaces.push_back(index);
        }
    }
    if (surfaces.empty()) {
        return;
    }

    auto slot = static_cast<uint32_t>(frameIndex % FRAMES_IN_FLIGHT);
    VkDescriptorSet bindlessSet = bindless.set();
    auto drawFluid = [&](uint32_t index, uint32_t thicknessSlot) {
        const scene::Fluid& settings = scene.fluids[index];
        FluidDrawPushConstants push{};
        push.camera = frame.cameraBuffer.address;
        push.vertices = fluid->surfaceVertexAddress(slot, index);
        push.lights = frame.lightBuffer.address;
        push.waterColor = glm::vec4{settings.waterColor, settings.surfaceRoughness};
        push.absorption = glm::vec4{settings.absorption, settings.thicknessScale};
        push.thicknessTexture = thicknessSlot;
        vkCmdPushConstants(commandBuffer,
                           fluidSurfaceLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           sizeof(push),
                           &push);
        // GPU 백엔드는 정점 수를 GPU 만 아니까 간접 그리기로 낸다. CPU 백엔드도 같은 버퍼를 채워
        // 두므로 그리기 경로가 하나다.
        vkCmdDrawIndirect(commandBuffer, fluid->surfaceDrawBuffer(slot, index), 0, 1, sizeof(VkDrawIndirectCommand));
    };
    auto beginPass = [&](const VkRenderingInfo& info, VkPipeline pipeline) {
        vkCmdBeginRendering(commandBuffer, &info);
        setFullViewport(commandBuffer, currentRenderExtent);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, fluidSurfaceLayout, 0, 1, &bindlessSet, 0, nullptr);
    };

    uint32_t zone = frameProfiler.begin("물 표면", commandBuffer);
    // 앞선 패스가 깊이에 쓴 것을 다 마쳐야 읽을 수 있다. 하늘을 건너뛰면 불투명 패스가 마지막이다.
    imageBarrier(commandBuffer,
                 targets.depth.handle,
                 VK_IMAGE_ASPECT_DEPTH_BIT,
                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
    // 반사 가중치는 광선 반사 컴퓨트가 읽으려고 이미 읽기 전용으로 넘어가 있다. 덮어쓰려면 다시
    // 첨부물로 되돌린다.
    imageBarrier(commandBuffer,
                 targets.guideSpecularAlbedo.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    // 앞선 패스(불투명·하늘)가 색상과 모션 벡터에 쓴 것을 다 마쳐야 그 위에 섞을 수 있다. 렌더링
    // 구간이 나뉘어 있으면 자동으로 이어지지 않는다.
    for (const Image* image : {&targets.color, &targets.velocity}) {
        imageBarrier(commandBuffer,
                     image->handle,
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    }
    imageBarrier(commandBuffer,
                 targets.fluidThickness.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo depthLoad{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthLoad.imageView = targets.depth.view;
    depthLoad.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthLoad.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    depthLoad.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    // 1) 두께. 앞뒤 면을 모두 그려 «뒷면 거리 − 앞면 거리» 를 가산 혼합으로 쌓는다. 깊이는 읽기만 한다.
    VkRenderingAttachmentInfo thicknessColor =
        colorAttachment(targets.fluidThickness.view, VK_ATTACHMENT_LOAD_OP_CLEAR, {{0.0F, 0.0F, 0.0F, 0.0F}});
    VkRenderingAttachmentInfo thicknessDepth = depthLoad;
    thicknessDepth.storeOp = VK_ATTACHMENT_STORE_OP_NONE;
    VkRenderingInfo thicknessPass{VK_STRUCTURE_TYPE_RENDERING_INFO};
    thicknessPass.renderArea.extent = currentRenderExtent;
    thicknessPass.layerCount = 1;
    thicknessPass.colorAttachmentCount = 1;
    thicknessPass.pColorAttachments = &thicknessColor;
    thicknessPass.pDepthAttachment = &thicknessDepth;

    beginPass(thicknessPass, fluidThicknessPipeline);
    for (uint32_t index : surfaces) {
        drawFluid(index, asset::INVALID_TEXTURE);
    }
    vkCmdEndRendering(commandBuffer);

    // 2) 깊이 프리패스. 앞면만 그려 «가장 가까운 물 표면» 의 깊이를 박아 둔다. 이것이 없으면 마칭이
    //    내놓는 삼각형 순서가 프레임마다 달라 한 화소가 한 번 섞이기도 두 번 섞이기도 한다.
    VkRenderingAttachmentInfo prepassDepth = depthLoad;
    VkRenderingInfo prepass{VK_STRUCTURE_TYPE_RENDERING_INFO};
    prepass.renderArea.extent = currentRenderExtent;
    prepass.layerCount = 1;
    prepass.pDepthAttachment = &prepassDepth;

    // 두께는 깊이를 읽고 프리패스는 깊이를 쓴다. 사이를 묶어 준다.
    VkMemoryBarrier2 depthBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    depthBarrier.srcStageMask =
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    depthBarrier.srcAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT;
    depthBarrier.dstStageMask =
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    depthBarrier.dstAccessMask =
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    VkDependencyInfo depthDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    depthDependency.memoryBarrierCount = 1;
    depthDependency.pMemoryBarriers = &depthBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &depthDependency);

    beginPass(prepass, fluidDepthPipeline);
    for (uint32_t index : surfaces) {
        drawFluid(index, asset::INVALID_TEXTURE);
    }
    vkCmdEndRendering(commandBuffer);

    imageBarrier(commandBuffer,
                 targets.fluidThickness.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    vkCmdPipelineBarrier2(commandBuffer, &depthDependency);

    // 3) 표면. 프리패스가 박아 둔 깊이와 «같은» 조각만 남으므로 화소마다 딱 한 번 섞인다.
    //    반사 가중치를 0 으로 덮어 광선 반사가 물에 가려진 표면의 법선으로 광선을 쏘지 않게 한다.
    std::array<VkRenderingAttachmentInfo, 3> surfaceColor{
        colorAttachment(targets.color.view, VK_ATTACHMENT_LOAD_OP_LOAD, {}),
        colorAttachment(targets.velocity.view, VK_ATTACHMENT_LOAD_OP_LOAD, {}),
        colorAttachment(targets.guideSpecularAlbedo.view, VK_ATTACHMENT_LOAD_OP_LOAD, {})};
    VkRenderingAttachmentInfo surfaceDepth = depthLoad;
    surfaceDepth.storeOp = VK_ATTACHMENT_STORE_OP_NONE;
    VkRenderingInfo surfacePass{VK_STRUCTURE_TYPE_RENDERING_INFO};
    surfacePass.renderArea.extent = currentRenderExtent;
    surfacePass.layerCount = 1;
    surfacePass.colorAttachmentCount = static_cast<uint32_t>(surfaceColor.size());
    surfacePass.pColorAttachments = surfaceColor.data();
    surfacePass.pDepthAttachment = &surfaceDepth;

    beginPass(surfacePass, fluidSurfacePipeline);
    for (uint32_t index : surfaces) {
        drawFluid(index, targets.fluidThicknessSlot);
    }
    vkCmdEndRendering(commandBuffer);

    // 광선 반사 컴퓨트가 다시 읽을 수 있게 되돌린다.
    imageBarrier(commandBuffer,
                 targets.guideSpecularAlbedo.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    frameProfiler.end(zone, commandBuffer);
}

void Renderer::recordFluidPass(VkCommandBuffer commandBuffer,
                               const Frame& frame,
                               const FrameBatches& batches,
                               const scene::Scene& scene,
                               bool wantsTlas) {
    auto slot = static_cast<uint32_t>(frameIndex % FRAMES_IN_FLIGHT);
    uint32_t particles = fluid->totalParticles();
    VkDeviceAddress tlasAddress = 0;
    VkDeviceAddress sphereBlas = 0;
    if (wantsTlas) {
        sphereBlas = rayTracer->bottomLevelAddress(fluidSphereMesh);
        if (sphereBlas != 0) {
            // updateTopLevel 이 기록 중에 버퍼를 다시 잡으면 여기서 넘긴 주소가 낡는다. 미리 잡아 둔다.
            rayTracer->reserveInstances(slot, particles + batches.instanceCount);
            tlasAddress = rayTracer->instanceBufferAddress(slot);
            fluidTlasPrepended = particles;
        }
    }
    // CPU 백엔드는 매핑된 버퍼에 직접 쓴다. TLAS 인스턴스 버퍼도 호스트에서 보이는 자리다.
    void* tlasMapped = tlasAddress != 0 ? rayTracer->instanceBufferMapped(slot) : nullptr;

    uint32_t base = 0;
    for (uint32_t f = 0; f < batches.fluidDraws.count; ++f) {
        uint32_t count = fluid->particleCount(f);
        if (count == 0) {
            continue;
        }
        if (fluid->onCpu(f)) {
            fluid->writeCpuInstances(f,
                                     scene,
                                     frameDeltaSeconds,
                                     frame.instanceBuffer.mapped,
                                     batches.fluidInstanceBase + base,
                                     tlasMapped,
                                     base,
                                     sphereBlas,
                                     fluidSphereMesh,
                                     temporalResetThisFrame);
        } else {
            fluid->record(commandBuffer,
                          slot,
                          f,
                          scene,
                          frameDeltaSeconds,
                          frame.instanceBuffer.address,
                          batches.fluidInstanceBase + base,
                          tlasAddress,
                          base,
                          sphereBlas,
                          fluidSphereMesh,
                          temporalResetThisFrame);
        }
        // 표면은 시뮬레이션이 끝난 위치로 만든다. 둘 다 여기서 해야 장면 패스가 읽기 전에 준비된다.
        if (fluid->surfaceActive(f)) {
            if (fluid->onCpu(f)) {
                fluid->buildCpuSurface(slot, f, scene);
            } else {
                fluid->recordSurface(commandBuffer, slot, f, scene);
            }
        }
        base += count;
    }
}

} // namespace gfx
