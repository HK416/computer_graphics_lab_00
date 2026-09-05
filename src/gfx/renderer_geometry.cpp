// 메쉬 파이프라인(고전·mesh shader·광선 질의 변종)과 장면 지오메트리 패스.
// Renderer 의 멤버 함수 정의만 나눠 담은 번역 단위다. 선언은 renderer.h 하나에 있다.

#include "gfx/renderer_internal.h"

namespace gfx {

void Renderer::createMeshPipelines() {
    scenePushStages = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    if (context.caps.meshShader) {
        scenePushStages |= VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;
    }

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = scenePushStages;
    pushConstantRange.size = sizeof(ScenePushConstants);

    VkDescriptorSetLayout bindlessLayout = bindless.layout();
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &bindlessLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &meshPipelineLayout));

    VkShaderModule vertexModule =
        createShaderModule(context.device, context.caps.shaderDrawIndex ? "mesh.vert.spv" : "mesh_nodrawid.vert.spv");
    VkShaderModule opaqueFragment = createShaderModule(context.device, "mesh.frag.spv");
    VkShaderModule oitFragment = createShaderModule(context.device, "mesh_oit.frag.spv");

    // 광선 질의는 SPIR-V 능력을 요구해 런타임 분기가 안 된다. 같은 셰이더를 정의만 바꿔 한 벌 더
    // 컴파일해 두고, 하드웨어가 지원할 때만 파이프라인을 만든다.
    VkShaderModule rayQueryOpaque = VK_NULL_HANDLE;
    VkShaderModule rayQueryOit = VK_NULL_HANDLE;
    if (rayQueryShadowsAvailable()) {
        std::array<VkDescriptorSetLayout, 2> rayQuerySets{bindlessLayout, rayTracer->accelerationLayout()};
        layoutInfo.setLayoutCount = static_cast<uint32_t>(rayQuerySets.size());
        layoutInfo.pSetLayouts = rayQuerySets.data();
        VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &meshRayQueryPipelineLayout));
        rayQueryOpaque = createShaderModule(context.device, "mesh_rq.frag.spv");
        rayQueryOit = createShaderModule(context.device, "mesh_oit_rq.frag.spv");
    }

    // 프래그먼트 셰이더의 알파 경로를 특수화 상수로 고정해, 불투명 경로에서는 discard 가 사라진다.
    uint32_t alphaVariant = 0;
    VkSpecializationMapEntry specializationEntry{0, 0, sizeof(uint32_t)};
    VkSpecializationInfo specialization{};
    specialization.mapEntryCount = 1;
    specialization.pMapEntries = &specializationEntry;
    specialization.dataSize = sizeof(alphaVariant);
    specialization.pData = &alphaVariant;

    std::array<VkPipelineShaderStageCreateInfo, 2> stages{shaderStage(VK_SHADER_STAGE_VERTEX_BIT, vertexModule),
                                                          shaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, opaqueFragment)};
    stages[1].pSpecializationInfo = &specialization;

    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterization{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.lineWidth = 1.0F;

    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // reverse-Z 이므로 깊이 버퍼는 0 으로 지우고 더 큰 값을 통과시킨다.
    VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;

    constexpr VkColorComponentFlags ALL_CHANNELS =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    constexpr VkColorComponentFlags VELOCITY_CHANNELS = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT;
    // 불투명 경로는 색상, 모션 벡터, 노멀·거칠기, 반사 가중치 넷이고 반투명 경로는 누적과 잔여
    // 투과율 둘이다.
    constexpr uint32_t OPAQUE_ATTACHMENTS = 4;
    constexpr uint32_t TRANSLUCENT_ATTACHMENTS = 2;
    std::array<VkPipelineColorBlendAttachmentState, OPAQUE_ATTACHMENTS> blendAttachments{};
    std::array<VkFormat, OPAQUE_ATTACHMENTS> colorFormats{};
    auto resetOpaqueAttachments = [&]() {
        blendAttachments = {};
        blendAttachments[0].colorWriteMask = ALL_CHANNELS;
        blendAttachments[1].colorWriteMask = VELOCITY_CHANNELS;
        blendAttachments[2].colorWriteMask = ALL_CHANNELS;
        blendAttachments[3].colorWriteMask = ALL_CHANNELS;
        colorFormats = {COLOR_FORMAT, VELOCITY_FORMAT, COLOR_FORMAT, COLOR_FORMAT};
    };
    // 반투명은 누적과 잔여 투과율 두 대상에 기록한다. 누적은 더하고, 잔여 투과율은 (1 - 알파)를 곱해
    // 나간다. 고전 경로와 mesh shader 경로가 같은 상태를 써야 하므로 한 곳에서 정한다.
    auto setTranslucentAttachments = [&]() {
        blendAttachments = {};
        colorFormats = {OIT_ACCUMULATION_FORMAT, OIT_REVEALAGE_FORMAT, COLOR_FORMAT, COLOR_FORMAT};
        blendAttachments[0].colorWriteMask = ALL_CHANNELS;
        blendAttachments[0].blendEnable = VK_TRUE;
        blendAttachments[0].srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachments[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachments[0].colorBlendOp = VK_BLEND_OP_ADD;
        blendAttachments[0].srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachments[0].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachments[0].alphaBlendOp = VK_BLEND_OP_ADD;
        blendAttachments[1].blendEnable = VK_TRUE;
        blendAttachments[1].srcColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        blendAttachments[1].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
        blendAttachments[1].colorBlendOp = VK_BLEND_OP_ADD;
        blendAttachments[1].srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        blendAttachments[1].dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAttachments[1].alphaBlendOp = VK_BLEND_OP_ADD;
        blendAttachments[1].colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
    };
    resetOpaqueAttachments();

    VkPipelineColorBlendStateCreateInfo colorBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlend.attachmentCount = OPAQUE_ATTACHMENTS;
    colorBlend.pAttachments = blendAttachments.data();

    // 양면 재질은 컬 모드만 다르므로 파이프라인을 늘리지 않고 동적 상태로 전환한다.
    VkDynamicState dynamicStates[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_CULL_MODE};
    VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = 3;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    renderingInfo.colorAttachmentCount = OPAQUE_ATTACHMENTS;
    renderingInfo.pColorAttachmentFormats = colorFormats.data();
    renderingInfo.depthAttachmentFormat = DEPTH_FORMAT;

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
    pipelineInfo.layout = meshPipelineLayout;

    // 알파 경로 세 벌. 광선 질의 변종은 프래그먼트 모듈과 배치만 다르므로 상태를 되돌리고 한 번 더 돈다.
    auto buildAlphaVariants =
        [&](VkShaderModule opaque, VkShaderModule oit, VkPipelineLayout layout, VkPipeline* target) {
            stages[1].module = opaque;
            depthStencil.depthWriteEnable = VK_TRUE;
            resetOpaqueAttachments();
            colorBlend.attachmentCount = OPAQUE_ATTACHMENTS;
            renderingInfo.colorAttachmentCount = OPAQUE_ATTACHMENTS;
            pipelineInfo.layout = layout;
            for (uint32_t variant = 0; variant < ALPHA_MODE_COUNT; ++variant) {
                alphaVariant = variant;
                if (variant == TRANSLUCENT_MODE) {
                    // 반투명은 깊이를 읽기만 한다. 모션 벡터는 불투명 표면만 남기므로 기록하지 않는다.
                    stages[1].module = oit;
                    depthStencil.depthWriteEnable = VK_FALSE;
                    setTranslucentAttachments();
                    colorBlend.attachmentCount = TRANSLUCENT_ATTACHMENTS;
                    renderingInfo.colorAttachmentCount = TRANSLUCENT_ATTACHMENTS;
                }
                VK_CHECK(vkCreateGraphicsPipelines(
                    context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &target[variant]));
            }
        };

    buildAlphaVariants(opaqueFragment, oitFragment, meshPipelineLayout, meshPipelines.data());
    if (rayQueryShadowsAvailable()) {
        buildAlphaVariants(rayQueryOpaque, rayQueryOit, meshRayQueryPipelineLayout, meshRayQueryPipelines.data());
    }

    // 와이어프레임 디버그 뷰는 불투명 경로를 선 모드로 한 벌 더 만든다.
    alphaVariant = 0;
    pipelineInfo.layout = meshPipelineLayout;
    stages[1].module = opaqueFragment;
    depthStencil.depthWriteEnable = VK_TRUE;
    resetOpaqueAttachments();
    colorBlend.attachmentCount = OPAQUE_ATTACHMENTS;
    renderingInfo.colorAttachmentCount = OPAQUE_ATTACHMENTS;
    rasterization.polygonMode = VK_POLYGON_MODE_LINE;
    VK_CHECK(vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &wireframePipeline));

    if (context.caps.meshShader) {
        // 태스크와 메쉬 셰이더 경로는 정점 입력과 입력 조립 상태를 쓰지 않는다.
        VkShaderModule taskModule = createShaderModule(context.device, "mesh.task.spv");
        VkShaderModule meshModule = createShaderModule(context.device, "mesh.mesh.spv");

        std::array<VkPipelineShaderStageCreateInfo, 3> meshStages{
            shaderStage(VK_SHADER_STAGE_TASK_BIT_EXT, taskModule),
            shaderStage(VK_SHADER_STAGE_MESH_BIT_EXT, meshModule),
            shaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, opaqueFragment)};
        meshStages[2].pSpecializationInfo = &specialization;

        pipelineInfo.stageCount = static_cast<uint32_t>(meshStages.size());
        pipelineInfo.pStages = meshStages.data();
        pipelineInfo.pVertexInputState = nullptr;
        pipelineInfo.pInputAssemblyState = nullptr;
        rasterization.polygonMode = VK_POLYGON_MODE_FILL;

        // 프래그먼트 셰이더만 다르므로 광선 질의 변종도 같은 방식으로 한 벌 더 만든다.
        auto buildMeshVariants =
            [&](VkShaderModule opaque, VkShaderModule oit, VkPipelineLayout layout, VkPipeline* target) {
                pipelineInfo.layout = layout;
                for (uint32_t variant = 0; variant < ALPHA_MODE_COUNT; ++variant) {
                    alphaVariant = variant;
                    bool translucent = variant == static_cast<uint32_t>(asset::AlphaMode::TRANSLUCENT);
                    meshStages[2].module = translucent ? oit : opaque;
                    depthStencil.depthWriteEnable = translucent ? VK_FALSE : VK_TRUE;
                    // 블렌드 상태는 앞 루프가 남긴 것에 기대지 않고 매번 통째로 정한다.
                    if (translucent) {
                        setTranslucentAttachments();
                    } else {
                        resetOpaqueAttachments();
                    }
                    colorBlend.attachmentCount = translucent ? TRANSLUCENT_ATTACHMENTS : OPAQUE_ATTACHMENTS;
                    renderingInfo.colorAttachmentCount = translucent ? TRANSLUCENT_ATTACHMENTS : OPAQUE_ATTACHMENTS;
                    VK_CHECK(vkCreateGraphicsPipelines(
                        context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &target[variant]));
                }
            };

        buildMeshVariants(opaqueFragment, oitFragment, meshPipelineLayout, meshShaderPipelines.data());
        if (rayQueryShadowsAvailable()) {
            buildMeshVariants(
                rayQueryOpaque, rayQueryOit, meshRayQueryPipelineLayout, meshShaderRayQueryPipelines.data());
        }

        vkDestroyShaderModule(context.device, meshModule, nullptr);
        vkDestroyShaderModule(context.device, taskModule, nullptr);

        drawMeshTasksIndirect = reinterpret_cast<PFN_vkCmdDrawMeshTasksIndirectEXT>(
            vkGetDeviceProcAddr(context.device, "vkCmdDrawMeshTasksIndirectEXT"));
        if (drawMeshTasksIndirect == nullptr) {
            core::fatal("vkCmdDrawMeshTasksIndirectEXT 를 찾을 수 없습니다");
        }
        settings.useMeshShader = true;
        spdlog::info("mesh shader 경로 사용 가능");
    }

    vkDestroyShaderModule(context.device, rayQueryOit, nullptr);
    vkDestroyShaderModule(context.device, rayQueryOpaque, nullptr);
    vkDestroyShaderModule(context.device, oitFragment, nullptr);
    vkDestroyShaderModule(context.device, opaqueFragment, nullptr);
    vkDestroyShaderModule(context.device, vertexModule, nullptr);
}

void Renderer::recordGeometryPass(VkCommandBuffer commandBuffer,
                                  const FrameBatches& batches,
                                  bool translucentPass,
                                  uint32_t cullPhase) {
    constexpr VkDeviceSize DRAW_STRIDE = sizeof(VkDrawIndexedIndirectCommand);
    constexpr VkDeviceSize TASK_STRIDE = sizeof(VkDrawMeshTasksIndirectCommandEXT);

    Frame& frame = frames[frameIndex % FRAMES_IN_FLIGHT];
    bool meshPath = useMeshPath();
    bool cullPath = settings.useComputeCulling && !meshPath;

    size_t firstMode = translucentPass ? TRANSLUCENT_MODE : 0;
    size_t lastMode = translucentPass ? TRANSLUCENT_MODE + 1 : TRANSLUCENT_MODE;

    // 태스크 셰이더의 두 패스 판정과 프래그먼트의 컬 패스 디버그 뷰가 읽는다. 두 배치는 푸시 상수
    // 구간이 같아 어느 쪽으로 밀어도 된다.
    vkCmdPushConstants(commandBuffer,
                       meshPipelineLayout,
                       scenePushStages,
                       offsetof(ScenePushConstants, cullPhase),
                       sizeof(uint32_t),
                       &cullPhase);

    for (size_t mode = firstMode; mode < lastMode; ++mode) {
        bool bound = false;
        for (size_t sided = 0; sided < 2; ++sided) {
            const DrawBatch& batch = meshPath   ? batches.groups[mode][sided]
                                     : cullPath ? batches.meshletDraws[mode][sided]
                                                : batches.draws[mode][sided];
            if (batch.count == 0) {
                continue;
            }
            if (!bound) {
                VkPipeline pipeline = rayQueryPass ? meshShaderRayQueryPipelines[mode] : meshShaderPipelines[mode];
                if (!meshPath) {
                    pipeline = settings.wireframe && !translucentPass ? wireframePipeline
                               : rayQueryPass                         ? meshRayQueryPipelines[mode]
                                                                      : meshPipelines[mode];
                }
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                bound = true;
            }
            vkCmdSetCullMode(commandBuffer, sided == 1 ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT);

            // 구간 시작을 알려 준다. 태스크 셰이더는 gl_WorkGroupID 가, 고전 경로의 정점 셰이더는
            // gl_DrawID 가 호출마다 0 부터 시작해 meshlet 그룹과 명령별 meshlet 번호를 이 값에 더해 찾는다.
            vkCmdPushConstants(commandBuffer,
                               meshPipelineLayout,
                               scenePushStages,
                               offsetof(ScenePushConstants, meshletGroupBase),
                               sizeof(uint32_t),
                               &batch.first);
            if (meshPath) {
                drawMeshTasksIndirect(commandBuffer,
                                      frame.meshTaskIndirectBuffer.handle,
                                      (mode * 2 + sided) * TASK_STRIDE,
                                      1,
                                      static_cast<uint32_t>(TASK_STRIDE));
            } else if (cullPath) {
                VkDeviceSize offset = batch.first * DRAW_STRIDE;
                if (drawIndexedIndirectCount != nullptr) {
                    // 압축 간접 그리기가 있으면 GPU 가 센 개수만 처리한다.
                    drawIndexedIndirectCount(commandBuffer,
                                             frame.meshletDrawBuffer.handle,
                                             offset,
                                             frame.drawCountBuffer.handle,
                                             (mode * 2 + sided) * sizeof(uint32_t),
                                             batch.count,
                                             static_cast<uint32_t>(DRAW_STRIDE));
                } else {
                    // 없으면 상한만큼 넘기고 컬링된 자리는 0 으로 채워 둔 무효 명령이 걸러 낸다.
                    vkCmdDrawIndexedIndirect(commandBuffer,
                                             frame.meshletDrawBuffer.handle,
                                             offset,
                                             batch.count,
                                             static_cast<uint32_t>(DRAW_STRIDE));
                }
            } else {
                vkCmdDrawIndexedIndirect(commandBuffer,
                                         frame.drawBuffer.handle,
                                         batch.first * DRAW_STRIDE,
                                         batch.count,
                                         static_cast<uint32_t>(DRAW_STRIDE));
            }
        }
    }

    // 유체 입자. 어느 래스터 경로든 고전 정점 셰이더의 인스턴스 드로우 하나로 그린다. 두 패스 컬링에서는
    // 1차에만 그린다(오클루전 컬링 대상이 아니다).
    // ponytail: 입자 단위 오클루전 컬링은 없다. 필요하면 입자마다 meshlet 그룹을 두되 디스패치 한도를 본다.
    if (!translucentPass && cullPhase != CULL_PHASE_SECOND && batches.fluidDraws.count > 0) {
        VkPipeline pipeline = settings.wireframe ? wireframePipeline
                              : rayQueryPass     ? meshRayQueryPipelines[0]
                                                 : meshPipelines[0];
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        vkCmdSetCullMode(commandBuffer, VK_CULL_MODE_BACK_BIT);
        vkCmdPushConstants(commandBuffer,
                           meshPipelineLayout,
                           scenePushStages,
                           offsetof(ScenePushConstants, meshletGroupBase),
                           sizeof(uint32_t),
                           &batches.fluidDraws.first);
        // 컴퓨트 컬링 경로는 명령별 meshlet 을 GPU 가 쓴 버퍼로 가리킨다. 유체 명령은 CPU 것에 있다.
        VkDeviceAddress cpuDrawMeshlets = frame.drawMeshletBuffer.address;
        if (cullPath) {
            vkCmdPushConstants(commandBuffer,
                               meshPipelineLayout,
                               scenePushStages,
                               offsetof(ScenePushConstants, drawMeshlets),
                               sizeof(VkDeviceAddress),
                               &cpuDrawMeshlets);
        }
        vkCmdDrawIndexedIndirect(commandBuffer,
                                 frame.drawBuffer.handle,
                                 batches.fluidDraws.first * DRAW_STRIDE,
                                 batches.fluidDraws.count,
                                 static_cast<uint32_t>(DRAW_STRIDE));
        if (cullPath) {
            VkDeviceAddress gpuDrawMeshlets = frame.meshletDrawMeshletBuffer.address;
            vkCmdPushConstants(commandBuffer,
                               meshPipelineLayout,
                               scenePushStages,
                               offsetof(ScenePushConstants, drawMeshlets),
                               sizeof(VkDeviceAddress),
                               &gpuDrawMeshlets);
        }
    }
}

} // namespace gfx
