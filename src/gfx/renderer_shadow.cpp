// 그림자 드로우 구성과 그림자 아틀라스 패스.
// Renderer 의 멤버 함수 정의만 나눠 담은 번역 단위다. 선언은 renderer.h 하나에 있다.

#include "gfx/renderer_internal.h"

namespace gfx {

// 시점마다 캐스터를 걸러 압축한 그리기 명령을 만든다.
//
// GPU 컬링이 아니라 CPU 인 이유: 이 저장소의 컴퓨트 컬링은 meshlet 단위로 명령을 뱉는데, 압축
// 간접 그리기(drawIndirectCount)가 없는 장치에서는 상한만큼 발행해야 해서 시점 하나에 수백 개의
// 무효 드로우가 생긴다. 그림자는 오브젝트 단위 드로우가 훨씬 싸고, 편집기 규모에서 시점 × 오브젝트
// 순회는 무시할 수준이다.
//
// ponytail: 압축 간접 그리기가 있는 장치라면 시점을 컬 컴퓨트의 두 번째 디스패치 축으로 넘겨
// meshlet 단위까지 걸러 내는 편이 낫다.
void Renderer::buildShadowDraws(Frame& frame, const FrameBatches& batches, const glm::mat4& cameraViewProjection) {
    shadowBatches.assign(shadowViews.size() * TRANSLUCENT_MODE, DrawBatch{});
    shadowDrawsIssued = 0;
    shadowDrawsTotal = 0;
    shadowLayerDirty.assign(MAX_SHADOW_VIEWS, 0);
    if (shadowViews.empty()) {
        return;
    }

    // 어떤 층을 다시 그릴지 먼저 정한다. 설정이 바뀌면 그리는 집합 자체가 달라지므로 전부 무효화한다.
    uint64_t settings = static_cast<uint64_t>(lodLevel) | (static_cast<uint64_t>(automaticLod) << 8U) |
                        (static_cast<uint64_t>(shadowViews.size()) << 16U) |
                        (static_cast<uint64_t>(shadowViewCulling) << 24U) |
                        (static_cast<uint64_t>(shadowCasterCulling) << 25U) |
                        (static_cast<uint64_t>(std::bit_cast<uint32_t>(lodErrorThreshold)) << 32U);
    //
    // ponytail: 장면이 조금이라도 바뀌면 층을 전부 다시 그린다. 움직인 캐스터의 이전/현재 경계구를
    // 시점 절두체와 비교해 걸린 층만 무효화하면 더 아낄 수 있지만, 지금은 카메라만 움직이는 경우가
    // 대부분이고 그때는 이 조건이 걸리지 않아 이득이 이미 나온다.
    bool invalidateAll = !shadowCaching || settings != lastShadowSettings || sceneChangedThisFrame;
    lastShadowSettings = settings;
    for (uint32_t layer = 0; layer < shadowViews.size(); ++layer) {
        // 캐스케이드 텍셀 스냅 덕에 카메라가 텍셀 안에서 움직이는 동안에는 행렬이 그대로다.
        bool changed = invalidateAll || !shadowLayers[layer].valid ||
                       shadowLayers[layer].drawnViewProjection != shadowViews[layer].viewProjection;
        shadowLayerDirty[layer] = changed ? 1 : 0;
    }

    uint32_t casterCount = batches.fluidDraws.count;
    for (size_t mode = 0; mode < TRANSLUCENT_MODE; ++mode) {
        casterCount += batches.draws[mode][0].count + batches.draws[mode][1].count;
    }
    shadowDrawsTotal = casterCount * static_cast<uint32_t>(shadowViews.size());
    reserveShadowDraws(frame, shadowDrawsTotal);
    if (casterCount == 0) {
        return;
    }

    // 카메라 절두체는 무한 원거리라 원평면이 없다. 스윕 길이는 장면을 가로지르면 충분하다.
    std::array<glm::vec4, MAX_FRUSTUM_PLANES> cameraPlanes{};
    uint32_t cameraPlaneCount = extractFrustumPlanes(cameraViewProjection, cameraPlanes, false);
    float sweep = 2.0F * sceneRadius;

    // 매핑된 버퍼가 아니라 CPU 사본에서 읽는다. buildDrawCommands 가 같은 프레임에 채워 둔다.
    // 시점마다 casterCount 칸짜리 구간을 미리 나눠 두어 워커가 서로 다른 자리에 쓴다. 잠금도 원자 증가도
    // 필요 없고, 남는 칸은 아무 구간도 가리키지 않아 그리지 않는다.
    const VkDrawIndexedIndirectCommand* source = drawCommands.data();
    shadowDrawData.resize(shadowDrawsTotal);
    auto* destination = shadowDrawData.data();
    auto* mapped = static_cast<VkDrawIndexedIndirectCommand*>(frame.shadowDrawBuffer.mapped);
    std::atomic<uint32_t> issued{0};

    // 절두체 판정을 먼저 «오브젝트 전체 × 시점 전체» 로 한 번에 돌린다. 시점 수가 캐스케이드 넷뿐이라
    // 시점으로만 나누면 워커 넷만 일하는데, 오브젝트로 나누면 전부 일한다. 아래 시점별 압축은 이 표를
    // 비트로 읽기만 하므로 훨씬 싸다.
    uint32_t opaqueFirst = batches.draws[0][0].first;
    uint32_t opaqueCount = 0;
    for (size_t mode = 0; mode < TRANSLUCENT_MODE; ++mode) {
        opaqueCount += batches.draws[mode][0].count + batches.draws[mode][1].count;
    }
    auto viewCount = static_cast<uint32_t>(shadowViews.size());
    shadowVisibleMask.assign(static_cast<size_t>(opaqueCount) * viewCount, 0);
    std::vector<std::array<glm::vec4, MAX_FRUSTUM_PLANES>> viewPlanes(viewCount);
    std::vector<uint32_t> viewPlaneCounts(viewCount);
    for (uint32_t view = 0; view < viewCount; ++view) {
        viewPlaneCounts[view] = extractFrustumPlanes(shadowViews[view].viewProjection, viewPlanes[view], true);
    }
    if (opaqueCount > 0) {
        jobs.parallelFor(opaqueCount, 512, [&](uint32_t begin, uint32_t end) {
            for (uint32_t i = begin; i < end; ++i) {
                const glm::vec4& sphere = instanceBounds[opaqueFirst + i];
                for (uint32_t view = 0; view < viewCount; ++view) {
                    const ShadowView& shadowView = shadowViews[view];
                    // 광원 절두체 밖이면 이 시점에 아무것도 남기지 않는다.
                    if (shadowViewCulling &&
                        !sphereInFrustum(viewPlanes[view], viewPlaneCounts[view], glm::vec3(sphere), sphere.w)) {
                        continue;
                    }
                    // 그림자가 뻗어 나갈 범위까지 부풀려도 카메라에 닿지 않으면 화면에 나올 수 없다.
                    if (shadowCasterCulling) {
                        glm::vec3 direction = shadowView.directional
                                                  ? shadowView.sweepDirection
                                                  : glm::normalize(glm::vec3(sphere) - shadowView.origin);
                        if (!sweptSphereInFrustum(
                                cameraPlanes, cameraPlaneCount, glm::vec3(sphere), sphere.w, direction, sweep)) {
                            continue;
                        }
                    }
                    shadowVisibleMask[static_cast<size_t>(view) * opaqueCount + i] = 1;
                }
            }
        });
    }

    auto cullView = [&](uint32_t view) {
        const std::array<glm::vec4, MAX_FRUSTUM_PLANES>& planes = viewPlanes[view];
        uint32_t planeCount = viewPlaneCounts[view];
        uint32_t base = view * casterCount;
        uint32_t cursor = base;
        const uint8_t* mask = shadowVisibleMask.data() + static_cast<size_t>(view) * opaqueCount;

        for (size_t mode = 0; mode < TRANSLUCENT_MODE; ++mode) {
            DrawBatch& batch = shadowBatches[view * TRANSLUCENT_MODE + mode];
            batch.first = cursor;
            uint32_t first = batches.draws[mode][0].first;
            uint32_t count = batches.draws[mode][0].count + batches.draws[mode][1].count;

            for (uint32_t i = 0; i < count; ++i) {
                uint32_t slot = first + i;
                if (mask[slot - opaqueFirst] == 0) {
                    continue;
                }
                destination[cursor++] = source[slot];
            }
            // 유체 입자는 불투명 단면 구간에 인스턴스 드로우로 붙는다. 용기 구가 시점에 걸리면 통째로 넣는다.
            if (mode == 0) {
                for (uint32_t f = 0; f < batches.fluidDraws.count; ++f) {
                    const glm::vec4& sphere = fluidBounds[f];
                    if (shadowViewCulling && !sphereInFrustum(planes, planeCount, glm::vec3(sphere), sphere.w)) {
                        continue;
                    }
                    destination[cursor++] = source[batches.fluidDraws.first + f];
                }
            }
            batch.count = cursor - batch.first;
        }
        // 이 시점 몫만 매핑된 버퍼로 옮긴다. 쓰기 결합 메모리라 순서대로 한 번에 쓴다.
        std::copy_n(destination + base, cursor - base, mapped + base);
        issued.fetch_add(cursor - base, std::memory_order_relaxed);
    };
    jobs.parallelFor(static_cast<uint32_t>(shadowViews.size()), 1, [&cullView](uint32_t begin, uint32_t end) {
        for (uint32_t view = begin; view < end; ++view) {
            cullView(view);
        }
    });
    shadowDrawsIssued = issued.load(std::memory_order_relaxed);
}

void Renderer::createShadowPipeline() {
    // 컷오프 캐스터가 기저 색 텍스처를 읽어야 하므로 bindless 셋을 붙인다. 불투명 파이프라인도
    // 같은 레이아웃을 쓴다. 안 쓰는 셋을 바인드하는 비용은 없고 레이아웃이 둘로 늘지 않는다.
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.size = sizeof(DepthPushConstants);

    VkDescriptorSetLayout bindlessLayout = bindless.layout();
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &bindlessLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &depthPipelineLayout));

    VkShaderModule vertexModule = createShaderModule(context.device, "depth_only.vert.spv");
    VkShaderModule cutoffModule = createShaderModule(context.device, "shadow_cutoff.frag.spv");
    std::array<VkPipelineShaderStageCreateInfo, 2> stages{shaderStage(VK_SHADER_STAGE_VERTEX_BIT, vertexModule),
                                                          shaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, cutoffModule)};

    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    // 그림자는 양면을 모두 그려야 얇은 물체가 사라지지 않는다. 경사 편향으로 자기 그림자를 줄인다.
    VkPipelineRasterizationStateCreateInfo rasterization{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.lineWidth = 1.0F;
    rasterization.depthBiasEnable = VK_TRUE;
    rasterization.depthBiasConstantFactor = 1.25F;
    rasterization.depthBiasSlopeFactor = 2.5F;
    // 근평면 앞의 캐스터를 잘라 내지 않고 눌러 담는다. 없으면 그림자가 사라진다.
    rasterization.depthClampEnable = context.caps.depthClamp ? VK_TRUE : VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // 그림자 아틀라스는 일반 깊이라 0 에서 1 로 커진다.
    VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendStateCreateInfo colorBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};

    VkDynamicState dynamicStates[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    renderingInfo.depthAttachmentFormat = SHADOW_FORMAT;

    VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.pNext = &renderingInfo;
    // 불투명 캐스터는 프래그먼트 셰이더 없이 그린다. discard 가 없으니 조기 깊이 판정을 온전히 쓴다.
    pipelineInfo.stageCount = 1;
    pipelineInfo.pStages = stages.data();
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterization;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = depthPipelineLayout;
    VK_CHECK(vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &shadowPipeline));

    pipelineInfo.stageCount = 2;
    VK_CHECK(
        vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &shadowCutoffPipeline));

    vkDestroyShaderModule(context.device, cutoffModule, nullptr);
    vkDestroyShaderModule(context.device, vertexModule, nullptr);
}

void Renderer::recordShadowPass(VkCommandBuffer commandBuffer) {
    constexpr VkDeviceSize DRAW_STRIDE = sizeof(VkDrawIndexedIndirectCommand);
    Frame& frame = frames[frameIndex % FRAMES_IN_FLIGHT];
    std::array<VkPipeline, TRANSLUCENT_MODE> casterPipelines{shadowPipeline, shadowCutoffPipeline};

    // 다시 그릴 층이 하나도 없으면 지난 프레임 내용을 그대로 쓴다. 렌더 패스를 시작하지 않으므로
    // 타일 기반 GPU 에서도 읽기/쓰기 비용이 없다.
    shadowLayersRedrawn = 0;
    for (uint32_t layer = 0; layer < shadowViews.size(); ++layer) {
        shadowLayersRedrawn += shadowLayerDirty[layer] != 0 ? 1 : 0;
    }
    if (shadowLayersRedrawn == 0) {
        return;
    }

    // 캐시된 층의 내용이 살아남아야 하므로 다시 그릴 층만 골라 전이한다. 이미지 전체를
    // UNDEFINED 로 전이하면 캐시가 통째로 날아간다.
    for (uint32_t layer = 0; layer < shadowViews.size(); ++layer) {
        if (shadowLayerDirty[layer] == 0) {
            continue;
        }
        imageBarrier(commandBuffer,
                     targets.shadowAtlas.handle,
                     VK_IMAGE_ASPECT_DEPTH_BIT,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                     VK_QUEUE_FAMILY_IGNORED,
                     VK_QUEUE_FAMILY_IGNORED,
                     0,
                     VK_REMAINING_MIP_LEVELS,
                     layer,
                     1);
    }

    VkDescriptorSet bindlessSet = bindless.set();
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, depthPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdBindIndexBuffer(commandBuffer, geometry.indexBuffer.handle, 0, VK_INDEX_TYPE_UINT32);

    DepthPushConstants pushConstants{};
    pushConstants.vertices = geometry.vertexBuffer.address;
    pushConstants.instances = frame.instanceBuffer.address;
    pushConstants.skinnedVertices = skinnedVertexBuffer.address;
    pushConstants.meshes = geometry.meshBuffer.address;
    pushConstants.materials = geometry.materialBuffer.address;

    VkViewport viewport{
        0.0F, 0.0F, static_cast<float>(SHADOW_MAP_SIZE), static_cast<float>(SHADOW_MAP_SIZE), 0.0F, 1.0F};
    VkRect2D scissor{{0, 0}, {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE}};

    for (uint32_t layer = 0; layer < shadowViews.size(); ++layer) {
        if (shadowLayerDirty[layer] == 0) {
            continue;
        }

        VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depthAttachment.imageView = targets.shadowLayerViews[layer];
        depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depthAttachment.clearValue.depthStencil.depth = 1.0F;

        VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering.renderArea = scissor;
        rendering.layerCount = 1;
        rendering.pDepthAttachment = &depthAttachment;
        vkCmdBeginRendering(commandBuffer, &rendering);

        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        pushConstants.viewProjection = shadowViews[layer].viewProjection;
        vkCmdPushConstants(commandBuffer,
                           depthPipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           sizeof(pushConstants),
                           &pushConstants);

        for (size_t mode = 0; mode < TRANSLUCENT_MODE; ++mode) {
            const DrawBatch& batch = shadowBatches[layer * TRANSLUCENT_MODE + mode];
            if (batch.count == 0) {
                continue;
            }
            vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, casterPipelines[mode]);
            vkCmdDrawIndexedIndirect(commandBuffer,
                                     frame.shadowDrawBuffer.handle,
                                     batch.first * DRAW_STRIDE,
                                     batch.count,
                                     static_cast<uint32_t>(DRAW_STRIDE));
        }
        vkCmdEndRendering(commandBuffer);

        shadowLayers[layer].drawnViewProjection = shadowViews[layer].viewProjection;
        shadowLayers[layer].valid = true;
    }

    for (uint32_t layer = 0; layer < shadowViews.size(); ++layer) {
        if (shadowLayerDirty[layer] == 0) {
            continue;
        }
        imageBarrier(commandBuffer,
                     targets.shadowAtlas.handle,
                     VK_IMAGE_ASPECT_DEPTH_BIT,
                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                     VK_QUEUE_FAMILY_IGNORED,
                     VK_QUEUE_FAMILY_IGNORED,
                     0,
                     VK_REMAINING_MIP_LEVELS,
                     layer,
                     1);
    }
}

} // namespace gfx
