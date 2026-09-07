#include "app/plugins/debug_lines_plugin.h"

#include <algorithm>

#include <glm/mat4x4.hpp>
#include <imgui.h>

#include "app/application.h"
#include "gfx/render_graph.h"
#include "gfx/vk_check.h"

namespace app {

namespace {

// shaders/debug_line_common.glsl 의 DebugLinePushConstants 와 배치가 같아야 한다(scalar).
struct DebugLinePushConstants {
    glm::mat4 viewProjection;
    VkDeviceAddress vertices;
    uint32_t depthTexture;
    uint32_t occlude;
    float viewportSize[2];
};
static_assert(sizeof(DebugLinePushConstants) == 88, "디버그 선 푸시 상수 배치가 셰이더와 어긋난다");

} // namespace

DebugLinesPlugin::~DebugLinesPlugin() {
    if (context == nullptr) {
        return;
    }
    for (gfx::Buffer& buffer : buffers) {
        gfx::destroyBuffer(*context, buffer);
    }
    vkDestroyPipeline(context->device, pipeline, nullptr);
    vkDestroyPipelineLayout(context->device, pipelineLayout, nullptr);
}

void DebugLinesPlugin::build(Services& services) {
    services.settings.showColliders = services.options.showColliders;
    // 그릴 곳이 없으면 아무 것도 만들지 않는다. 헤드리스는 GPU 물리 때문에 장치만 있고 렌더러가
    // 없을 수 있으므로 렌더러로 판정한다.
    if (services.renderer == nullptr) {
        return;
    }
    context = services.context;
    bindless = services.bindless;
    renderer = services.renderer;
    editor = services.editor;
    createPipeline(*renderer, *bindless);

    renderer->addPass([this](gfx::RenderGraph& graph, const gfx::Renderer::FrameInfo& info) {
        // 선은 그래프를 짤 때 만든다. 비어 있으면 노드가 돌지 않는다.
        const gfx::RenderSettings& settings = renderer->settings;
        gfx::DebugLineOptions options;
        options.colliders = settings.showColliders;
        options.fluidBounds = settings.showColliders;
        options.selected = editor != nullptr ? editor->selectedObject : -1;
        vertices.clear();
        if (settings.showColliders) {
            gfx::buildDebugLines(info.scene, options, vertices);
        }
        // 톤 매핑과 업스케일이 끝난 표시 해상도에 덧그린다. 여기서 그려야 노출과 업스케일이 색을 흔들지 않고,
        // 깊이 버퍼를 텍스처로 읽어 물체 뒤로 숨을 수 있다. 앞 패스가 같은 이미지를 첨부물로 썼으므로 writes
        // 선언이 그 사이 배리어를 낸다.
        graph.addAfter("업스케일",
                       gfx::RenderNode{"디버그 선",
                                       "디버그 선",
                                       [this] { return !vertices.empty(); },
                                       {gfx::ImageUse{info.depth.handle,
                                                      VK_IMAGE_ASPECT_DEPTH_BIT,
                                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                                      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT}},
                                       {gfx::ImageUse{info.present.handle,
                                                      VK_IMAGE_ASPECT_COLOR_BIT,
                                                      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                                      VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                                                          VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT}},
                                       {},
                                       [this, &info](VkCommandBuffer cmd) { record(cmd, info); }});
    });
}

void DebugLinesPlugin::createPipeline(gfx::Renderer& renderer, gfx::BindlessTextures& bindless) {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.size = sizeof(DebugLinePushConstants);

    VkDescriptorSetLayout bindlessLayout = bindless.layout();
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &bindlessLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK(vkCreatePipelineLayout(context->device, &layoutInfo, nullptr, &pipelineLayout));

    VkShaderModule vertexModule = gfx::createShaderModule(context->device, "debug_line.vert.spv");
    VkShaderModule fragmentModule = gfx::createShaderModule(context->device, "debug_line.frag.spv");
    std::array<VkPipelineShaderStageCreateInfo, 2> stages{
        gfx::shaderStage(VK_SHADER_STAGE_VERTEX_BIT, vertexModule),
        gfx::shaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, fragmentModule)};

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
    rasterization.lineWidth = context->caps.wideLines ? 2.0F : 1.0F;

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

    VkFormat colorFormat = renderer.presentFormat();
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
    pipelineInfo.layout = pipelineLayout;
    VK_CHECK(vkCreateGraphicsPipelines(context->device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline));

    vkDestroyShaderModule(context->device, fragmentModule, nullptr);
    vkDestroyShaderModule(context->device, vertexModule, nullptr);
}

void DebugLinesPlugin::reserve(uint32_t slot, uint32_t vertexCount) {
    uint32_t needed = std::max(vertexCount, 1U);
    if (needed <= capacities[slot]) {
        return;
    }
    uint32_t capacity = std::max(needed, capacities[slot] * 2);
    // 프레임 슬롯마다 하나라 이 시점에는 아무도 읽지 않는다. 타임라인 대기를 이미 지났다.
    gfx::destroyBuffer(*context, buffers[slot]);
    buffers[slot] = gfx::createBuffer(*context,
                                      static_cast<VkDeviceSize>(capacity) * sizeof(gfx::DebugLineVertex),
                                      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                      gfx::MemoryLocation::HOST_WRITE,
                                      "디버그 선");
    capacities[slot] = capacity;
}

void DebugLinesPlugin::record(VkCommandBuffer commandBuffer, const gfx::Renderer::FrameInfo& info) {
    auto vertexCount = static_cast<uint32_t>(vertices.size());
    reserve(info.frameSlot, vertexCount);
    gfx::Buffer& buffer = buffers[info.frameSlot];
    std::copy_n(vertices.data(), vertexCount, static_cast<gfx::DebugLineVertex*>(buffer.mapped));

    DebugLinePushConstants pushConstants{};
    // 종횡비는 장면을 그린 렌더 해상도로 잡는다. 표시 해상도로 잡으면 배율 반올림만큼 선이 어긋난다.
    float aspect = static_cast<float>(info.renderExtent.width) / static_cast<float>(info.renderExtent.height);
    // 지터는 넣지 않는다. 톤 매핑과 업스케일이 끝난 뒤에 그리므로 흔들 이유가 없다.
    //
    // ponytail: 깊이 버퍼는 지터가 들어간 투영으로 쓰였다. 시간축 업스케일을 켜면 실루엣 경계에서
    // 가림 판정이 프레임마다 한 화소쯤 흔들린다. 정확히 하려면 깊이를 읽을 uv 에서 지터를 빼야 한다.
    pushConstants.viewProjection = info.scene.camera.projectionMatrix(aspect) * info.scene.camera.viewMatrix();
    pushConstants.vertices = buffer.address;
    pushConstants.depthTexture = info.depthSlot;
    // 경로 추적은 깊이 버퍼에 아무것도 쓰지 않는다(레이아웃만 맞춰 둔다). 그 내용으로 가림을
    // 판정하면 미정의 값이나 직전 래스터 프레임의 깊이를 읽으므로 그 모드에서는 늘 그린다.
    bool depthAvailable = !(renderer->settings.usePathTracing && renderer->pathTracingAvailable());
    pushConstants.occlude = renderer->settings.colliderOcclusion && depthAvailable ? 1U : 0U;
    pushConstants.viewportSize[0] = static_cast<float>(info.displayExtent.width);
    pushConstants.viewportSize[1] = static_cast<float>(info.displayExtent.height);

    VkRenderingAttachmentInfo lineColor = gfx::colorAttachment(info.present.view, VK_ATTACHMENT_LOAD_OP_LOAD, {});
    VkRenderingInfo linePass{VK_STRUCTURE_TYPE_RENDERING_INFO};
    linePass.renderArea.extent = info.displayExtent;
    linePass.layerCount = 1;
    linePass.colorAttachmentCount = 1;
    linePass.pColorAttachments = &lineColor;

    VkDescriptorSet bindlessSet = bindless->set();
    vkCmdBeginRendering(commandBuffer, &linePass);
    gfx::setFullViewport(commandBuffer, info.displayExtent);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer,
                       pipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       sizeof(pushConstants),
                       &pushConstants);
    vkCmdDraw(commandBuffer, vertexCount, 1, 0, 0);
    vkCmdEndRendering(commandBuffer);
}

void DebugLinesPlugin::ui(Services& services) {
    if (!services.editor->settingsSection("콜라이더 표시")) {
        return;
    }
    ImGui::Checkbox("콜라이더 표시", &services.settings.showColliders);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip("강체 콜라이더는 초록, 유체 용기는 청록, 방출 상자는 노랑으로 덧그린다");
    }
    // Path Tracing은 깊이 버퍼를 채우지 않아 가림을 판정할 수 없다. 그 모드에서는 늘 보인다.
    bool depthAvailable = !services.settings.usePathTracing;
    ImGui::BeginDisabled(!services.settings.showColliders || !depthAvailable);
    ImGui::SameLine();
    ImGui::Checkbox("가림 판정", &services.settings.colliderOcclusion);
    if (ImGui::IsItemHovered()) {
        ImGui::SetTooltip(depthAvailable ? "끄면 물체에 가려도 선이 그대로 보인다"
                                         : "Path Tracing은 깊이 버퍼를 채우지 않아 늘 보인다");
    }
    ImGui::EndDisabled();
}

} // namespace app
