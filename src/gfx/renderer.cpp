#include "gfx/renderer.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/mat4x4.hpp>
#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>
#include <stb_image_write.h>

#include "asset/model.h"
#include "core/error.h"
#include "gfx/bindless.h"
#include "gfx/context.h"
#include "gfx/geometry.h"
#include "gfx/swapchain.h"
#include "gfx/vk_check.h"
#include "scene/scene.h"

namespace gfx {
namespace {

constexpr VkFormat COLOR_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat DEPTH_FORMAT = VK_FORMAT_D32_SFLOAT;
constexpr VkFormat OIT_ACCUMULATION_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
constexpr VkFormat OIT_REVEALAGE_FORMAT = VK_FORMAT_R16_SFLOAT;
constexpr VkFormat PRESENT_FORMAT = VK_FORMAT_R8G8B8A8_UNORM;
constexpr uint32_t MINIMUM_INSTANCE_CAPACITY = 1024;
constexpr size_t TRANSLUCENT_MODE = 2;
// shaders/meshlet_task.glsl 의 MESHLET_GROUP_SIZE 와 같아야 한다.
constexpr uint32_t MESHLET_GROUP_SIZE = 32;

// shaders/scene_data.glsl 의 CameraBuffer 와 배치가 같아야 한다.
struct GpuCamera {
    glm::mat4 viewProjection;
    glm::vec4 position;
    glm::vec4 parameters; // x: 근평면
};

// shaders/scene_data.glsl 의 MeshletGroup 과 배치가 같아야 한다.
struct GpuMeshletGroup {
    uint32_t instanceIndex;
    uint32_t firstMeshlet;
    uint32_t meshletCount;
    uint32_t padding;
};

struct ScenePushConstants {
    VkDeviceAddress vertices;
    VkDeviceAddress meshes;
    VkDeviceAddress instances;
    VkDeviceAddress materials;
    VkDeviceAddress camera;
    VkDeviceAddress meshlets;
    VkDeviceAddress meshletTriangles;
    VkDeviceAddress vertexMeshlets;
    VkDeviceAddress meshletGroups;
    uint32_t meshletGroupBase;
    uint32_t debugMode;
};

struct CompositePushConstants {
    uint32_t accumulationTexture;
    uint32_t revealageTexture;
};

struct TonemapPushConstants {
    uint32_t colorTexture;
    float exposure;
};

std::vector<uint32_t> readSpirv(const std::string& name) {
    std::filesystem::path path = std::filesystem::path(CG_LAB_SHADER_ROOT) / name;
    std::error_code error;
    auto size = std::filesystem::file_size(path, error);
    if (error || size == 0 || size % sizeof(uint32_t) != 0) {
        core::fatal("셰이더를 읽을 수 없습니다: {}", path.string());
    }

    std::vector<uint32_t> code(size / sizeof(uint32_t));
    std::FILE* file = std::fopen(path.string().c_str(), "rb");
    if (file == nullptr || std::fread(code.data(), 1, size, file) != size) {
        core::fatal("셰이더 읽기에 실패했습니다: {}", path.string());
    }
    std::fclose(file);
    return code;
}

VkShaderModule createShaderModule(VkDevice device, const std::string& name) {
    std::vector<uint32_t> code = readSpirv(name);
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = code.size() * sizeof(uint32_t);
    info.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &info, nullptr, &module));
    return module;
}

VkPipelineShaderStageCreateInfo shaderStage(VkShaderStageFlagBits stage, VkShaderModule module) {
    VkPipelineShaderStageCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    info.stage = stage;
    info.module = module;
    info.pName = "main";
    return info;
}

VkRenderingAttachmentInfo colorAttachment(VkImageView view, VkAttachmentLoadOp loadOp, VkClearColorValue clear) {
    VkRenderingAttachmentInfo info{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    info.imageView = view;
    info.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    info.loadOp = loadOp;
    info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    info.clearValue.color = clear;
    return info;
}

void setFullViewport(VkCommandBuffer commandBuffer, VkExtent2D extent) {
    VkViewport viewport{};
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.maxDepth = 1.0F;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = extent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
}

} // namespace

Renderer::Renderer(Context& context, GeometryStore& geometry, BindlessTextures& bindless, SDL_Window* window)
    : context(context), geometry(geometry), bindless(bindless) {
    swapchain = std::make_unique<Swapchain>(context, window, vsync);
    currentRenderExtent = swapchain->extent;

    // 오프스크린 대상을 셰이더에서 읽을 때 쓰는 샘플러. 화면 해상도 그대로 읽으므로 보간이 필요 없다.
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(context.device, &samplerInfo, nullptr, &postSampler));

    createRenderTargets();
    createFrames();
    createPresentSemaphores();
    createMeshPipelines();
    createPostPipelines();

    VkSemaphoreTypeCreateInfo timelineInfo{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphoreInfo.pNext = &timelineInfo;
    VK_CHECK(vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &frameTimeline));
}

Renderer::~Renderer() {
    waitIdle();
    vkDestroyPipeline(context.device, tonemapPipeline, nullptr);
    vkDestroyPipeline(context.device, compositePipeline, nullptr);
    vkDestroyPipelineLayout(context.device, postPipelineLayout, nullptr);
    for (VkPipeline pipeline : meshShaderPipelines) {
        vkDestroyPipeline(context.device, pipeline, nullptr);
    }
    vkDestroyPipeline(context.device, wireframePipeline, nullptr);
    for (VkPipeline pipeline : meshPipelines) {
        vkDestroyPipeline(context.device, pipeline, nullptr);
    }
    vkDestroyPipelineLayout(context.device, meshPipelineLayout, nullptr);
    vkDestroySemaphore(context.device, frameTimeline, nullptr);
    destroyPresentSemaphores();
    for (Frame& frame : frames) {
        destroyBuffer(context, frame.meshTaskIndirectBuffer);
        destroyBuffer(context, frame.meshletGroupBuffer);
        destroyBuffer(context, frame.drawBuffer);
        destroyBuffer(context, frame.instanceBuffer);
        destroyBuffer(context, frame.cameraBuffer);
        vkDestroySemaphore(context.device, frame.imageAvailable, nullptr);
        vkDestroyCommandPool(context.device, frame.commandPool, nullptr);
    }
    destroyBuffer(context, captureBuffer);
    destroyImage(context, targets.present);
    destroyImage(context, targets.oitRevealage);
    destroyImage(context, targets.oitAccumulation);
    destroyImage(context, targets.depth);
    destroyImage(context, targets.color);
    vkDestroySampler(context.device, postSampler, nullptr);
    swapchain.reset();
}

void Renderer::waitIdle() {
    VK_CHECK(vkDeviceWaitIdle(context.device));
}

void Renderer::setVsync(bool enabled) {
    if (enabled == vsync) {
        return;
    }
    vsync = enabled;
    resizeRequested = true;
}

void Renderer::setRenderExtent(VkExtent2D extent) {
    extent.width = std::max(extent.width, 1U);
    extent.height = std::max(extent.height, 1U);
    if (extent.width == currentRenderExtent.width && extent.height == currentRenderExtent.height) {
        return;
    }
    waitIdle();
    currentRenderExtent = extent;
    createRenderTargets();
}

std::vector<Renderer::TargetView> Renderer::targetViews() const {
    std::vector<TargetView> views{
        {"색상 (HDR)", targets.color.view}, {"표시 (톤 매핑)", targets.present.view}, {"깊이", targets.depth.view}};
    if (oitTargetsValid) {
        views.push_back({"OIT 누적", targets.oitAccumulation.view});
        views.push_back({"OIT 잔여 투과율", targets.oitRevealage.view});
    }
    return views;
}

VkFormat Renderer::swapchainFormat() const {
    return swapchain->format;
}

uint32_t Renderer::swapchainImageCount() const {
    return static_cast<uint32_t>(swapchain->images.size());
}

void Renderer::createRenderTargets() {
    destroyImage(context, targets.present);
    destroyImage(context, targets.oitRevealage);
    destroyImage(context, targets.oitAccumulation);
    destroyImage(context, targets.depth);
    destroyImage(context, targets.color);

    VkExtent3D extent{currentRenderExtent.width, currentRenderExtent.height, 1};

    ImageDesc colorDesc;
    colorDesc.extent = extent;
    colorDesc.format = COLOR_FORMAT;
    colorDesc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    targets.color = createImage(context, colorDesc, "HDR 색상");

    ImageDesc depthDesc;
    depthDesc.extent = extent;
    depthDesc.format = DEPTH_FORMAT;
    depthDesc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    depthDesc.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    targets.depth = createImage(context, depthDesc, "깊이");

    ImageDesc accumulationDesc = colorDesc;
    accumulationDesc.format = OIT_ACCUMULATION_FORMAT;
    targets.oitAccumulation = createImage(context, accumulationDesc, "OIT 누적");

    ImageDesc revealageDesc = colorDesc;
    revealageDesc.format = OIT_REVEALAGE_FORMAT;
    targets.oitRevealage = createImage(context, revealageDesc, "OIT 잔여 투과율");

    ImageDesc presentDesc = colorDesc;
    presentDesc.format = PRESENT_FORMAT;
    targets.present = createImage(context, presentDesc, "표시");

    ++generation;

    if (!targets.slotsAllocated) {
        targets.colorSlot = bindless.add(targets.color.view, postSampler);
        targets.accumulationSlot = bindless.add(targets.oitAccumulation.view, postSampler);
        targets.revealageSlot = bindless.add(targets.oitRevealage.view, postSampler);
        targets.slotsAllocated = true;
    } else {
        bindless.update(targets.colorSlot, targets.color.view, postSampler);
        bindless.update(targets.accumulationSlot, targets.oitAccumulation.view, postSampler);
        bindless.update(targets.revealageSlot, targets.oitRevealage.view, postSampler);
    }
}

void Renderer::createFrames() {
    for (Frame& frame : frames) {
        VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolInfo.queueFamilyIndex = context.queueFamilies.graphics;
        VK_CHECK(vkCreateCommandPool(context.device, &poolInfo, nullptr, &frame.commandPool));

        VkCommandBufferAllocateInfo allocateInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocateInfo.commandPool = frame.commandPool;
        allocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateInfo.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(context.device, &allocateInfo, &frame.commandBuffer));

        VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &frame.imageAvailable));

        frame.cameraBuffer = createBuffer(
            context, sizeof(GpuCamera), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryLocation::HOST_WRITE, "카메라");
        reserveInstances(frame, MINIMUM_INSTANCE_CAPACITY);
        reserveMeshletGroups(frame, MINIMUM_INSTANCE_CAPACITY);
    }
}

void Renderer::reserveInstances(Frame& frame, uint32_t instanceCount) {
    if (instanceCount <= frame.instanceCapacity) {
        return;
    }
    uint32_t capacity = std::max(instanceCount, std::max(frame.instanceCapacity * 2, MINIMUM_INSTANCE_CAPACITY));
    destroyBuffer(context, frame.instanceBuffer);
    destroyBuffer(context, frame.drawBuffer);
    frame.instanceBuffer = createBuffer(context,
                                        static_cast<VkDeviceSize>(capacity) * sizeof(GpuInstance),
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                        MemoryLocation::HOST_WRITE,
                                        "인스턴스");
    frame.drawBuffer = createBuffer(context,
                                    static_cast<VkDeviceSize>(capacity) * sizeof(VkDrawIndexedIndirectCommand),
                                    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    MemoryLocation::HOST_WRITE,
                                    "간접 그리기 명령");
    frame.instanceCapacity = capacity;
}

void Renderer::reserveMeshletGroups(Frame& frame, uint32_t groupCount) {
    if (frame.meshTaskIndirectBuffer.handle == VK_NULL_HANDLE) {
        frame.meshTaskIndirectBuffer =
            createBuffer(context,
                         sizeof(VkDrawMeshTasksIndirectCommandEXT) * ALPHA_MODE_COUNT * 2,
                         VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         MemoryLocation::HOST_WRITE,
                         "mesh task 간접 명령");
    }
    if (groupCount <= frame.groupCapacity) {
        return;
    }
    uint32_t capacity = std::max(groupCount, std::max(frame.groupCapacity * 2, MINIMUM_INSTANCE_CAPACITY));
    destroyBuffer(context, frame.meshletGroupBuffer);
    frame.meshletGroupBuffer = createBuffer(context,
                                            static_cast<VkDeviceSize>(capacity) * sizeof(GpuMeshletGroup),
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            MemoryLocation::HOST_WRITE,
                                            "meshlet 그룹");
    frame.groupCapacity = capacity;
}

void Renderer::createPresentSemaphores() {
    destroyPresentSemaphores();
    presentReady.resize(swapchain->images.size());
    for (VkSemaphore& semaphore : presentReady) {
        VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &semaphore));
    }
}

void Renderer::destroyPresentSemaphores() {
    for (VkSemaphore semaphore : presentReady) {
        vkDestroySemaphore(context.device, semaphore, nullptr);
    }
    presentReady.clear();
}

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

    VkShaderModule vertexModule = createShaderModule(context.device, "mesh.vert.spv");
    VkShaderModule opaqueFragment = createShaderModule(context.device, "mesh.frag.spv");
    VkShaderModule oitFragment = createShaderModule(context.device, "mesh_oit.frag.spv");

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
    std::array<VkPipelineColorBlendAttachmentState, 2> blendAttachments{};
    blendAttachments[0].colorWriteMask = ALL_CHANNELS;

    VkPipelineColorBlendStateCreateInfo colorBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = blendAttachments.data();

    // 양면 재질은 컬 모드만 다르므로 파이프라인을 늘리지 않고 동적 상태로 전환한다.
    VkDynamicState dynamicStates[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_CULL_MODE};
    VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = 3;
    dynamicState.pDynamicStates = dynamicStates;

    std::array<VkFormat, 2> colorFormats{COLOR_FORMAT, OIT_REVEALAGE_FORMAT};
    VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    renderingInfo.colorAttachmentCount = 1;
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

    for (uint32_t variant = 0; variant < ALPHA_MODE_COUNT; ++variant) {
        alphaVariant = variant;
        if (variant == TRANSLUCENT_MODE) {
            // 반투명은 누적과 잔여 투과율 두 대상에 기록하고 깊이는 읽기만 한다.
            stages[1].module = oitFragment;
            depthStencil.depthWriteEnable = VK_FALSE;
            renderingInfo.colorAttachmentCount = 2;
            colorFormats[0] = OIT_ACCUMULATION_FORMAT;
            colorBlend.attachmentCount = 2;

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
        }
        VK_CHECK(vkCreateGraphicsPipelines(
            context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &meshPipelines[variant]));
    }

    // 와이어프레임 디버그 뷰는 불투명 경로를 선 모드로 한 벌 더 만든다.
    alphaVariant = 0;
    stages[1].module = opaqueFragment;
    depthStencil.depthWriteEnable = VK_TRUE;
    renderingInfo.colorAttachmentCount = 1;
    colorFormats[0] = COLOR_FORMAT;
    colorBlend.attachmentCount = 1;
    blendAttachments[0].blendEnable = VK_FALSE;
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

        for (uint32_t variant = 0; variant < ALPHA_MODE_COUNT; ++variant) {
            alphaVariant = variant;
            bool translucent = variant == static_cast<uint32_t>(asset::AlphaMode::TRANSLUCENT);
            meshStages[2].module = translucent ? oitFragment : opaqueFragment;
            depthStencil.depthWriteEnable = translucent ? VK_FALSE : VK_TRUE;
            renderingInfo.colorAttachmentCount = translucent ? 2 : 1;
            colorFormats[0] = translucent ? OIT_ACCUMULATION_FORMAT : COLOR_FORMAT;
            colorBlend.attachmentCount = translucent ? 2 : 1;
            blendAttachments[0].blendEnable = translucent ? VK_TRUE : VK_FALSE;
            VK_CHECK(vkCreateGraphicsPipelines(
                context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &meshShaderPipelines[variant]));
        }

        vkDestroyShaderModule(context.device, meshModule, nullptr);
        vkDestroyShaderModule(context.device, taskModule, nullptr);

        drawMeshTasksIndirect = reinterpret_cast<PFN_vkCmdDrawMeshTasksIndirectEXT>(
            vkGetDeviceProcAddr(context.device, "vkCmdDrawMeshTasksIndirectEXT"));
        if (drawMeshTasksIndirect == nullptr) {
            core::fatal("vkCmdDrawMeshTasksIndirectEXT 를 찾을 수 없습니다");
        }
        useMeshShader = true;
        spdlog::info("mesh shader 경로 사용 가능");
    }

    vkDestroyShaderModule(context.device, oitFragment, nullptr);
    vkDestroyShaderModule(context.device, opaqueFragment, nullptr);
    vkDestroyShaderModule(context.device, vertexModule, nullptr);
}

void Renderer::createPostPipelines() {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.size = sizeof(TonemapPushConstants);

    VkDescriptorSetLayout bindlessLayout = bindless.layout();
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &bindlessLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &postPipelineLayout));

    VkShaderModule vertexModule = createShaderModule(context.device, "fullscreen.vert.spv");
    VkShaderModule compositeFragment = createShaderModule(context.device, "oit_composite.frag.spv");
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
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = ALL_CHANNELS;
    // 합성은 알파에 담긴 잔여 투과율로 src*(1-a) + dst*a 를 만든다.
    blendAttachment.blendEnable = VK_TRUE;
    blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

    VkPipelineColorBlendStateCreateInfo colorBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &blendAttachment;

    VkDynamicState dynamicStates[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = 2;
    dynamicState.pDynamicStates = dynamicStates;

    VkFormat colorFormat = COLOR_FORMAT;
    VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat;

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

    stages[1].module = tonemapFragment;
    blendAttachment.blendEnable = VK_FALSE;
    colorFormat = PRESENT_FORMAT;
    VK_CHECK(vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &tonemapPipeline));

    vkDestroyShaderModule(context.device, tonemapFragment, nullptr);
    vkDestroyShaderModule(context.device, compositeFragment, nullptr);
    vkDestroyShaderModule(context.device, vertexModule, nullptr);
}

void Renderer::recreateSwapchain() {
    waitIdle();
    swapchain->recreate(vsync);
    createRenderTargets();
    // 이미지 개수가 달라질 수 있으므로 표시 완료 세마포어도 다시 만든다.
    createPresentSemaphores();
    resizeRequested = false;
}

FrameBatches Renderer::buildDrawCommands(Frame& frame, const scene::Scene& scene) {
    reserveInstances(frame, static_cast<uint32_t>(scene.objects.size()));

    // 재질 경로와 면 방향 조합마다 명령이 연속 구간을 이루도록 두 번 순회한다.
    auto bucketOf = [this](const scene::Object& object) {
        const asset::Material& material = geometry.material(geometry.mesh(object.meshIndex).materialIndex);
        return std::pair<size_t, size_t>{static_cast<size_t>(material.alphaMode), material.doubleSided ? 1U : 0U};
    };
    auto lodFor = [this](const scene::Object& object) -> const GpuMeshLod& {
        const GpuMesh& mesh = geometry.mesh(object.meshIndex);
        return geometry.lod(mesh.lodOffset + std::min(lodLevel, mesh.lodCount - 1));
    };
    auto groupsFor = [&lodFor](const scene::Object& object) {
        return (lodFor(object).meshletCount + MESHLET_GROUP_SIZE - 1) / MESHLET_GROUP_SIZE;
    };
    auto drawable = [this](const scene::Object& object) {
        return object.visible && object.meshIndex < geometry.meshCount();
    };

    FrameBatches batches{};
    uint32_t totalGroups = 0;
    for (const scene::Object& object : scene.objects) {
        if (!drawable(object)) {
            continue;
        }
        auto [mode, sided] = bucketOf(object);
        ++batches.draws[mode][sided].count;
        uint32_t groups = groupsFor(object);
        batches.groups[mode][sided].count += groups;
        totalGroups += groups;
    }
    reserveMeshletGroups(frame, totalGroups);

    uint32_t drawOffset = 0;
    uint32_t groupOffset = 0;
    for (size_t mode = 0; mode < ALPHA_MODE_COUNT; ++mode) {
        for (size_t sided = 0; sided < 2; ++sided) {
            batches.draws[mode][sided].first = drawOffset;
            drawOffset += batches.draws[mode][sided].count;
            batches.groups[mode][sided].first = groupOffset;
            groupOffset += batches.groups[mode][sided].count;
        }
    }

    auto* instances = static_cast<GpuInstance*>(frame.instanceBuffer.mapped);
    auto* draws = static_cast<VkDrawIndexedIndirectCommand*>(frame.drawBuffer.mapped);
    auto* groups = static_cast<GpuMeshletGroup*>(frame.meshletGroupBuffer.mapped);

    std::array<std::array<uint32_t, 2>, ALPHA_MODE_COUNT> drawCursors{};
    std::array<std::array<uint32_t, 2>, ALPHA_MODE_COUNT> groupCursors{};
    for (size_t mode = 0; mode < ALPHA_MODE_COUNT; ++mode) {
        for (size_t sided = 0; sided < 2; ++sided) {
            drawCursors[mode][sided] = batches.draws[mode][sided].first;
            groupCursors[mode][sided] = batches.groups[mode][sided].first;
        }
    }

    for (const scene::Object& object : scene.objects) {
        if (!drawable(object)) {
            continue;
        }
        auto [mode, sided] = bucketOf(object);
        uint32_t slot = drawCursors[mode][sided]++;

        const GpuMesh& mesh = geometry.mesh(object.meshIndex);
        const GpuMeshLod& lod = lodFor(object);
        glm::mat4 model = object.transform.matrix();

        instances[slot].model = model;
        instances[slot].normalMatrix = glm::mat4(glm::inverseTranspose(glm::mat3(model)));
        instances[slot].meshIndex = object.meshIndex;

        draws[slot].indexCount = lod.indexCount;
        draws[slot].instanceCount = 1;
        draws[slot].firstIndex = lod.indexOffset;
        draws[slot].vertexOffset = mesh.vertexOffset;
        // 셰이더는 gl_InstanceIndex 로 인스턴스 배열을 참조한다.
        draws[slot].firstInstance = slot;

        for (uint32_t group = 0; group < groupsFor(object); ++group) {
            uint32_t groupSlot = groupCursors[mode][sided]++;
            uint32_t first = group * MESHLET_GROUP_SIZE;
            groups[groupSlot].instanceIndex = slot;
            groups[groupSlot].firstMeshlet = lod.meshletOffset + first;
            groups[groupSlot].meshletCount = std::min(MESHLET_GROUP_SIZE, lod.meshletCount - first);
            groups[groupSlot].padding = 0;
        }
    }

    auto* meshTasks = static_cast<VkDrawMeshTasksIndirectCommandEXT*>(frame.meshTaskIndirectBuffer.mapped);
    for (size_t mode = 0; mode < ALPHA_MODE_COUNT; ++mode) {
        for (size_t sided = 0; sided < 2; ++sided) {
            size_t bucket = mode * 2 + sided;
            meshTasks[bucket].groupCountX = batches.groups[mode][sided].count;
            meshTasks[bucket].groupCountY = 1;
            meshTasks[bucket].groupCountZ = 1;
        }
    }

    auto* camera = static_cast<GpuCamera*>(frame.cameraBuffer.mapped);
    float aspect = static_cast<float>(currentRenderExtent.width) / static_cast<float>(currentRenderExtent.height);
    camera->viewProjection = scene.camera.projectionMatrix(aspect) * scene.camera.viewMatrix();
    camera->position = glm::vec4{scene.camera.position, 1.0F};
    camera->parameters = glm::vec4{scene.camera.nearPlane, 0.0F, 0.0F, 0.0F};

    return batches;
}

void Renderer::recordGeometryPass(VkCommandBuffer commandBuffer, const FrameBatches& batches, bool translucentPass) {
    constexpr VkDeviceSize DRAW_STRIDE = sizeof(VkDrawIndexedIndirectCommand);
    constexpr VkDeviceSize TASK_STRIDE = sizeof(VkDrawMeshTasksIndirectCommandEXT);

    Frame& frame = frames[frameIndex % FRAMES_IN_FLIGHT];
    // 와이어프레임 디버그 뷰는 고전 경로에만 있으므로 그때는 mesh shader 경로를 쓰지 않는다.
    bool meshPath = useMeshShader && meshShaderAvailable() && !wireframe;

    size_t firstMode = translucentPass ? TRANSLUCENT_MODE : 0;
    size_t lastMode = translucentPass ? TRANSLUCENT_MODE + 1 : TRANSLUCENT_MODE;

    for (size_t mode = firstMode; mode < lastMode; ++mode) {
        bool bound = false;
        for (size_t sided = 0; sided < 2; ++sided) {
            const DrawBatch& batch = meshPath ? batches.groups[mode][sided] : batches.draws[mode][sided];
            if (batch.count == 0) {
                continue;
            }
            if (!bound) {
                VkPipeline pipeline = meshShaderPipelines[mode];
                if (!meshPath) {
                    pipeline = wireframe && !translucentPass ? wireframePipeline : meshPipelines[mode];
                }
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
                bound = true;
            }
            vkCmdSetCullMode(commandBuffer, sided == 1 ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT);

            if (meshPath) {
                // 태스크 셰이더는 gl_WorkGroupID 가 0 부터 시작하므로 구간 시작을 따로 알려 준다.
                vkCmdPushConstants(commandBuffer,
                                   meshPipelineLayout,
                                   scenePushStages,
                                   offsetof(ScenePushConstants, meshletGroupBase),
                                   sizeof(uint32_t),
                                   &batch.first);
                drawMeshTasksIndirect(commandBuffer,
                                      frame.meshTaskIndirectBuffer.handle,
                                      (mode * 2 + sided) * TASK_STRIDE,
                                      1,
                                      static_cast<uint32_t>(TASK_STRIDE));
            } else {
                vkCmdDrawIndexedIndirect(commandBuffer,
                                         frame.drawBuffer.handle,
                                         batch.first * DRAW_STRIDE,
                                         batch.count,
                                         static_cast<uint32_t>(DRAW_STRIDE));
            }
        }
    }
}

void Renderer::recordUiPass(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
    VkRenderingAttachmentInfo uiColor =
        colorAttachment(swapchain->imageViews[imageIndex], VK_ATTACHMENT_LOAD_OP_CLEAR, {{0.0F, 0.0F, 0.0F, 1.0F}});
    VkRenderingInfo uiPass{VK_STRUCTURE_TYPE_RENDERING_INFO};
    uiPass.renderArea.extent = swapchain->extent;
    uiPass.layerCount = 1;
    uiPass.colorAttachmentCount = 1;
    uiPass.pColorAttachments = &uiColor;

    vkCmdBeginRendering(commandBuffer, &uiPass);
    setFullViewport(commandBuffer, swapchain->extent);
    if (uiCallback) {
        uiCallback(commandBuffer);
    }
    vkCmdEndRendering(commandBuffer);
}

void Renderer::recordCommands(Frame& frame, uint32_t imageIndex, const FrameBatches& batches) {
    VkCommandBuffer commandBuffer = frame.commandBuffer;
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    bool hasTranslucent = batches.draws[TRANSLUCENT_MODE][0].count + batches.draws[TRANSLUCENT_MODE][1].count > 0;
    oitTargetsValid = hasTranslucent;

    imageBarrier(commandBuffer,
                 targets.color.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                 0,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    imageBarrier(commandBuffer,
                 targets.depth.handle,
                 VK_IMAGE_ASPECT_DEPTH_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                 0,
                 VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    ScenePushConstants scenePushConstants{geometry.vertexBuffer.address,
                                          geometry.meshBuffer.address,
                                          frame.instanceBuffer.address,
                                          geometry.materialBuffer.address,
                                          frame.cameraBuffer.address,
                                          geometry.meshletBuffer.address,
                                          geometry.meshletTriangleBuffer.address,
                                          geometry.vertexMeshletBuffer.address,
                                          frame.meshletGroupBuffer.address,
                                          0,
                                          debugMode};
    VkDescriptorSet bindlessSet = bindless.set();
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdPushConstants(
        commandBuffer, meshPipelineLayout, scenePushStages, 0, sizeof(scenePushConstants), &scenePushConstants);
    vkCmdBindIndexBuffer(commandBuffer, geometry.indexBuffer.handle, 0, VK_INDEX_TYPE_UINT32);

    // 1) 불투명과 컷오프 경로를 HDR 색상 대상에 그린다.
    VkRenderingAttachmentInfo opaqueColor =
        colorAttachment(targets.color.view, VK_ATTACHMENT_LOAD_OP_CLEAR, {{0.05F, 0.05F, 0.07F, 1.0F}});

    VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAttachment.imageView = targets.depth.view;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue.depthStencil.depth = 0.0F;

    VkRenderingInfo opaquePass{VK_STRUCTURE_TYPE_RENDERING_INFO};
    opaquePass.renderArea.extent = currentRenderExtent;
    opaquePass.layerCount = 1;
    opaquePass.colorAttachmentCount = 1;
    opaquePass.pColorAttachments = &opaqueColor;
    opaquePass.pDepthAttachment = &depthAttachment;

    vkCmdBeginRendering(commandBuffer, &opaquePass);
    setFullViewport(commandBuffer, currentRenderExtent);
    recordGeometryPass(commandBuffer, batches, false);
    vkCmdEndRendering(commandBuffer);

    if (hasTranslucent) {
        // 2) 반투명은 누적과 잔여 투과율 대상에 순서 독립으로 기록한다.
        imageBarrier(commandBuffer,
                     targets.oitAccumulation.handle,
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                     0,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        imageBarrier(commandBuffer,
                     targets.oitRevealage.handle,
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                     0,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        imageBarrier(commandBuffer,
                     targets.depth.handle,
                     VK_IMAGE_ASPECT_DEPTH_BIT,
                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT);

        std::array<VkRenderingAttachmentInfo, 2> oitAttachments{
            colorAttachment(targets.oitAccumulation.view, VK_ATTACHMENT_LOAD_OP_CLEAR, {{0.0F, 0.0F, 0.0F, 0.0F}}),
            colorAttachment(targets.oitRevealage.view, VK_ATTACHMENT_LOAD_OP_CLEAR, {{1.0F, 0.0F, 0.0F, 0.0F}})};

        VkRenderingAttachmentInfo readOnlyDepth = depthAttachment;
        readOnlyDepth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        readOnlyDepth.storeOp = VK_ATTACHMENT_STORE_OP_NONE;

        VkRenderingInfo translucentPass{VK_STRUCTURE_TYPE_RENDERING_INFO};
        translucentPass.renderArea.extent = currentRenderExtent;
        translucentPass.layerCount = 1;
        translucentPass.colorAttachmentCount = static_cast<uint32_t>(oitAttachments.size());
        translucentPass.pColorAttachments = oitAttachments.data();
        translucentPass.pDepthAttachment = &readOnlyDepth;

        vkCmdBeginRendering(commandBuffer, &translucentPass);
        setFullViewport(commandBuffer, currentRenderExtent);
        recordGeometryPass(commandBuffer, batches, true);
        vkCmdEndRendering(commandBuffer);

        // 3) 누적 결과를 HDR 색상 위에 합성한다.
        imageBarrier(commandBuffer,
                     targets.oitAccumulation.handle,
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        imageBarrier(commandBuffer,
                     targets.oitRevealage.handle,
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        imageBarrier(commandBuffer,
                     targets.color.handle,
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

        VkRenderingAttachmentInfo compositeColor = colorAttachment(targets.color.view, VK_ATTACHMENT_LOAD_OP_LOAD, {});
        VkRenderingInfo compositePass{VK_STRUCTURE_TYPE_RENDERING_INFO};
        compositePass.renderArea.extent = currentRenderExtent;
        compositePass.layerCount = 1;
        compositePass.colorAttachmentCount = 1;
        compositePass.pColorAttachments = &compositeColor;

        CompositePushConstants compositePushConstants{targets.accumulationSlot, targets.revealageSlot};
        vkCmdBeginRendering(commandBuffer, &compositePass);
        setFullViewport(commandBuffer, currentRenderExtent);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeline);
        vkCmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, postPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
        vkCmdPushConstants(commandBuffer,
                           postPipelineLayout,
                           VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           sizeof(compositePushConstants),
                           &compositePushConstants);
        vkCmdDraw(commandBuffer, 3, 1, 0, 0);
        vkCmdEndRendering(commandBuffer);
    }

    // 4) HDR 색상을 톤 매핑해 표시 이미지에 옮긴다. 편집기 뷰포트가 이 이미지를 그대로 보여준다.
    imageBarrier(commandBuffer,
                 targets.color.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    imageBarrier(commandBuffer,
                 targets.present.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                 0,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo presentColor = colorAttachment(targets.present.view, VK_ATTACHMENT_LOAD_OP_DONT_CARE, {});
    VkRenderingInfo tonemapPass{VK_STRUCTURE_TYPE_RENDERING_INFO};
    tonemapPass.renderArea.extent = currentRenderExtent;
    tonemapPass.layerCount = 1;
    tonemapPass.colorAttachmentCount = 1;
    tonemapPass.pColorAttachments = &presentColor;

    TonemapPushConstants tonemapPushConstants{targets.colorSlot, exposure};
    vkCmdBeginRendering(commandBuffer, &tonemapPass);
    setFullViewport(commandBuffer, currentRenderExtent);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemapPipeline);
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, postPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer,
                       postPipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       sizeof(tonemapPushConstants),
                       &tonemapPushConstants);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    vkCmdEndRendering(commandBuffer);

    // 5) 편집기 UI 를 스왑체인에 그린다. 오프스크린 대상들은 UI 가 샘플링할 수 있는 레이아웃으로 옮긴다.
    imageBarrier(commandBuffer,
                 targets.present.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    imageBarrier(commandBuffer,
                 targets.depth.handle,
                 VK_IMAGE_ASPECT_DEPTH_BIT,
                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    imageBarrier(commandBuffer,
                 swapchain->images[imageIndex],
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                 0,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    recordUiPass(commandBuffer, imageIndex);

    if (!capturePath.empty()) {
        imageBarrier(commandBuffer,
                     swapchain->images[imageIndex],
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COPY_BIT,
                     VK_ACCESS_2_TRANSFER_READ_BIT);

        VkBufferImageCopy2 region{VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {swapchain->extent.width, swapchain->extent.height, 1};
        VkCopyImageToBufferInfo2 copyInfo{VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2};
        copyInfo.srcImage = swapchain->images[imageIndex];
        copyInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        copyInfo.dstBuffer = captureBuffer.handle;
        copyInfo.regionCount = 1;
        copyInfo.pRegions = &region;
        vkCmdCopyImageToBuffer2(commandBuffer, &copyInfo);

        imageBarrier(commandBuffer,
                     swapchain->images[imageIndex],
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     VK_PIPELINE_STAGE_2_COPY_BIT,
                     VK_ACCESS_2_TRANSFER_READ_BIT,
                     VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                     0);
    } else {
        imageBarrier(commandBuffer,
                     swapchain->images[imageIndex],
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                     0);
    }

    VK_CHECK(vkEndCommandBuffer(commandBuffer));
}

void Renderer::writeCapture() {
    uint32_t width = swapchain->extent.width;
    uint32_t height = swapchain->extent.height;
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4);
    std::memcpy(pixels.data(), captureBuffer.mapped, pixels.size());

    bool isBgra = swapchain->format == VK_FORMAT_B8G8R8A8_SRGB || swapchain->format == VK_FORMAT_B8G8R8A8_UNORM;
    if (isBgra) {
        for (size_t i = 0; i < pixels.size(); i += 4) {
            std::swap(pixels[i], pixels[i + 2]);
        }
    }

    std::string path = capturePath.string();
    int written = stbi_write_png(
        path.c_str(), static_cast<int>(width), static_cast<int>(height), 4, pixels.data(), static_cast<int>(width) * 4);
    if (written == 0) {
        core::fatal("화면 캡처 저장에 실패했습니다: {}", path);
    }
    spdlog::info("화면 캡처 저장: {} ({}x{})", path, width, height);
    capturePath.clear();
}

void Renderer::drawFrame(const scene::Scene& scene) {
    if (swapchain->extent.width == 0 || swapchain->extent.height == 0) {
        return;
    }
    if (resizeRequested) {
        recreateSwapchain();
    }

    Frame& frame = frames[frameIndex % FRAMES_IN_FLIGHT];

    // 같은 프레임 자원을 다시 쓰기 전에 FRAMES_IN_FLIGHT 이전 프레임의 완료를 기다린다.
    if (frameIndex >= FRAMES_IN_FLIGHT) {
        uint64_t waitValue = frameIndex - FRAMES_IN_FLIGHT + 1;
        VkSemaphoreWaitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
        waitInfo.semaphoreCount = 1;
        waitInfo.pSemaphores = &frameTimeline;
        waitInfo.pValues = &waitValue;
        VK_CHECK(vkWaitSemaphores(context.device, &waitInfo, UINT64_MAX));
    }

    uint32_t imageIndex = 0;
    VkResult acquireResult = vkAcquireNextImageKHR(
        context.device, swapchain->handle, UINT64_MAX, frame.imageAvailable, VK_NULL_HANDLE, &imageIndex);
    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return;
    }
    if (acquireResult != VK_SUCCESS && acquireResult != VK_SUBOPTIMAL_KHR) {
        core::fatal("스왑체인 이미지 획득에 실패했습니다: {}", toString(acquireResult));
    }

    FrameBatches batches = buildDrawCommands(frame, scene);

    if (!capturePath.empty()) {
        VkDeviceSize required = static_cast<VkDeviceSize>(swapchain->extent.width) * swapchain->extent.height * 4;
        if (captureBuffer.size < required) {
            destroyBuffer(context, captureBuffer);
            captureBuffer = createBuffer(
                context, required, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryLocation::HOST_READ, "화면 캡처");
        }
    }

    VK_CHECK(vkResetCommandPool(context.device, frame.commandPool, 0));
    recordCommands(frame, imageIndex, batches);

    VkSemaphoreSubmitInfo waitSemaphore{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
    waitSemaphore.semaphore = frame.imageAvailable;
    waitSemaphore.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSemaphoreSubmitInfo signalSemaphores[2]{};
    signalSemaphores[0].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signalSemaphores[0].semaphore = presentReady[imageIndex];
    signalSemaphores[0].stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    signalSemaphores[1].sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signalSemaphores[1].semaphore = frameTimeline;
    signalSemaphores[1].value = frameIndex + 1;
    signalSemaphores[1].stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

    VkCommandBufferSubmitInfo commandBufferInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO};
    commandBufferInfo.commandBuffer = frame.commandBuffer;

    VkSubmitInfo2 submitInfo{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
    submitInfo.waitSemaphoreInfoCount = 1;
    submitInfo.pWaitSemaphoreInfos = &waitSemaphore;
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &commandBufferInfo;
    submitInfo.signalSemaphoreInfoCount = 2;
    submitInfo.pSignalSemaphoreInfos = signalSemaphores;
    VK_CHECK(vkQueueSubmit2(context.graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE));

    VkPresentInfoKHR presentInfo{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &presentReady[imageIndex];
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &swapchain->handle;
    presentInfo.pImageIndices = &imageIndex;
    VkResult presentResult = vkQueuePresentKHR(context.graphicsQueue, &presentInfo);

    ++frameIndex;

    if (!capturePath.empty()) {
        waitIdle();
        writeCapture();
    }

    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR) {
        resizeRequested = true;
    } else if (presentResult != VK_SUCCESS) {
        core::fatal("스왑체인 표시에 실패했습니다: {}", toString(presentResult));
    }
}

} // namespace gfx
