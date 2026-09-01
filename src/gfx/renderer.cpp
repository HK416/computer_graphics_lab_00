#include "gfx/renderer.h"

#include <algorithm>
#include <array>
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

constexpr VkFormat DEPTH_FORMAT = VK_FORMAT_D32_SFLOAT;
constexpr uint32_t MINIMUM_INSTANCE_CAPACITY = 1024;

// shaders/scene_data.glsl 의 CameraBuffer 와 배치가 같아야 한다.
struct GpuCamera {
    glm::mat4 viewProjection;
    glm::vec4 position;
};

struct ScenePushConstants {
    VkDeviceAddress vertices;
    VkDeviceAddress meshes;
    VkDeviceAddress instances;
    VkDeviceAddress materials;
    VkDeviceAddress camera;
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

} // namespace

Renderer::Renderer(Context& context, GeometryStore& geometry, BindlessTextures& bindless, SDL_Window* window)
    : context(context), geometry(geometry), bindless(bindless) {
    swapchain = std::make_unique<Swapchain>(context, window, vsync);
    createDepthImage();
    createFrames();
    createPresentSemaphores();
    createMeshPipelines();

    VkSemaphoreTypeCreateInfo timelineInfo{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphoreInfo.pNext = &timelineInfo;
    VK_CHECK(vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &frameTimeline));
}

Renderer::~Renderer() {
    waitIdle();
    for (VkPipeline pipeline : meshPipelines) {
        vkDestroyPipeline(context.device, pipeline, nullptr);
    }
    vkDestroyPipelineLayout(context.device, meshPipelineLayout, nullptr);
    vkDestroySemaphore(context.device, frameTimeline, nullptr);
    destroyPresentSemaphores();
    for (Frame& frame : frames) {
        destroyBuffer(context, frame.drawBuffer);
        destroyBuffer(context, frame.instanceBuffer);
        destroyBuffer(context, frame.cameraBuffer);
        vkDestroySemaphore(context.device, frame.imageAvailable, nullptr);
        vkDestroyCommandPool(context.device, frame.commandPool, nullptr);
    }
    destroyBuffer(context, captureBuffer);
    destroyImage(context, depthImage);
    swapchain.reset();
}

void Renderer::waitIdle() {
    VK_CHECK(vkDeviceWaitIdle(context.device));
}

void Renderer::createDepthImage() {
    destroyImage(context, depthImage);
    ImageDesc desc;
    desc.extent = {swapchain->extent.width, swapchain->extent.height, 1};
    desc.format = DEPTH_FORMAT;
    desc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    desc.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    depthImage = createImage(context, desc, "깊이");
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
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.size = sizeof(ScenePushConstants);

    VkDescriptorSetLayout bindlessLayout = bindless.layout();
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &bindlessLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &meshPipelineLayout));

    VkShaderModule vertexModule = createShaderModule(context.device, "mesh.vert.spv");
    VkShaderModule fragmentModule = createShaderModule(context.device, "mesh.frag.spv");

    // 프래그먼트 셰이더의 알파 경로를 특수화 상수로 고정해, 불투명 경로에서는 discard 가 사라진다.
    uint32_t alphaVariant = 0;
    VkSpecializationMapEntry specializationEntry{0, 0, sizeof(uint32_t)};
    VkSpecializationInfo specialization{};
    specialization.mapEntryCount = 1;
    specialization.pMapEntries = &specializationEntry;
    specialization.dataSize = sizeof(alphaVariant);
    specialization.pData = &alphaVariant;

    VkPipelineShaderStageCreateInfo stages[2]{};
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vertexModule;
    stages[0].pName = "main";
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fragmentModule;
    stages[1].pName = "main";
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

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    VkPipelineColorBlendStateCreateInfo colorBlend{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &blendAttachment;

    // 양면 재질은 컬 모드만 다르므로 파이프라인을 늘리지 않고 동적 상태로 전환한다.
    VkDynamicState dynamicStates[]{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_CULL_MODE};
    VkPipelineDynamicStateCreateInfo dynamicState{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamicState.dynamicStateCount = 3;
    dynamicState.pDynamicStates = dynamicStates;

    VkPipelineRenderingCreateInfo renderingInfo{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &swapchain->format;
    renderingInfo.depthAttachmentFormat = DEPTH_FORMAT;

    VkGraphicsPipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pipelineInfo.pNext = &renderingInfo;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = stages;
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
        bool translucent = variant == static_cast<uint32_t>(asset::AlphaMode::TRANSLUCENT);
        // 반투명은 깊이를 쓰지 않고 알파 블렌딩한다. 순서 독립 처리는 다음 단계에서 대체한다.
        depthStencil.depthWriteEnable = translucent ? VK_FALSE : VK_TRUE;
        blendAttachment.blendEnable = translucent ? VK_TRUE : VK_FALSE;
        blendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        blendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
        VK_CHECK(vkCreateGraphicsPipelines(
            context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &meshPipelines[variant]));
    }

    vkDestroyShaderModule(context.device, fragmentModule, nullptr);
    vkDestroyShaderModule(context.device, vertexModule, nullptr);
}

void Renderer::recreateSwapchain() {
    waitIdle();
    swapchain->recreate(vsync);
    createDepthImage();
    // 이미지 개수가 달라질 수 있으므로 표시 완료 세마포어도 다시 만든다.
    createPresentSemaphores();
    resizeRequested = false;
}

DrawBatches Renderer::buildDrawCommands(Frame& frame, const scene::Scene& scene) {
    reserveInstances(frame, static_cast<uint32_t>(scene.objects.size()));

    auto* instances = static_cast<GpuInstance*>(frame.instanceBuffer.mapped);
    auto* draws = static_cast<VkDrawIndexedIndirectCommand*>(frame.drawBuffer.mapped);

    // 재질 경로와 면 방향 조합마다 간접 명령이 연속 구간을 이루도록 두 번 순회한다.
    auto bucketOf = [this](const scene::Object& object) {
        const asset::Material& material = geometry.material(geometry.mesh(object.meshIndex).materialIndex);
        return std::pair<size_t, size_t>{static_cast<size_t>(material.alphaMode), material.doubleSided ? 1U : 0U};
    };

    DrawBatches batches{};
    for (const scene::Object& object : scene.objects) {
        if (!object.visible || object.meshIndex >= geometry.meshCount()) {
            continue;
        }
        auto [mode, sided] = bucketOf(object);
        ++batches[mode][sided].count;
    }

    uint32_t offset = 0;
    for (auto& modeBatches : batches) {
        for (DrawBatch& batch : modeBatches) {
            batch.first = offset;
            offset += batch.count;
        }
    }

    std::array<std::array<uint32_t, 2>, ALPHA_MODE_COUNT> cursors{};
    for (size_t mode = 0; mode < ALPHA_MODE_COUNT; ++mode) {
        for (size_t sided = 0; sided < 2; ++sided) {
            cursors[mode][sided] = batches[mode][sided].first;
        }
    }

    for (const scene::Object& object : scene.objects) {
        if (!object.visible || object.meshIndex >= geometry.meshCount()) {
            continue;
        }
        auto [mode, sided] = bucketOf(object);
        uint32_t slot = cursors[mode][sided]++;

        const GpuMesh& mesh = geometry.mesh(object.meshIndex);
        glm::mat4 model = object.transform.matrix();

        instances[slot].model = model;
        instances[slot].normalMatrix = glm::mat4(glm::inverseTranspose(glm::mat3(model)));
        instances[slot].meshIndex = object.meshIndex;

        draws[slot].indexCount = mesh.indexCount;
        draws[slot].instanceCount = 1;
        draws[slot].firstIndex = mesh.indexOffset;
        draws[slot].vertexOffset = mesh.vertexOffset;
        // 셰이더는 gl_InstanceIndex 로 인스턴스 배열을 참조한다.
        draws[slot].firstInstance = slot;
    }

    auto* camera = static_cast<GpuCamera*>(frame.cameraBuffer.mapped);
    float aspect = static_cast<float>(swapchain->extent.width) / static_cast<float>(swapchain->extent.height);
    camera->viewProjection = scene.camera.projectionMatrix(aspect) * scene.camera.viewMatrix();
    camera->position = glm::vec4{scene.camera.position, 1.0F};

    return batches;
}

void Renderer::recordCommands(Frame& frame, uint32_t imageIndex, const DrawBatches& batches) {
    VkCommandBuffer commandBuffer = frame.commandBuffer;
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));

    imageBarrier(commandBuffer,
                 swapchain->images[imageIndex],
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                 0,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    imageBarrier(commandBuffer,
                 depthImage.handle,
                 VK_IMAGE_ASPECT_DEPTH_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                 0,
                 VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo colorAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    colorAttachment.imageView = swapchain->imageViews[imageIndex];
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.clearValue.color = {{0.05F, 0.05F, 0.07F, 1.0F}};

    VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAttachment.imageView = depthImage.view;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue.depthStencil.depth = 0.0F;

    VkRenderingInfo renderingInfo{VK_STRUCTURE_TYPE_RENDERING_INFO};
    renderingInfo.renderArea.extent = swapchain->extent;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachment;
    renderingInfo.pDepthAttachment = &depthAttachment;
    vkCmdBeginRendering(commandBuffer, &renderingInfo);

    VkViewport viewport{};
    viewport.width = static_cast<float>(swapchain->extent.width);
    viewport.height = static_cast<float>(swapchain->extent.height);
    viewport.maxDepth = 1.0F;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = swapchain->extent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

    ScenePushConstants pushConstants{geometry.vertexBuffer.address,
                                     geometry.meshBuffer.address,
                                     frame.instanceBuffer.address,
                                     geometry.materialBuffer.address,
                                     frame.cameraBuffer.address};
    VkDescriptorSet bindlessSet = bindless.set();
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer,
                       meshPipelineLayout,
                       VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       sizeof(pushConstants),
                       &pushConstants);
    vkCmdBindIndexBuffer(commandBuffer, geometry.indexBuffer.handle, 0, VK_INDEX_TYPE_UINT32);

    constexpr VkDeviceSize DRAW_STRIDE = sizeof(VkDrawIndexedIndirectCommand);
    for (size_t mode = 0; mode < ALPHA_MODE_COUNT; ++mode) {
        bool bound = false;
        for (size_t sided = 0; sided < 2; ++sided) {
            const DrawBatch& batch = batches[mode][sided];
            if (batch.count == 0) {
                continue;
            }
            if (!bound) {
                vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, meshPipelines[mode]);
                bound = true;
            }
            vkCmdSetCullMode(commandBuffer, sided == 1 ? VK_CULL_MODE_NONE : VK_CULL_MODE_BACK_BIT);
            vkCmdDrawIndexedIndirect(commandBuffer,
                                     frame.drawBuffer.handle,
                                     batch.first * DRAW_STRIDE,
                                     batch.count,
                                     static_cast<uint32_t>(DRAW_STRIDE));
        }
    }

    vkCmdEndRendering(commandBuffer);

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

    DrawBatches batches = buildDrawCommands(frame, scene);

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
