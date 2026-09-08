#include "gfx/observation.h"

#include <algorithm>
#include <cmath>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <spdlog/spdlog.h>

#include "gfx/bindless.h"
#include "gfx/context.h"
#include "gfx/geometry.h"
#include "gfx/vk_check.h"
#include "scene/scene.h"

namespace gfx {
namespace {

constexpr VkFormat COLOR_FORMAT = VK_FORMAT_R8G8B8A8_UNORM;
constexpr VkFormat DEPTH_FORMAT = VK_FORMAT_D32_SFLOAT;
// 인코드 컴퓨트의 작업 그룹. shaders/observation_encode.comp 의 local_size 와 같아야 한다.
constexpr uint32_t ENCODE_GROUP = 8;

// 관측이 쓰는 조명. 장면의 첫 방향광을 따르고 없으면 이 값이다. **자동 노출도 IBL 도 타지 않으므로**
// 밝기가 여기서만 정해진다 — 장면을 바꿔도 관측의 밝기가 흔들리지 않는 것이 요점이다.
constexpr glm::vec3 DEFAULT_LIGHT_DIRECTION{0.45F, 0.8F, 0.35F};
constexpr float DEFAULT_LIGHT_INTENSITY = 3.0F;
constexpr float AMBIENT_INTENSITY = 0.15F;

} // namespace

ObservationLayout buildObservationLayout(const scene::Scene& scene) {
    ObservationLayout layout;
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        const scene::CameraComponent* camera = scene.component<scene::CameraComponent>(index);
        // **관측 플래그가 선 것만 모은다.** 화면용 카메라까지 뷰로 삼으면 장면에 카메라를 하나 더
        // 놓는 것만으로 정책의 입력 모양이 바뀌어, 학습한 가중치를 못 읽게 된다.
        if (camera == nullptr || !camera->observation) {
            continue;
        }
        ObservationLayout::View view;
        view.object = index;
        view.fovYDegrees = camera->fovYDegrees;
        view.nearPlane = camera->nearPlane;
        layout.views.push_back(view);
    }
    return layout;
}

ObservationRenderer::ObservationRenderer(Context& context, BindlessTextures& bindless, GeometryStore& geometry)
    : context(context), bindless(bindless), geometry(geometry) {
    createPipelines();
}

ObservationRenderer::~ObservationRenderer() {
    destroyTargets();
    destroyBuffer(context, instances);
    destroyBuffer(context, viewBuffer);
    destroyBuffer(context, history);
    destroyBuffer(context, features);
    destroyBuffer(context, featureReadback);
    vkDestroyPipeline(context.device, drawPipeline, nullptr);
    vkDestroyPipeline(context.device, encodePipeline, nullptr);
    vkDestroyPipelineLayout(context.device, drawLayout, nullptr);
    vkDestroyPipelineLayout(context.device, encodeLayout, nullptr);
    vkDestroySampler(context.device, sampler, nullptr);
}

void ObservationRenderer::createPipelines() {
    VkShaderModule vertex = tryCreateShaderModule(context.device, "observation.vert.spv");
    VkShaderModule fragment = tryCreateShaderModule(context.device, "observation.frag.spv");
    VkShaderModule encode = tryCreateShaderModule(context.device, "observation_encode.comp.spv");
    if (vertex == VK_NULL_HANDLE || fragment == VK_NULL_HANDLE || encode == VK_NULL_HANDLE) {
        spdlog::warn("관측 셰이더를 읽지 못했습니다. 관측 렌더를 끕니다");
        vkDestroyShaderModule(context.device, vertex, nullptr);
        vkDestroyShaderModule(context.device, fragment, nullptr);
        vkDestroyShaderModule(context.device, encode, nullptr);
        return;
    }

    VkDescriptorSetLayout bindlessLayout = bindless.layout();
    VkPushConstantRange drawRange{
        VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(ObservationPushConstants)};
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &bindlessLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &drawRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &drawLayout));

    VkPushConstantRange encodeRange{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ObservationEncodePushConstants)};
    layoutInfo.pPushConstantRanges = &encodeRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &encodeLayout));

    VkPipelineShaderStageCreateInfo stages[]{shaderStage(VK_SHADER_STAGE_VERTEX_BIT, vertex),
                                             shaderStage(VK_SHADER_STAGE_FRAGMENT_BIT, fragment)};
    // 정점은 전부 buffer device address 로 읽는다. 정점 입력 상태가 비어 있는 것이 이 저장소의 규약이다.
    VkPipelineVertexInputStateCreateInfo vertexInput{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo inputAssembly{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewportState{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rasterization{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    // **면을 자르지 않는다.** 재질별로 파이프라인을 나누지 않으려는 것이고, 깊이 검사가 있으므로 닫힌
    // 메쉬에서는 결과가 같다. 열린 메쉬(천, 판)는 뒷면도 그려져 오히려 관측이 온전하다.
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.lineWidth = 1.0F;
    VkPipelineMultisampleStateCreateInfo multisample{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depthStencil{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    // 역 Z 다. 장면 카메라의 투영이 무한 원거리 역 Z 라 근평면이 1, 무한이 0 이다.
    depthStencil.depthCompareOp = VK_COMPARE_OP_GREATER_OR_EQUAL;
    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
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
    pipelineInfo.layout = drawLayout;
    VK_CHECK(vkCreateGraphicsPipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &drawPipeline));

    VkComputePipelineCreateInfo encodeInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    encodeInfo.stage = shaderStage(VK_SHADER_STAGE_COMPUTE_BIT, encode);
    encodeInfo.layout = encodeLayout;
    VK_CHECK(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &encodeInfo, nullptr, &encodePipeline));

    vkDestroyShaderModule(context.device, vertex, nullptr);
    vkDestroyShaderModule(context.device, fragment, nullptr);
    vkDestroyShaderModule(context.device, encode, nullptr);

    // **NEAREST 다.** 인코드가 texelFetch 로 읽으므로 필터는 쓰이지 않지만, 슬롯 인코딩에 샘플러 번호가
    // 들어가야 해서 하나는 있어야 한다.
    VkSamplerCreateInfo samplerInfo{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(context.device, &samplerInfo, nullptr, &sampler));

    ready = true;
}

void ObservationRenderer::destroyTargets() {
    for (VkImageView view : colorLayerViews) {
        vkDestroyImageView(context.device, view, nullptr);
    }
    for (VkImageView view : depthLayerViews) {
        vkDestroyImageView(context.device, view, nullptr);
    }
    colorLayerViews.clear();
    depthLayerViews.clear();
    destroyImage(context, color);
    destroyImage(context, depth);
}

bool ObservationRenderer::reserve(const ObservationLayout& layout) {
    if (!ready || layout.empty()) {
        return false;
    }
    if (layout == activeLayout) {
        return true;
    }
    // 시점이 바뀌었으면 이미지를 다시 잡지 않더라도 스택은 버린다.
    stackDirty = true;
    uint32_t views = layout.count();
    activeLayout = layout;
    if (views == layers) {
        return true;
    }
    destroyTargets();
    layers = views;
    step = 0;

    ImageDesc colorDesc;
    colorDesc.extent = {OBSERVATION_SIZE, OBSERVATION_SIZE, 1};
    colorDesc.format = COLOR_FORMAT;
    colorDesc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    colorDesc.arrayLayers = layers;
    colorDesc.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    color = createImage(context, colorDesc, "관측 색");

    ImageDesc depthDesc = colorDesc;
    depthDesc.format = DEPTH_FORMAT;
    depthDesc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    depthDesc.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
    depth = createImage(context, depthDesc, "관측 깊이");

    // 층 하나씩 붙이는 뷰. 그림자 아틀라스와 같은 꼴이다.
    auto makeLayerView = [&](const Image& image, VkImageAspectFlags aspect, uint32_t layer) {
        VkImageViewCreateInfo info{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        info.image = image.handle;
        info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        info.format = image.format;
        info.subresourceRange = {aspect, 0, 1, layer, 1};
        VkImageView view = VK_NULL_HANDLE;
        VK_CHECK(vkCreateImageView(context.device, &info, nullptr, &view));
        return view;
    };
    colorLayerViews.resize(layers);
    depthLayerViews.resize(layers);
    for (uint32_t layer = 0; layer < layers; ++layer) {
        colorLayerViews[layer] = makeLayerView(color, VK_IMAGE_ASPECT_COLOR_BIT, layer);
        depthLayerViews[layer] = makeLayerView(depth, VK_IMAGE_ASPECT_DEPTH_BIT, layer);
    }
    // **슬롯은 한 번만 잡고 그 뒤로는 덮어쓴다.** 배열 슬롯에는 자유 목록이 없어 다시 잡을 때마다 새
    // 번호를 물고, 옛 슬롯에는 파괴된 뷰가 남는다. 한계가 16 이라 뷰 수가 열몇 번 바뀌면 죽는다.
    // 렌더 타겟이 slotsAllocated 가드로 하는 것과 같은 규약이다.
    if (colorSlotAllocated) {
        bindless.updateArray(colorBindlessSlot, color.view, sampler);
    } else {
        colorBindlessSlot = bindless.addArray(color.view, sampler);
        colorSlotAllocated = true;
    }

    destroyBuffer(context, viewBuffer);
    viewBuffer = createBuffer(context,
                              static_cast<VkDeviceSize>(layers) * sizeof(GpuObservationView),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              MemoryLocation::HOST_WRITE,
                              "관측 뷰");

    VkDeviceSize featureBytes = static_cast<VkDeviceSize>(featureFloatCount()) * sizeof(float);
    destroyBuffer(context, history);
    history = createBuffer(context,
                           featureBytes,
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                           MemoryLocation::DEVICE,
                           "관측 프레임 스택");
    destroyBuffer(context, features);
    features = createBuffer(context,
                            featureBytes,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                            MemoryLocation::DEVICE,
                            "관측 특징");
    destroyBuffer(context, featureReadback);
    featureReadback =
        createBuffer(context, featureBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryLocation::HOST_READ, "관측 되읽기");
    featureReadbackFloats = static_cast<float*>(featureReadback.mapped);
    return true;
}

void ObservationRenderer::record(VkCommandBuffer commandBuffer,
                                 const scene::Scene& scene,
                                 const ObservationLayout& layout,
                                 VkDeviceAddress skinnedVertices,
                                 bool reset) {
    if (!ready || layers == 0 || layout.count() != layers) {
        return;
    }
    reset = reset || stackDirty;
    stackDirty = false;

    // 그릴 오브젝트를 모은다. 오브젝트 번호가 곧 인스턴스 번호라 draw 의 firstInstance 로 그대로 쓴다.
    drawObjects.clear();
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        uint32_t mesh = scene.meshOf(index);
        if (mesh == scene::INVALID_MESH || !geometry.meshLive(mesh) || !scene.visibleCached(index)) {
            continue;
        }
        drawObjects.push_back(index);
    }

    VkDeviceSize needed = static_cast<VkDeviceSize>(std::max<size_t>(scene.objects.size(), 1)) * sizeof(GpuInstance);
    if (instances.size < needed) {
        // 자라기만 한다. 관측은 장면이 고정된 채로 수만 걸음을 도므로 줄이는 쪽은 값이 없다.
        //
        // **바로 파괴한다.** 관측은 제출마다 펜스를 기다리므로 옛 버퍼를 읽는 명령이 남아 있지 않다.
        // Context 의 은퇴 목록에 맡기면 창이 있는 실행에서 렌더러의 은퇴 자원까지 덩달아 거둬진다.
        destroyBuffer(context, instances);
        instances = createBuffer(context,
                                 needed,
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                 MemoryLocation::HOST_WRITE,
                                 "관측 인스턴스");
    }
    auto* gpuInstances = static_cast<GpuInstance*>(instances.mapped);
    for (uint32_t index : drawObjects) {
        GpuInstance& instance = gpuInstances[index];
        instance = GpuInstance{};
        instance.model = scene.world(index);
        instance.previousModel = instance.model;
        instance.normalMatrix = glm::mat4(glm::inverseTranspose(glm::mat3(instance.model)));
        instance.meshIndex = scene.meshOf(index);
        // ponytail: 스킨은 스킨 컴퓨트가 돈 뒤에만 뜻이 있다. 헤드리스 학습에는 그 패스가 없으므로
        // 스킨 메쉬는 바인드 포즈로 그려진다. 진자·리처는 강체라 해당이 없다.
        instance.skinnedVertexOffset = NO_SKINNED_VERTICES;
        instance.previousSkinnedVertexOffset = NO_SKINNED_VERTICES;
    }

    // 뷰마다 시점과 조명. 조명은 장면의 첫 방향광을 따른다.
    glm::vec3 lightDirection = glm::normalize(DEFAULT_LIGHT_DIRECTION);
    glm::vec3 lightColor{1.0F};
    float lightIntensity = DEFAULT_LIGHT_INTENSITY;
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        const scene::Light* light = scene.component<scene::Light>(index);
        if (light == nullptr || light->type != scene::LightType::DIRECTIONAL) {
            continue;
        }
        // 방향광은 -Z 를 앞으로 본다. 셰이딩이 쓰는 것은 «표면에서 광원으로» 라 부호가 뒤집힌다.
        glm::vec3 forward = -glm::vec3(scene.world(index)[2]);
        if (glm::dot(forward, forward) > 1.0e-8F) {
            lightDirection = glm::normalize(-forward);
            lightColor = light->color;
            lightIntensity = light->intensity;
        }
        break;
    }

    auto* gpuViews = static_cast<GpuObservationView*>(viewBuffer.mapped);
    for (uint32_t view = 0; view < layers; ++view) {
        glm::mat4 world = scene.world(layout.views[view].object);
        gpuViews[view].cameraPosition = glm::vec4(glm::vec3(world[3]), 0.0F);
        gpuViews[view].lightDirection = glm::vec4(lightDirection, lightIntensity);
        gpuViews[view].lightColor = glm::vec4(lightColor, AMBIENT_INTENSITY);
    }

    imageBarrier(commandBuffer,
                 color.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    imageBarrier(commandBuffer,
                 depth.handle,
                 VK_IMAGE_ASPECT_DEPTH_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    VkDescriptorSet bindlessSet = bindless.set();
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, drawLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, drawPipeline);
    vkCmdBindIndexBuffer(commandBuffer, geometry.indexBuffer.handle, 0, VK_INDEX_TYPE_UINT32);

    ObservationPushConstants push{};
    push.vertices = geometry.vertexBuffer.address;
    push.instances = instances.address;
    push.skinnedVertices = skinnedVertices != 0 ? skinnedVertices : geometry.vertexBuffer.address;
    push.meshes = geometry.meshBuffer.address;
    push.materials = geometry.materialBuffer.address;
    push.views = viewBuffer.address;

    VkRect2D scissor{{0, 0}, {OBSERVATION_SIZE, OBSERVATION_SIZE}};
    VkViewport viewport{
        0.0F, 0.0F, static_cast<float>(OBSERVATION_SIZE), static_cast<float>(OBSERVATION_SIZE), 0.0F, 1.0F};

    for (uint32_t view = 0; view < layers; ++view) {
        const ObservationLayout::View& item = layout.views[view];
        glm::mat4 world = scene.world(item.object);
        glm::vec3 position = glm::vec3(world[3]);
        glm::vec3 forward = -glm::vec3(world[2]);
        if (glm::dot(forward, forward) < 1.0e-8F) {
            forward = glm::vec3(0.0F, 0.0F, -1.0F);
        }
        glm::vec3 up = glm::vec3(world[1]);
        if (glm::dot(up, up) < 1.0e-8F) {
            up = glm::vec3(0.0F, 1.0F, 0.0F);
        }
        glm::mat4 viewMatrix = glm::lookAt(position, position + glm::normalize(forward), glm::normalize(up));
        // 장면 카메라와 **같은 투영**이다(무한 원거리 역 Z). 관측만 다른 규약을 쓰면 깊이 비교 방향이
        // 어긋나 아무 것도 안 그려진다.
        float focal = 1.0F / std::tan(glm::radians(item.fovYDegrees) * 0.5F);
        glm::mat4 projection(0.0F);
        projection[0][0] = focal;
        projection[1][1] = -focal;
        projection[2][3] = -1.0F;
        projection[3][2] = item.nearPlane;
        push.viewProjection = projection * viewMatrix;
        push.view = view;

        VkRenderingAttachmentInfo colorInfo = colorAttachment(
            colorLayerViews[view], VK_ATTACHMENT_LOAD_OP_CLEAR, VkClearColorValue{{0.0F, 0.0F, 0.0F, 1.0F}});
        VkRenderingAttachmentInfo depthInfo{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        depthInfo.imageView = depthLayerViews[view];
        depthInfo.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        depthInfo.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthInfo.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthInfo.clearValue.depthStencil.depth = 0.0F;

        VkRenderingInfo rendering{VK_STRUCTURE_TYPE_RENDERING_INFO};
        rendering.renderArea = scissor;
        rendering.layerCount = 1;
        rendering.colorAttachmentCount = 1;
        rendering.pColorAttachments = &colorInfo;
        rendering.pDepthAttachment = &depthInfo;
        vkCmdBeginRendering(commandBuffer, &rendering);
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        vkCmdPushConstants(commandBuffer,
                           drawLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           sizeof(push),
                           &push);
        for (uint32_t object : drawObjects) {
            const GpuMesh& mesh = geometry.mesh(scene.meshOf(object));
            // **LOD 0 만 그린다.** GpuMesh::indexCount 는 0단계가 아니라 **모든 단계를 이어 붙인 합**이라
            // 그대로 그리면 원본과 단순화 단계가 같은 자리에 겹쳐 그려진다. 역 Z 깊이 검사 덕에 얼추 맞아
            // 보이지만 실루엣에서 거친 단계가 삐져나오고 곡면에서 z 파이팅이 난다 — 시점에 따라 깜빡이는
            // 관측은 이 경로가 피하려던 바로 그 문제다. 주 래스터 경로도 lod.* 로 그린다.
            const GpuMeshLod& lod = geometry.lod(mesh.lodOffset);
            vkCmdDrawIndexed(commandBuffer, lod.indexCount, 1, lod.indexOffset, mesh.vertexOffset, object);
        }
        vkCmdEndRendering(commandBuffer);
    }

    imageBarrier(commandBuffer,
                 color.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);

    ObservationEncodePushConstants encodePush{};
    encodePush.history = history.address;
    encodePush.features = features.address;
    encodePush.colorSlot = colorBindlessSlot;
    encodePush.views = layers;
    encodePush.slot = static_cast<uint32_t>(step % OBSERVATION_STACK);
    encodePush.reset = reset ? 1U : 0U;
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, encodeLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, encodePipeline);
    vkCmdPushConstants(commandBuffer, encodeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(encodePush), &encodePush);
    uint32_t groups = (OBSERVATION_SIZE + ENCODE_GROUP - 1) / ENCODE_GROUP;
    vkCmdDispatch(commandBuffer, groups, groups, layers);
    memoryBarrier(commandBuffer,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT);
    ++step;
}

void ObservationRenderer::recordDownload(VkCommandBuffer commandBuffer) {
    if (!ready || layers == 0) {
        return;
    }
    VkBufferCopy region{};
    region.size = static_cast<VkDeviceSize>(featureFloatCount()) * sizeof(float);
    vkCmdCopyBuffer(commandBuffer, features.handle, featureReadback.handle, 1, &region);
}

} // namespace gfx
