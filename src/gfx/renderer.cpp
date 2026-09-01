#include "gfx/renderer.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/mat4x4.hpp>
#include <glm/trigonometric.hpp>
#include <glm/vec4.hpp>
#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>
#include <stb_image_write.h>

#include "asset/model.h"
#include "core/error.h"
#include "gfx/bindless.h"
#include "gfx/context.h"
#include "gfx/geometry.h"
#include "gfx/raytracing.h"
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
// 그림자 시점의 근평면. 광원에 아주 가까운 물체는 그림자를 만들지 못한다.
constexpr float SHADOW_NEAR_PLANE = 0.05F;
constexpr VkFormat SHADOW_FORMAT = VK_FORMAT_D32_SFLOAT;
constexpr VkFormat SSAO_FORMAT = VK_FORMAT_R32_SFLOAT;
constexpr size_t TRANSLUCENT_MODE = 2;
// shaders/meshlet_task.glsl 의 MESHLET_GROUP_SIZE 와 같아야 한다.
constexpr uint32_t MESHLET_GROUP_SIZE = 32;

// shaders/scene_data.glsl 의 CameraBuffer 와 배치가 같아야 한다.
struct GpuCamera {
    glm::mat4 viewProjection;
    glm::vec4 position;
    glm::vec4 parameters; // x: 근평면
    // 균일 환경광. w 는 쓰지 않는다.
    glm::vec4 ambient;
    // xy 렌더 해상도, zw 그 역수.
    glm::vec4 viewport;
    // 깊이에서 월드 위치를 되돌릴 때 쓴다.
    glm::mat4 inverseViewProjection;
    // x: 조명 수, y: 그림자 아틀라스 슬롯, z: SSAO 슬롯, w: 아틀라스 한 변의 타일 수.
    // bindless 슬롯은 상위 8비트가 샘플러 번호라 float 로는 정확히 담기지 않는다.
    glm::uvec4 shading;
    // x: 조도 큐브 슬롯, y: 프리필터 큐브 슬롯, z: BRDF 표 슬롯, w: 프리필터 밉 수(0 이면 IBL 꺼짐).
    glm::uvec4 environment;
};

// shaders/scene_data.glsl 의 MeshletGroup 과 배치가 같아야 한다.
struct GpuMeshletGroup {
    uint32_t instanceIndex;
    uint32_t firstMeshlet;
    uint32_t meshletCount;
    uint32_t padding;
};

struct HzbPushConstants {
    uint32_t sourceTexture;
    uint32_t destinationStorage;
    int32_t sourceSize[2];
    int32_t destinationSize[2];
};

constexpr VkFormat HZB_FORMAT = VK_FORMAT_R32_SFLOAT;
constexpr VkFormat ACCUMULATION_FORMAT = VK_FORMAT_R32G32B32A32_SFLOAT;

struct CullPushConstants {
    VkDeviceAddress instances;
    VkDeviceAddress meshes;
    VkDeviceAddress meshlets;
    VkDeviceAddress camera;
    VkDeviceAddress drawCommands;
    VkDeviceAddress drawCounts;
    uint32_t instanceCount;
    uint32_t flags;
    uint32_t hzbTexture;
    float hzbMaxLevel;
    float hzbSize[2];
    VkDeviceAddress network;
};

struct SkinPushConstants {
    VkDeviceAddress source;
    VkDeviceAddress destination;
    VkDeviceAddress joints;
    uint32_t sourceOffset;
    uint32_t destinationOffset;
    uint32_t jointOffset;
    uint32_t vertexCount;
};

// skin.comp 의 local_size_x 와 같아야 한다.
constexpr uint32_t SKIN_GROUP_SIZE = 64;

constexpr uint32_t CULL_FLAG_FRUSTUM = 1;
constexpr uint32_t CULL_FLAG_CONE = 2;
constexpr uint32_t CULL_FLAG_OCCLUSION = 4;
constexpr uint32_t CULL_FLAG_NEURAL_LOD = 8;
// 학습 표본 수. 이보다 많으면 일정 간격으로 건너뛰며 뽑는다.
constexpr uint32_t MAX_LOD_SAMPLES = 1024;
constexpr uint32_t BUCKET_COUNT = ALPHA_MODE_COUNT * 2;

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
    VkDeviceAddress joints;
    VkDeviceAddress lights;
    VkDeviceAddress shadowMatrices;
    uint32_t meshletGroupBase;
    uint32_t debugMode;
};

// shaders/depth_only.vert 의 DepthPushConstants 와 배치가 같아야 한다.
struct DepthPushConstants {
    glm::mat4 viewProjection;
    VkDeviceAddress vertices;
    VkDeviceAddress instances;
    VkDeviceAddress joints;
    VkDeviceAddress meshes;
    VkDeviceAddress materials;
};

struct SsaoPushConstants {
    VkDeviceAddress camera;
    uint32_t depthTexture;
    uint32_t occlusionStorage;
    int32_t size[2];
    float radius;
    float intensity;
    float bias;
    uint32_t sampleCount;
};

struct SsaoBlurPushConstants {
    uint32_t sourceTexture;
    uint32_t destinationStorage;
    int32_t size[2];
};

struct CompositePushConstants {
    uint32_t accumulationTexture;
    uint32_t revealageTexture;
};

struct TonemapPushConstants {
    VkDeviceAddress camera;
    uint32_t colorTexture;
    float exposure;
    uint32_t sampleCount;
    uint32_t depthTexture;
    uint32_t environmentCube;
};

struct UpscalePushConstants {
    uint32_t sourceTexture;
    float sharpness;
    float sourceSize[2];
    float destinationSize[2];
};

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
    : context(context), geometry(geometry), bindless(bindless), frameProfiler(context, FRAMES_IN_FLIGHT) {
    swapchain = std::make_unique<Swapchain>(context, window, vsync);
    currentDisplayExtent = swapchain->extent;
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
    // 광선 질의 그림자 파이프라인이 TLAS 디스크립터 배치를 필요로 하므로 먼저 만든다.
    if (context.caps.rayTracingPipeline) {
        rayTracer = std::make_unique<RayTracer>(context, geometry, bindless);
        rayTracer->buildBottomLevel();
    }
    createMeshPipelines();
    createPostPipelines();
    createCullPipeline();
    createSkinPipeline();
    createShadowPipeline();
    createSsaoPipelines();
    environment = std::make_unique<EnvironmentMap>(context, bindless);

    VkSemaphoreTypeCreateInfo timelineInfo{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphoreInfo.pNext = &timelineInfo;
    VK_CHECK(vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &frameTimeline));
}

Renderer::~Renderer() {
    waitIdle();
    for (VkPipeline pipeline : upscalePipelines) {
        vkDestroyPipeline(context.device, pipeline, nullptr);
    }
    vkDestroyPipeline(context.device, tonemapPipeline, nullptr);
    vkDestroyPipeline(context.device, compositePipeline, nullptr);
    vkDestroyPipelineLayout(context.device, postPipelineLayout, nullptr);
    for (VkPipeline pipeline : meshShaderPipelines) {
        vkDestroyPipeline(context.device, pipeline, nullptr);
    }
    for (VkPipeline pipeline : meshRayQueryPipelines) {
        vkDestroyPipeline(context.device, pipeline, nullptr);
    }
    for (VkPipeline pipeline : meshShaderRayQueryPipelines) {
        vkDestroyPipeline(context.device, pipeline, nullptr);
    }
    vkDestroyPipelineLayout(context.device, meshRayQueryPipelineLayout, nullptr);
    vkDestroyPipeline(context.device, wireframePipeline, nullptr);
    for (VkPipeline pipeline : meshPipelines) {
        vkDestroyPipeline(context.device, pipeline, nullptr);
    }
    vkDestroyPipeline(context.device, hzbPipeline, nullptr);
    vkDestroyPipelineLayout(context.device, hzbPipelineLayout, nullptr);
    vkDestroyPipeline(context.device, cullPipeline, nullptr);
    vkDestroyPipeline(context.device, skinPipeline, nullptr);
    vkDestroyPipelineLayout(context.device, skinPipelineLayout, nullptr);
    destroyBuffer(context, skinnedVertexBuffer);
    vkDestroyPipeline(context.device, ssaoBlurPipeline, nullptr);
    vkDestroyPipelineLayout(context.device, ssaoBlurPipelineLayout, nullptr);
    vkDestroyPipeline(context.device, ssaoPipeline, nullptr);
    vkDestroyPipelineLayout(context.device, ssaoPipelineLayout, nullptr);
    vkDestroyPipeline(context.device, shadowCutoffPipeline, nullptr);
    vkDestroyPipeline(context.device, shadowPipeline, nullptr);
    vkDestroyPipelineLayout(context.device, depthPipelineLayout, nullptr);
    vkDestroyPipelineLayout(context.device, cullPipelineLayout, nullptr);
    vkDestroyPipelineLayout(context.device, meshPipelineLayout, nullptr);
    vkDestroySemaphore(context.device, frameTimeline, nullptr);
    destroyPresentSemaphores();
    for (Frame& frame : frames) {
        destroyBuffer(context, frame.shadowDrawBuffer);
        destroyBuffer(context, frame.shadowMatrixBuffer);
        destroyBuffer(context, frame.lightBuffer);
        destroyBuffer(context, frame.jointBuffer);
        destroyBuffer(context, frame.lodNetworkBuffer);
        destroyBuffer(context, frame.drawCountBuffer);
        destroyBuffer(context, frame.meshletDrawBuffer);
        destroyBuffer(context, frame.meshTaskIndirectBuffer);
        destroyBuffer(context, frame.meshletGroupBuffer);
        destroyBuffer(context, frame.drawBuffer);
        destroyBuffer(context, frame.instanceBuffer);
        destroyBuffer(context, frame.cameraBuffer);
        vkDestroySemaphore(context.device, frame.imageAvailable, nullptr);
        vkDestroyCommandPool(context.device, frame.commandPool, nullptr);
    }
    destroyBuffer(context, captureBuffer);
    for (VkImageView view : targets.hzbMipViews) {
        vkDestroyImageView(context.device, view, nullptr);
    }
    destroyImage(context, targets.hzb);
    for (VkImageView view : targets.shadowLayerViews) {
        vkDestroyImageView(context.device, view, nullptr);
    }
    destroyImage(context, targets.shadowAtlas);
    destroyImage(context, targets.ssao);
    destroyImage(context, targets.ssaoRaw);
    destroyImage(context, targets.pathAccumulation);
    destroyImage(context, targets.tonemapped);
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

void Renderer::updateRenderExtent() {
    float scale = std::clamp(renderScale, 0.25F, 2.0F);
    VkExtent2D scaled{std::max(static_cast<uint32_t>(static_cast<float>(currentDisplayExtent.width) * scale), 1U),
                      std::max(static_cast<uint32_t>(static_cast<float>(currentDisplayExtent.height) * scale), 1U)};
    if (scaled.width == currentRenderExtent.width && scaled.height == currentRenderExtent.height) {
        return;
    }
    waitIdle();
    currentRenderExtent = scaled;
    createRenderTargets();
}

void Renderer::setDisplayExtent(VkExtent2D extent) {
    extent.width = std::max(extent.width, 1U);
    extent.height = std::max(extent.height, 1U);
    if (extent.width != currentDisplayExtent.width || extent.height != currentDisplayExtent.height) {
        waitIdle();
        currentDisplayExtent = extent;
        currentRenderExtent = {};
    }
    updateRenderExtent();
}

std::vector<UpscalerInfo> Renderer::upscalers() const {
    bool isNvidia = context.properties.vendorID == 0x10DE;
    return {
        {Upscaler::NONE, "없음 (통과)", true, ""},
        {Upscaler::SPATIAL, "내장 공간 업스케일", true, ""},
        {Upscaler::FSR, "FSR", false, "FidelityFX SDK 미포함"},
        {Upscaler::DLSS, "DLSS", false, isNvidia ? "NGX SDK 미포함" : "NVIDIA 장치 아님"},
    };
}

std::vector<Renderer::TargetView> Renderer::targetViews() const {
    constexpr VkImageLayout READ_ONLY = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    std::vector<TargetView> views{{"색상 (HDR)", targets.color.view, READ_ONLY},
                                  {"그림자 아틀라스", targets.shadowAtlas.view, READ_ONLY},
                                  {"SSAO", targets.ssao.view, VK_IMAGE_LAYOUT_GENERAL},
                                  {"톤 매핑", targets.tonemapped.view, READ_ONLY},
                                  {"표시 (업스케일)", targets.present.view, READ_ONLY},
                                  {"깊이", targets.depth.view, READ_ONLY},
                                  // HZB 는 컴퓨트가 쓰고 읽으므로 계속 GENERAL 이다.
                                  {"HZB", targets.hzb.view, VK_IMAGE_LAYOUT_GENERAL}};
    if (oitTargetsValid) {
        views.push_back({"OIT 누적", targets.oitAccumulation.view, READ_ONLY});
        views.push_back({"OIT 잔여 투과율", targets.oitRevealage.view, READ_ONLY});
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
    destroyImage(context, targets.tonemapped);
    destroyImage(context, targets.present);
    destroyImage(context, targets.oitRevealage);
    destroyImage(context, targets.oitAccumulation);
    destroyImage(context, targets.depth);
    destroyImage(context, targets.color);

    VkExtent3D extent{currentRenderExtent.width, currentRenderExtent.height, 1};

    ImageDesc colorDesc;
    colorDesc.extent = extent;
    colorDesc.format = COLOR_FORMAT;
    colorDesc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
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

    destroyImage(context, targets.pathAccumulation);
    ImageDesc pathAccumulationDesc = colorDesc;
    pathAccumulationDesc.format = ACCUMULATION_FORMAT;
    pathAccumulationDesc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    targets.pathAccumulation = createImage(context, pathAccumulationDesc, "경로 추적 누적");

    destroyImage(context, targets.tonemapped);
    ImageDesc tonemappedDesc;
    tonemappedDesc.extent = extent;
    tonemappedDesc.format = PRESENT_FORMAT;
    tonemappedDesc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    targets.tonemapped = createImage(context, tonemappedDesc, "톤 매핑");

    ImageDesc presentDesc = tonemappedDesc;
    presentDesc.extent = {currentDisplayExtent.width, currentDisplayExtent.height, 1};
    targets.present = createImage(context, presentDesc, "표시");

    for (VkImageView view : targets.hzbMipViews) {
        vkDestroyImageView(context.device, view, nullptr);
    }
    targets.hzbMipViews.clear();
    destroyImage(context, targets.hzb);

    targets.hzbExtent = {std::max(currentRenderExtent.width / 2, 1U), std::max(currentRenderExtent.height / 2, 1U)};
    uint32_t hzbLevels =
        1U + static_cast<uint32_t>(std::floor(std::log2(std::max(targets.hzbExtent.width, targets.hzbExtent.height))));

    ImageDesc hzbDesc;
    hzbDesc.extent = {targets.hzbExtent.width, targets.hzbExtent.height, 1};
    hzbDesc.format = HZB_FORMAT;
    hzbDesc.mipLevels = hzbLevels;
    hzbDesc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    targets.hzb = createImage(context, hzbDesc, "HZB");

    // 그림자 맵은 화면 크기와 무관하므로 한 번만 만든다. 층 하나가 시점 하나다.
    if (targets.shadowAtlas.handle == VK_NULL_HANDLE) {
        ImageDesc shadowDesc;
        shadowDesc.extent = {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE, 1};
        shadowDesc.format = SHADOW_FORMAT;
        shadowDesc.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        shadowDesc.aspect = VK_IMAGE_ASPECT_DEPTH_BIT;
        shadowDesc.arrayLayers = MAX_SHADOW_VIEWS;
        shadowDesc.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        targets.shadowAtlas = createImage(context, shadowDesc, "그림자 맵");

        targets.shadowLayerViews.resize(MAX_SHADOW_VIEWS);
        for (uint32_t layer = 0; layer < MAX_SHADOW_VIEWS; ++layer) {
            VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
            viewInfo.image = targets.shadowAtlas.handle;
            viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            viewInfo.format = SHADOW_FORMAT;
            viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;
            viewInfo.subresourceRange.levelCount = 1;
            viewInfo.subresourceRange.baseArrayLayer = layer;
            viewInfo.subresourceRange.layerCount = 1;
            VK_CHECK(vkCreateImageView(context.device, &viewInfo, nullptr, &targets.shadowLayerViews[layer]));
        }
    }

    destroyImage(context, targets.ssao);
    destroyImage(context, targets.ssaoRaw);
    targets.ssaoExtent = {std::max(currentRenderExtent.width / 2, 1U), std::max(currentRenderExtent.height / 2, 1U)};
    ImageDesc ssaoDesc;
    ssaoDesc.extent = {targets.ssaoExtent.width, targets.ssaoExtent.height, 1};
    ssaoDesc.format = SSAO_FORMAT;
    ssaoDesc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    targets.ssaoRaw = createImage(context, ssaoDesc, "SSAO 원본");
    targets.ssao = createImage(context, ssaoDesc, "SSAO");

    targets.hzbMipViews.resize(hzbLevels);
    for (uint32_t level = 0; level < hzbLevels; ++level) {
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = targets.hzb.handle;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = HZB_FORMAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = level;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(context.device, &viewInfo, nullptr, &targets.hzbMipViews[level]));
    }

    ++generation;

    if (!targets.slotsAllocated) {
        targets.colorSlot = bindless.add(targets.color.view, postSampler);
        targets.accumulationSlot = bindless.add(targets.oitAccumulation.view, postSampler);
        targets.revealageSlot = bindless.add(targets.oitRevealage.view, postSampler);
        targets.depthSlot = bindless.add(targets.depth.view, postSampler);
        targets.tonemappedSlot = bindless.add(targets.tonemapped.view, postSampler);
        // HZB 는 컴퓨트가 쓰고 읽으므로 계속 GENERAL 레이아웃에 둔다.
        targets.hzbSampledSlot = bindless.add(targets.hzb.view, postSampler, VK_IMAGE_LAYOUT_GENERAL);
        targets.pathAccumulationStorageSlot = bindless.addStorageImageRgba(targets.pathAccumulation.view);
        targets.pathAccumulationSampledSlot =
            bindless.add(targets.pathAccumulation.view, postSampler, VK_IMAGE_LAYOUT_GENERAL);
        targets.shadowAtlasSlot = bindless.addArray(targets.shadowAtlas.view, postSampler);
        // SSAO 도 컴퓨트가 쓰고 프래그먼트가 읽으므로 계속 GENERAL 에 둔다.
        targets.ssaoRawSlot = bindless.add(targets.ssaoRaw.view, postSampler, VK_IMAGE_LAYOUT_GENERAL);
        targets.ssaoSlot = bindless.add(targets.ssao.view, postSampler, VK_IMAGE_LAYOUT_GENERAL);
        targets.ssaoRawStorageSlot = bindless.addStorageImage(targets.ssaoRaw.view);
        targets.ssaoStorageSlot = bindless.addStorageImage(targets.ssao.view);
        targets.hzbStorageSlots.resize(targets.hzbMipViews.size());
        for (size_t level = 0; level < targets.hzbMipViews.size(); ++level) {
            targets.hzbStorageSlots[level] = bindless.addStorageImage(targets.hzbMipViews[level]);
        }
        targets.slotsAllocated = true;
    } else {
        bindless.update(targets.colorSlot, targets.color.view, postSampler);
        bindless.update(targets.accumulationSlot, targets.oitAccumulation.view, postSampler);
        bindless.update(targets.revealageSlot, targets.oitRevealage.view, postSampler);
        bindless.update(targets.depthSlot, targets.depth.view, postSampler);
        bindless.update(targets.tonemappedSlot, targets.tonemapped.view, postSampler);
        bindless.update(targets.hzbSampledSlot, targets.hzb.view, postSampler, VK_IMAGE_LAYOUT_GENERAL);
        bindless.updateStorageImageRgba(targets.pathAccumulationStorageSlot, targets.pathAccumulation.view);
        bindless.update(
            targets.pathAccumulationSampledSlot, targets.pathAccumulation.view, postSampler, VK_IMAGE_LAYOUT_GENERAL);
        bindless.update(targets.ssaoRawSlot, targets.ssaoRaw.view, postSampler, VK_IMAGE_LAYOUT_GENERAL);
        bindless.update(targets.ssaoSlot, targets.ssao.view, postSampler, VK_IMAGE_LAYOUT_GENERAL);
        bindless.updateStorageImage(targets.ssaoRawStorageSlot, targets.ssaoRaw.view);
        bindless.updateStorageImage(targets.ssaoStorageSlot, targets.ssao.view);
        // 밉 수가 줄어들면 남는 슬롯은 마지막 뷰로 채워 유효한 상태를 유지한다.
        for (size_t level = 0; level < targets.hzbStorageSlots.size(); ++level) {
            size_t source = std::min(level, targets.hzbMipViews.size() - 1);
            bindless.updateStorageImage(targets.hzbStorageSlots[level], targets.hzbMipViews[source]);
        }
    }
    hzbNeedsClear = true;
    ssaoNeedsClear = true;
    pathSampleCount = 0;
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
        reserveMeshletDraws(frame, MINIMUM_INSTANCE_CAPACITY);
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

void Renderer::reserveJoints(Frame& frame, uint32_t jointCount) {
    // 스킨이 없는 장면에서도 셰이더가 주소를 읽으므로 최소 하나는 잡아 둔다.
    uint32_t needed = std::max(jointCount, 1U);
    if (needed <= frame.jointCapacity) {
        return;
    }
    uint32_t capacity = std::max(needed, frame.jointCapacity * 2);
    destroyBuffer(context, frame.jointBuffer);
    frame.jointBuffer = createBuffer(context,
                                     static_cast<VkDeviceSize>(capacity) * sizeof(glm::mat4),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     MemoryLocation::HOST_WRITE,
                                     "조인트 행렬");
    frame.jointCapacity = capacity;
}

void Renderer::reserveLights(Frame& frame, uint32_t lightCount) {
    if (frame.shadowMatrixBuffer.handle == VK_NULL_HANDLE) {
        frame.shadowMatrixBuffer = createBuffer(context,
                                                sizeof(glm::mat4) * MAX_SHADOW_VIEWS,
                                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                MemoryLocation::HOST_WRITE,
                                                "그림자 시점 행렬");
    }
    // 조명이 없는 장면에서도 셰이더가 주소를 읽으므로 최소 하나는 잡아 둔다.
    uint32_t needed = std::max(lightCount, 1U);
    if (needed <= frame.lightCapacity) {
        return;
    }
    uint32_t capacity = std::max(needed, frame.lightCapacity * 2);
    destroyBuffer(context, frame.lightBuffer);
    frame.lightBuffer = createBuffer(context,
                                     static_cast<VkDeviceSize>(capacity) * sizeof(GpuLight),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     MemoryLocation::HOST_WRITE,
                                     "조명");
    frame.lightCapacity = capacity;
}

void Renderer::reserveShadowDraws(Frame& frame, uint32_t drawCount) {
    uint32_t needed = std::max(drawCount, 1U);
    if (needed <= frame.shadowDrawCapacity) {
        return;
    }
    uint32_t capacity = std::max(needed, frame.shadowDrawCapacity * 2);
    destroyBuffer(context, frame.shadowDrawBuffer);
    frame.shadowDrawBuffer = createBuffer(context,
                                          static_cast<VkDeviceSize>(capacity) * sizeof(VkDrawIndexedIndirectCommand),
                                          VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                                          MemoryLocation::HOST_WRITE,
                                          "그림자 그리기 명령");
    frame.shadowDrawCapacity = capacity;
}

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

    uint32_t casterCount = 0;
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

    const auto* source = static_cast<const VkDrawIndexedIndirectCommand*>(frame.drawBuffer.mapped);
    auto* destination = static_cast<VkDrawIndexedIndirectCommand*>(frame.shadowDrawBuffer.mapped);
    uint32_t cursor = 0;

    for (size_t view = 0; view < shadowViews.size(); ++view) {
        const ShadowView& shadowView = shadowViews[view];
        std::array<glm::vec4, MAX_FRUSTUM_PLANES> planes{};
        uint32_t planeCount = extractFrustumPlanes(shadowView.viewProjection, planes, true);

        for (size_t mode = 0; mode < TRANSLUCENT_MODE; ++mode) {
            DrawBatch& batch = shadowBatches[view * TRANSLUCENT_MODE + mode];
            batch.first = cursor;
            uint32_t first = batches.draws[mode][0].first;
            uint32_t count = batches.draws[mode][0].count + batches.draws[mode][1].count;

            for (uint32_t i = 0; i < count; ++i) {
                uint32_t slot = first + i;
                const glm::vec4& sphere = instanceBounds[slot];
                // 광원 절두체 밖이면 이 시점에 아무것도 남기지 않는다.
                if (shadowViewCulling && !sphereInFrustum(planes, planeCount, glm::vec3(sphere), sphere.w)) {
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
                destination[cursor++] = source[slot];
            }
            batch.count = cursor - batch.first;
        }
    }
    shadowDrawsIssued = cursor;
}

void Renderer::reserveMeshletDraws(Frame& frame, uint32_t drawCount) {
    if (frame.lodNetworkBuffer.handle == VK_NULL_HANDLE) {
        frame.lodNetworkBuffer = createBuffer(context,
                                              sizeof(GpuLodNetwork),
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                              MemoryLocation::HOST_WRITE,
                                              "LOD 신경망 가중치");
    }
    if (frame.drawCountBuffer.handle == VK_NULL_HANDLE) {
        frame.drawCountBuffer = createBuffer(context,
                                             sizeof(uint32_t) * BUCKET_COUNT,
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                             MemoryLocation::DEVICE,
                                             "그리기 개수");
    }
    if (drawCount <= frame.meshletDrawCapacity) {
        return;
    }
    uint32_t capacity = std::max(drawCount, std::max(frame.meshletDrawCapacity * 2, MINIMUM_INSTANCE_CAPACITY));
    destroyBuffer(context, frame.meshletDrawBuffer);
    frame.meshletDrawBuffer = createBuffer(context,
                                           static_cast<VkDeviceSize>(capacity) * sizeof(VkDrawIndexedIndirectCommand),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           MemoryLocation::DEVICE,
                                           "meshlet 그리기 명령");
    frame.meshletDrawCapacity = capacity;
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

void Renderer::createSsaoPipelines() {
    VkDescriptorSetLayout bindlessLayout = bindless.layout();

    VkPushConstantRange ssaoRange{};
    ssaoRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    ssaoRange.size = sizeof(SsaoPushConstants);
    VkPipelineLayoutCreateInfo ssaoLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    ssaoLayoutInfo.setLayoutCount = 1;
    ssaoLayoutInfo.pSetLayouts = &bindlessLayout;
    ssaoLayoutInfo.pushConstantRangeCount = 1;
    ssaoLayoutInfo.pPushConstantRanges = &ssaoRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &ssaoLayoutInfo, nullptr, &ssaoPipelineLayout));

    VkShaderModule ssaoModule = createShaderModule(context.device, "ssao.comp.spv");
    VkComputePipelineCreateInfo ssaoInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ssaoInfo.stage = shaderStage(VK_SHADER_STAGE_COMPUTE_BIT, ssaoModule);
    ssaoInfo.layout = ssaoPipelineLayout;
    VK_CHECK(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &ssaoInfo, nullptr, &ssaoPipeline));
    vkDestroyShaderModule(context.device, ssaoModule, nullptr);

    VkPushConstantRange blurRange{};
    blurRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    blurRange.size = sizeof(SsaoBlurPushConstants);
    VkPipelineLayoutCreateInfo blurLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    blurLayoutInfo.setLayoutCount = 1;
    blurLayoutInfo.pSetLayouts = &bindlessLayout;
    blurLayoutInfo.pushConstantRangeCount = 1;
    blurLayoutInfo.pPushConstantRanges = &blurRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &blurLayoutInfo, nullptr, &ssaoBlurPipelineLayout));

    VkShaderModule blurModule = createShaderModule(context.device, "ssao_blur.comp.spv");
    VkComputePipelineCreateInfo blurInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    blurInfo.stage = shaderStage(VK_SHADER_STAGE_COMPUTE_BIT, blurModule);
    blurInfo.layout = ssaoBlurPipelineLayout;
    VK_CHECK(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &blurInfo, nullptr, &ssaoBlurPipeline));
    vkDestroyShaderModule(context.device, blurModule, nullptr);
}

void Renderer::createSkinPipeline() {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.size = sizeof(SkinPushConstants);

    VkDescriptorSetLayout skinBindlessLayout = bindless.layout();
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &skinBindlessLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &skinPipelineLayout));

    VkShaderModule module = createShaderModule(context.device, "skin.comp.spv");
    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = shaderStage(VK_SHADER_STAGE_COMPUTE_BIT, module);
    pipelineInfo.layout = skinPipelineLayout;
    VK_CHECK(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &skinPipeline));
    vkDestroyShaderModule(context.device, module, nullptr);
}

void Renderer::createCullPipeline() {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.size = sizeof(CullPushConstants);

    VkDescriptorSetLayout cullBindlessLayout = bindless.layout();
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &cullBindlessLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstantRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &cullPipelineLayout));

    VkShaderModule module = createShaderModule(context.device, "cull_meshlets.comp.spv");
    VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pipelineInfo.stage = shaderStage(VK_SHADER_STAGE_COMPUTE_BIT, module);
    pipelineInfo.layout = cullPipelineLayout;
    VK_CHECK(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &cullPipeline));
    vkDestroyShaderModule(context.device, module, nullptr);

    if (context.caps.drawIndirectCount) {
        drawIndexedIndirectCount = reinterpret_cast<PFN_vkCmdDrawIndexedIndirectCount>(
            vkGetDeviceProcAddr(context.device, "vkCmdDrawIndexedIndirectCount"));
    }
    VkPushConstantRange hzbRange{};
    hzbRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    hzbRange.size = sizeof(HzbPushConstants);

    VkDescriptorSetLayout bindlessLayout = bindless.layout();
    VkPipelineLayoutCreateInfo hzbLayoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    hzbLayoutInfo.setLayoutCount = 1;
    hzbLayoutInfo.pSetLayouts = &bindlessLayout;
    hzbLayoutInfo.pushConstantRangeCount = 1;
    hzbLayoutInfo.pPushConstantRanges = &hzbRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &hzbLayoutInfo, nullptr, &hzbPipelineLayout));

    VkShaderModule hzbModule = createShaderModule(context.device, "hzb_reduce.comp.spv");
    VkComputePipelineCreateInfo hzbPipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    hzbPipelineInfo.stage = shaderStage(VK_SHADER_STAGE_COMPUTE_BIT, hzbModule);
    hzbPipelineInfo.layout = hzbPipelineLayout;
    VK_CHECK(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &hzbPipelineInfo, nullptr, &hzbPipeline));
    vkDestroyShaderModule(context.device, hzbModule, nullptr);

    spdlog::info("컴퓨트 컬링 준비 완료 (압축 간접 그리기: {})", drawIndexedIndirectCount != nullptr);
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
    pushConstants.joints = frame.jointBuffer.address;
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

void Renderer::recordSsaoPass(VkCommandBuffer commandBuffer, const Frame& frame) {
    VkDescriptorSet bindlessSet = bindless.set();
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ssaoPipeline);
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ssaoPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);

    SsaoPushConstants pushConstants{};
    pushConstants.camera = frame.cameraBuffer.address;
    pushConstants.depthTexture = targets.depthSlot;
    pushConstants.occlusionStorage = targets.ssaoRawStorageSlot;
    pushConstants.size[0] = static_cast<int32_t>(targets.ssaoExtent.width);
    pushConstants.size[1] = static_cast<int32_t>(targets.ssaoExtent.height);
    // 반지름은 장면 크기에 대한 비율이라 여우든 헬멧이든 비슷하게 보인다.
    pushConstants.radius = ssaoRadius * sceneRadius;
    pushConstants.intensity = ssaoIntensity;
    pushConstants.bias = ssaoBias;
    pushConstants.sampleCount = ssaoSamples;
    vkCmdPushConstants(
        commandBuffer, ssaoPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, (targets.ssaoExtent.width + 7) / 8, (targets.ssaoExtent.height + 7) / 8, 1);

    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ssaoBlurPipeline);
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ssaoBlurPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    SsaoBlurPushConstants blurConstants{};
    blurConstants.sourceTexture = targets.ssaoRawSlot;
    blurConstants.destinationStorage = targets.ssaoStorageSlot;
    blurConstants.size[0] = pushConstants.size[0];
    blurConstants.size[1] = pushConstants.size[1];
    vkCmdPushConstants(
        commandBuffer, ssaoBlurPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(blurConstants), &blurConstants);
    vkCmdDispatch(commandBuffer, (targets.ssaoExtent.width + 7) / 8, (targets.ssaoExtent.height + 7) / 8, 1);
}

void Renderer::recordHzbPass(VkCommandBuffer commandBuffer) {
    VkDescriptorSet bindlessSet = bindless.set();
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, hzbPipeline);
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, hzbPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);

    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;

    uint32_t sourceWidth = currentRenderExtent.width;
    uint32_t sourceHeight = currentRenderExtent.height;
    for (size_t level = 0; level < targets.hzbStorageSlots.size(); ++level) {
        uint32_t destinationWidth = std::max(targets.hzbExtent.width >> level, 1U);
        uint32_t destinationHeight = std::max(targets.hzbExtent.height >> level, 1U);

        HzbPushConstants pushConstants{};
        // 0단계는 깊이 버퍼에서, 나머지는 바로 위 단계에서 줄인다.
        pushConstants.sourceTexture = level == 0 ? targets.depthSlot : targets.hzbSampledSlot;
        pushConstants.destinationStorage = targets.hzbStorageSlots[level];
        pushConstants.sourceSize[0] = static_cast<int32_t>(sourceWidth);
        pushConstants.sourceSize[1] = static_cast<int32_t>(sourceHeight);
        pushConstants.destinationSize[0] = static_cast<int32_t>(destinationWidth);
        pushConstants.destinationSize[1] = static_cast<int32_t>(destinationHeight);

        vkCmdPushConstants(
            commandBuffer, hzbPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(commandBuffer, (destinationWidth + 7) / 8, (destinationHeight + 7) / 8, 1);
        vkCmdPipelineBarrier2(commandBuffer, &dependency);

        sourceWidth = destinationWidth;
        sourceHeight = destinationHeight;
    }
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

// 광선 질의는 가속 구조와 함께여야 쓸모가 있다. 이 저장소는 TLAS 를 RayTracer 가 들고 있으므로
// 경로 추적 파이프라인이 있는 장치에서만 켤 수 있다.
//
// ponytail: 광선 질의는 있는데 경로 추적 파이프라인이 없는 장치도 규격상 가능하다. 그런 장치까지
// 받으려면 가속 구조 관리만 RayTracer 에서 떼어내야 한다.
bool Renderer::rayQueryShadowsAvailable() const {
    return context.caps.rayQuery && rayTracer != nullptr;
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

    // 알파 경로 세 벌. 광선 질의 변종은 프래그먼트 모듈과 배치만 다르므로 상태를 되돌리고 한 번 더 돈다.
    auto buildAlphaVariants =
        [&](VkShaderModule opaque, VkShaderModule oit, VkPipelineLayout layout, VkPipeline* target) {
            stages[1].module = opaque;
            depthStencil.depthWriteEnable = VK_TRUE;
            renderingInfo.colorAttachmentCount = 1;
            colorFormats[0] = COLOR_FORMAT;
            colorBlend.attachmentCount = 1;
            blendAttachments = {};
            blendAttachments[0].colorWriteMask = ALL_CHANNELS;
            pipelineInfo.layout = layout;
            for (uint32_t variant = 0; variant < ALPHA_MODE_COUNT; ++variant) {
                alphaVariant = variant;
                if (variant == TRANSLUCENT_MODE) {
                    // 반투명은 누적과 잔여 투과율 두 대상에 기록하고 깊이는 읽기만 한다.
                    stages[1].module = oit;
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

        // 프래그먼트 셰이더만 다르므로 광선 질의 변종도 같은 방식으로 한 벌 더 만든다.
        auto buildMeshVariants =
            [&](VkShaderModule opaque, VkShaderModule oit, VkPipelineLayout layout, VkPipeline* target) {
                pipelineInfo.layout = layout;
                for (uint32_t variant = 0; variant < ALPHA_MODE_COUNT; ++variant) {
                    alphaVariant = variant;
                    bool translucent = variant == static_cast<uint32_t>(asset::AlphaMode::TRANSLUCENT);
                    meshStages[2].module = translucent ? oit : opaque;
                    depthStencil.depthWriteEnable = translucent ? VK_FALSE : VK_TRUE;
                    renderingInfo.colorAttachmentCount = translucent ? 2 : 1;
                    colorFormats[0] = translucent ? OIT_ACCUMULATION_FORMAT : COLOR_FORMAT;
                    colorBlend.attachmentCount = translucent ? 2 : 1;
                    blendAttachments[0].blendEnable = translucent ? VK_TRUE : VK_FALSE;
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
        useMeshShader = true;
        spdlog::info("mesh shader 경로 사용 가능");
    }

    vkDestroyShaderModule(context.device, rayQueryOit, nullptr);
    vkDestroyShaderModule(context.device, rayQueryOpaque, nullptr);
    vkDestroyShaderModule(context.device, oitFragment, nullptr);
    vkDestroyShaderModule(context.device, opaqueFragment, nullptr);
    vkDestroyShaderModule(context.device, vertexModule, nullptr);
}

void Renderer::createPostPipelines() {
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    pushConstantRange.size = std::max(sizeof(TonemapPushConstants), sizeof(UpscalePushConstants));

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

namespace {
// 광원에서 direction 을 바라보는 시점 행렬. 방향이 위쪽과 나란하면 기준 축을 바꾼다.
glm::mat4 lookAlong(glm::vec3 eye, glm::vec3 direction) {
    glm::vec3 up = std::abs(direction.y) > 0.99F ? glm::vec3{0.0F, 0.0F, 1.0F} : glm::vec3{0.0F, 1.0F, 0.0F};
    return glm::lookAt(eye, eye + direction, up);
}

// shaders/shadow.glsl 의 cubeFaceIndex 와 같은 순서여야 한다.
constexpr std::array<glm::vec3, 6> CUBE_FACE_DIRECTIONS{glm::vec3{1.0F, 0.0F, 0.0F},
                                                        glm::vec3{-1.0F, 0.0F, 0.0F},
                                                        glm::vec3{0.0F, 1.0F, 0.0F},
                                                        glm::vec3{0.0F, -1.0F, 0.0F},
                                                        glm::vec3{0.0F, 0.0F, 1.0F},
                                                        glm::vec3{0.0F, 0.0F, -1.0F}};
} // namespace

void Renderer::buildLights(Frame& frame, const scene::Scene& scene) {
    frameLights.clear();
    shadowViews.clear();

    // 방향광의 그림자 절두체를 맞추려면 보이는 메쉬 전체의 세계 경계가 필요하다.
    glm::vec3 minimum{std::numeric_limits<float>::max()};
    glm::vec3 maximum{std::numeric_limits<float>::lowest()};
    bool hasBounds = false;
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        const scene::Object& object = scene.objects[index];
        if (object.meshIndex >= geometry.meshCount() || !scene.visibleCached(index)) {
            continue;
        }
        const glm::mat4& world = scene.world(index);
        glm::vec4 sphere = geometry.mesh(object.meshIndex).boundingSphere;
        glm::vec3 center = glm::vec3(world * glm::vec4{glm::vec3(sphere), 1.0F});
        float scale = std::sqrt(std::max({glm::dot(glm::vec3(world[0]), glm::vec3(world[0])),
                                          glm::dot(glm::vec3(world[1]), glm::vec3(world[1])),
                                          glm::dot(glm::vec3(world[2]), glm::vec3(world[2]))}));
        minimum = glm::min(minimum, center - sphere.w * scale);
        maximum = glm::max(maximum, center + sphere.w * scale);
        hasBounds = true;
    }
    glm::vec3 sceneCenter = hasBounds ? (minimum + maximum) * 0.5F : glm::vec3{0.0F};
    // 멤버에 담아 SSAO 반지름을 장면 크기에 맞추는 데도 쓴다.
    sceneRadius = hasBounds ? std::max(glm::length(maximum - minimum) * 0.5F, 1.0F) : 1.0F;

    bool sunAssigned = false;
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        const scene::Object& object = scene.objects[index];
        if (object.light < 0 || static_cast<size_t>(object.light) >= scene.lights.size() ||
            !scene.visibleCached(index)) {
            continue;
        }
        const scene::Light& source = scene.lights[static_cast<size_t>(object.light)];
        const glm::mat4& world = scene.world(index);
        glm::vec3 position = glm::vec3(world[3]);
        // glTF 와 Unity 처럼 -Z 를 앞으로 본다.
        glm::vec3 direction = glm::normalize(-glm::vec3(world[2]));

        GpuLight light{};
        light.positionRange = glm::vec4{position, source.range};
        light.directionIntensity = glm::vec4{direction, source.intensity};
        light.colorType = glm::vec4{source.color, static_cast<float>(source.type)};
        light.coneSize = glm::vec4{std::cos(glm::radians(source.innerConeDegrees)),
                                   std::cos(glm::radians(source.outerConeDegrees)),
                                   source.size.x * 0.5F,
                                   source.size.y * 0.5F};
        light.rightShadow = glm::vec4{glm::normalize(glm::vec3(world[0])), -1.0F};
        light.up = glm::vec4{glm::normalize(glm::vec3(world[1])), 0.0F};

        // 하늘의 태양과 그림자 방향이 어긋나면 곧바로 눈에 띈다. 첫 방향광을 따라간다.
        if (source.type == scene::LightType::DIRECTIONAL && !sunAssigned) {
            sunDirection = direction;
            sunAssigned = true;
        }

        // 영역광은 반구 전체로 빛을 내보내 시점 하나로 담을 수 없어 그림자를 만들지 않는다.
        uint32_t viewsNeeded = 0;
        if (shadowsEnabled && source.castsShadow) {
            switch (source.type) {
            case scene::LightType::DIRECTIONAL:
            case scene::LightType::SPOT:
                viewsNeeded = 1;
                break;
            case scene::LightType::POINT:
                viewsNeeded = 6;
                break;
            default:
                break;
            }
        }
        // 방향광은 캐스케이드 수만큼 층을 쓴다. 층이 모자라면 버리지 말고 캐스케이드를 줄인다.
        if (source.type == scene::LightType::DIRECTIONAL && viewsNeeded > 0) {
            viewsNeeded = std::min(std::clamp(shadowCascades, 1U, MAX_SHADOW_CASCADES),
                                   static_cast<uint32_t>(MAX_SHADOW_VIEWS - shadowViews.size()));
        }

        if (viewsNeeded > 0 && shadowViews.size() + viewsNeeded <= MAX_SHADOW_VIEWS) {
            light.rightShadow.w = static_cast<float>(shadowViews.size());
            light.up.w = static_cast<float>(viewsNeeded);
            ShadowView view{};
            view.origin = position;
            view.sweepDirection = direction;
            if (source.type == scene::LightType::DIRECTIONAL) {
                view.directional = true;
                // 평행이동 없는 회전만의 광 시점. 광원을 카메라 쪽으로 옮겨 가며 스냅하면
                // 격자가 카메라를 따라다녀 스냅이 무의미해진다.
                glm::mat4 lightRotation = lookAlong(glm::vec3{0.0F}, direction);
                glm::vec3 sceneInLight = glm::vec3(lightRotation * glm::vec4{sceneCenter, 1.0F});
                float depthNear = -(sceneInLight.z + sceneRadius);
                float depthFar = -(sceneInLight.z - sceneRadius);

                float farDistance = shadowDistance > 0.0F ? shadowDistance : std::min(4.0F * sceneRadius, 500.0F);
                std::array<float, MAX_SHADOW_CASCADES> splits{};
                cascadeSplits(scene.camera.nearPlane, farDistance, viewsNeeded, shadowSplitLambda, splits);

                float aspect =
                    static_cast<float>(currentRenderExtent.width) / static_cast<float>(currentRenderExtent.height);
                float fov = glm::radians(scene.camera.fovYDegrees);
                glm::vec3 forward = scene.camera.forward();
                float previous = scene.camera.nearPlane;
                for (uint32_t cascade = 0; cascade < viewsNeeded; ++cascade) {
                    CascadeSphere sphere = fitCascadeSphere(previous, splits[cascade], fov, aspect);
                    glm::vec3 center = scene.camera.position + forward * sphere.distance;
                    view.viewProjection =
                        snapCascadeMatrix(lightRotation, center, sphere.radius, depthNear, depthFar, SHADOW_MAP_SIZE);
                    shadowViews.push_back(view);

                    light.cascadeSplits[cascade] = splits[cascade];
                    light.cascadeTexelSizes[cascade] = 2.0F * sphere.radius / static_cast<float>(SHADOW_MAP_SIZE);
                    previous = splits[cascade];
                }
            } else if (source.type == scene::LightType::SPOT) {
                float fov = std::min(glm::radians(source.outerConeDegrees) * 2.0F, glm::radians(170.0F));
                view.viewProjection =
                    glm::perspectiveRH_ZO(fov, 1.0F, SHADOW_NEAR_PLANE, source.range) * lookAlong(position, direction);
                shadowViews.push_back(view);
            } else {
                glm::mat4 projection =
                    glm::perspectiveRH_ZO(glm::radians(90.0F), 1.0F, SHADOW_NEAR_PLANE, source.range);
                for (const glm::vec3& face : CUBE_FACE_DIRECTIONS) {
                    view.viewProjection = projection * lookAlong(position, face);
                    shadowViews.push_back(view);
                }
            }
        }
        frameLights.push_back(light);
    }

    reserveLights(frame, static_cast<uint32_t>(frameLights.size()));
    if (!frameLights.empty()) {
        std::ranges::copy(frameLights, static_cast<GpuLight*>(frame.lightBuffer.mapped));
    }
    auto* shadowMatrices = static_cast<glm::mat4*>(frame.shadowMatrixBuffer.mapped);
    for (size_t view = 0; view < shadowViews.size(); ++view) {
        shadowMatrices[view] = shadowViews[view].viewProjection;
    }
}

FrameBatches Renderer::buildDrawCommands(Frame& frame, const scene::Scene& scene) {
    reserveInstances(frame, static_cast<uint32_t>(scene.objects.size()));

    // 장면이 통째로 바뀌면(장면 전환) 프레임 캐시가 다른 장면 것이다.
    sceneChangedThisFrame = &scene != lastScene || scene.revision() != lastSceneRevision;
    lastScene = &scene;
    lastSceneRevision = scene.revision();
    objectInstanceSlots.assign(scene.objects.size(), INVALID_INSTANCE_SLOT);
    objectSkinnedBlas.assign(scene.objects.size(), RayTracer::NO_SKINNED_BLAS);
    instanceBounds.assign(scene.objects.size(), glm::vec4{0.0F});
    skinDispatches.clear();
    skinnedInstances.clear();

    // 변형 정점은 가속 구조를 세울 때만 쓴다. 래스터만 그리는 프레임에는 만들지 않는다.
    bool rayTracedFrame =
        rayTracer != nullptr && (usePathTracing || (useRayQueryShadows && rayQueryShadowsAvailable()));
    uint32_t skinnedVertexCursor = 0;

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
    // 조상이 숨겨져 있으면 자식도 그리지 않는다. 변환만 담는 노드는 메쉬가 없어 걸러진다.
    auto drawable = [this, &scene](uint32_t index) {
        return scene.visibleCached(index) && scene.objects[index].meshIndex < geometry.meshCount();
    };

    FrameBatches batches{};
    uint32_t totalGroups = 0;
    uint32_t totalMeshletDraws = 0;
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        if (!drawable(index)) {
            continue;
        }
        const scene::Object& object = scene.objects[index];
        auto [mode, sided] = bucketOf(object);
        ++batches.draws[mode][sided].count;
        uint32_t groups = groupsFor(object);
        batches.groups[mode][sided].count += groups;
        totalGroups += groups;
        // 컴퓨트 컬링은 모든 단계의 meshlet 을 후보로 보므로 상한도 전체 개수로 잡는다.
        uint32_t candidates = geometry.mesh(object.meshIndex).meshletCount;
        batches.meshletDraws[mode][sided].count += candidates;
        totalMeshletDraws += candidates;
        ++batches.instanceCount;
    }
    reserveMeshletGroups(frame, totalGroups);
    reserveMeshletDraws(frame, totalMeshletDraws);

    // 조인트 행렬은 (애니메이터, 스킨) 마다 한 번만 올리고 인스턴스는 그 구간의 시작점만 가리킨다.
    std::vector<std::vector<uint32_t>> skinOffsets(scene.animators.size());
    uint32_t totalJoints = 0;
    for (size_t animator = 0; animator < scene.animators.size(); ++animator) {
        const std::vector<std::vector<glm::mat4>>& matrices = scene.animators[animator].jointMatrices;
        skinOffsets[animator].assign(matrices.size(), NO_JOINTS);
        for (size_t skin = 0; skin < matrices.size(); ++skin) {
            if (matrices[skin].empty()) {
                continue;
            }
            skinOffsets[animator][skin] = totalJoints;
            totalJoints += static_cast<uint32_t>(matrices[skin].size());
        }
    }
    reserveJoints(frame, totalJoints);
    auto* joints = static_cast<glm::mat4*>(frame.jointBuffer.mapped);
    for (size_t animator = 0; animator < skinOffsets.size(); ++animator) {
        for (size_t skin = 0; skin < skinOffsets[animator].size(); ++skin) {
            if (skinOffsets[animator][skin] != NO_JOINTS) {
                std::ranges::copy(scene.animators[animator].jointMatrices[skin], joints + skinOffsets[animator][skin]);
            }
        }
    }
    auto jointOffsetFor = [&skinOffsets](const scene::Object& object) {
        if (object.animator < 0 || object.skin < 0) {
            return NO_JOINTS;
        }
        const std::vector<uint32_t>& offsets = skinOffsets[static_cast<size_t>(object.animator)];
        return static_cast<size_t>(object.skin) < offsets.size() ? offsets[static_cast<size_t>(object.skin)]
                                                                 : NO_JOINTS;
    };

    uint32_t drawOffset = 0;
    uint32_t groupOffset = 0;
    uint32_t meshletDrawOffset = 0;
    for (size_t mode = 0; mode < ALPHA_MODE_COUNT; ++mode) {
        for (size_t sided = 0; sided < 2; ++sided) {
            batches.draws[mode][sided].first = drawOffset;
            drawOffset += batches.draws[mode][sided].count;
            batches.groups[mode][sided].first = groupOffset;
            groupOffset += batches.groups[mode][sided].count;
            batches.meshletDraws[mode][sided].first = meshletDrawOffset;
            meshletDrawOffset += batches.meshletDraws[mode][sided].count;
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

    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        if (!drawable(index)) {
            continue;
        }
        const scene::Object& object = scene.objects[index];
        auto [mode, sided] = bucketOf(object);
        uint32_t slot = drawCursors[mode][sided]++;

        const GpuMesh& mesh = geometry.mesh(object.meshIndex);
        const GpuMeshLod& lod = lodFor(object);
        const glm::mat4& model = scene.world(index);
        objectInstanceSlots[index] = slot;

        instances[slot].model = model;
        instances[slot].normalMatrix = glm::mat4(glm::inverseTranspose(glm::mat3(model)));
        instanceBounds[slot] = transformBoundingSphere(model, mesh.boundingSphere);
        instances[slot].meshIndex = object.meshIndex;
        instances[slot].bucket = static_cast<uint32_t>(mode * 2 + sided);
        instances[slot].bucketBase = batches.meshletDraws[mode][sided].first;
        uint32_t jointOffset = jointOffsetFor(object);
        instances[slot].jointOffset = jointOffset;

        // 스킨 인스턴스는 변형 정점을 따로 뽑아 두고, 그 구간으로 자기 하위 가속 구조를 세운다.
        // 같은 메쉬를 여러 오브젝트가 서로 다른 포즈로 쓸 수 있어 오브젝트마다 하나씩 잡는다.
        uint32_t skinnedVertexOffset = NO_SKINNED_VERTICES;
        if (rayTracedFrame && jointOffset != NO_JOINTS) {
            uint32_t vertexCount = geometry.meshVertexCount(object.meshIndex);
            skinnedVertexOffset = skinnedVertexCursor;
            skinnedVertexCursor += vertexCount;
            objectSkinnedBlas[index] = static_cast<uint32_t>(skinnedInstances.size());
            skinnedInstances.push_back(SkinnedInstance{object.meshIndex, skinnedVertexOffset});
            skinDispatches.push_back(SkinDispatch{static_cast<uint32_t>(mesh.vertexOffset),
                                                  skinnedVertexOffset,
                                                  jointOffset,
                                                  vertexCount});
        }
        instances[slot].skinnedVertexOffset = skinnedVertexOffset;

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

    // 커질 때만 다시 잡는다. 지난 프레임이 아직 읽고 있을 수 있어 그때는 장치를 세운다. 스킨
    // 오브젝트가 늘어나는 순간에만 일어나므로 프레임마다 드는 비용은 아니다.
    if (skinnedVertexCursor > skinnedVertexCapacity) {
        waitIdle();
        destroyBuffer(context, skinnedVertexBuffer);
        skinnedVertexCapacity = skinnedVertexCursor * 2;
        skinnedVertexBuffer =
            createBuffer(context,
                         static_cast<VkDeviceSize>(skinnedVertexCapacity) * sizeof(asset::Vertex),
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                         MemoryLocation::DEVICE,
                         "스킨 정점");
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

    // 카메라 변환을 먼저 구한다. 그림자 캐스터 컬링이 이번 프레임의 카메라 절두체를 봐야 한다.
    float aspect = static_cast<float>(currentRenderExtent.width) / static_cast<float>(currentRenderExtent.height);
    glm::mat4 cameraViewProjection = scene.camera.projectionMatrix(aspect) * scene.camera.viewMatrix();

    buildLights(frame, scene);
    if (usePathTracing && rayTracer != nullptr) {
        // 경로 추적은 그림자 아틀라스를 읽지 않는다. 시점마다 캐스터를 걸러 명령을 짜는 CPU 비용을
        // 통째로 아끼고, 편집기 표시도 실제로 그리는 양과 어긋나지 않게 0 으로 둔다.
        shadowBatches.clear();
        shadowLayerDirty.assign(MAX_SHADOW_VIEWS, 0);
        shadowDrawsIssued = 0;
        shadowDrawsTotal = 0;
    } else {
        buildShadowDraws(frame, batches, cameraViewProjection);
    }

    auto* camera = static_cast<GpuCamera*>(frame.cameraBuffer.mapped);
    camera->viewProjection = cameraViewProjection;
    camera->position = glm::vec4{scene.camera.position, 1.0F};
    // 화면 공간 오차 = 월드 오차 * projectionScale / 거리.
    float projectionScale = static_cast<float>(currentRenderExtent.height) /
                            (2.0F * std::tan(glm::radians(scene.camera.fovYDegrees) * 0.5F));
    camera->parameters = glm::vec4{scene.camera.nearPlane,
                                   projectionScale,
                                   lodErrorThreshold,
                                   automaticLod ? -1.0F : static_cast<float>(lodLevel)};
    camera->shading = glm::uvec4{static_cast<uint32_t>(frameLights.size()),
                                 shadowViews.empty() ? asset::INVALID_TEXTURE : targets.shadowAtlasSlot,
                                 useSsao ? targets.ssaoSlot : asset::INVALID_TEXTURE,
                                 environment->environmentSlot()};
    bool iblReady = useIbl && environment->ready();
    camera->environment = glm::uvec4{environment->irradianceSlot(),
                                     environment->prefilterSlot(),
                                     environment->brdfSlot(),
                                     iblReady ? environment->prefilterMipCount() : 0U};
    bool rayShadows = useRayQueryShadows && rayQueryShadowsAvailable();
    camera->ambient = glm::vec4{scene.ambientColor * scene.ambientIntensity, rayShadows ? rayShadowDistance : 0.0F};
    camera->viewport = glm::vec4{static_cast<float>(currentRenderExtent.width),
                                 static_cast<float>(currentRenderExtent.height),
                                 1.0F / static_cast<float>(currentRenderExtent.width),
                                 1.0F / static_cast<float>(currentRenderExtent.height)};
    camera->inverseViewProjection = glm::inverse(camera->viewProjection);

    // 카메라나 화면, 장면, 경로 추적 설정이 바뀌면 누적을 처음부터 다시 쌓는다. 설정 변경을 빼면
    // 수백 표본이 쌓인 뒤에는 새 표본이 1/N 밖에 못 섞여 화면이 멈춘 것처럼 보인다.
    bool traceInputsChanged = pathTrace != lastPathTrace || useIbl != lastUseIbl;
    lastPathTrace = pathTrace;
    lastUseIbl = useIbl;
    if (camera->viewProjection != lastViewProjection || sceneChangedThisFrame || traceInputsChanged) {
        lastViewProjection = camera->viewProjection;
        pathSampleCount = 0;
    }

    updateLodNetwork(scene, frame, projectionScale);

    return batches;
}

void Renderer::recordGeometryPass(VkCommandBuffer commandBuffer, const FrameBatches& batches, bool translucentPass) {
    constexpr VkDeviceSize DRAW_STRIDE = sizeof(VkDrawIndexedIndirectCommand);
    constexpr VkDeviceSize TASK_STRIDE = sizeof(VkDrawMeshTasksIndirectCommandEXT);

    Frame& frame = frames[frameIndex % FRAMES_IN_FLIGHT];
    bool meshPath = useMeshPath();
    bool cullPath = useComputeCulling && !meshPath;

    size_t firstMode = translucentPass ? TRANSLUCENT_MODE : 0;
    size_t lastMode = translucentPass ? TRANSLUCENT_MODE + 1 : TRANSLUCENT_MODE;

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
                    pipeline = wireframe && !translucentPass ? wireframePipeline
                               : rayQueryPass                ? meshRayQueryPipelines[mode]
                                                             : meshPipelines[mode];
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
}

void Renderer::updateLodNetwork(const scene::Scene& scene, Frame& frame, float projectionScale) {
    // 가중치는 매 프레임 올린다. 학습을 꺼도 GPU 가 마지막 가중치를 그대로 쓰게 된다.
    auto uploadWeights = [&frame, this]() {
        std::memcpy(frame.lodNetworkBuffer.mapped, &lodNetwork.weights(), sizeof(GpuLodNetwork));
    };
    if (!useNeuralLod || !trainLodNetwork) {
        uploadWeights();
        return;
    }

    uint32_t candidateCount = 0;
    for (const scene::Object& object : scene.objects) {
        if (object.visible && object.meshIndex < geometry.meshCount()) {
            candidateCount += geometry.mesh(object.meshIndex).meshletCount;
        }
    }
    if (candidateCount == 0) {
        uploadWeights();
        return;
    }

    uint32_t stride = std::max(candidateCount / MAX_LOD_SAMPLES, 1U);
    std::vector<LodSample> samples;
    samples.reserve(std::min(candidateCount, MAX_LOD_SAMPLES) + 1);

    glm::vec3 cameraPosition = scene.camera.position;
    uint32_t candidateIndex = 0;
    for (const scene::Object& object : scene.objects) {
        if (!object.visible || object.meshIndex >= geometry.meshCount()) {
            continue;
        }
        const GpuMesh& mesh = geometry.mesh(object.meshIndex);
        glm::mat4 model = object.transform.matrix();
        float modelScale = std::sqrt(std::max({glm::dot(glm::vec3(model[0]), glm::vec3(model[0])),
                                               glm::dot(glm::vec3(model[1]), glm::vec3(model[1])),
                                               glm::dot(glm::vec3(model[2]), glm::vec3(model[2]))}));

        for (uint32_t i = 0; i < mesh.meshletCount; ++i, ++candidateIndex) {
            if (candidateIndex % stride != 0) {
                continue;
            }
            const GpuMeshlet& meshlet = geometry.meshlet(mesh.meshletOffset + i);
            // 최상위 meshlet 은 오차가 무한대라 학습 표본에서 뺀다.
            if (!std::isfinite(meshlet.error) || meshlet.error <= 0.0F) {
                continue;
            }

            glm::vec3 center = glm::vec3(model * glm::vec4{glm::vec3(meshlet.errorSphere), 1.0F});
            float radius = meshlet.errorSphere.w * modelScale;
            float viewDistance = std::max(glm::distance(center, cameraPosition) - radius, 1e-3F);
            float projected = meshlet.error * projectionScale / viewDistance;

            LodSample sample;
            sample.features[0] = std::log2(std::max(viewDistance, 1e-3F)) / 8.0F;
            sample.features[1] = std::log2(std::max(radius, 1e-6F)) / 8.0F;
            sample.features[2] = std::log2(std::max(meshlet.error, 1e-6F)) / 8.0F;
            sample.features[3] = std::log2(std::max(projected, 1e-6F)) / 8.0F;
            sample.projectedError = projected;
            sample.triangleCount = static_cast<float>(meshlet.triangleCount);
            samples.push_back(sample);
        }
    }

    // 표본은 전체의 일부이므로 예산도 같은 비율로 줄인다.
    float sampleRatio = static_cast<float>(samples.size() * stride) / static_cast<float>(candidateCount);
    lodNetwork.train(samples, lodErrorThreshold, triangleBudget * std::max(sampleRatio, 1e-3F));
    uploadWeights();
}

void Renderer::recordCullPass(VkCommandBuffer commandBuffer, const FrameBatches& batches) {
    Frame& frame = frames[frameIndex % FRAMES_IN_FLIGHT];

    vkCmdFillBuffer(commandBuffer, frame.drawCountBuffer.handle, 0, VK_WHOLE_SIZE, 0);
    if (drawIndexedIndirectCount == nullptr) {
        // 압축 간접 그리기가 없으면 상한만큼 그리므로, 남은 자리는 0 으로 채워 무효 명령으로 만든다.
        vkCmdFillBuffer(commandBuffer, frame.meshletDrawBuffer.handle, 0, VK_WHOLE_SIZE, 0);
    }

    VkMemoryBarrier2 clearBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    clearBarrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    clearBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    clearBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    clearBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo clearDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    clearDependency.memoryBarrierCount = 1;
    clearDependency.pMemoryBarriers = &clearBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &clearDependency);

    CullPushConstants pushConstants{};
    pushConstants.instances = frame.instanceBuffer.address;
    pushConstants.meshes = geometry.meshBuffer.address;
    pushConstants.meshlets = geometry.meshletBuffer.address;
    pushConstants.camera = frame.cameraBuffer.address;
    pushConstants.drawCommands = frame.meshletDrawBuffer.address;
    pushConstants.drawCounts = frame.drawCountBuffer.address;
    pushConstants.instanceCount = batches.instanceCount;
    pushConstants.flags = (frustumCulling ? CULL_FLAG_FRUSTUM : 0U) | (coneCulling ? CULL_FLAG_CONE : 0U) |
                          (occlusionCulling ? CULL_FLAG_OCCLUSION : 0U);
    pushConstants.hzbTexture = targets.hzbSampledSlot;
    pushConstants.hzbMaxLevel = static_cast<float>(targets.hzbStorageSlots.size() - 1);
    pushConstants.hzbSize[0] = static_cast<float>(targets.hzbExtent.width);
    pushConstants.hzbSize[1] = static_cast<float>(targets.hzbExtent.height);
    pushConstants.network = frame.lodNetworkBuffer.address;
    if (useNeuralLod) {
        pushConstants.flags |= CULL_FLAG_NEURAL_LOD;
    }

    VkDescriptorSet cullBindlessSet = bindless.set();
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cullPipelineLayout, 0, 1, &cullBindlessSet, 0, nullptr);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, cullPipeline);
    vkCmdPushConstants(
        commandBuffer, cullPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    vkCmdDispatch(commandBuffer, std::max(batches.instanceCount, 1U), 1, 1);

    VkMemoryBarrier2 cullBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    cullBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    cullBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    cullBarrier.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    cullBarrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
    VkDependencyInfo cullDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    cullDependency.memoryBarrierCount = 1;
    cullDependency.pMemoryBarriers = &cullBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &cullDependency);
}

void Renderer::recordSkinPass(VkCommandBuffer commandBuffer, const Frame& frame) {
    if (skinDispatches.empty()) {
        return;
    }

    VkDescriptorSet bindlessSet = bindless.set();
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, skinPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, skinPipeline);
    for (const SkinDispatch& dispatch : skinDispatches) {
        SkinPushConstants pushConstants{geometry.vertexBuffer.address,
                                        skinnedVertexBuffer.address,
                                        frame.jointBuffer.address,
                                        dispatch.sourceOffset,
                                        dispatch.destinationOffset,
                                        dispatch.jointOffset,
                                        dispatch.vertexCount};
        vkCmdPushConstants(
            commandBuffer, skinPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
        vkCmdDispatch(commandBuffer, (dispatch.vertexCount + SKIN_GROUP_SIZE - 1) / SKIN_GROUP_SIZE, 1, 1);
    }

    // 하위 가속 구조 구축이 이 정점을 그대로 읽는다.
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

void Renderer::updateAccelerationStructures(VkCommandBuffer commandBuffer,
                                            const Frame& frame,
                                            const scene::Scene& scene) {
    // 장면이 그대로면 가속 구조도 그대로다. 포즈가 바뀌면 애니메이터가 장면 리비전을 올리므로
    // 같은 조건으로 걸러진다. 스킨 목록 자체가 달라졌으면(경로 추적을 막 켰거나 오브젝트가
    // 늘었으면) 리비전과 무관하게 다시 세워야 한다.
    if (!sceneChangedThisFrame && skinnedInstances == builtSkinnedInstances && rayTracer->ready()) {
        return;
    }
    builtSkinnedInstances = skinnedInstances;
    recordSkinPass(commandBuffer, frame);
    rayTracer->updateSkinnedBottomLevel(commandBuffer, skinnedVertexBuffer, skinnedInstances);
    rayTracer->updateTopLevel(commandBuffer,
                              scene,
                              objectInstanceSlots,
                              objectSkinnedBlas,
                              static_cast<uint32_t>(frameIndex % FRAMES_IN_FLIGHT));
}

void Renderer::recordPathTracePass(VkCommandBuffer commandBuffer, Frame& frame, const scene::Scene& scene) {
    updateAccelerationStructures(commandBuffer, frame, scene);
    if (!rayTracer->ready()) {
        return;
    }

    imageBarrier(commandBuffer,
                 targets.pathAccumulation.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 pathSampleCount == 0 ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL,
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                 VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    // 표본 상한에 닿으면 더 쏘지 않고 쌓아 둔 결과를 그대로 보여준다.
    if (pathTrace.maxSamples == 0 || pathSampleCount < pathTrace.maxSamples) {
        rayTracer->trace(commandBuffer,
                         currentRenderExtent,
                         frame.cameraBuffer.address,
                         frame.instanceBuffer.address,
                         frame.lightBuffer.address,
                         skinnedVertexBuffer.address,
                         targets.pathAccumulationStorageSlot,
                         0,
                         static_cast<uint32_t>(frameIndex),
                         pathSampleCount,
                         pathTrace);
        ++pathSampleCount;
    }

    imageBarrier(commandBuffer,
                 targets.pathAccumulation.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    // 깊이는 쓰지 않지만 UI 뷰어가 샘플링하므로 레이아웃만 맞춰 둔다.
    imageBarrier(commandBuffer,
                 targets.depth.handle,
                 VK_IMAGE_ASPECT_DEPTH_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                 0,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
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

void Renderer::recordCommands(Frame& frame,
                              uint32_t imageIndex,
                              const FrameBatches& batches,
                              const scene::Scene& scene) {
    VkCommandBuffer commandBuffer = frame.commandBuffer;
    VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(commandBuffer, &beginInfo));
    uint32_t frameZone = frameProfiler.begin("프레임 전체", commandBuffer);

    bool hasTranslucent = batches.draws[TRANSLUCENT_MODE][0].count + batches.draws[TRANSLUCENT_MODE][1].count > 0;
    oitTargetsValid = hasTranslucent;

    if (hzbNeedsClear) {
        // 첫 프레임에는 이전 깊이가 없으므로 아무것도 가리지 않도록 가장 먼 값으로 채운다.
        imageBarrier(commandBuffer,
                     targets.hzb.handle,
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                     0,
                     VK_PIPELINE_STAGE_2_CLEAR_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT);

        VkClearColorValue clear{};
        VkImageSubresourceRange range{};
        range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        range.levelCount = VK_REMAINING_MIP_LEVELS;
        range.layerCount = VK_REMAINING_ARRAY_LAYERS;
        vkCmdClearColorImage(commandBuffer, targets.hzb.handle, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);

        VkMemoryBarrier2 clearBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        clearBarrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        clearBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        clearBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        clearBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        VkDependencyInfo clearDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        clearDependency.memoryBarrierCount = 1;
        clearDependency.pMemoryBarriers = &clearBarrier;
        vkCmdPipelineBarrier2(commandBuffer, &clearDependency);
        hzbNeedsClear = false;
    }

    if (shadowNeedsInit) {
        // 층 전체를 한 번 읽기 좋은 레이아웃으로 옮긴다. 이후 프레임은 층마다 따로 전이하므로
        // 여기서 맞춰 두지 않으면 한 번도 안 그린 층이 잘못된 레이아웃으로 남는다.
        imageBarrier(commandBuffer,
                     targets.shadowAtlas.handle,
                     VK_IMAGE_ASPECT_DEPTH_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                     0,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        shadowNeedsInit = false;
    }

    if (ssaoNeedsClear) {
        // 첫 프레임에는 이전 깊이가 없으므로 차폐 없음(1)으로 채운다.
        for (const Image* image : {&targets.ssaoRaw, &targets.ssao}) {
            imageBarrier(commandBuffer,
                         image->handle,
                         VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                         0,
                         VK_PIPELINE_STAGE_2_CLEAR_BIT,
                         VK_ACCESS_2_TRANSFER_WRITE_BIT);

            VkClearColorValue clear{};
            clear.float32[0] = 1.0F;
            VkImageSubresourceRange range{};
            range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            range.levelCount = VK_REMAINING_MIP_LEVELS;
            range.layerCount = VK_REMAINING_ARRAY_LAYERS;
            vkCmdClearColorImage(commandBuffer, image->handle, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
        }

        VkMemoryBarrier2 clearBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        clearBarrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        clearBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        clearBarrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        clearBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        VkDependencyInfo clearDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        clearDependency.memoryBarrierCount = 1;
        clearDependency.pMemoryBarriers = &clearBarrier;
        vkCmdPipelineBarrier2(commandBuffer, &clearDependency);
        ssaoNeedsClear = false;
    }

    bool pathTracing = usePathTracing && rayTracer != nullptr;
    rayQueryPass = false;

    // 환경 맵은 설정이 바뀔 때만 다시 굽는다. 래스터와 경로 추적이 같은 환경을 본다.
    {
        uint32_t environmentZone = frameProfiler.begin("환경", commandBuffer);
        if (environment->update(commandBuffer, scene.environment, sunDirection)) {
            // 하늘이 바뀌었으면 쌓아 둔 경로 추적 표본은 옛 환경의 것이다.
            pathSampleCount = 0;
        }
        frameProfiler.end(environmentZone, commandBuffer);
    }

    // 그림자 아틀라스는 장면 패스보다 먼저 채워야 한다. 경로 추적은 아틀라스를 읽지 않으므로 건너뛴다.
    if (!pathTracing && !shadowViews.empty()) {
        uint32_t shadowZone = frameProfiler.begin("그림자", commandBuffer);
        recordShadowPass(commandBuffer);
        frameProfiler.end(shadowZone, commandBuffer);
    }

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

    VkDescriptorSet bindlessSet = bindless.set();
    if (pathTracing) {
        uint32_t pathZone = frameProfiler.begin("경로 추적", commandBuffer);
        recordPathTracePass(commandBuffer, frame, scene);
        frameProfiler.end(pathZone, commandBuffer);
    } else {
        // mesh shader 경로는 태스크 셰이더가 직접 컬링한다. 컬 컴퓨트 결과를 아무도 읽지 않으므로
        // 그때는 디스패치 자체를 하지 않는다.
        if (useComputeCulling && !useMeshPath()) {
            uint32_t cullZone = frameProfiler.begin("컬링", commandBuffer);
            recordCullPass(commandBuffer, batches);
            frameProfiler.end(cullZone, commandBuffer);
        }

        ScenePushConstants scenePushConstants{geometry.vertexBuffer.address,
                                              geometry.meshBuffer.address,
                                              frame.instanceBuffer.address,
                                              geometry.materialBuffer.address,
                                              frame.cameraBuffer.address,
                                              geometry.meshletBuffer.address,
                                              geometry.meshletTriangleBuffer.address,
                                              geometry.vertexMeshletBuffer.address,
                                              frame.meshletGroupBuffer.address,
                                              frame.jointBuffer.address,
                                              frame.lightBuffer.address,
                                              frame.shadowMatrixBuffer.address,
                                              0,
                                              debugMode};
        // 광선 질의 그림자를 쓰면 TLAS 를 집합 1 로 함께 묶고, 장면이 바뀌었으면 먼저 다시 만든다.
        rayQueryPass = useRayQueryShadows && rayQueryShadowsAvailable();
        if (rayQueryPass) {
            updateAccelerationStructures(commandBuffer, frame, scene);
            rayQueryPass = rayTracer->ready();
        }
        VkPipelineLayout sceneLayout = rayQueryPass ? meshRayQueryPipelineLayout : meshPipelineLayout;

        vkCmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, sceneLayout, 0, 1, &bindlessSet, 0, nullptr);
        if (rayQueryPass) {
            VkDescriptorSet accelerationSet = rayTracer->accelerationSet();
            vkCmdBindDescriptorSets(
                commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, sceneLayout, 1, 1, &accelerationSet, 0, nullptr);
        }
        vkCmdPushConstants(
            commandBuffer, sceneLayout, scenePushStages, 0, sizeof(scenePushConstants), &scenePushConstants);
        vkCmdBindIndexBuffer(commandBuffer, geometry.indexBuffer.handle, 0, VK_INDEX_TYPE_UINT32);

        // 1) 불투명과 컷오프 경로를 HDR 색상 대상에 그린다.
        uint32_t opaqueZone = frameProfiler.begin("불투명", commandBuffer);
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
        frameProfiler.end(opaqueZone, commandBuffer);

        if (hasTranslucent) {
            // 2) 반투명은 누적과 잔여 투과율 대상에 순서 독립으로 기록한다.
            uint32_t oitZone = frameProfiler.begin("OIT", commandBuffer);
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
            frameProfiler.end(oitZone, commandBuffer);

            // 3) 누적 결과를 HDR 색상 위에 합성한다.
            uint32_t compositeZone = frameProfiler.begin("OIT 합성", commandBuffer);
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

            VkRenderingAttachmentInfo compositeColor =
                colorAttachment(targets.color.view, VK_ATTACHMENT_LOAD_OP_LOAD, {});
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
            frameProfiler.end(compositeZone, commandBuffer);
        }

        // 4) 이번 프레임 깊이로 HZB 를 만들어 다음 프레임 컬링에 쓴다.
        imageBarrier(commandBuffer,
                     targets.depth.handle,
                     VK_IMAGE_ASPECT_DEPTH_BIT,
                     VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                     VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
        uint32_t hzbZone = frameProfiler.begin("HZB", commandBuffer);
        recordHzbPass(commandBuffer);
        frameProfiler.end(hzbZone, commandBuffer);
        // SSAO 도 같은 깊이를 읽는다. 결과는 다음 프레임의 셰이딩이 쓴다.
        //
        // ponytail: 한 프레임 늦은 깊이라 카메라가 빠르게 움직이면 차폐가 살짝 밀린다. 정확히
        // 하려면 불투명 깊이 선행 패스를 넣고 같은 프레임 안에서 계산해야 한다.
        if (useSsao) {
            uint32_t ssaoZone = frameProfiler.begin("SSAO", commandBuffer);
            recordSsaoPass(commandBuffer, frame);
            frameProfiler.end(ssaoZone, commandBuffer);
        }
    }

    // 5) HDR 색상을 톤 매핑해 표시 이미지에 옮긴다. 편집기 뷰포트가 이 이미지를 그대로 보여준다.
    uint32_t tonemapZone = frameProfiler.begin("톤 매핑", commandBuffer);
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
                 targets.tonemapped.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                 0,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    VkRenderingAttachmentInfo tonemappedColor =
        colorAttachment(targets.tonemapped.view, VK_ATTACHMENT_LOAD_OP_DONT_CARE, {});
    VkRenderingInfo tonemapPass{VK_STRUCTURE_TYPE_RENDERING_INFO};
    tonemapPass.renderArea.extent = currentRenderExtent;
    tonemapPass.layerCount = 1;
    tonemapPass.colorAttachmentCount = 1;
    tonemapPass.pColorAttachments = &tonemappedColor;

    TonemapPushConstants tonemapPushConstants{};
    tonemapPushConstants.colorTexture = pathTracing ? targets.pathAccumulationSampledSlot : targets.colorSlot;
    tonemapPushConstants.exposure = exposure;
    tonemapPushConstants.sampleCount = pathTracing ? pathSampleCount : 0U;
    tonemapPushConstants.camera = frame.cameraBuffer.address;
    tonemapPushConstants.depthTexture = targets.depthSlot;
    tonemapPushConstants.environmentCube =
        useIbl && environment->ready() ? environment->environmentSlot() : asset::INVALID_TEXTURE;
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

    frameProfiler.end(tonemapZone, commandBuffer);

    // 6) 렌더 해상도의 톤 매핑 결과를 표시 해상도로 확대한다.
    uint32_t upscaleZone = frameProfiler.begin("업스케일", commandBuffer);
    imageBarrier(commandBuffer,
                 targets.tonemapped.handle,
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

    VkRenderingAttachmentInfo upscaleColor = colorAttachment(targets.present.view, VK_ATTACHMENT_LOAD_OP_DONT_CARE, {});
    VkRenderingInfo upscalePass{VK_STRUCTURE_TYPE_RENDERING_INFO};
    upscalePass.renderArea.extent = currentDisplayExtent;
    upscalePass.layerCount = 1;
    upscalePass.colorAttachmentCount = 1;
    upscalePass.pColorAttachments = &upscaleColor;

    UpscalePushConstants upscalePushConstants{};
    upscalePushConstants.sourceTexture = targets.tonemappedSlot;
    upscalePushConstants.sharpness = upscaleSharpness;
    upscalePushConstants.sourceSize[0] = static_cast<float>(currentRenderExtent.width);
    upscalePushConstants.sourceSize[1] = static_cast<float>(currentRenderExtent.height);
    upscalePushConstants.destinationSize[0] = static_cast<float>(currentDisplayExtent.width);
    upscalePushConstants.destinationSize[1] = static_cast<float>(currentDisplayExtent.height);

    // 구현된 방식은 통과와 내장 공간 업스케일 둘뿐이다. 나머지는 벤더 SDK 가 필요해 통과로 떨어진다.
    size_t upscaleVariant = upscaler == Upscaler::SPATIAL ? 1 : 0;

    vkCmdBeginRendering(commandBuffer, &upscalePass);
    setFullViewport(commandBuffer, currentDisplayExtent);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, upscalePipelines[upscaleVariant]);
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, postPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer,
                       postPipelineLayout,
                       VK_SHADER_STAGE_FRAGMENT_BIT,
                       0,
                       sizeof(upscalePushConstants),
                       &upscalePushConstants);
    vkCmdDraw(commandBuffer, 3, 1, 0, 0);
    vkCmdEndRendering(commandBuffer);

    frameProfiler.end(upscaleZone, commandBuffer);

    // 7) 편집기 UI 를 스왑체인에 그린다. 오프스크린 대상들은 UI 가 샘플링할 수 있는 레이아웃으로 옮긴다.
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
                 swapchain->images[imageIndex],
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                 0,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    uint32_t uiZone = frameProfiler.begin("UI", commandBuffer);
    recordUiPass(commandBuffer, imageIndex);
    frameProfiler.end(uiZone, commandBuffer);

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

    frameProfiler.end(frameZone, commandBuffer);
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

void Renderer::onGeometryChanged() {
    if (rayTracer != nullptr) {
        rayTracer->buildBottomLevel();
    }
}

void Renderer::prepareFrame() {
    if (resizeRequested) {
        recreateSwapchain();
    }
}

void Renderer::drawFrame(const scene::Scene& scene) {
    if (swapchain->extent.width == 0 || swapchain->extent.height == 0) {
        return;
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

    // 타임라인 대기를 통과했으므로 이 슬롯의 지난 쿼리 결과를 대기 없이 읽을 수 있다.
    frameProfiler.collect();

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

    uint32_t buildZone = frameProfiler.begin("그리기 명령 구성");
    FrameBatches batches = buildDrawCommands(frame, scene);
    frameProfiler.end(buildZone);

    if (!capturePath.empty()) {
        VkDeviceSize required = static_cast<VkDeviceSize>(swapchain->extent.width) * swapchain->extent.height * 4;
        if (captureBuffer.size < required) {
            destroyBuffer(context, captureBuffer);
            captureBuffer = createBuffer(
                context, required, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryLocation::HOST_READ, "화면 캡처");
        }
    }

    VK_CHECK(vkResetCommandPool(context.device, frame.commandPool, 0));
    uint32_t recordZone = frameProfiler.begin("명령 기록");
    recordCommands(frame, imageIndex, batches, scene);
    frameProfiler.end(recordZone);
    frameProfiler.endFrame();

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
