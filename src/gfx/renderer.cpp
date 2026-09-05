#include "gfx/renderer.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>
#include <glm/trigonometric.hpp>
#include <glm/vec4.hpp>
#include <SDL3/SDL.h>
#include <spdlog/spdlog.h>
#include <stb_image_write.h>

#include "asset/model.h"
#include "core/error.h"
#include "core/job_system.h"
#include "gfx/bindless.h"
#include "gfx/context.h"
#include "gfx/geometry.h"
#include "gfx/raytracing.h"
#include "gfx/renderer_internal.h"
#include "gfx/swapchain.h"
#include "gfx/upscaler_math.h"
#include "gfx/vk_check.h"
#include "scene/scene.h"

namespace gfx {

Renderer::Renderer(Context& context,
                   GeometryStore& geometry,
                   BindlessTextures& bindless,
                   SDL_Window* window,
                   core::JobSystem& jobs,
                   RenderSettings& settings)
    : context(context), geometry(geometry), bindless(bindless), jobs(jobs), settings(settings),
      frameProfiler(context, FRAMES_IN_FLIGHT) {
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
    // HZB 는 밉 단계를 명시적으로 골라 읽는다. maxLod 가 0 이면 전부 밉 0 으로 잘린다.
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
    VK_CHECK(vkCreateSampler(context.device, &samplerInfo, nullptr, &postSampler));

    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.maxLod = VK_LOD_CLAMP_NONE;
    VK_CHECK(vkCreateSampler(context.device, &samplerInfo, nullptr, &linearSampler));

    histogramBuffer = createBuffer(context,
                                   sizeof(uint32_t) * HISTOGRAM_BINS,
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                   MemoryLocation::DEVICE,
                                   "휘도 히스토그램");
    exposureBuffer =
        createBuffer(context, sizeof(float), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryLocation::DEVICE, "자동 노출");

    // NGX 는 초기화해 봐야 쓸 수 있는지 알 수 있다. 편집기가 항목을 켤 수 있으려면 고르기 전에
    // 미리 띄워 둬야 한다.
    startDlssRuntime(context);

    createRenderTargets();
    createFrames();
    createPresentSemaphores();
    // 광선 질의 그림자 파이프라인이 TLAS 디스크립터 배치를 필요로 하므로 먼저 만든다.
    if (context.caps.rayTracingPipeline) {
        // 하위 가속 구조는 광선 기능이 처음 필요할 때 세운다. 큰 모델은 예산을 넘겨 못 세울 수 있다.
        rayTracer = std::make_unique<RayTracer>(context, geometry, bindless);
    }
    createMeshPipelines();
    createPostPipelines();
    createBloomPipelines();
    createDebugLinePipeline();
    // 두께 대상은 가산 혼합으로 쌓으므로 혼합을 지원하는 포맷이어야 한다.
    VkFormatProperties thicknessProperties{};
    vkGetPhysicalDeviceFormatProperties(context.physicalDevice, THICKNESS_FORMAT, &thicknessProperties);
    bool blendable = (thicknessProperties.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) != 0;
    thicknessFormat = blendable ? THICKNESS_FORMAT : THICKNESS_FALLBACK_FORMAT;
    if (!blendable) {
        spdlog::info("R32 색상 혼합을 지원하지 않아 물 두께를 반정밀도로 잽니다");
    }
    createFluidSurfacePipelines();
    createReflectionPipelines();
    createCullPipeline();
    createSkinPipeline();
    createShadowPipeline();
    createSsaoPipelines();
    environment = std::make_unique<EnvironmentMap>(context, bindless);
    fluid = std::make_unique<FluidSimulator>(context, bindless, jobs);

    VkSemaphoreTypeCreateInfo timelineInfo{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    timelineInfo.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    VkSemaphoreCreateInfo semaphoreInfo{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    semaphoreInfo.pNext = &timelineInfo;
    VK_CHECK(vkCreateSemaphore(context.device, &semaphoreInfo, nullptr, &frameTimeline));
}

Renderer::~Renderer() {
    waitIdle();
    temporalUpscaler.reset();
    // NGX 는 장치가 살아 있을 때 정리해야 한다.
    shutdownDlssRuntime();
    vkDestroyPipeline(context.device, skyPipeline, nullptr);
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
    vkDestroyPipeline(context.device, debugLinePipeline, nullptr);
    vkDestroyPipelineLayout(context.device, debugLinePipelineLayout, nullptr);
    vkDestroyPipeline(context.device, fluidThicknessPipeline, nullptr);
    vkDestroyPipeline(context.device, fluidDepthPipeline, nullptr);
    vkDestroyPipeline(context.device, fluidSurfacePipeline, nullptr);
    vkDestroyPipelineLayout(context.device, fluidSurfaceLayout, nullptr);
    for (VkPipeline pipeline : meshPipelines) {
        vkDestroyPipeline(context.device, pipeline, nullptr);
    }
    vkDestroyPipeline(context.device, hzbPipeline, nullptr);
    vkDestroyPipelineLayout(context.device, hzbPipelineLayout, nullptr);
    destroyBuffer(context, meshletVisibilityBuffer);
    vkDestroyPipeline(context.device, cullPipeline, nullptr);
    vkDestroyPipeline(context.device, skinBoundsPipeline, nullptr);
    vkDestroyPipelineLayout(context.device, skinBoundsPipelineLayout, nullptr);
    vkDestroyPipeline(context.device, skinPipeline, nullptr);
    vkDestroyPipelineLayout(context.device, skinPipelineLayout, nullptr);
    destroyBuffer(context, skinnedBoundsBuffer);
    destroyBuffer(context, skinnedVertexBuffer);
    vkDestroyPipeline(context.device, reflectionResolvePipeline, nullptr);
    vkDestroyPipeline(context.device, reflectionTracePipeline, nullptr);
    vkDestroyPipelineLayout(context.device, reflectionPipelineLayout, nullptr);
    vkDestroyPipeline(context.device, exposurePipeline, nullptr);
    vkDestroyPipelineLayout(context.device, exposurePipelineLayout, nullptr);
    vkDestroyPipeline(context.device, histogramPipeline, nullptr);
    vkDestroyPipelineLayout(context.device, histogramPipelineLayout, nullptr);
    vkDestroyPipeline(context.device, bloomUpsamplePipeline, nullptr);
    vkDestroyPipeline(context.device, bloomDownsamplePipeline, nullptr);
    vkDestroyPipelineLayout(context.device, bloomPipelineLayout, nullptr);
    destroyBuffer(context, exposureBuffer);
    destroyBuffer(context, histogramBuffer);
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
        destroyBuffer(context, frame.debugLineBuffer);
        destroyBuffer(context, frame.shadowDrawBuffer);
        destroyBuffer(context, frame.shadowMatrixBuffer);
        destroyBuffer(context, frame.lightBuffer);
        destroyBuffer(context, frame.jointBuffer);
        destroyBuffer(context, frame.lodNetworkBuffer);
        destroyBuffer(context, frame.drawCountBuffer);
        destroyBuffer(context, frame.meshletDrawMeshletBuffer);
        destroyBuffer(context, frame.meshletDrawBuffer);
        destroyBuffer(context, frame.meshTaskIndirectBuffer);
        destroyBuffer(context, frame.meshletGroupBuffer);
        destroyBuffer(context, frame.drawMeshletBuffer);
        destroyBuffer(context, frame.drawBuffer);
        destroyBuffer(context, frame.instanceBuffer);
        destroyBuffer(context, frame.cameraBuffer);
        vkDestroySemaphore(context.device, frame.imageAvailable, nullptr);
        vkDestroyCommandPool(context.device, frame.commandPool, nullptr);
    }
    destroyBuffer(context, captureBuffer);
    for (Buffer& buffer : fluidSurfaceTables) {
        destroyBuffer(context, buffer);
    }
    for (VkImageView view : targets.hzbMipViews) {
        vkDestroyImageView(context.device, view, nullptr);
    }
    destroyImage(context, targets.hzb);
    for (VkImageView view : targets.bloomMipViews) {
        vkDestroyImageView(context.device, view, nullptr);
    }
    destroyImage(context, targets.bloom);
    for (VkImageView view : targets.shadowLayerViews) {
        vkDestroyImageView(context.device, view, nullptr);
    }
    destroyImage(context, targets.shadowAtlas);
    destroyImage(context, targets.ssao);
    destroyImage(context, targets.ssaoRaw);
    destroyImage(context, targets.pathAccumulation);
    destroyImage(context, targets.reflectionHistory[1]);
    destroyImage(context, targets.reflectionHistory[0]);
    destroyImage(context, targets.reflectionRaw);
    destroyImage(context, targets.guideDepth);
    destroyImage(context, targets.guideRoughness);
    destroyImage(context, targets.guideNormal);
    destroyImage(context, targets.guideSpecularAlbedo);
    destroyImage(context, targets.guideDiffuseAlbedo);
    destroyImage(context, targets.tonemapped);
    destroyImage(context, targets.upscaledColor);
    destroyImage(context, targets.present);
    destroyImage(context, targets.oitRevealage);
    destroyImage(context, targets.fluidThickness);
    destroyImage(context, targets.oitAccumulation);
    destroyImage(context, targets.velocity);
    destroyImage(context, targets.depth);
    destroyImage(context, targets.color);
    vkDestroySampler(context.device, linearSampler, nullptr);
    vkDestroySampler(context.device, postSampler, nullptr);
    swapchain.reset();
}

void Renderer::waitIdle() {
    VK_CHECK(vkDeviceWaitIdle(context.device));
    // 어느 제출도 남아 있지 않으니 맡아 둔 자원을 여기서 전부 돌려준다. 장면을 열거나 스왑체인을
    // 다시 만드는 자리가 모두 이 함수를 거친다.
    context.collectRetired();
}

void Renderer::setVsync(bool enabled) {
    if (enabled == vsync) {
        return;
    }
    vsync = enabled;
    resizeRequested = true;
}

VkFormat Renderer::swapchainFormat() const {
    return swapchain->format;
}

uint32_t Renderer::swapchainImageCount() const {
    return static_cast<uint32_t>(swapchain->images.size());
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

void Renderer::recreateSwapchain() {
    waitIdle();
    swapchain->recreate(vsync);
    createRenderTargets();
    // 이미지 개수가 달라질 수 있으므로 표시 완료 세마포어도 다시 만든다.
    createPresentSemaphores();
    resizeRequested = false;
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

    // 프레임의 패스를 그래프 노드로 짠다. 노드는 조건과 무관하게 늘 등록하고 enabled 가 실행을 정한다.
    // 이미지 레이아웃 전이는 노드의 reads/writes 선언에서 그래프가 만든다. 패스 안에서 스스로 전이하는
    // 것(층·밉 단위, 컴퓨트가 도로 돌려 놓는 것)은 leaves 로 남긴 상태를 알린다.
    //
    // 아래 지역 변수들은 노드가 참조로 잡으므로 execute 가 이 함수 안에서 끝나야 한다. 앞 노드가 정하고
    // 뒤 노드가 읽는 값(rayQueryPass, sceneLayout, 톤 매핑 푸시 상수 등)도 같은 이유로 지역 변수다.
    graph.clear();

    bool hasTranslucent = batches.draws[TRANSLUCENT_MODE][0].count + batches.draws[TRANSLUCENT_MODE][1].count > 0;
    bool pathTracing = settings.usePathTracing && rayTracer != nullptr;
    rayQueryPass = false;
    VkDescriptorSet bindlessSet = bindless.set();

    // 자주 쓰는 이미지 사용 꼴. 깊이 첨부물은 이른·늦은 조각 검사 둘 다에서 쓰이므로 두 단계를 함께 적는다.
    constexpr VkPipelineStageFlags2 DEPTH_STAGES =
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
    auto colorWrite = [](const Image& image, bool discard) {
        return ImageUse{image.handle,
                        VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                        discard ? VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT
                                : VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                        discard};
    };
    auto sampled = [](const Image& image, VkPipelineStageFlags2 stages) {
        return ImageUse{image.handle,
                        VK_IMAGE_ASPECT_COLOR_BIT,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        stages,
                        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT};
    };
    auto storage = [](const Image& image, VkPipelineStageFlags2 stages, VkAccessFlags2 access) {
        return ImageUse{image.handle, VK_IMAGE_ASPECT_COLOR_BIT, VK_IMAGE_LAYOUT_GENERAL, stages, access};
    };
    auto depthWrite = [&](bool discard) {
        return ImageUse{targets.depth.handle,
                        VK_IMAGE_ASPECT_DEPTH_BIT,
                        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                        DEPTH_STAGES,
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                        discard};
    };
    // 깊이를 첨부물로 묶되 검사만 하는 패스(하늘, OIT).
    ImageUse depthTest{targets.depth.handle,
                       VK_IMAGE_ASPECT_DEPTH_BIT,
                       VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
                       DEPTH_STAGES,
                       VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT};
    auto depthSampled = [&](VkPipelineStageFlags2 stages) {
        return ImageUse{targets.depth.handle,
                        VK_IMAGE_ASPECT_DEPTH_BIT,
                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        stages,
                        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT};
    };

    // ---- 첫 프레임 초기화. 한 번 돌고 표식을 내린다.
    // 첫 프레임에는 이전 깊이가 없으므로 아무것도 가리지 않도록 가장 먼 값으로 채운다.
    graph.add({"HZB 초기화",
               nullptr,
               [&] { return hzbNeedsClear; },
               {},
               {storage(targets.hzb, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT)},
               {},
               [&](VkCommandBuffer cmd) {
                   VkClearColorValue clear{};
                   VkImageSubresourceRange range{};
                   range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                   range.levelCount = VK_REMAINING_MIP_LEVELS;
                   range.layerCount = VK_REMAINING_ARRAY_LAYERS;
                   vkCmdClearColorImage(cmd, targets.hzb.handle, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);

                   VkMemoryBarrier2 clearBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
                   clearBarrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
                   clearBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                   clearBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                   clearBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                   VkDependencyInfo clearDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                   clearDependency.memoryBarrierCount = 1;
                   clearDependency.pMemoryBarriers = &clearBarrier;
                   vkCmdPipelineBarrier2(cmd, &clearDependency);
                   hzbNeedsClear = false;
               }});

    // 층 전체를 한 번 읽기 좋은 레이아웃으로 옮긴다. 이후 프레임은 층마다 따로 전이하므로 여기서 맞춰
    // 두지 않으면 한 번도 안 그린 층이 잘못된 레이아웃으로 남는다.
    graph.add({"그림자 초기화",
               nullptr,
               [&] { return shadowNeedsInit; },
               {ImageUse{targets.shadowAtlas.handle,
                         VK_IMAGE_ASPECT_DEPTH_BIT,
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                         VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                         VK_ACCESS_2_SHADER_SAMPLED_READ_BIT}},
               {},
               {},
               [&](VkCommandBuffer) { shadowNeedsInit = false; }});

    // 첫 프레임에는 이전 깊이가 없으므로 차폐 없음(1)으로 채운다.
    graph.add({"SSAO 초기화",
               nullptr,
               [&] { return ssaoNeedsClear; },
               {},
               {storage(targets.ssaoRaw, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT),
                storage(targets.ssao, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT)},
               {},
               [&](VkCommandBuffer cmd) {
                   for (const Image* image : {&targets.ssaoRaw, &targets.ssao}) {
                       VkClearColorValue clear{};
                       clear.float32[0] = 1.0F;
                       VkImageSubresourceRange range{};
                       range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                       range.levelCount = VK_REMAINING_MIP_LEVELS;
                       range.layerCount = VK_REMAINING_ARRAY_LAYERS;
                       vkCmdClearColorImage(cmd, image->handle, VK_IMAGE_LAYOUT_GENERAL, &clear, 1, &range);
                   }

                   VkMemoryBarrier2 clearBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
                   clearBarrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
                   clearBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                   clearBarrier.dstStageMask =
                       VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                   clearBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
                   VkDependencyInfo clearDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                   clearDependency.memoryBarrierCount = 1;
                   clearDependency.pMemoryBarriers = &clearBarrier;
                   vkCmdPipelineBarrier2(cmd, &clearDependency);
                   ssaoNeedsClear = false;
               }});

    // 톤 매핑 대상과 시간축 업스케일 대상은 고른 방식에 따라 한쪽만 쓰인다. 쓰이지 않는 쪽도 디버그 뷰어가
    // 읽으므로 처음 한 번 레이아웃을 맞춰 둔다.
    graph.add(
        {"후처리 초기화",
         nullptr,
         [&] { return postTargetsNeedInit; },
         {sampled(targets.tonemapped, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT),
          storage(targets.upscaledColor, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT)},
         {storage(targets.bloom, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
          storage(targets.reflectionRaw, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
          storage(targets.reflectionHistory[0],
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
          storage(targets.reflectionHistory[1],
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT)},
         {},
         [&](VkCommandBuffer) { postTargetsNeedInit = false; }});

    // ---- 프레임 공통 준비.
    // 환경 맵은 설정이 바뀔 때만 다시 굽는다. 래스터와 경로 추적이 같은 환경을 본다.
    graph.add({"환경", "환경", {}, {}, {}, {}, [&](VkCommandBuffer cmd) {
                   if (environment->update(cmd, scene.environment, sunDirection)) {
                       // 하늘이 바뀌었으면 쌓아 둔 경로 추적 표본은 옛 환경의 것이다.
                       pathSampleCount = 0;
                   }
               }});

    // 변형 정점은 그림자·장면·광선 경로가 모두 읽으므로 맨 먼저 만든다.
    graph.add({"스킨", nullptr, {}, {}, {}, {}, [&](VkCommandBuffer cmd) { recordSkinPass(cmd, frame); }});

    // 유체 입자 인스턴스도 같은 이유로 그림자보다 앞이다. 광선 기능이 켜져 있고 구의 하위 구조가 있으면
    // 상위 가속 구조 인스턴스도 함께 써 둔다. ensureBottomLevel 은 drawFrame 이 이미 불렀다.
    fluidTlasPrepended = 0;
    graph.add({"유체",
               "유체",
               [&] { return batches.fluidDraws.count > 0; },
               {},
               {},
               {},
               [&](VkCommandBuffer cmd) {
                   bool wantsTlas = rayTracer != nullptr && rayTracer->bottomLevelReady() &&
                                    (pathTracing || ((settings.useRayQueryShadows || reflectionsActive()) &&
                                                     rayQueryShadowsAvailable()));
                   recordFluidPass(cmd, frame, batches, scene, wantsTlas);
               }});

    // 그림자 아틀라스는 장면 패스보다 먼저 채워야 한다. 경로 추적은 아틀라스를 읽지 않으므로 건너뛴다.
    // 층 단위 전이는 패스 안에 있고 아틀라스는 읽기 전용으로 돌아온다.
    graph.add({"그림자",
               "그림자",
               [&] { return !pathTracing && !shadowViews.empty(); },
               {},
               {},
               {},
               [&](VkCommandBuffer cmd) { recordShadowPass(cmd); }});

    // ---- 경로 추적 경로. 모션 벡터와 깊이는 광선 생성 셰이더가 직접 쓰고 읽기 전용으로 남긴다.
    graph.add({"경로 추적",
               "경로 추적",
               [&] { return pathTracing; },
               {},
               {},
               {sampled(targets.velocity, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT),
                depthSampled(VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)},
               [&](VkCommandBuffer cmd) { recordPathTracePass(cmd, frame, scene); }});

    // ---- 래스터 경로. 오클루전 컬링은 두 패스로 돈다. 1차는 지난 프레임 가시 집합, HZB 구축, 2차는 나머지.
    // 고전 경로는 GPU 컬링이 없어 한 패스다.
    bool computeCullPath = settings.useComputeCulling && !useMeshPath();
    bool twoPhase = settings.occlusionCulling && (useMeshPath() || computeCullPath);
    uint32_t firstPhase = twoPhase ? CULL_PHASE_FIRST : CULL_PHASE_NONE;
    auto raster = [&] { return !pathTracing; };
    auto rasterTwoPhase = [&] { return !pathTracing && twoPhase; };

    // 가시성 비트는 프레임을 넘어 살아남는다. 지난 프레임 2차 패스의 쓰기가 이번 읽기에 앞서고,
    // 새로 잡은 버퍼는 0 으로 채운다.
    graph.add({"가시성 비트", nullptr, raster, {}, {}, {}, [&](VkCommandBuffer cmd) {
                   VkMemoryBarrier2 bitsBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
                   bitsBarrier.srcStageMask =
                       VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                   bitsBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                   bitsBarrier.dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
                                              VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                   bitsBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                               VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                   VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
                   dependency.memoryBarrierCount = 1;
                   dependency.pMemoryBarriers = &bitsBarrier;
                   vkCmdPipelineBarrier2(cmd, &dependency);
                   if (visibilityNeedsClear) {
                       vkCmdFillBuffer(cmd, meshletVisibilityBuffer.handle, 0, VK_WHOLE_SIZE, 0);
                       bitsBarrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
                       bitsBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                       vkCmdPipelineBarrier2(cmd, &dependency);
                       visibilityNeedsClear = false;
                   }
               }});

    // mesh shader 경로는 태스크 셰이더가 직접 컬링한다. 컬 컴퓨트 결과를 아무도 읽지 않으므로
    // 그때는 디스패치 자체를 하지 않는다.
    graph.add({"컬링",
               "컬링",
               [&] { return !pathTracing && computeCullPath; },
               {},
               {},
               {},
               [&](VkCommandBuffer cmd) { recordCullPass(cmd, batches, firstPhase); }});

    // 고전 경로의 명령별 meshlet 은 컴퓨트 컬링이면 컬 셰이더가, 아니면 CPU 가 채운 것을 쓴다.
    ScenePushConstants scenePushConstants{geometry.vertexBuffer.address,
                                          geometry.meshBuffer.address,
                                          frame.instanceBuffer.address,
                                          geometry.materialBuffer.address,
                                          frame.cameraBuffer.address,
                                          geometry.meshletBuffer.address,
                                          geometry.meshletTriangleBuffer.address,
                                          geometry.meshletVertexBuffer.address,
                                          computeCullPath ? frame.meshletDrawMeshletBuffer.address
                                                          : frame.drawMeshletBuffer.address,
                                          frame.meshletGroupBuffer.address,
                                          skinnedVertexBuffer.address,
                                          skinnedBoundsBuffer.address,
                                          frame.lightBuffer.address,
                                          frame.shadowMatrixBuffer.address,
                                          0,
                                          firstPhase,
                                          meshletVisibilityBuffer.address};
    VkPipelineLayout sceneLayout = meshPipelineLayout;

    VkRenderingAttachmentInfo depthAttachment{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    depthAttachment.imageView = targets.depth.view;
    depthAttachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depthAttachment.clearValue.depthStencil.depth = 0.0F;

    auto bindScene = [&](VkCommandBuffer cmd) {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sceneLayout, 0, 1, &bindlessSet, 0, nullptr);
        if (rayQueryPass) {
            VkDescriptorSet accelerationSet = rayTracer->accelerationSet();
            vkCmdBindDescriptorSets(
                cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, sceneLayout, 1, 1, &accelerationSet, 0, nullptr);
        }
        vkCmdPushConstants(cmd, sceneLayout, scenePushStages, 0, sizeof(scenePushConstants), &scenePushConstants);
    };
    auto drawOpaque = [&](VkCommandBuffer cmd, VkAttachmentLoadOp loadOp, uint32_t phase) {
        std::array<VkRenderingAttachmentInfo, 4> opaqueColor{
            colorAttachment(targets.color.view, loadOp, {{0.05F, 0.05F, 0.07F, 1.0F}}),
            colorAttachment(targets.velocity.view, loadOp, {{0.0F, 0.0F, 0.0F, 0.0F}}),
            colorAttachment(targets.guideNormal.view, loadOp, {{0.0F, 0.0F, 0.0F, 0.0F}}),
            colorAttachment(targets.guideSpecularAlbedo.view, loadOp, {{0.0F, 0.0F, 0.0F, 0.0F}})};
        VkRenderingAttachmentInfo depth = depthAttachment;
        depth.loadOp = loadOp;

        VkRenderingInfo opaquePass{VK_STRUCTURE_TYPE_RENDERING_INFO};
        opaquePass.renderArea.extent = currentRenderExtent;
        opaquePass.layerCount = 1;
        opaquePass.colorAttachmentCount = static_cast<uint32_t>(opaqueColor.size());
        opaquePass.pColorAttachments = opaqueColor.data();
        opaquePass.pDepthAttachment = &depth;

        vkCmdBeginRendering(cmd, &opaquePass);
        setFullViewport(cmd, currentRenderExtent);
        recordGeometryPass(cmd, batches, false, phase);
        vkCmdEndRendering(cmd);
    };

    // 1) 불투명과 컷오프 경로를 HDR 색상 대상에 그린다. 두 패스 컬링이면 1차 뒤에 HZB 를 만들고
    //    2차가 같은 첨부물에 이어 그린다. 아무것도 그리지 않은 화소는 변위 0 이다. 하늘 패스가 나중에
    //    그 자리를 채운다. 색상·모션 벡터·깊이·노멀·반사 가중치는 프레임마다 처음부터 채운다(discard).
    graph.add({"불투명",
               "불투명",
               raster,
               {},
               {colorWrite(targets.color, true),
                colorWrite(targets.velocity, true),
                colorWrite(targets.guideNormal, true),
                colorWrite(targets.guideSpecularAlbedo, true),
                depthWrite(true)},
               {},
               [&](VkCommandBuffer cmd) {
                   // 광선 질의 그림자나 반사를 쓰면 TLAS 를 집합 1 로 함께 묶고, 장면이 바뀌었으면 먼저 다시
                   // 만든다. 반사만 켜도 광선 질의 변종 프래그먼트가 돌지만, ambient.w 가 0 이라 그림자는
                   // 그림자 맵을 그대로 쓴다.
                   rayQueryPass = (settings.useRayQueryShadows || reflectionsActive()) && rayQueryShadowsAvailable();
                   if (rayQueryPass) {
                       updateAccelerationStructures(cmd, scene);
                       rayQueryPass = rayTracer->ready();
                   }
                   sceneLayout = rayQueryPass ? meshRayQueryPipelineLayout : meshPipelineLayout;
                   bindScene(cmd);
                   vkCmdBindIndexBuffer(cmd, geometry.indexBuffer.handle, 0, VK_INDEX_TYPE_UINT32);
                   drawOpaque(cmd, VK_ATTACHMENT_LOAD_OP_CLEAR, firstPhase);
               }});

    // 1차 깊이로 HZB 를 만든다. 지난 프레임에 보였던 것만 그렸으므로 가리개가 적어 보수적이다.
    graph.add({"HZB",
               "HZB",
               rasterTwoPhase,
               {depthSampled(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)},
               {},
               {},
               [&](VkCommandBuffer cmd) { recordHzbPass(cmd); }});

    // 2) 나머지 중 HZB 로 보이는 것만 이어 그린다. 대개 비어 있거나 가장자리 몇 개다.
    graph.add({"컬링 2차",
               "컬링 2차",
               [&] { return !pathTracing && twoPhase && computeCullPath; },
               {},
               {},
               {},
               [&](VkCommandBuffer cmd) { recordCullPass(cmd, batches, CULL_PHASE_SECOND); }});
    graph.add({"불투명 2차",
               "불투명 2차",
               rasterTwoPhase,
               {},
               {colorWrite(targets.color, false),
                colorWrite(targets.velocity, false),
                colorWrite(targets.guideNormal, false),
                colorWrite(targets.guideSpecularAlbedo, false),
                depthWrite(false)},
               {},
               [&](VkCommandBuffer cmd) { drawOpaque(cmd, VK_ATTACHMENT_LOAD_OP_LOAD, CULL_PHASE_SECOND); }});

    // 불투명이 끝났으니 노멀·거칠기와 반사 가중치는 읽기 전용이다. 반사 컴퓨트와 디버그 뷰가 읽는다.
    graph.add({"가이드 읽기 전용",
               nullptr,
               raster,
               {sampled(targets.guideNormal, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT),
                sampled(targets.guideSpecularAlbedo, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)},
               {},
               {},
               [](VkCommandBuffer) {}});

    // 2) 아무것도 그려지지 않은 화소를 하늘로 채운다. 톤 매핑이 아니라 여기서 채워야 시간축
    //    업스케일러가 하늘까지 함께 누적하고, 반투명도 하늘 위에 합성된다.
    graph.add(
        {"하늘",
         "하늘",
         [&] { return !pathTracing && settings.useIbl && environment->ready(); },
         {depthTest},
         {colorWrite(targets.color, false), colorWrite(targets.velocity, false)},
         {},
         [&](VkCommandBuffer cmd) {
             std::array<VkRenderingAttachmentInfo, 2> skyColor{
                 colorAttachment(targets.color.view, VK_ATTACHMENT_LOAD_OP_LOAD, {}),
                 colorAttachment(targets.velocity.view, VK_ATTACHMENT_LOAD_OP_LOAD, {})};
             VkRenderingAttachmentInfo skyDepth = depthAttachment;
             skyDepth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
             skyDepth.storeOp = VK_ATTACHMENT_STORE_OP_NONE;

             VkRenderingInfo skyPass{VK_STRUCTURE_TYPE_RENDERING_INFO};
             skyPass.renderArea.extent = currentRenderExtent;
             skyPass.layerCount = 1;
             skyPass.colorAttachmentCount = static_cast<uint32_t>(skyColor.size());
             skyPass.pColorAttachments = skyColor.data();
             skyPass.pDepthAttachment = &skyDepth;

             SkyPushConstants skyPushConstants{frame.cameraBuffer.address, environment->environmentSlot()};
             vkCmdBeginRendering(cmd, &skyPass);
             setFullViewport(cmd, currentRenderExtent);
             vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyPipeline);
             vkCmdBindDescriptorSets(
                 cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, postPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
             vkCmdPushConstants(
                 cmd, postPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(skyPushConstants), &skyPushConstants);
             vkCmdDraw(cmd, 3, 1, 0, 0);
             vkCmdEndRendering(cmd);
         }});

    // 2-2) 물 표면은 하늘 뒤다. 뒤에 있는 배경이 이미 색상 대상에 들어 있어야 미리 곱해진 알파로
    //      섞을 때 투과가 맞는다. 하늘 블록 «밖» 이라 IBL 을 꺼도 물이 사라지지 않는다.
    //      반사 가중치를 첨부물로 되돌려 쓰고 다시 읽기 전용으로 남긴다(패스 안에서 전이).
    graph.add({"유체 표면",
               nullptr,
               raster,
               {sampled(targets.guideSpecularAlbedo, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)},
               {},
               {sampled(targets.guideSpecularAlbedo, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT),
                colorWrite(targets.color, false),
                colorWrite(targets.velocity, false),
                depthWrite(false)},
               [&](VkCommandBuffer cmd) { recordFluidSurfacePass(cmd, frame, batches, scene); }});

    // 3) 광선 반사. 불투명 깊이·노멀로 추적해 색상에 더한다. 하늘이 먼저 채워져 있어야 반사가
    //    되짚는 히스토리와 색상이 맞고, 반투명은 이 위에 합성된다. 패스 안에서 깊이·색상을 스토리지로
    //    옮겼다가 첨부물로 되돌리고, 모션 벡터는 읽기 전용으로 남긴다.
    auto reflectionRuns = [&] { return !pathTracing && reflectionsActive() && rayQueryPass; };
    graph.add(
        {"반사",
         nullptr,
         reflectionRuns,
         {sampled(targets.guideNormal, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT),
          sampled(targets.guideSpecularAlbedo, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)},
         {},
         {sampled(targets.velocity, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT),
          colorWrite(targets.color, false),
          depthWrite(false)},
         [&](VkCommandBuffer cmd) { recordReflectionPass(cmd, frame); }});
    // 반사가 돌지 않은 프레임은 히스토리가 끊긴다.
    graph.add({"반사 히스토리 무효",
               nullptr,
               [&] { return !reflectionRuns(); },
               {},
               {},
               {},
               [&](VkCommandBuffer) { reflectionHistoryValid = false; }});

    // 2) 반투명은 누적과 잔여 투과율 대상에 순서 독립으로 기록한다.
    graph.add(
        {"OIT",
         "OIT",
         [&] { return !pathTracing && hasTranslucent; },
         {depthTest},
         {colorWrite(targets.oitAccumulation, true), colorWrite(targets.oitRevealage, true)},
         {},
         [&](VkCommandBuffer cmd) {
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

             // 하늘 패스가 후처리 배치로 집합 0 을 다시 묶었다. 장면 배치와 푸시 상수를 되돌린다.
             bindScene(cmd);

             vkCmdBeginRendering(cmd, &translucentPass);
             setFullViewport(cmd, currentRenderExtent);
             recordGeometryPass(cmd, batches, true, CULL_PHASE_NONE);
             vkCmdEndRendering(cmd);
         }});

    // 3) 누적 결과를 HDR 색상 위에 합성한다.
    graph.add({"OIT 합성",
               "OIT 합성",
               [&] { return !pathTracing && hasTranslucent; },
               {sampled(targets.oitAccumulation, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT),
                sampled(targets.oitRevealage, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)},
               {colorWrite(targets.color, false)},
               {},
               [&](VkCommandBuffer cmd) {
                   VkRenderingAttachmentInfo compositeColor =
                       colorAttachment(targets.color.view, VK_ATTACHMENT_LOAD_OP_LOAD, {});
                   VkRenderingInfo compositePass{VK_STRUCTURE_TYPE_RENDERING_INFO};
                   compositePass.renderArea.extent = currentRenderExtent;
                   compositePass.layerCount = 1;
                   compositePass.colorAttachmentCount = 1;
                   compositePass.pColorAttachments = &compositeColor;

                   CompositePushConstants compositePushConstants{targets.accumulationSlot, targets.revealageSlot};
                   vkCmdBeginRendering(cmd, &compositePass);
                   setFullViewport(cmd, currentRenderExtent);
                   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, compositePipeline);
                   vkCmdBindDescriptorSets(
                       cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, postPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
                   vkCmdPushConstants(cmd,
                                      postPipelineLayout,
                                      VK_SHADER_STAGE_FRAGMENT_BIT,
                                      0,
                                      sizeof(compositePushConstants),
                                      &compositePushConstants);
                   vkCmdDraw(cmd, 3, 1, 0, 0);
                   vkCmdEndRendering(cmd);
               }});

    // 4) 깊이는 이제 읽기 전용이다. SSAO 와 콜라이더 표시의 프래그먼트, 디버그 뷰가 읽는다. SSAO 결과는
    //    다음 프레임의 셰이딩이 쓴다.
    //
    // ponytail: 한 프레임 늦은 깊이라 카메라가 빠르게 움직이면 차폐가 살짝 밀린다. 정확히
    // 하려면 불투명 깊이 선행 패스를 넣고 같은 프레임 안에서 계산해야 한다.
    graph.add({"깊이 읽기 전용",
               nullptr,
               raster,
               {depthSampled(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)},
               {},
               {},
               [](VkCommandBuffer) {}});
    graph.add({"SSAO",
               "SSAO",
               [&] { return !pathTracing && settings.useSsao; },
               {},
               {},
               {},
               [&](VkCommandBuffer cmd) { recordSsaoPass(cmd, frame); }});

    // ---- 여기부터 두 경로가 다시 합쳐진다.
    // 5) 시간축 업스케일러는 톤 매핑 앞에서 선형 HDR 을 받아 표시 해상도로 늘린다. 노출과 톤
    //    곡선이 흔들려도 히스토리가 따라 흔들리지 않으려면 이 순서여야 한다. 공간 업스케일은
    //    톤 매핑 뒤에서 도므로 여기서 두 경로가 갈린다.
    // 모션 벡터와 색상은 읽기 전용이어야 한다. 반사 패스나 경로 추적이 이미 옮겼으면 배리어가 나가지
    // 않는다. 톤 매핑(프래그먼트)뿐 아니라 Bloom 다운샘플과 자동 노출 히스토그램(컴퓨트)도 색상을 읽는다.
    graph.add(
        {"후처리 입력 준비",
         nullptr,
         {},
         {sampled(targets.velocity, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT),
          sampled(targets.color, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT)},
         {},
         {},
         [](VkCommandBuffer) {}});

    // 경로 추적은 누적 버퍼가 이미 표본을 쌓고 있어 시간축 업스케일과 겹친다. 지터도 꺼져 있다.
    // Ray Reconstruction 만 예외다. 누적 대신 1표본을 받아 스스로 디노이즈한다.
    bool rayReconstruction = rayReconstructionActive();
    bool temporalUpscale = temporalReady() && (!pathTracing || rayReconstruction);

    TonemapPushConstants tonemapPushConstants{};
    tonemapPushConstants.colorTexture = pathTracing ? targets.pathAccumulationSampledSlot : targets.colorSlot;
    tonemapPushConstants.exposure = settings.exposure;
    tonemapPushConstants.camera = frame.cameraBuffer.address;

    // 지난 프레임 톤 매핑이 아직 읽고 있을 수 있다. 덮어쓰기 전에 그 읽기를 끝낸다(그래프가 상태를 안다).
    graph.add(
        {"시간축 업스케일",
         "시간축 업스케일",
         [&] { return temporalUpscale; },
         {},
         {storage(targets.upscaledColor, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT)},
         {},
         [&](VkCommandBuffer cmd) {
             UpscaleInputs inputs{};
             // RR 은 경로 추적의 1표본 색상과 안내 버퍼를 받는다. 깊이도 경로 추적이 따로 적은 것을 쓴다.
             inputs.color = rayReconstruction ? &targets.pathAccumulation : &targets.color;
             inputs.depth = rayReconstruction ? &targets.guideDepth : &targets.depth;
             if (rayReconstruction) {
                 inputs.guideDiffuseAlbedo = &targets.guideDiffuseAlbedo;
                 inputs.guideSpecularAlbedo = &targets.guideSpecularAlbedo;
                 inputs.guideNormal = &targets.guideNormal;
                 inputs.guideRoughness = &targets.guideRoughness;
                 inputs.guideDepth = &targets.guideDepth;
             }
             inputs.velocity = &targets.velocity;
             inputs.output = &targets.upscaledColor;
             inputs.colorTexture = targets.colorSlot;
             inputs.depthTexture = targets.depthSlot;
             inputs.velocityTexture = targets.velocitySlot;
             inputs.outputStorage = targets.upscaledColorStorageSlot;
             inputs.bindlessSet = bindlessSet;
             inputs.jitter = currentJitter;
             inputs.deltaSeconds = frameDeltaSeconds;
             inputs.nearPlane = scene.camera.nearPlane;
             inputs.verticalFovRadians = glm::radians(scene.camera.fovYDegrees);
             inputs.reset = temporalResetThisFrame;
             temporalUpscaler->evaluate(cmd, inputs);

             tonemapPushConstants.colorTexture = targets.upscaledColorSlot;
         }});

    // Bloom 과 자동 노출은 톤 매핑이 읽을 바로 그 이미지에서 만든다. 세 갈래(HDR 색상, 시간축
    // 업스케일 결과, 경로 추적 누적) 어느 쪽이든 같은 자리에서 같은 결과를 낸다.
    graph.add({"후처리", nullptr, {}, {}, {}, {}, [&](VkCommandBuffer cmd) {
                   // 경로 추적 노드가 이번 프레임 표본을 더한 뒤의 값을 써야 한다. 그래프를 짤 때 잡으면 하나 모자란다.
                   tonemapPushConstants.sampleCount = pathTracing ? pathSampleCount : 0U;
                   recordPostEffects(cmd,
                                     scene.post,
                                     tonemapPushConstants.colorTexture,
                                     temporalUpscale ? currentDisplayExtent : currentRenderExtent,
                                     tonemapPushConstants.sampleCount);
                   tonemapPushConstants.exposureBuffer = exposureBuffer.address;
                   tonemapPushConstants.bloomTexture = targets.bloomSampledSlots[0];
                   tonemapPushConstants.bloomIntensity = scene.post.bloomIntensity;
                   tonemapPushConstants.autoExposure = scene.post.autoExposure ? 1U : 0U;
               }});

    // 시간축 경로는 이미 표시 해상도라 톤 매핑이 곧바로 표시 이미지를 채운다. 그렇지 않으면
    // 렌더 해상도 톤 매핑 결과를 만들고 공간 업스케일이 확대한다.
    const Image& tonemapTarget = temporalUpscale ? targets.present : targets.tonemapped;
    VkExtent2D tonemapExtent = temporalUpscale ? currentDisplayExtent : currentRenderExtent;
    std::vector<ImageUse> tonemapReads;
    if (temporalUpscale) {
        tonemapReads.push_back(storage(
            targets.upscaledColor, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT));
    }
    graph.add({"톤 매핑",
               "톤 매핑",
               {},
               std::move(tonemapReads),
               {colorWrite(tonemapTarget, true)},
               {},
               [&](VkCommandBuffer cmd) {
                   VkRenderingAttachmentInfo tonemappedColor =
                       colorAttachment(tonemapTarget.view, VK_ATTACHMENT_LOAD_OP_DONT_CARE, {});
                   VkRenderingInfo tonemapPass{VK_STRUCTURE_TYPE_RENDERING_INFO};
                   tonemapPass.renderArea.extent = tonemapExtent;
                   tonemapPass.layerCount = 1;
                   tonemapPass.colorAttachmentCount = 1;
                   tonemapPass.pColorAttachments = &tonemappedColor;

                   vkCmdBeginRendering(cmd, &tonemapPass);
                   setFullViewport(cmd, tonemapExtent);
                   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, tonemapPipeline);
                   vkCmdBindDescriptorSets(
                       cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, postPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
                   vkCmdPushConstants(cmd,
                                      postPipelineLayout,
                                      VK_SHADER_STAGE_FRAGMENT_BIT,
                                      0,
                                      sizeof(tonemapPushConstants),
                                      &tonemapPushConstants);
                   vkCmdDraw(cmd, 3, 1, 0, 0);
                   vkCmdEndRendering(cmd);
               }});

    // 6) 렌더 해상도의 톤 매핑 결과를 표시 해상도로 확대한다.
    graph.add({"업스케일",
               "업스케일",
               [&] { return !temporalUpscale; },
               {sampled(targets.tonemapped, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)},
               {colorWrite(targets.present, true)},
               {},
               [&](VkCommandBuffer cmd) {
                   VkRenderingAttachmentInfo upscaleColor =
                       colorAttachment(targets.present.view, VK_ATTACHMENT_LOAD_OP_DONT_CARE, {});
                   VkRenderingInfo upscalePass{VK_STRUCTURE_TYPE_RENDERING_INFO};
                   upscalePass.renderArea.extent = currentDisplayExtent;
                   upscalePass.layerCount = 1;
                   upscalePass.colorAttachmentCount = 1;
                   upscalePass.pColorAttachments = &upscaleColor;

                   UpscalePushConstants upscalePushConstants{};
                   upscalePushConstants.sourceTexture = targets.tonemappedSlot;
                   upscalePushConstants.sharpness = settings.upscaleSharpness;
                   upscalePushConstants.sourceSize[0] = static_cast<float>(currentRenderExtent.width);
                   upscalePushConstants.sourceSize[1] = static_cast<float>(currentRenderExtent.height);
                   upscalePushConstants.destinationSize[0] = static_cast<float>(currentDisplayExtent.width);
                   upscalePushConstants.destinationSize[1] = static_cast<float>(currentDisplayExtent.height);

                   size_t upscaleVariant = settings.upscaler == Upscaler::SPATIAL ? 1 : 0;

                   vkCmdBeginRendering(cmd, &upscalePass);
                   setFullViewport(cmd, currentDisplayExtent);
                   vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, upscalePipelines[upscaleVariant]);
                   vkCmdBindDescriptorSets(
                       cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, postPipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
                   vkCmdPushConstants(cmd,
                                      postPipelineLayout,
                                      VK_SHADER_STAGE_FRAGMENT_BIT,
                                      0,
                                      sizeof(upscalePushConstants),
                                      &upscalePushConstants);
                   vkCmdDraw(cmd, 3, 1, 0, 0);
                   vkCmdEndRendering(cmd);
               }});

    // 7) 콜라이더 표시. 톤 매핑과 업스케일이 끝난 표시 해상도에 덧그린다. 여기서 그려야 노출과
    // 업스케일이 색을 흔들지 않고, 깊이 버퍼를 텍스처로 읽어 물체 뒤로 숨을 수 있다.
    graph.add({"디버그 선",
               nullptr,
               [&] { return settings.showColliders; },
               {},
               {},
               {},
               [&](VkCommandBuffer cmd) { recordDebugLines(cmd, frame, scene, currentDisplayExtent); }});

    // 8) 편집기 UI 를 스왑체인에 그린다. 표시 대상은 UI 가 샘플링할 수 있는 레이아웃으로 옮긴다.
    // 스왑체인 이미지는 프레임마다 새로 받아 그래프가 추적하지 않는다. 노드 안에서 전이한다.
    graph.add({"표시 준비",
               nullptr,
               {},
               {sampled(targets.present, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)},
               {},
               {},
               [&](VkCommandBuffer cmd) {
                   imageBarrier(cmd,
                                swapchain->images[imageIndex],
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                VK_IMAGE_LAYOUT_UNDEFINED,
                                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                                0,
                                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
               }});

    // 표시 대상 캡처. UI 가 없는 렌더 결과만 떠서 실행 간 바이트 비교가 된다. 복사 뒤 UI 가 샘플링할 수
    // 있게 되돌린다.
    graph.add({"표시 캡처",
               nullptr,
               [&] { return !capturePath.empty() && capturePresent; },
               {ImageUse{targets.present.handle,
                         VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                         VK_PIPELINE_STAGE_2_COPY_BIT,
                         VK_ACCESS_2_TRANSFER_READ_BIT}},
               {},
               {sampled(targets.present, VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT)},
               [&](VkCommandBuffer cmd) {
                   VkBufferImageCopy2 region{VK_STRUCTURE_TYPE_BUFFER_IMAGE_COPY_2};
                   region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                   region.imageSubresource.layerCount = 1;
                   region.imageExtent = {currentDisplayExtent.width, currentDisplayExtent.height, 1};
                   VkCopyImageToBufferInfo2 copyInfo{VK_STRUCTURE_TYPE_COPY_IMAGE_TO_BUFFER_INFO_2};
                   copyInfo.srcImage = targets.present.handle;
                   copyInfo.srcImageLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
                   copyInfo.dstBuffer = captureBuffer.handle;
                   copyInfo.regionCount = 1;
                   copyInfo.pRegions = &region;
                   vkCmdCopyImageToBuffer2(cmd, &copyInfo);
                   imageBarrier(cmd,
                                targets.present.handle,
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                VK_PIPELINE_STAGE_2_COPY_BIT,
                                VK_ACCESS_2_TRANSFER_READ_BIT,
                                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
               }});

    graph.add({"UI", "UI", {}, {}, {}, {}, [&](VkCommandBuffer cmd) { recordUiPass(cmd, imageIndex); }});

    // 스왑체인 캡처(UI 포함)를 뜨고 표시 레이아웃으로 넘긴다. 캡처가 없으면 전이만 한다.
    graph.add({"스왑체인 캡처",
               nullptr,
               [&] { return !capturePath.empty() && !capturePresent; },
               {},
               {},
               {},
               [&](VkCommandBuffer cmd) {
                   imageBarrier(cmd,
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
                   vkCmdCopyImageToBuffer2(cmd, &copyInfo);

                   imageBarrier(cmd,
                                swapchain->images[imageIndex],
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                VK_PIPELINE_STAGE_2_COPY_BIT,
                                VK_ACCESS_2_TRANSFER_READ_BIT,
                                VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                                0);
               }});
    graph.add({"표시 전이",
               nullptr,
               [&] { return capturePath.empty() || capturePresent; },
               {},
               {},
               {},
               [&](VkCommandBuffer cmd) {
                   imageBarrier(cmd,
                                swapchain->images[imageIndex],
                                VK_IMAGE_ASPECT_COLOR_BIT,
                                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                                VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                                VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
                                0);
               }});

    // 플러그인의 패스. 앵커 뒤에 끼우므로 위 노드가 모두 등록된 뒤에 부른다.
    FrameInfo info{scene, frameIndex, static_cast<uint32_t>(frameIndex % FRAMES_IN_FLIGHT)};
    for (PassHook& hook : passHooks) {
        hook(graph, info);
    }

    graph.execute(commandBuffer, frameProfiler);

    frameProfiler.end(frameZone, commandBuffer);
    VK_CHECK(vkEndCommandBuffer(commandBuffer));
}

void Renderer::writeCapture() {
    VkExtent2D extent = capturePresent ? currentDisplayExtent : swapchain->extent;
    VkFormat format = capturePresent ? targets.present.format : swapchain->format;
    uint32_t width = extent.width;
    uint32_t height = extent.height;
    std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4);
    std::memcpy(pixels.data(), captureBuffer.mapped, pixels.size());

    bool isBgra = format == VK_FORMAT_B8G8R8A8_SRGB || format == VK_FORMAT_B8G8R8A8_UNORM;
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
        rayTracer->invalidateBottomLevel();
    }
    // 모델이 더해져 예산이 바뀌었으니 다음 요청 때 다시 재 본다.
    rayTracingBlockedReason.clear();
}

bool Renderer::ensureBottomLevel() {
    if (rayTracer == nullptr || !rayTracingBlockedReason.empty()) {
        return false;
    }
    if (rayTracer->bottomLevelReady()) {
        return true;
    }
    std::string reason;
    if (rayTracer->buildBottomLevel(reason)) {
        return true;
    }
    // 폴백은 두지 않는다. 광선 기능을 끄고 편집기에 사유를 보인다.
    rayTracingBlockedReason = reason;
    settings.usePathTracing = false;
    settings.useRayQueryShadows = false;
    settings.useReflections = false;
    spdlog::warn("광선 기능을 끕니다: {}", reason);
    return false;
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

    // 편집기가 방식을 바꿨으면 여기서 갈아 끼운다. 지터를 정하기 전이어야 한다.
    updateUpscaler();

    auto now = std::chrono::steady_clock::now();
    if (lastFrameTime.time_since_epoch().count() != 0) {
        // 창을 옮기거나 장치가 멈추면 간격이 크게 튄다. 업스케일러가 히스토리를 통째로 버리지
        // 않도록 한 자리에서 막아 둔다.
        frameDeltaSeconds = std::clamp(std::chrono::duration<float>(now - lastFrameTime).count(), 1e-4F, 0.1F);
    }
    if (fixedFrameDelta > 0.0F) {
        frameDeltaSeconds = fixedFrameDelta;
    }
    lastFrameTime = now;

    Frame& frame = frames[frameIndex % FRAMES_IN_FLIGHT];

    // 같은 프레임 자원을 다시 쓰기 전에 FRAMES_IN_FLIGHT 이전 프레임의 완료를 기다린다. 지난 프레임에
    // 맡긴 자원이 있으면(buildDrawCommands 에서 유체 상태가 사라지는 때) 바로 앞 프레임까지 기다려
    // 제출이 하나도 남지 않게 한 뒤 지운다. 자원이 사라지는 프레임은 드물어 그때만 한 프레임 값을 낸다.
    if (frameIndex > 0) {
        bool drain = context.hasRetired();
        if (drain || frameIndex >= FRAMES_IN_FLIGHT) {
            uint64_t waitValue = drain ? frameIndex : frameIndex - FRAMES_IN_FLIGHT + 1;
            VkSemaphoreWaitInfo waitInfo{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
            waitInfo.semaphoreCount = 1;
            waitInfo.pSemaphores = &frameTimeline;
            waitInfo.pValues = &waitValue;
            VK_CHECK(vkWaitSemaphores(context.device, &waitInfo, UINT64_MAX));
        }
        if (drain) {
            context.collectRetired();
        }
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
    // 광선 기능을 쓸 프레임이면 하위 가속 구조를 먼저 확보한다. 예산을 넘겨 못 세우면 여기서 광선
    // 기능이 꺼지므로 뒤의 모드 판정이 래스터로 떨어져 검은 프레임이 나오지 않는다.
    if (settings.usePathTracing || settings.useRayQueryShadows || settings.useReflections) {
        ensureBottomLevel();
    }
    FrameBatches batches = buildDrawCommands(frame, scene);
    frameProfiler.end(buildZone);

    if (!capturePath.empty()) {
        VkExtent2D captureExtent = capturePresent ? currentDisplayExtent : swapchain->extent;
        VkDeviceSize required = static_cast<VkDeviceSize>(captureExtent.width) * captureExtent.height * 4;
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
