// DDGI 프로브 GI 컴퓨트.
// Renderer 의 멤버 함수 정의만 나눠 담은 번역 단위다. 선언은 renderer.h 하나에 있다.

#include "gfx/renderer_internal.h"

namespace gfx {

namespace {

// shaders/lighting.glsl 의 PROBE_*_TEXELS 와 같아야 한다.
constexpr uint32_t PROBE_IRRADIANCE_TEXELS = 8;
constexpr uint32_t PROBE_VISIBILITY_TEXELS = 16;
constexpr uint32_t DDGI_GROUP_SIZE = 64;

} // namespace

bool Renderer::ddgiActive() const {
    return settings.useDdgi && rayQueryShadowsAvailable() && !settings.usePathTracing;
}

void Renderer::createDdgiPipelines() {
    if (!rayQueryShadowsAvailable()) {
        return;
    }
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.size = sizeof(DdgiPushConstants);
    std::array<VkDescriptorSetLayout, 2> sets{bindless.layout(), rayTracer->accelerationLayout()};
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = static_cast<uint32_t>(sets.size());
    layoutInfo.pSetLayouts = sets.data();
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &range;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &ddgiPipelineLayout));

    // 네 단계가 한 셰이더의 특수화 상수로 갈린다.
    VkShaderModule module = createShaderModule(context.device, "ddgi.comp.spv");
    uint32_t stage = 0;
    VkSpecializationMapEntry entry{0, 0, sizeof(uint32_t)};
    VkSpecializationInfo specialization{};
    specialization.mapEntryCount = 1;
    specialization.pMapEntries = &entry;
    specialization.dataSize = sizeof(stage);
    specialization.pData = &stage;
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = shaderStage(VK_SHADER_STAGE_COMPUTE_BIT, module);
    info.stage.pSpecializationInfo = &specialization;
    info.layout = ddgiPipelineLayout;
    for (stage = 0; stage < static_cast<uint32_t>(ddgiPipelines.size()); ++stage) {
        VK_CHECK(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &ddgiPipelines[stage]));
    }
    vkDestroyShaderModule(context.device, module, nullptr);
}

void Renderer::destroyDdgiResources() {
    destroyImage(context, targets.ddgiIrradiance);
    destroyImage(context, targets.ddgiVisibility);
    destroyBuffer(context, targets.ddgiResults);
}

void Renderer::ensureDdgiResources(uint32_t probesPerAxis, uint32_t raysPerProbe) {
    if (probesPerAxis == ddgiProbesPerAxis && raysPerProbe == ddgiRaysPerProbe &&
        targets.ddgiIrradiance.handle != VK_NULL_HANDLE) {
        return;
    }
    // 아틀라스 크기가 바뀐다. 지난 프레임이 아직 읽고 있을 수 있어 장치를 세운다. 설정을 바꿀 때만 일어난다.
    waitIdle();
    destroyDdgiResources();
    uint32_t total = probesPerAxis * probesPerAxis * probesPerAxis;
    auto makeAtlas = [&](uint32_t texels, VkFormat format, const char* name) {
        ImageDesc desc;
        desc.extent = {probesPerAxis * probesPerAxis * (texels + 2), probesPerAxis * (texels + 2), 1};
        desc.format = format;
        desc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        return createImage(context, desc, name);
    };
    targets.ddgiIrradiance = makeAtlas(PROBE_IRRADIANCE_TEXELS, COLOR_FORMAT, "DDGI 조도");
    targets.ddgiVisibility = makeAtlas(PROBE_VISIBILITY_TEXELS, VELOCITY_FORMAT, "DDGI 가시성");
    targets.ddgiResults = createBuffer(context,
                                       static_cast<VkDeviceSize>(total) * raysPerProbe * sizeof(glm::vec4),
                                       VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                       MemoryLocation::DEVICE,
                                       "DDGI 광선 결과");
    if (!ddgiSlotsAllocated) {
        ddgiSlotsAllocated = true;
        // 아틀라스는 이중 선형으로 읽는다. 테두리 텍셀이 있어 CLAMP_TO_EDGE 면 된다.
        targets.ddgiIrradianceSlot = bindless.add(targets.ddgiIrradiance.view, linearSampler, VK_IMAGE_LAYOUT_GENERAL);
        targets.ddgiVisibilitySlot = bindless.add(targets.ddgiVisibility.view, linearSampler, VK_IMAGE_LAYOUT_GENERAL);
        targets.ddgiIrradianceStorageSlot = bindless.addStorageImageRgba16(targets.ddgiIrradiance.view);
        targets.ddgiVisibilityStorageSlot = bindless.addStorageImageRg16(targets.ddgiVisibility.view);
    } else {
        bindless.update(
            targets.ddgiIrradianceSlot, targets.ddgiIrradiance.view, linearSampler, VK_IMAGE_LAYOUT_GENERAL);
        bindless.update(
            targets.ddgiVisibilitySlot, targets.ddgiVisibility.view, linearSampler, VK_IMAGE_LAYOUT_GENERAL);
        bindless.updateStorageImageRgba16(targets.ddgiIrradianceStorageSlot, targets.ddgiIrradiance.view);
        bindless.updateStorageImageRg16(targets.ddgiVisibilityStorageSlot, targets.ddgiVisibility.view);
    }
    ddgiProbesPerAxis = probesPerAxis;
    ddgiRaysPerProbe = raysPerProbe;
    ddgiAtlasInitialized = false;
    ddgiVolumeValid = false;
}

void Renderer::recordDdgiPass(VkCommandBuffer commandBuffer, const Frame& frame) {
    uint32_t zone = frameProfiler.begin("DDGI", commandBuffer);

    if (!ddgiAtlasInitialized) {
        // 새로 만든 아틀라스는 UNDEFINED 다. 한 번 GENERAL 로 옮기고 그대로 둔다.
        for (Image* image : {&targets.ddgiIrradiance, &targets.ddgiVisibility}) {
            imageBarrier(commandBuffer,
                         image->handle,
                         VK_IMAGE_ASPECT_COLOR_BIT,
                         VK_IMAGE_LAYOUT_UNDEFINED,
                         VK_IMAGE_LAYOUT_GENERAL,
                         VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                         0,
                         VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }
        ddgiAtlasInitialized = true;
    }

    auto memoryBarrier = [&](VkPipelineStageFlags2 sourceStage,
                             VkAccessFlags2 sourceAccess,
                             VkPipelineStageFlags2 destinationStage,
                             VkAccessFlags2 destinationAccess) {
        VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        barrier.srcStageMask = sourceStage;
        barrier.srcAccessMask = sourceAccess;
        barrier.dstStageMask = destinationStage;
        barrier.dstAccessMask = destinationAccess;
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependency);
    };
    // 지난 프레임의 래스터·컴퓨트가 아틀라스를 읽고 있었다. 이번 갱신이 그 뒤에 온다.
    constexpr VkPipelineStageFlags2 READER_STAGES =
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memoryBarrier(READER_STAGES,
                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    DdgiPushConstants pushConstants{};
    pushConstants.vertices = geometry.vertexBuffer.address;
    pushConstants.skinnedVertices = skinnedVertexBuffer.address;
    pushConstants.indices = geometry.indexBuffer.address;
    pushConstants.meshes = geometry.meshBuffer.address;
    pushConstants.instances = frame.instanceBuffer.address;
    pushConstants.materials = geometry.materialBuffer.address;
    pushConstants.lods = geometry.lodBuffer.address;
    pushConstants.camera = frame.cameraBuffer.address;
    pushConstants.lights = frame.lightBuffer.address;
    pushConstants.fluidSurfaces = fluidSurfaceTables[frameIndex % FRAMES_IN_FLIGHT].address;
    pushConstants.results = targets.ddgiResults.address;
    pushConstants.frameIndex = static_cast<uint32_t>(frameIndex);
    pushConstants.params = std::min(std::max(settings.restirCandidates, 1U), 255U) |
                           (ddgiResetThisFrame ? 1U << 8U : 0U) | (ddgiRaysPerProbe << 16U);
    pushConstants.atlasStorage =
        (targets.ddgiIrradianceStorageSlot & 0xFFFFU) | (targets.ddgiVisibilityStorageSlot << 16U);

    std::array<VkDescriptorSet, 2> sets{bindless.set(), rayTracer->accelerationSet()};
    vkCmdBindDescriptorSets(commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            ddgiPipelineLayout,
                            0,
                            static_cast<uint32_t>(sets.size()),
                            sets.data(),
                            0,
                            nullptr);
    vkCmdPushConstants(
        commandBuffer, ddgiPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);

    uint32_t total = ddgiProbesPerAxis * ddgiProbesPerAxis * ddgiProbesPerAxis;
    auto dispatch = [&](size_t stage, uint32_t threads) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ddgiPipelines[stage]);
        vkCmdDispatch(commandBuffer, (threads + DDGI_GROUP_SIZE - 1) / DDGI_GROUP_SIZE, 1, 1);
    };
    auto computeToCompute = [&] {
        memoryBarrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    };
    dispatch(0, total * ddgiRaysPerProbe);
    computeToCompute();
    // 조도와 가시성은 서로 다른 아틀라스를 쓰므로 배리어 없이 나란히 간다.
    dispatch(1, total * PROBE_IRRADIANCE_TEXELS * PROBE_IRRADIANCE_TEXELS);
    dispatch(2, total * PROBE_VISIBILITY_TEXELS * PROBE_VISIBILITY_TEXELS);
    computeToCompute();
    dispatch(3, total);
    // 래스터 프래그먼트와 반사·ReSTIR 컴퓨트가 이번 프레임 아틀라스를 읽는다.
    memoryBarrier(VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  READER_STAGES,
                  VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    frameProfiler.end(zone, commandBuffer);
}

} // namespace gfx
