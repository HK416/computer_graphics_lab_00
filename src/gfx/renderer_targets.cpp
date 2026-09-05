// 렌더 대상·업스케일러 선택·디버그 뷰 대상 목록. 크기가 바뀔 때 대상을 통째로 다시 만든다.
// Renderer 의 멤버 함수 정의만 나눠 담은 번역 단위다. 선언은 renderer.h 하나에 있다.

#include "gfx/renderer_internal.h"

namespace gfx {

void Renderer::updateRenderExtent() {
    float scale = std::clamp(renderScale, 0.25F, 2.0F);
    // DLSS 는 렌더가 표시보다 크면 기능 생성이 InvalidParameter 로 거부된다. 배율을 1 로 자른다.
    if (upscaler == Upscaler::DLSS || upscaler == Upscaler::DLSS_RR) {
        scale = std::min(scale, 1.0F);
    }
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
    std::vector<UpscalerInfo> infos;
    for (Upscaler kind :
         {Upscaler::NONE, Upscaler::SPATIAL, Upscaler::TAAU, Upscaler::FSR, Upscaler::DLSS, Upscaler::DLSS_RR}) {
        infos.push_back(upscalerInfo(kind, context));
    }
    return infos;
}

void Renderer::updateUpscaler() {
    Upscaler wanted = effectiveUpscaler();
    if (wanted == activeUpscaler) {
        return;
    }
    // 파이프라인과 히스토리를 갈아 끼운다. 지난 프레임이 아직 쓰고 있을 수 있어 먼저 세운다.
    waitIdle();
    temporalUpscaler.reset();
    activeUpscaler = wanted;
    if (!isTemporal(wanted)) {
        return;
    }
    // DLSS 로 바꾸면 배율 상한이 달라진다. 컨텍스트를 만들기 전에 렌더 해상도를 그 상한에 맞춘다.
    updateRenderExtent();
    temporalUpscaler = createUpscaler(wanted, context, bindless);
    if (temporalUpscaler == nullptr) {
        // 편집기는 쓸 수 없는 방식을 고르지 못하게 하지만 실행 인자로는 들어올 수 있다.
        UpscalerInfo info = upscalerInfo(wanted, context);
        spdlog::warn("{} 을(를) 쓸 수 없어 내장 공간 업스케일로 돌아갑니다: {}", info.name, info.reason);
        upscaler = Upscaler::SPATIAL;
        activeUpscaler = upscaler;
        return;
    }
    temporalUpscaler->resize(currentRenderExtent, currentDisplayExtent);
}

std::vector<Renderer::TargetView> Renderer::targetViews() const {
    constexpr VkImageLayout READ_ONLY = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    constexpr VkImageLayout GENERAL = VK_IMAGE_LAYOUT_GENERAL;
    VkExtent2D render = currentRenderExtent;
    // 어느 대상이 이번 프레임에 채워지는지는 recordCommands 의 분기와 같아야 한다. 경로 추적은
    // 래스터 패스를 통째로 건너뛰어 깊이·모션 벡터·HZB 가 낡거나 다른 레이아웃에 남는다.
    bool pathTracing = usePathTracing && rayTracer != nullptr;
    bool temporal = temporalReady() && (!pathTracing || rayReconstructionActive());
    bool raster = !pathTracing;
    bool occlusion = raster && occlusionCulling && (useMeshPath() || useComputeCulling);
    std::vector<TargetView> views{
        {"표시 (Upscaling)", {targets.present.view}, READ_ONLY, currentDisplayExtent},
        {"색상 (HDR)", {targets.color.view}, READ_ONLY, render, nullptr, raster},
        {"경로 추적 누적", {targets.pathAccumulation.view}, GENERAL, render, nullptr, pathTracing},
        {"톤 매핑", {targets.tonemapped.view}, READ_ONLY, render, nullptr, !temporal},
        {"시간축 업스케일 (HDR)", {targets.upscaledColor.view}, GENERAL, currentDisplayExtent, nullptr, temporal},
        {"깊이", {targets.depth.view}, READ_ONLY, render, nullptr, raster},
        {"모션 벡터", {targets.velocity.view}, READ_ONLY, render, nullptr, raster},
        {"그림자 아틀라스",
         targets.shadowLayerViews,
         READ_ONLY,
         {SHADOW_MAP_SIZE, SHADOW_MAP_SIZE},
         "층",
         raster && shadowsEnabled && !shadowViews.empty()},
        {"SSAO", {targets.ssao.view}, GENERAL, targets.ssaoExtent, nullptr, raster && useSsao},
        // HZB 와 Bloom 은 컴퓨트가 쓰고 읽으므로 계속 GENERAL 이고, 밉을 골라 본다.
        {"HZB", targets.hzbMipViews, GENERAL, targets.hzbExtent, "밉", occlusion},
        {"Bloom", targets.bloomMipViews, GENERAL, targets.bloomExtent, "밉", bloomActive},
        {"반사 원본", {targets.reflectionRaw.view}, GENERAL, render, nullptr, reflectionsActive()},
    };
    if (graph.ran("OIT")) {
        views.push_back({"OIT 누적", {targets.oitAccumulation.view}, READ_ONLY, render, nullptr, raster});
        views.push_back({"OIT 잔여 투과율", {targets.oitRevealage.view}, READ_ONLY, render, nullptr, raster});
    }
    return views;
}

void Renderer::createRenderTargets() {
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

    ImageDesc velocityDesc;
    velocityDesc.extent = extent;
    velocityDesc.format = VELOCITY_FORMAT;
    velocityDesc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
    targets.velocity = createImage(context, velocityDesc, "모션 벡터");

    ImageDesc accumulationDesc = colorDesc;
    accumulationDesc.format = OIT_ACCUMULATION_FORMAT;
    targets.oitAccumulation = createImage(context, accumulationDesc, "OIT 누적");

    ImageDesc revealageDesc = colorDesc;
    revealageDesc.format = OIT_REVEALAGE_FORMAT;
    targets.oitRevealage = createImage(context, revealageDesc, "OIT 잔여 투과율");

    // 두께는 미터 단위 하나짜리 값이다. 가산 혼합으로 쌓으므로 첨부물이다.
    ImageDesc thicknessDesc;
    thicknessDesc.extent = extent;
    thicknessDesc.format = thicknessFormat;
    thicknessDesc.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    targets.fluidThickness = createImage(context, thicknessDesc, "물 두께");

    destroyImage(context, targets.reflectionRaw);
    destroyImage(context, targets.reflectionHistory[0]);
    destroyImage(context, targets.reflectionHistory[1]);
    ImageDesc reflectionDesc = colorDesc;
    reflectionDesc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    targets.reflectionRaw = createImage(context, reflectionDesc, "반사 원본");
    targets.reflectionHistory[0] = createImage(context, reflectionDesc, "반사 누적 0");
    targets.reflectionHistory[1] = createImage(context, reflectionDesc, "반사 누적 1");
    reflectionHistoryValid = false;

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
    // --capture present 가 이 이미지를 버퍼로 복사한다.
    presentDesc.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    targets.present = createImage(context, presentDesc, "표시");

    // 시간축 업스케일 결과. 아직 톤 매핑 전이라 선형 HDR 이고, 컴퓨트가 쓰므로 스토리지다.
    ImageDesc upscaledDesc;
    upscaledDesc.extent = presentDesc.extent;
    upscaledDesc.format = COLOR_FORMAT;
    // NGX 는 출력 이미지를 vkCmdClearColorImage 로 지우는 경로가 있어 전송 대상 자격이 필요하다.
    // 렌더 배율 1.0(DLAA)일 때만 타서 축소 배율 시험에서는 드러나지 않았다.
    upscaledDesc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    targets.upscaledColor = createImage(context, upscaledDesc, "시간축 업스케일 결과");

    // RR 안내 버퍼는 전부 렌더 해상도이고 광선 생성 셰이더가 스토리지로 쓴다. 깊이까지 여기 두는
    // 이유는 경로 추적이 깊이 첨부물을 쓰지 않아 읽을 깊이가 없기 때문이다.
    // 노멀과 반사 알베도는 래스터 불투명 패스도 첨부물로 채운다. 광선 반사 컴퓨트가 읽는다.
    ImageDesc guideDesc;
    guideDesc.extent = extent;
    guideDesc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    guideDesc.format = COLOR_FORMAT;
    targets.guideDiffuseAlbedo = createImage(context, guideDesc, "안내: 확산 알베도");
    targets.guideSpecularAlbedo = createImage(context, guideDesc, "안내: 반사 알베도");
    targets.guideNormal = createImage(context, guideDesc, "안내: 노멀");
    guideDesc.format = SSAO_FORMAT;
    targets.guideRoughness = createImage(context, guideDesc, "안내: 거칠기");
    targets.guideDepth = createImage(context, guideDesc, "안내: 깊이");

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

    for (VkImageView view : targets.bloomMipViews) {
        vkDestroyImageView(context.device, view, nullptr);
    }
    targets.bloomMipViews.clear();
    destroyImage(context, targets.bloom);

    targets.bloomExtent = targets.hzbExtent;
    uint32_t bloomLevels =
        std::min(BLOOM_MAX_LEVELS,
                 1U + static_cast<uint32_t>(
                          std::floor(std::log2(std::min(targets.bloomExtent.width, targets.bloomExtent.height)))));
    ImageDesc bloomDesc;
    bloomDesc.extent = {targets.bloomExtent.width, targets.bloomExtent.height, 1};
    bloomDesc.format = COLOR_FORMAT;
    bloomDesc.mipLevels = bloomLevels;
    bloomDesc.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    targets.bloom = createImage(context, bloomDesc, "Bloom");
    targets.bloomMipViews.resize(bloomLevels);
    for (uint32_t level = 0; level < bloomLevels; ++level) {
        VkImageViewCreateInfo viewInfo{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        viewInfo.image = targets.bloom.handle;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = COLOR_FORMAT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = level;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(context.device, &viewInfo, nullptr, &targets.bloomMipViews[level]));
    }

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
    // 대상을 전부 새로 만들었으니 그래프가 기억하는 레이아웃도 버린다.
    graph.resetStates();

    if (!targets.slotsAllocated) {
        targets.colorSlot = bindless.add(targets.color.view, postSampler);
        targets.accumulationSlot = bindless.add(targets.oitAccumulation.view, postSampler);
        targets.revealageSlot = bindless.add(targets.oitRevealage.view, postSampler);
        targets.fluidThicknessSlot = bindless.add(targets.fluidThickness.view, postSampler);
        targets.depthSlot = bindless.add(targets.depth.view, postSampler);
        targets.tonemappedSlot = bindless.add(targets.tonemapped.view, postSampler);
        targets.velocitySlot = bindless.add(targets.velocity.view, postSampler);
        targets.velocityStorageSlot = bindless.addStorageImageRg16(targets.velocity.view);
        // 컴퓨트가 쓰고 프래그먼트가 읽으므로 계속 GENERAL 에 둔다.
        targets.upscaledColorSlot = bindless.add(targets.upscaledColor.view, postSampler, VK_IMAGE_LAYOUT_GENERAL);
        targets.upscaledColorStorageSlot = bindless.addStorageImageRgba16(targets.upscaledColor.view);
        // HZB 는 컴퓨트가 쓰고 읽으므로 계속 GENERAL 레이아웃에 둔다.
        targets.hzbSampledSlot = bindless.add(targets.hzb.view, postSampler, VK_IMAGE_LAYOUT_GENERAL);
        targets.guideDiffuseAlbedoStorageSlot = bindless.addStorageImageRgba16(targets.guideDiffuseAlbedo.view);
        targets.guideSpecularAlbedoStorageSlot = bindless.addStorageImageRgba16(targets.guideSpecularAlbedo.view);
        targets.guideNormalStorageSlot = bindless.addStorageImageRgba16(targets.guideNormal.view);
        targets.guideNormalSlot = bindless.add(targets.guideNormal.view, postSampler);
        targets.guideSpecularAlbedoSlot = bindless.add(targets.guideSpecularAlbedo.view, postSampler);
        targets.guideRoughnessStorageSlot = bindless.addStorageImage(targets.guideRoughness.view);
        targets.guideDepthStorageSlot = bindless.addStorageImage(targets.guideDepth.view);
        targets.colorStorageSlot = bindless.addStorageImageRgba16(targets.color.view);
        targets.reflectionRawSlot = bindless.add(targets.reflectionRaw.view, postSampler, VK_IMAGE_LAYOUT_GENERAL);
        targets.reflectionRawStorageSlot = bindless.addStorageImageRgba16(targets.reflectionRaw.view);
        for (size_t i = 0; i < 2; ++i) {
            // 히스토리는 되짚은 자리를 이중 선형으로 읽으므로 선형 샘플러다.
            targets.reflectionHistorySlots[i] =
                bindless.add(targets.reflectionHistory[i].view, linearSampler, VK_IMAGE_LAYOUT_GENERAL);
            targets.reflectionHistoryStorageSlots[i] =
                bindless.addStorageImageRgba16(targets.reflectionHistory[i].view);
        }
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
        targets.bloomStorageSlots.resize(targets.bloomMipViews.size());
        targets.bloomSampledSlots.resize(targets.bloomMipViews.size());
        for (size_t level = 0; level < targets.bloomMipViews.size(); ++level) {
            targets.bloomStorageSlots[level] = bindless.addStorageImageRgba16(targets.bloomMipViews[level]);
            targets.bloomSampledSlots[level] =
                bindless.add(targets.bloomMipViews[level], linearSampler, VK_IMAGE_LAYOUT_GENERAL);
        }
        targets.slotsAllocated = true;
    } else {
        bindless.update(targets.colorSlot, targets.color.view, postSampler);
        bindless.update(targets.accumulationSlot, targets.oitAccumulation.view, postSampler);
        bindless.update(targets.revealageSlot, targets.oitRevealage.view, postSampler);
        bindless.update(targets.fluidThicknessSlot, targets.fluidThickness.view, postSampler);
        bindless.update(targets.depthSlot, targets.depth.view, postSampler);
        bindless.update(targets.tonemappedSlot, targets.tonemapped.view, postSampler);
        bindless.update(targets.velocitySlot, targets.velocity.view, postSampler);
        bindless.updateStorageImageRg16(targets.velocityStorageSlot, targets.velocity.view);
        bindless.update(targets.upscaledColorSlot, targets.upscaledColor.view, postSampler, VK_IMAGE_LAYOUT_GENERAL);
        bindless.updateStorageImageRgba16(targets.upscaledColorStorageSlot, targets.upscaledColor.view);
        bindless.update(targets.hzbSampledSlot, targets.hzb.view, postSampler, VK_IMAGE_LAYOUT_GENERAL);
        bindless.updateStorageImageRgba16(targets.guideDiffuseAlbedoStorageSlot, targets.guideDiffuseAlbedo.view);
        bindless.updateStorageImageRgba16(targets.guideSpecularAlbedoStorageSlot, targets.guideSpecularAlbedo.view);
        bindless.updateStorageImageRgba16(targets.guideNormalStorageSlot, targets.guideNormal.view);
        bindless.update(targets.guideNormalSlot, targets.guideNormal.view, postSampler);
        bindless.update(targets.guideSpecularAlbedoSlot, targets.guideSpecularAlbedo.view, postSampler);
        bindless.updateStorageImage(targets.guideRoughnessStorageSlot, targets.guideRoughness.view);
        bindless.updateStorageImage(targets.guideDepthStorageSlot, targets.guideDepth.view);
        bindless.updateStorageImageRgba16(targets.colorStorageSlot, targets.color.view);
        bindless.update(targets.reflectionRawSlot, targets.reflectionRaw.view, postSampler, VK_IMAGE_LAYOUT_GENERAL);
        bindless.updateStorageImageRgba16(targets.reflectionRawStorageSlot, targets.reflectionRaw.view);
        for (size_t i = 0; i < 2; ++i) {
            bindless.update(targets.reflectionHistorySlots[i],
                            targets.reflectionHistory[i].view,
                            linearSampler,
                            VK_IMAGE_LAYOUT_GENERAL);
            bindless.updateStorageImageRgba16(targets.reflectionHistoryStorageSlots[i],
                                              targets.reflectionHistory[i].view);
        }
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
        // Bloom 도 같다. 단계가 늘어나면 슬롯을 더 잡고, 줄어들면 남는 슬롯을 마지막 뷰로 채운다.
        for (size_t level = targets.bloomStorageSlots.size(); level < targets.bloomMipViews.size(); ++level) {
            targets.bloomStorageSlots.push_back(bindless.addStorageImageRgba16(targets.bloomMipViews[level]));
            targets.bloomSampledSlots.push_back(
                bindless.add(targets.bloomMipViews[level], linearSampler, VK_IMAGE_LAYOUT_GENERAL));
        }
        for (size_t level = 0; level < targets.bloomStorageSlots.size(); ++level) {
            size_t source = std::min(level, targets.bloomMipViews.size() - 1);
            bindless.updateStorageImageRgba16(targets.bloomStorageSlots[level], targets.bloomMipViews[source]);
            bindless.update(targets.bloomSampledSlots[level],
                            targets.bloomMipViews[source],
                            linearSampler,
                            VK_IMAGE_LAYOUT_GENERAL);
        }
    }
    hzbNeedsClear = true;
    ssaoNeedsClear = true;
    postTargetsNeedInit = true;
    exposureNeedsReset = true;
    pathSampleCount = 0;
    if (temporalUpscaler != nullptr) {
        temporalUpscaler->resize(currentRenderExtent, currentDisplayExtent);
    }
}

// 광선 질의는 가속 구조와 함께여야 쓸모가 있다. 이 저장소는 TLAS 를 RayTracer 가 들고 있으므로
// 경로 추적 파이프라인이 있는 장치에서만 켤 수 있다.
//
// ponytail: 광선 질의는 있는데 경로 추적 파이프라인이 없는 장치도 규격상 가능하다. 그런 장치까지
// 받으려면 가속 구조 관리만 RayTracer 에서 떼어내야 한다.
bool Renderer::reflectionsActive() const {
    return useReflections && rayQueryShadowsAvailable() && useIbl && environment != nullptr && environment->ready() &&
           !(usePathTracing && rayTracer != nullptr);
}

bool Renderer::rayQueryShadowsAvailable() const {
    return context.caps.rayQuery && rayTracer != nullptr && rayTracingBlockedReason.empty();
}

const char* Renderer::debugModeBlockedReason(uint32_t mode) const {
    if (usePathTracing && rayTracer != nullptr && !pathTraceSupportsDebugMode(mode)) {
        return "경로 추적에는 이 값이 없어 셰이딩으로 그린다";
    }
    if ((mode == DEBUG_MODE_REFLECTION_RAW || mode == DEBUG_MODE_REFLECTION) && !rayQueryShadowsAvailable()) {
        return "광선 질의가 없어 반사를 계산하지 않는다";
    }
    // mesh shader 경로는 meshlet 번호를 mesh 셰이더가 직접 넘기므로 gl_DrawID 가 필요 없다.
    if ((mode == DEBUG_MODE_MESHLET || mode == DEBUG_MODE_LOD) && !useMeshPath() && !context.caps.shaderDrawIndex) {
        return "gl_DrawID 가 없어 고전 경로는 meshlet 을 메쉬 단위로만 안다";
    }
    return nullptr;
}

} // namespace gfx
