// 경로 추적 패스.
// Renderer 의 멤버 함수 정의만 나눠 담은 번역 단위다. 선언은 renderer.h 하나에 있다.

#include "gfx/renderer_internal.h"

namespace gfx {

void Renderer::recordPathTracePass(VkCommandBuffer commandBuffer, Frame& frame, const scene::Scene& scene) {
    updateAccelerationStructures(commandBuffer, scene);
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
    if (settings.pathTrace.maxSamples == 0 || pathSampleCount < settings.pathTrace.maxSamples) {
        // 광선 생성 셰이더가 모든 화소를 덮어쓰므로 지난 내용은 버려도 된다.
        imageBarrier(commandBuffer,
                     targets.velocity.handle,
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_UNDEFINED,
                     VK_IMAGE_LAYOUT_GENERAL,
                     VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                     0,
                     VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        // RR 은 누적된 결과가 아니라 이번 프레임 1표본을 디노이즈한다. 표본 수를 0 으로 두면
        // 광선 생성 셰이더가 더하지 않고 덮어쓰고, 톤 매핑도 나누지 않는다.
        bool guides = rayReconstructionActive();
        // 표시용 디노이즈도 안내 노멀·깊이만 쓴다(누적은 그대로).
        bool writeGuides = guides || pathDenoiseThisFrame;
        PathGuideTargets guideTargets{};
        guideTargets.write = writeGuides;
        guideTargets.fixedJitter = guides;
        guideTargets.diffuseAlbedo = targets.guideDiffuseAlbedoStorageSlot;
        guideTargets.specularAlbedo = targets.guideSpecularAlbedoStorageSlot;
        guideTargets.normal = targets.guideNormalStorageSlot;
        guideTargets.roughness = targets.guideRoughnessStorageSlot;
        guideTargets.depth = targets.guideDepthStorageSlot;
        if (writeGuides) {
            for (const Image* image : {&targets.guideDiffuseAlbedo,
                                       &targets.guideSpecularAlbedo,
                                       &targets.guideNormal,
                                       &targets.guideRoughness,
                                       &targets.guideDepth}) {
                imageBarrier(commandBuffer,
                             image->handle,
                             VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_UNDEFINED,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
                             0,
                             VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            }
        }
        rayTracer->trace(commandBuffer,
                         currentRenderExtent,
                         frame.cameraBuffer.address,
                         frame.instanceBuffer.address,
                         frame.lightBuffer.address,
                         skinnedVertexBuffer.address,
                         fluidSurfaceTables[frameIndex % FRAMES_IN_FLIGHT].address,
                         targets.pathAccumulationStorageSlot,
                         targets.velocityStorageSlot,
                         static_cast<uint32_t>(frameIndex),
                         guides ? 0U : pathSampleCount,
                         settings.pathTrace,
                         guideTargets);
        if (guides) {
            for (const Image* image : {&targets.guideDiffuseAlbedo,
                                       &targets.guideSpecularAlbedo,
                                       &targets.guideNormal,
                                       &targets.guideRoughness,
                                       &targets.guideDepth}) {
                imageBarrier(commandBuffer,
                             image->handle,
                             VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            }
        } else {
            ++pathSampleCount;
        }
        if (pathDenoiseThisFrame) {
            // à-trous 가 안내 노멀·깊이를 샘플러로 읽는다. 다음 프레임은 UNDEFINED 에서 GENERAL 로 옮기며 통째로
            // 다시 쓰므로 여기 남긴 레이아웃은 되돌리지 않는다.
            for (const Image* image : {&targets.guideNormal, &targets.guideDepth}) {
                imageBarrier(commandBuffer,
                             image->handle,
                             VK_IMAGE_ASPECT_COLOR_BIT,
                             VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                             VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                             VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
            }
        }
        imageBarrier(commandBuffer,
                     targets.velocity.handle,
                     VK_IMAGE_ASPECT_COLOR_BIT,
                     VK_IMAGE_LAYOUT_GENERAL,
                     VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                     VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                     VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                     VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                     VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }

    imageBarrier(commandBuffer,
                 targets.pathAccumulation.handle,
                 VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_IMAGE_LAYOUT_GENERAL,
                 VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                 VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    if (pathDenoiseThisFrame) {
        // 누적을 표본 수로 나눠 세 번 누른다. 결과는 반사 필터 이미지에 남고 톤 매핑·Bloom 이 그것을 읽는다.
        ProfilerScope scope(frameProfiler, "경로 추적 디노이즈", commandBuffer);
        float scale = 1.0F / static_cast<float>(std::max(pathSampleCount, 1U));
        recordAtrous(commandBuffer,
                     frame.cameraBuffer.address,
                     targets.guideNormalSlot,
                     targets.guideDepthSlot,
                     {{targets.pathAccumulationSampledSlot, targets.reflectionFilteredStorageSlot, 1, false, scale},
                      {targets.reflectionFilteredSlot, targets.reflectionRawStorageSlot, 2, false, 1.0F},
                      {targets.reflectionRawSlot, targets.reflectionFilteredStorageSlot, 4, false, 1.0F}});
        memoryBarrier(commandBuffer,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
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

} // namespace gfx
