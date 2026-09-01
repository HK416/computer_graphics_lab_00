#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include "gfx/resources.h"
#include "scene/scene.h"

namespace gfx {

struct Context;
class BindlessTextures;

// 환경 큐브맵과 그 조도·프리필터·BRDF 표를 컴퓨트로 굽는다. 굽기는 프레임 커맨드 버퍼에
// 그대로 기록하며, 설정이 바뀌지 않으면 아무것도 하지 않는다.
class EnvironmentMap {
public:
    EnvironmentMap(Context& context, BindlessTextures& bindless);
    ~EnvironmentMap();
    EnvironmentMap(const EnvironmentMap&) = delete;
    EnvironmentMap& operator=(const EnvironmentMap&) = delete;

    // 다시 구웠으면 true. HDR 파일 적재는 이 안에서 동기로 처리한다.
    bool update(VkCommandBuffer commandBuffer, const scene::Environment& desired, const glm::vec3& sunDirection);
    // 다음 update 에서 설정이 같아도 다시 굽는다.
    void invalidate() { baked = false; }

    bool ready() const { return baked; }
    uint32_t environmentSlot() const { return environmentCubeSlot; }
    uint32_t irradianceSlot() const { return irradianceCubeSlot; }
    uint32_t prefilterSlot() const { return prefilterCubeSlot; }
    uint32_t brdfSlot() const { return brdfArraySlot; }
    uint32_t prefilterMipCount() const;

private:
    void createImages();
    void createPipelines();
    // HDR 파일을 읽어 등정방형 텍스처로 올린다. 실패하면 절차적 하늘로 되돌린다.
    bool loadHdr(const std::filesystem::path& path);
    void bakeBrdf(VkCommandBuffer commandBuffer);

    Context& context;
    BindlessTextures& bindless;

    Image environmentCube;
    Image irradianceCube;
    Image prefilterCube;
    Image brdfLut;
    Image equirect;
    VkImageView environmentStorageView = VK_NULL_HANDLE;
    VkImageView irradianceStorageView = VK_NULL_HANDLE;
    std::vector<VkImageView> prefilterStorageViews;
    VkImageView brdfStorageView = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;

    uint32_t environmentCubeSlot = 0;
    uint32_t irradianceCubeSlot = 0;
    uint32_t prefilterCubeSlot = 0;
    uint32_t brdfArraySlot = 0;
    uint32_t environmentStorageSlot = 0;
    uint32_t irradianceStorageSlot = 0;
    std::vector<uint32_t> prefilterStorageSlots;
    uint32_t brdfStorageSlot = 0;
    uint32_t equirectSlot = 0xFFFFFFFFU;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline skyPipeline = VK_NULL_HANDLE;
    VkPipeline irradiancePipeline = VK_NULL_HANDLE;
    VkPipeline prefilterPipeline = VK_NULL_HANDLE;
    VkPipeline brdfPipeline = VK_NULL_HANDLE;

    scene::Environment current;
    glm::vec3 currentSun{0.0F};
    std::filesystem::path loadedHdr;
    bool baked = false;
    bool brdfBaked = false;
};

} // namespace gfx
