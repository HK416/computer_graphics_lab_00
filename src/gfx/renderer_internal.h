#pragma once

// Renderer 의 여러 번역 단위(renderer_*.cpp)가 함께 쓰는 푸시 상수 배치·포맷·작은 도우미. 바깥에서는 쓰지 않는다.

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
#include "gfx/renderer.h"
#include "gfx/swapchain.h"
#include "gfx/upscaler_math.h"
#include "gfx/vk_check.h"
#include "scene/scene.h"

namespace gfx {

// CPU 사본을 매핑된 버퍼로 붓는다. 쓰기 결합 메모리라 한 스레드로는 대역폭을 다 못 쓴다. 작은 배열은
// 워커를 깨우는 값이 더 크므로 그대로 복사한다.
template <typename T> void parallelCopy(core::JobSystem& jobs, const T* source, T* destination, size_t count) {
    // 나눌 값어치는 원소 수가 아니라 바이트 수로 정한다. 덩어리 경계는 캐시 줄의 배수여야 쓰기 결합
    // 버퍼가 반쪽만 차지 않는다.
    constexpr size_t PARALLEL_BYTES = 128 * 1024;
    constexpr size_t CHUNK_BYTES = 32 * 1024;
    if (count * sizeof(T) < PARALLEL_BYTES) {
        std::copy_n(source, count, destination);
        return;
    }
    auto granularity = static_cast<uint32_t>(std::max<size_t>(CHUNK_BYTES / sizeof(T), 1));
    jobs.parallelFor(static_cast<uint32_t>(count), granularity, [source, destination](uint32_t begin, uint32_t end) {
        std::copy_n(source + begin, end - begin, destination + begin);
    });
}

inline constexpr VkFormat COLOR_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
inline constexpr VkFormat DEPTH_FORMAT = VK_FORMAT_D32_SFLOAT;
inline constexpr VkFormat OIT_ACCUMULATION_FORMAT = VK_FORMAT_R16G16B16A16_SFLOAT;
inline constexpr VkFormat OIT_REVEALAGE_FORMAT = VK_FORMAT_R16_SFLOAT;
inline constexpr VkFormat PRESENT_FORMAT = VK_FORMAT_R8G8B8A8_UNORM;
// 모션 벡터. 화면 UV 단위라 절댓값이 1 을 넘지 않아 반정밀도로 충분하다.
inline constexpr VkFormat VELOCITY_FORMAT = VK_FORMAT_R16G16_SFLOAT;
// 물 두께. 뒷면 거리에서 앞면 거리를 «빼서» 얻는 값이라 두 값이 비슷하면 자리가 날아간다. 단정밀도를
// 쓰고 싶지만 R32 색상 혼합은 규격이 보장하지 않으므로 기동 시 조회해 안 되면 반정밀도로 내려간다.
inline constexpr VkFormat THICKNESS_FORMAT = VK_FORMAT_R32_SFLOAT;
inline constexpr VkFormat THICKNESS_FALLBACK_FORMAT = VK_FORMAT_R16_SFLOAT;
inline constexpr uint32_t MINIMUM_INSTANCE_CAPACITY = 1024;
// 그림자 시점의 근평면. 광원에 아주 가까운 물체는 그림자를 만들지 못한다.
inline constexpr float SHADOW_NEAR_PLANE = 0.05F;
inline constexpr VkFormat SHADOW_FORMAT = VK_FORMAT_D32_SFLOAT;
inline constexpr VkFormat SSAO_FORMAT = VK_FORMAT_R32_SFLOAT;
inline constexpr size_t TRANSLUCENT_MODE = 2;
// shaders/meshlet_task.glsl 의 MESHLET_GROUP_SIZE 와 같아야 한다.
inline constexpr uint32_t MESHLET_GROUP_SIZE = 32;

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
    // 지난 프레임의 시점 변환. 모션 벡터가 이전 위치를 되짚는다.
    glm::mat4 previousViewProjection;
    // xy 이번 프레임의 NDC 지터. viewProjection 에만 들어 있어 모션 벡터가 다시 빼야 한다.
    glm::vec4 jitter;
    // x: 조명 수, y: 그림자 아틀라스 슬롯, z: SSAO 슬롯, w: 아틀라스 한 변의 타일 수.
    // bindless 슬롯은 상위 8비트가 샘플러 번호라 float 로는 정확히 담기지 않는다.
    glm::uvec4 shading;
    // x: 조도 큐브 슬롯, y: 프리필터 큐브 슬롯, z: BRDF 표 슬롯, w: 프리필터 밉 수(0 이면 IBL 꺼짐).
    glm::uvec4 environment;
    // x: HZB 샘플 슬롯, y: 최대 밉 단계, zw: 0단계 크기.
    glm::uvec4 hzb;
    // 높이 안개. rgb 색, w 밀도. fogParameters 는 x 기준 높이, y 감쇠.
    glm::vec4 fog;
    glm::vec4 fogParameters;
    // x 디버그 뷰. 푸시 상수가 128 바이트에 꽉 차서 여기로 옮겼다.
    glm::uvec4 flags;
};

// shaders/scene_data.glsl 의 MeshletGroup 과 배치가 같아야 한다.
struct GpuMeshletGroup {
    uint32_t instanceIndex;
    uint32_t firstMeshlet;
    uint32_t meshletCount;
    uint32_t padding;
};

// shaders/hzb_reduce.comp 의 푸시 상수와 배치가 같아야 한다.
struct HzbPushConstants {
    uint32_t sourceTexture;
    uint32_t destinationStorage;
    int32_t sourceSize[2];
    int32_t destinationSize[2];
    // 원본 슬롯에서 읽을 밉 단계. 깊이 버퍼는 0, HZB 는 바로 위 단계다.
    float sourceLevel;
};

inline constexpr VkFormat HZB_FORMAT = VK_FORMAT_R32_SFLOAT;
inline constexpr VkFormat ACCUMULATION_FORMAT = VK_FORMAT_R32G32B32A32_SFLOAT;

struct CullPushConstants {
    VkDeviceAddress instances;
    VkDeviceAddress meshes;
    VkDeviceAddress meshlets;
    VkDeviceAddress camera;
    VkDeviceAddress drawCommands;
    VkDeviceAddress drawCounts;
    uint32_t instanceCount;
    uint32_t flags;
    uint32_t phase;
    VkDeviceAddress network;
    VkDeviceAddress skinnedBounds;
    VkDeviceAddress visibility;
    VkDeviceAddress drawMeshlets;
};

// shaders/skin.comp 의 푸시 상수와 배치가 같아야 한다.
struct SkinPushConstants {
    VkDeviceAddress source;
    VkDeviceAddress destination;
    VkDeviceAddress joints;
    VkDeviceAddress weights;
    uint32_t sourceOffset;
    uint32_t destinationOffset;
    uint32_t jointOffset;
    uint32_t vertexCount;
    uint32_t weightOffset;
};

// shaders/skin_bounds.comp 의 푸시 상수와 배치가 같아야 한다.
struct SkinBoundsPushConstants {
    VkDeviceAddress skinnedVertices;
    VkDeviceAddress meshlets;
    VkDeviceAddress bounds;
    uint32_t meshletOffset;
    uint32_t meshletCount;
    uint32_t meshVertexOffset;
    uint32_t skinnedVertexOffset;
    uint32_t boundsOffset;
    VkDeviceAddress meshletVertices;
};

// skin.comp 의 local_size_x 와 같아야 한다.
inline constexpr uint32_t SKIN_GROUP_SIZE = 64;

inline constexpr uint32_t CULL_FLAG_FRUSTUM = 1;
inline constexpr uint32_t CULL_FLAG_CONE = 2;
inline constexpr uint32_t CULL_FLAG_NEURAL_LOD = 8;
// shaders/culling.glsl 의 CULL_PHASE_* 와 같아야 한다.
inline constexpr uint32_t CULL_PHASE_NONE = 0;
inline constexpr uint32_t CULL_PHASE_FIRST = 1;
inline constexpr uint32_t CULL_PHASE_SECOND = 2;
// 학습 표본 수. 이보다 많으면 일정 간격으로 건너뛰며 뽑는다.
inline constexpr uint32_t MAX_LOD_SAMPLES = 1024;
inline constexpr uint32_t BUCKET_COUNT = ALPHA_MODE_COUNT * 2;

struct ScenePushConstants {
    VkDeviceAddress vertices;
    VkDeviceAddress meshes;
    VkDeviceAddress instances;
    VkDeviceAddress materials;
    VkDeviceAddress camera;
    VkDeviceAddress meshlets;
    VkDeviceAddress meshletTriangles;
    VkDeviceAddress meshletVertices;
    VkDeviceAddress drawMeshlets;
    VkDeviceAddress meshletGroups;
    VkDeviceAddress skinnedVertices;
    VkDeviceAddress skinnedBounds;
    VkDeviceAddress lights;
    VkDeviceAddress shadowMatrices;
    uint32_t meshletGroupBase;
    uint32_t cullPhase;
    VkDeviceAddress meshletVisibility;
};
static_assert(sizeof(ScenePushConstants) == 128, "푸시 상수는 규격이 보장하는 128 바이트에 꼭 맞춰 두었다");

// shaders/depth_only.vert 의 DepthPushConstants 와 배치가 같아야 한다.
struct DepthPushConstants {
    glm::mat4 viewProjection;
    VkDeviceAddress vertices;
    VkDeviceAddress instances;
    VkDeviceAddress skinnedVertices;
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

// shaders/tonemap.frag 의 푸시 상수와 배치가 같아야 한다.
struct TonemapPushConstants {
    VkDeviceAddress camera;
    VkDeviceAddress exposureBuffer;
    uint32_t colorTexture;
    float exposure;
    uint32_t sampleCount;
    uint32_t bloomTexture;
    float bloomIntensity;
    uint32_t autoExposure;
};

// shaders/reflect.comp 의 푸시 상수와 배치가 같아야 한다.
struct ReflectPushConstants {
    VkDeviceAddress vertices;
    VkDeviceAddress skinnedVertices;
    VkDeviceAddress indices;
    VkDeviceAddress meshes;
    VkDeviceAddress instances;
    VkDeviceAddress materials;
    VkDeviceAddress lods;
    VkDeviceAddress camera;
    VkDeviceAddress lights;
    uint32_t normalRoughnessTexture;
    uint32_t weightTexture;
    uint32_t depthTexture;
    uint32_t velocityTexture;
    uint32_t rawTexture;
    uint32_t rawStorage;
    uint32_t historyTexture;
    uint32_t historyStorage;
    uint32_t colorStorage;
    uint32_t frameIndex;
    uint32_t maxSamples;
    uint32_t reset;
    uint32_t debugMode;
};
static_assert(sizeof(ReflectPushConstants) <= 128, "푸시 상수는 규격이 보장하는 128 바이트 안에 있어야 한다");

// shaders/bloom_downsample.comp, bloom_upsample.comp 와 배치가 같아야 한다.
struct BloomPushConstants {
    uint32_t sourceTexture;
    uint32_t destinationStorage;
    float sourceTexelSize[2];
    int32_t destinationSize[2];
    uint32_t sampleCount;
    uint32_t firstLevel;
    float threshold;
    float knee;
    float scatter;
};

// shaders/exposure_histogram.comp 와 배치가 같아야 한다.
struct HistogramPushConstants {
    VkDeviceAddress histogram;
    uint32_t sourceTexture;
    int32_t sourceSize[2];
    float minLog;
    float inverseLogRange;
    uint32_t sampleCount;
};

// shaders/debug_line_common.glsl 의 DebugLinePushConstants 와 배치가 같아야 한다(scalar).
struct DebugLinePushConstants {
    glm::mat4 viewProjection;
    VkDeviceAddress vertices;
    uint32_t depthTexture;
    uint32_t occlude;
    float viewportSize[2];
};
static_assert(sizeof(DebugLinePushConstants) == 88, "디버그 선 푸시 상수 배치가 셰이더와 어긋난다");

// shaders/fluid_draw_common.glsl 의 FluidDrawPushConstants 와 배치가 같아야 한다(scalar).
struct FluidDrawPushConstants {
    VkDeviceAddress camera = 0;
    VkDeviceAddress vertices = 0;
    VkDeviceAddress lights = 0;
    // rgb 물 색, w 표면 거칠기.
    glm::vec4 waterColor{0.0F};
    // rgb 흡수 계수, w 두께 배율.
    glm::vec4 absorption{0.0F};
    uint32_t thicknessTexture = 0;
    uint32_t pad0 = 0;
    uint32_t pad1 = 0;
    uint32_t pad2 = 0;
};
static_assert(sizeof(FluidDrawPushConstants) == 72, "물 표면 푸시 상수 배치가 셰이더와 어긋난다");

// shaders/exposure_average.comp 와 배치가 같아야 한다.
struct ExposurePushConstants {
    VkDeviceAddress histogram;
    VkDeviceAddress exposure;
    float minLog;
    float logRange;
    float deltaSeconds;
    float adaptationSpeed;
    float minEv;
    float maxEv;
    uint32_t reset;
};

// Bloom 밉 사슬 최대 단계. 절반 해상도에서 시작하므로 6단계면 1080p 에서 30x17 까지 내려간다.
inline constexpr uint32_t BLOOM_MAX_LEVELS = 6;
// 히스토그램이 담는 log2 휘도 범위. 바깥은 양 끝 빈으로 몰린다.
inline constexpr float EXPOSURE_MIN_LOG = -10.0F;
inline constexpr float EXPOSURE_MAX_LOG = 16.0F;
// shaders/exposure.glsl 의 HISTOGRAM_BINS 와 같아야 한다.
inline constexpr uint32_t HISTOGRAM_BINS = 256;

struct SkyPushConstants {
    VkDeviceAddress camera;
    uint32_t environmentCube;
};

struct UpscalePushConstants {
    uint32_t sourceTexture;
    float sharpness;
    float sourceSize[2];
    float destinationSize[2];
};

inline VkPipelineShaderStageCreateInfo shaderStage(VkShaderStageFlagBits stage, VkShaderModule module) {
    VkPipelineShaderStageCreateInfo info{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    info.stage = stage;
    info.module = module;
    info.pName = "main";
    return info;
}

inline VkRenderingAttachmentInfo colorAttachment(VkImageView view, VkAttachmentLoadOp loadOp, VkClearColorValue clear) {
    VkRenderingAttachmentInfo info{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    info.imageView = view;
    info.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    info.loadOp = loadOp;
    info.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    info.clearValue.color = clear;
    return info;
}

inline void setFullViewport(VkCommandBuffer commandBuffer, VkExtent2D extent) {
    VkViewport viewport{};
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.maxDepth = 1.0F;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.extent = extent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
}

} // namespace gfx
