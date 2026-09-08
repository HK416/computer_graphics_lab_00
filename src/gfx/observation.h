#pragma once

#include <cstdint>
#include <vector>

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include "gfx/resources.h"

namespace scene {
struct Scene;
}

namespace gfx {

struct Context;
class BindlessTextures;
class GeometryStore;

// 관측 한 판의 한 변(정사각). DrQ-v2 와 MAD 가 84 를 쓴다. 인코더 conv 넷의 모양이 여기서 나온다
// (84 -> 41 -> 39 -> 37 -> 35).
inline constexpr uint32_t OBSERVATION_SIZE = 84;
// 프레임 스택 깊이. 채널이 «최근 세 프레임의 회색» 이다. 정지 그림 하나로는 속도를 알 수 없어 진자를
// 세울 수 없으므로 시간축을 채널로 접는다.
inline constexpr uint32_t OBSERVATION_STACK = 3;
// 한 뷰가 내놓는 float 수.
inline constexpr uint32_t OBSERVATION_VIEW_FLOATS = OBSERVATION_STACK * OBSERVATION_SIZE * OBSERVATION_SIZE;

// 관측 카메라를 **오브젝트 번호 순**으로 모은 것. physics::buildRobotLayout 과 같은 규칙이다 — 순서가
// 장면 파일의 오브젝트 순서로만 정해져야 학습한 가중치를 다시 올렸을 때 뷰가 뒤바뀌지 않는다.
struct ObservationLayout {
    struct View {
        uint32_t object = 0;
        float fovYDegrees = 60.0F;
        float nearPlane = 0.05F;

        bool operator==(const View&) const = default;
    };

    std::vector<View> views;

    bool empty() const { return views.empty(); }
    uint32_t count() const { return static_cast<uint32_t>(views.size()); }
    bool operator==(const ObservationLayout&) const = default;
};

ObservationLayout buildObservationLayout(const scene::Scene& scene);

// shaders/observation_common.glsl 의 ObservationView 와 배치가 같아야 한다.
struct GpuObservationView {
    glm::vec4 cameraPosition;
    // xyz 표면에서 광원으로 향하는 방향, w 세기.
    glm::vec4 lightDirection;
    // xyz 광원 색, w 상수 환경광 세기.
    glm::vec4 lightColor;
};

// shaders/observation_common.glsl 의 동명 블록과 배치가 같아야 한다.
struct ObservationPushConstants {
    glm::mat4 viewProjection;
    VkDeviceAddress vertices = 0;
    VkDeviceAddress instances = 0;
    VkDeviceAddress skinnedVertices = 0;
    VkDeviceAddress meshes = 0;
    VkDeviceAddress materials = 0;
    VkDeviceAddress views = 0;
    uint32_t view = 0;
};

// shaders/observation_encode.comp 의 동명 블록과 배치가 같아야 한다.
struct ObservationEncodePushConstants {
    VkDeviceAddress history = 0;
    VkDeviceAddress features = 0;
    uint32_t colorSlot = 0;
    uint32_t views = 0;
    uint32_t size = OBSERVATION_SIZE;
    uint32_t stack = OBSERVATION_STACK;
    uint32_t slot = 0;
    uint32_t reset = 0;
};

// 「정책이 무엇을 보는가」를 그리는 전용 렌더. 주 렌더러를 타지 않는 이유는 observation_common.glsl 에
// 적어 두었다.
//
// 층 하나가 뷰 하나다. 뷰마다 시점 행렬만 바꿔 끼우는 꼴은 그림자 아틀라스(renderer_shadow.cpp)와 같다.
class ObservationRenderer {
public:
    ObservationRenderer(Context& context, BindlessTextures& bindless, GeometryStore& geometry);
    ~ObservationRenderer();
    ObservationRenderer(const ObservationRenderer&) = delete;
    ObservationRenderer& operator=(const ObservationRenderer&) = delete;

    // 셰이더와 파이프라인이 모두 만들어졌는지. 아니면 관측 경로 자체를 끈다(폴백을 만들지 않는다).
    bool available() const { return ready; }
    // 표에 맞춰 이미지와 버퍼를 잡는다. 같은 표면 아무 것도 하지 않는다.
    //
    // **뷰 «수» 가 아니라 표 자체를 견준다.** 카메라 부품을 떼었다 다른 오브젝트에 붙이면 수는 그대로인데
    // 시점이 딴 데를 본다. 그때 프레임 스택을 그냥 두면 지난 시점의 그림 둘이 새 시점의 «과거» 로 섞여
    // 들어가, 정책이 있지도 않은 움직임을 본다.
    bool reserve(const ObservationLayout& layout);
    uint32_t viewCount() const { return layers; }

    // 관측 한 판을 그리고 회색 NCHW 로 인코드한다. 표가 지난번과 다르면 프레임 스택을 이번 그림으로
    // 채운다(reserve 가 판정한다). reset 은 그 위에 얹는 강제다 — 에피소드가 바뀔 때 부르는 쪽이 준다.
    // skinnedVertices 는 스킨 컴퓨트가 채운 **같은 버퍼**의 주소이고, 없으면 0 이다.
    void record(VkCommandBuffer commandBuffer,
                const scene::Scene& scene,
                const ObservationLayout& layout,
                VkDeviceAddress skinnedVertices,
                bool reset);

    // 인코드 결과. 신경망 입력 텐서가 이 주소를 그대로 읽는다.
    VkDeviceAddress featureAddress() const { return features.address; }
    // 같은 버퍼의 핸들. 활성 배열로 복사할 때 쓴다(주소로는 vkCmdCopyBuffer 를 부를 수 없다).
    VkBuffer featureBuffer() const { return features.handle; }
    size_t featureFloatCount() const { return static_cast<size_t>(layers) * OBSERVATION_VIEW_FLOATS; }
    // 편집기가 「정책이 보는 그림」을 띄울 때 쓰는 bindless 슬롯.
    uint32_t colorSlot() const { return colorBindlessSlot; }

    // 디버그 덤프용 되읽기. record 뒤에 기록하고 제출이 끝난 다음에 결과를 읽는다.
    void recordDownload(VkCommandBuffer commandBuffer);
    const float* featureResult() const { return featureReadbackFloats; }

private:
    void createPipelines();
    void destroyTargets();

    Context& context;
    BindlessTextures& bindless;
    GeometryStore& geometry;

    bool ready = false;
    uint32_t layers = 0;
    // 지금 잡혀 있는 표. 다음 reserve 가 이것과 견준다.
    ObservationLayout activeLayout;
    // 다음 record 가 프레임 스택을 다시 채워야 하는지.
    bool stackDirty = true;
    // 몇 번째 판인지. 프레임 스택의 고리 칸을 고른다.
    uint64_t step = 0;

    VkPipelineLayout drawLayout = VK_NULL_HANDLE;
    VkPipeline drawPipeline = VK_NULL_HANDLE;
    VkPipelineLayout encodeLayout = VK_NULL_HANDLE;
    VkPipeline encodePipeline = VK_NULL_HANDLE;
    VkSampler sampler = VK_NULL_HANDLE;

    Image color;
    Image depth;
    // 층마다 하나씩. dynamic rendering 이 층 하나만 붙이는 데 쓴다.
    std::vector<VkImageView> colorLayerViews;
    std::vector<VkImageView> depthLayerViews;
    uint32_t colorBindlessSlot = 0;
    bool colorSlotAllocated = false;

    Buffer instances;
    Buffer viewBuffer;
    Buffer history;
    Buffer features;
    Buffer featureReadback;
    float* featureReadbackFloats = nullptr;

    // 호스트에서 채우는 임시 배열. 프레임마다 다시 잡지 않으려고 멤버로 둔다.
    std::vector<uint32_t> drawObjects;
};

} // namespace gfx
