#pragma once

#include <cstdint>
#include <vector>

#include <glm/vec3.hpp>

#include "scene/scene.h"

namespace gfx {

// 디버그 선 하나의 끝점. shaders/debug_line.vert 의 DebugLineVertex 와 배치가 같아야 한다(scalar).
struct DebugLineVertex {
    glm::vec3 position;
    // 0xAABBGGRR. 셰이더가 unpackUnorm4x8 로 푼다.
    uint32_t color;
};
static_assert(sizeof(DebugLineVertex) == 16, "디버그 선 정점 배치가 셰이더와 어긋난다");

// 강체 초록, 유체 용기 청록, 방출 상자 노랑. 고른 오브젝트는 밝게 그린다.
inline constexpr uint32_t DEBUG_COLOR_COLLIDER = 0xFF40D040U;
inline constexpr uint32_t DEBUG_COLOR_COLLIDER_SELECTED = 0xFF80FF80U;
inline constexpr uint32_t DEBUG_COLOR_FLUID_CONTAINER = 0xFFD0D040U;
inline constexpr uint32_t DEBUG_COLOR_FLUID_EMITTER = 0xFF40D0D0U;

struct DebugLineOptions {
    bool colliders = true;
    bool fluidBounds = true;
    // 밝게 그릴 오브젝트. 없으면 -1.
    int32_t selected = -1;
};

// 장면의 콜라이더와 유체 경계를 선분 목록(정점 두 개가 선 하나)으로 편다. Vulkan 을 타지 않아
// 테스트할 수 있다. scene 은 refresh 를 마친 상태여야 한다(세계 변환 캐시를 읽는다).
void buildDebugLines(const scene::Scene& scene, const DebugLineOptions& options, std::vector<DebugLineVertex>& out);

} // namespace gfx
