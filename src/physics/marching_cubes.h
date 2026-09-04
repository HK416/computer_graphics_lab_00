#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace core {
class JobSystem;
} // namespace core

namespace physics {

struct FluidParams;

// 한 셀이 낼 수 있는 삼각형 수와 표의 폭. shaders/marching_cubes.glsl 과 같아야 한다.
inline constexpr uint32_t MC_MAX_TRIANGLES = 5;
inline constexpr uint32_t MC_TABLE_WIDTH = MC_MAX_TRIANGLES * 3 + 1;

// 셀 꼭짓점 여덟 개의 지역 좌표(0 또는 1). 번호는 x + 2y + 4z 가 아니라 아래 순서다.
extern const std::array<glm::ivec3, 8> MC_CORNERS;
// 모서리 열두 개가 잇는 꼭짓점 짝.
extern const std::array<std::array<uint8_t, 2>, 12> MC_EDGES;
// 케이스마다 삼각형을 이루는 모서리 번호. -1 이 끝이다.
extern const std::array<std::array<int8_t, MC_TABLE_WIDTH>, 256> MC_TABLE;

// 표면 정점 하나. shaders/fluid_types.glsl 의 FluidSurfaceVertex 와 배치가 같아야 한다(scalar).
struct SurfaceVertex {
    glm::vec3 position{0.0F};
    // 8진법으로 접은 법선. asset::packUnitVector 와 같은 규칙이다.
    uint32_t normal = 0;
};
static_assert(sizeof(SurfaceVertex) == 16, "표면 정점 배치가 셰이더와 어긋난다");

// 입자에서 스칼라 장을 만든다. 격자는 (resolution+1)³ 개의 표본이고 x 가 가장 빠르다. 값은 커널을
// 합친 «밀도 비슷한 것» 이라 등치값은 무차원이다. shaders/fluid_field.comp 와 같은 식을 쓴다.
void buildFluidField(const std::vector<glm::vec4>& particles,
                     const FluidParams& params,
                     uint32_t resolution,
                     std::vector<float>& field,
                     core::JobSystem* jobs);

// 스칼라 장에서 등치면을 뽑아 삼각형 정점을 채운다. 돌려주는 값은 쓴 정점 수이며 capacity 를 넘지
// 않는다. shaders/fluid_marching.comp 와 같은 표·같은 보간을 쓴다.
uint32_t marchFluidField(const std::vector<float>& field,
                         uint32_t resolution,
                         glm::vec3 origin,
                         glm::vec3 cellSize,
                         float iso,
                         SurfaceVertex* out,
                         uint32_t capacity,
                         core::JobSystem* jobs);

} // namespace physics
