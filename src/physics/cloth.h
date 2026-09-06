#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

#include "physics/fluid_sph.h"
#include "physics/force_field.h"
#include "scene/scene.h"

namespace core {
class JobSystem;
} // namespace core

namespace physics {

// XPBD 천. 격자 메쉬(내장 «천 격자»)의 정점을 거리 제약(신장·전단·굽힘)으로 잇고 «색칠한 Gauss-Seidel» 로
// 푼다: 제약을 정점을 나눠 갖지 않는 12색으로 나누고 색마다 병렬로 바로 위치를 고친다. 한 색 안에서는 정점
// 하나를 제약 하나만 만지므로 원자 연산이 없고 순서도 무관해, 스레드 수와 백엔드(CPU/GPU)에 무관하게 같은
// 결과가 나온다. shaders/cloth_*.comp 가 같은 알고리즘을 같은 색 순서로 돈다.

// 제약 색 수. 격자는 신장 2×2 + 전단 2×2 + 굽힘 2×2 다. shaders/cloth_common.glsl 과 같아야 한다.
inline constexpr uint32_t CLOTH_COLORS = 12;
// 한 프레임에 진행할 최대 시간. 느린 프레임에 천이 폭주하지 않게 한다.
inline constexpr float CLOTH_MAX_FRAME_STEP = 1.0F / 30.0F;
inline constexpr uint32_t CLOTH_MAX_COLLIDERS = FLUID_MAX_COLLIDERS;

// shaders/cloth_common.glsl 의 ClothConstraint 와 배치가 같아야 한다(scalar, 16 바이트).
struct ClothConstraint {
    uint32_t a = 0;
    uint32_t b = 0;
    float restLength = 0.0F;
    // XPBD 컴플라이언스(m/N). 0 이면 강체 제약.
    float compliance = 0.0F;
};
static_assert(sizeof(ClothConstraint) == 16, "천 제약 배치가 셰이더와 어긋난다");

// shaders/cloth_common.glsl 의 ClothVertexInfo 와 배치가 같아야 한다(scalar, 16 바이트).
struct ClothVertexInfo {
    uint32_t gridX = 0;
    uint32_t gridZ = 0;
    uint32_t pad0 = 0;
    uint32_t pad1 = 0;
};
static_assert(sizeof(ClothVertexInfo) == 16, "천 정점 정보 배치가 셰이더와 어긋난다");

// 격자 위상. 정점 순서는 메쉬(LOD 구축 뒤)의 것을 그대로 따르고, 격자 좌표는 정지 위치를 양자화해 되찾는다.
struct ClothTopology {
    uint32_t resolution = 0;
    // 격자 (x + z·(n+1)) → 정점 번호.
    std::vector<uint32_t> gridToVertex;
    std::vector<ClothVertexInfo> vertices;
    // 정지 위치(월드)와 질량 역수(고정점은 0).
    std::vector<glm::vec4> rest;
    // 색 순서로 정렬된 제약. colorBegin[c] 부터 colorBegin[c+1] 앞까지가 색 c 다.
    std::vector<ClothConstraint> constraints;
    std::array<uint32_t, CLOTH_COLORS + 1> colorBegin{};
    uint32_t stretchCount = 0;
    uint32_t shearCount = 0;
    uint32_t bendCount = 0;
    // 월드 정지 자세의 경계 구. 그림자 컬링이 쓴다.
    glm::vec3 restCenter{0.0F};
    float restRadius = 0.0F;
};

// 부품 설정과 장면에서 끌어낸 시뮬레이션 상수. CPU 솔버와 GPU 셰이더가 «같은 함수»로 만든 같은 값을 본다.
struct ClothParams {
    glm::vec3 gravity{0.0F, -9.81F, 0.0F};
    float damping = 0.5F;
    glm::vec3 wind{0.0F};
    float thickness = 0.02F;
    float friction = 0.5F;
    uint32_t substeps = 4;
    uint32_t iterations = 8;
    // 한 프레임에 진행할 시간. CLOTH_MAX_FRAME_STEP 으로 잘라 둔 값이다.
    float frameStep = 0.0F;
    uint32_t colliderCount = 0;
    std::array<FluidCollider, CLOTH_MAX_COLLIDERS> colliders{};
    uint32_t fieldCount = 0;
    std::array<ForceFieldSample, MAX_FORCE_FIELDS> fields{};
};

// 세계 공간 삼각형. CPU 백엔드의 메쉬 콜라이더 충돌이 읽는다.
struct ClothTriangle {
    glm::vec3 a;
    glm::vec3 b;
    glm::vec3 c;
};

// 메쉬의 지역 정점을 격자로 되짚어 위상을 짓는다. localPositions 는 LOD 구축 뒤 정점 순서(scene::ColliderMesh
// 의 것), world 는 오브젝트 월드 변환이다. 정점 수가 (n+1)² 이 아니거나 격자로 안 풀리면 거짓을 돌려준다.
// 고정은 월드 Y 평균이 가장 높은 변(TOP_EDGE) 또는 그 양 끝(TWO_CORNERS)이라 오브젝트 방향에 무관하다.
bool buildClothTopology(const std::vector<glm::vec3>& localPositions,
                        const glm::mat4& world,
                        const scene::Cloth& settings,
                        ClothTopology& out);

// 장면의 강체 도형 콜라이더를 모은다(메쉬 제외, 보이는 것만, 상한 FLUID_MAX_COLLIDERS). 유체와 천이 함께 쓴다.
uint32_t collectShapeColliders(const scene::Scene& scene, std::array<FluidCollider, FLUID_MAX_COLLIDERS>& out);
// 보이는 MESH 강체의 삼각형을 세계 공간으로 모은다. skipObject 는 천 자신(자기 메쉬는 제외).
void collectMeshTriangles(const scene::Scene& scene, uint32_t skipObject, std::vector<ClothTriangle>& out);

ClothParams deriveClothParams(const scene::Cloth& settings, const scene::Scene& scene, float deltaSeconds);

// CPU 백엔드. 위치·속도를 들고 있고 프레임마다 step 을 부른다.
class ClothSolver {
public:
    void reset(const ClothTopology& topology);
    // 한 프레임(params.frameStep) 진행. jobs 가 있으면 제약·정점 패스를 나눈다. 결과는 스레드 수와 무관하다.
    void step(const ClothTopology& topology,
              const ClothParams& params,
              const std::vector<ClothTriangle>& triangles,
              core::JobSystem* jobs);
    // 격자 중앙 차분으로 법선·탄젠트를 낸다(GPU cloth_write.comp 와 같은 식).
    void computeNormals(const ClothTopology& topology);

    const std::vector<glm::vec3>& positions() const { return current; }
    const std::vector<glm::vec3>& velocities() const { return velocity; }
    const std::vector<glm::vec3>& normals() const { return normal; }
    const std::vector<glm::vec3>& tangents() const { return tangent; }
    // 현재 위치의 경계 구.
    glm::vec4 bounds() const;

private:
    std::vector<glm::vec3> current;
    std::vector<glm::vec3> predicted;
    std::vector<glm::vec3> velocity;
    std::vector<glm::vec3> normal;
    std::vector<glm::vec3> tangent;
    std::vector<float> lambda;
};

} // namespace physics
