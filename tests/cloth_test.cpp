#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include <glm/geometric.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "asset/primitives.h"
#include "core/job_system.h"
#include "physics/cloth.h"
#include "scene/scene.h"

namespace {

// 내장 천 격자의 지역 정점(LOD 구축 전이지만 순서는 어차피 위치로 되짚는다).
std::vector<glm::vec3> gridPositions(uint32_t resolution) {
    asset::Model model = asset::makePrimitive(asset::clothPrimitiveFor(resolution));
    std::vector<glm::vec3> positions;
    for (const asset::Vertex& vertex : model.meshes[0].vertices) {
        positions.push_back(vertex.position);
    }
    return positions;
}

// 정점 순서가 달라도 같은 위상이 나와야 한다. 뒤집어 넣어 본다.
std::vector<glm::vec3> reversed(std::vector<glm::vec3> positions) {
    std::reverse(positions.begin(), positions.end());
    return positions;
}

scene::Scene makeScene() {
    scene::Scene scene;
    scene::Object object;
    object.name = "천";
    object.transform.position = glm::vec3{0.0F, 2.0F, 0.0F};
    // 격자는 XZ 평면이다. X 축으로 90도 세워 윗변이 생기게 한다.
    object.transform.rotation = glm::angleAxis(glm::radians(90.0F), glm::vec3{1.0F, 0.0F, 0.0F});
    scene.objects.push_back(std::move(object));
    scene.attachCloth(0);
    scene.refresh();
    return scene;
}

float maxStretch(const physics::ClothTopology& topology, const std::vector<glm::vec3>& positions) {
    float worst = 0.0F;
    for (uint32_t c = 0; c < topology.stretchCount; ++c) {
        const physics::ClothConstraint& constraint = topology.constraints[c];
        float length = glm::distance(positions[constraint.a], positions[constraint.b]);
        worst = std::max(worst, length / constraint.restLength);
    }
    return worst;
}

bool finite(const std::vector<glm::vec3>& positions) {
    for (const glm::vec3& p : positions) {
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) {
            return false;
        }
    }
    return true;
}

void run(physics::ClothSolver& solver,
         const physics::ClothTopology& topology,
         const scene::Scene& scene,
         uint32_t frames,
         core::JobSystem* jobs) {
    std::vector<physics::ClothTriangle> triangles;
    physics::collectMeshTriangles(scene, 0, triangles);
    for (uint32_t frame = 0; frame < frames; ++frame) {
        physics::ClothParams params = physics::deriveClothParams(scene.cloths[0], scene, 1.0F / 60.0F);
        solver.step(topology, params, triangles, jobs);
    }
}

} // namespace

int main() {
    // 워커를 넉넉히 둔다. 결정성 검사는 스레드가 많을수록 인터리빙이 다양해져 잘 깨진다.
    core::JobSystem jobs(8);
    core::JobSystem single(1);

    // ---- 힘 마당: 바람은 방향 × 세기, 반지름 밖은 0, 소용돌이는 축과 반지름에 수직, 점은 중심을 향한다 ----
    {
        physics::ForceFieldSample wind;
        wind.type = scene::ForceFieldType::WIND;
        wind.axis = glm::vec3{1.0F, 0.0F, 0.0F};
        wind.strength = 4.0F;
        wind.radius = 2.0F;
        wind.falloff = 1.0F;
        glm::vec3 inside = physics::forceFieldAcceleration(wind, glm::vec3{0.0F, 1.0F, 0.0F});
        assert(std::abs(inside.x - 2.0F) < 1e-5F && inside.y == 0.0F && "반지름 절반에서 감쇠 1 이면 세기의 절반");
        assert((physics::forceFieldAcceleration(wind, glm::vec3{0.0F, 3.0F, 0.0F}) == glm::vec3{0.0F}) &&
               "반지름 밖은 0");
        wind.radius = 0.0F;
        assert((physics::forceFieldAcceleration(wind, glm::vec3{0.0F, 30.0F, 0.0F}) == glm::vec3{4.0F, 0.0F, 0.0F}));

        physics::ForceFieldSample vortex;
        vortex.type = scene::ForceFieldType::VORTEX;
        vortex.axis = glm::vec3{0.0F, 1.0F, 0.0F};
        vortex.strength = 3.0F;
        glm::vec3 swirl = physics::forceFieldAcceleration(vortex, glm::vec3{2.0F, 5.0F, 0.0F});
        assert(std::abs(glm::dot(swirl, vortex.axis)) < 1e-5F && std::abs(swirl.x) < 1e-5F &&
               std::abs(std::abs(swirl.z) - 3.0F) < 1e-5F);

        physics::ForceFieldSample point;
        point.type = scene::ForceFieldType::POINT;
        point.strength = 2.0F;
        glm::vec3 pull = physics::forceFieldAcceleration(point, glm::vec3{0.0F, 0.0F, 4.0F});
        assert(std::abs(pull.z + 2.0F) < 1e-5F && "점은 중심 쪽으로 세기만큼");
        std::printf("  힘 마당 식 통과\n");
    }

    // ---- 위상: 격자 16 의 제약 수와 정지 길이, 슬롯 ----
    {
        scene::Scene scene = makeScene();
        scene.cloths[0].resolution = 16;
        physics::ClothTopology topology;
        assert(physics::buildClothTopology(gridPositions(16), scene.world(0), scene.cloths[0], topology));
        assert(topology.stretchCount == 2 * 16 * 17 && "가로·세로 신장 제약");
        assert(topology.shearCount == 2 * 16 * 16 && "칸마다 대각 둘");
        assert(topology.bendCount == 2 * 15 * 17 && "한 칸 건너 굽힘");
        float spacing = 2.0F / 16.0F;
        for (uint32_t c = 0; c < topology.constraints.size(); ++c) {
            const physics::ClothConstraint& constraint = topology.constraints[c];
            float expected = c < topology.stretchCount                         ? spacing
                             : c < topology.stretchCount + topology.shearCount ? spacing * std::sqrt(2.0F)
                                                                               : spacing * 2.0F;
            assert(std::abs(constraint.restLength - expected) < 1e-4F && "정지 길이");
        }
        // 한 색 안에서는 어느 정점도 두 번 나오지 않는다. 그래야 색마다 병렬로 바로 고쳐도 경쟁이 없다.
        assert(topology.colorBegin[0] == 0 &&
               topology.colorBegin[physics::CLOTH_COLORS] == topology.constraints.size());
        for (uint32_t color = 0; color < physics::CLOTH_COLORS; ++color) {
            std::vector<uint8_t> touched(topology.rest.size(), 0);
            assert(topology.colorBegin[color + 1] > topology.colorBegin[color] && "빈 색이 없다");
            for (uint32_t c = topology.colorBegin[color]; c < topology.colorBegin[color + 1]; ++c) {
                const physics::ClothConstraint& constraint = topology.constraints[c];
                assert(touched[constraint.a] == 0 && touched[constraint.b] == 0 && "색 안에서 정점을 나눠 갖지 않는다");
                touched[constraint.a] = 1;
                touched[constraint.b] = 1;
            }
        }
        // 윗변(월드 Y 최대) 17개만 고정.
        uint32_t pinned = 0;
        float topY = -1e9F;
        for (const glm::vec4& rest : topology.rest) {
            topY = std::max(topY, rest.y);
        }
        for (const glm::vec4& rest : topology.rest) {
            if (rest.w == 0.0F) {
                ++pinned;
                assert(std::abs(rest.y - topY) < 1e-4F && "고정점은 가장 높은 변에 있다");
            }
        }
        assert(pinned == 17);

        // 정점 순서를 뒤집어도 격자는 같다.
        physics::ClothTopology mirrored;
        assert(physics::buildClothTopology(reversed(gridPositions(16)), scene.world(0), scene.cloths[0], mirrored));
        assert(mirrored.constraints.size() == topology.constraints.size());
        // 정점 수가 안 맞으면 거부.
        physics::ClothTopology bad;
        assert(!physics::buildClothTopology(gridPositions(32), scene.world(0), scene.cloths[0], bad));
        std::printf("천 위상 통과\n");
    }

    // ---- 윗변 고정 + 중력: 늘어나지 않고 아래로 처진다 ----
    {
        scene::Scene scene = makeScene();
        scene.cloths[0].resolution = 16;
        physics::ClothTopology topology;
        assert(physics::buildClothTopology(gridPositions(16), scene.world(0), scene.cloths[0], topology));
        physics::ClothSolver solver;
        solver.reset(topology);
        run(solver, topology, scene, 240, &jobs);
        assert(finite(solver.positions()));
        for (uint32_t i = 0; i < topology.rest.size(); ++i) {
            if (topology.rest[i].w == 0.0F) {
                assert(glm::distance(solver.positions()[i], glm::vec3{topology.rest[i]}) < 1e-6F &&
                       "고정점은 안 움직인다");
            }
        }
        assert(maxStretch(topology, solver.positions()) < 1.03F && "신장 컴플라이언스 0 이면 3% 넘게 늘지 않는다");
        float lowest = 1e9F;
        float pinY = topology.rest[0].y;
        for (uint32_t i = 0; i < topology.rest.size(); ++i) {
            if (topology.rest[i].w == 0.0F) {
                pinY = topology.rest[i].y;
            }
            lowest = std::min(lowest, solver.positions()[i].y);
        }
        assert(lowest < pinY - 1.5F && "아래 변은 고정점보다 훨씬 아래에 매달린다");
        // 법선은 단위 벡터다.
        for (const glm::vec3& n : solver.normals()) {
            assert(std::abs(glm::length(n) - 1.0F) < 1e-3F);
        }
        std::printf("천 매달림 통과\n");
    }

    // ---- 고정 없음 + 평면 강체: 두께만큼 떠서 멈춘다 ----
    {
        scene::Scene scene = makeScene();
        scene.cloths[0].resolution = 16;
        scene.cloths[0].pin = scene::ClothPin::NONE;
        scene.objects[0].transform.rotation = glm::quat{1.0F, 0.0F, 0.0F, 0.0F};
        scene::Object floor;
        floor.name = "바닥";
        scene.objects.push_back(std::move(floor));
        scene::RigidBody plane;
        plane.shape = scene::ColliderShape::PLANE;
        plane.kinematic = true;
        scene.attachRigidBody(1, plane);
        scene.refresh();
        physics::ClothTopology topology;
        assert(physics::buildClothTopology(gridPositions(16), scene.world(0), scene.cloths[0], topology));
        physics::ClothSolver solver;
        solver.reset(topology);
        run(solver, topology, scene, 240, &jobs);
        assert(finite(solver.positions()));
        for (const glm::vec3& p : solver.positions()) {
            assert(p.y >= scene.cloths[0].thickness - 1e-3F && "평면 아래로 뚫지 않는다");
            assert(p.y < 0.2F && "바닥까지 떨어졌다");
        }
        std::printf("천 바닥 충돌 통과\n");
    }

    // ---- 메쉬 콜라이더(내장 상자 삼각형): 뚫지 않는다 ----
    {
        scene::Scene scene = makeScene();
        scene.cloths[0].resolution = 16;
        scene.cloths[0].pin = scene::ClothPin::NONE;
        scene.objects[0].transform.rotation = glm::quat{1.0F, 0.0F, 0.0F, 0.0F};
        scene.objects[0].transform.position = glm::vec3{0.0F, 1.5F, 0.0F};
        scene::Object box;
        box.name = "상자";
        box.transform.scale = glm::vec3{0.5F};
        scene.objects.push_back(std::move(box));
        scene::RigidBody body;
        body.shape = scene::ColliderShape::MESH;
        scene.attachRigidBody(1, body);
        asset::Model boxModel = asset::makePrimitive(asset::Primitive::BOX);
        std::vector<scene::ColliderMesh> colliderMeshes(1);
        for (const asset::Vertex& vertex : boxModel.meshes[0].vertices) {
            colliderMeshes[0].positions.push_back(vertex.position);
        }
        colliderMeshes[0].indices = boxModel.meshes[0].indices;
        scene.colliderMeshes = &colliderMeshes;
        scene.attachMeshRenderer(1, 0);
        scene.refresh();
        physics::ClothTopology topology;
        assert(physics::buildClothTopology(gridPositions(16), scene.world(0), scene.cloths[0], topology));
        physics::ClothSolver solver;
        solver.reset(topology);
        run(solver, topology, scene, 240, &jobs);
        assert(finite(solver.positions()));
        // 상자 윗면(y = 0.5) 위에 걸친 정점은 두께만큼 떠 있다.
        bool anyOnTop = false;
        for (const glm::vec3& p : solver.positions()) {
            if (std::abs(p.x) < 0.45F && std::abs(p.z) < 0.45F) {
                assert(p.y >= 0.5F + scene.cloths[0].thickness - 2e-3F && "상자 윗면을 뚫지 않는다");
                anyOnTop = true;
            }
        }
        assert(anyOnTop);
        std::printf("천 메쉬 충돌 통과\n");
    }

    // ---- 결정성: 스레드 수와 무관하고 두 번 돌려도 같다 ----
    {
        scene::Scene scene = makeScene();
        scene.cloths[0].resolution = 32;
        scene.cloths[0].wind = glm::vec3{1.5F, 0.0F, 0.5F};
        physics::ClothTopology topology;
        assert(physics::buildClothTopology(gridPositions(32), scene.world(0), scene.cloths[0], topology));
        physics::ClothSolver a;
        physics::ClothSolver b;
        physics::ClothSolver c;
        a.reset(topology);
        b.reset(topology);
        c.reset(topology);
        run(a, topology, scene, 120, &jobs);
        run(b, topology, scene, 120, &single);
        run(c, topology, scene, 120, &jobs);
        size_t bytes = a.positions().size() * sizeof(glm::vec3);
        assert(std::memcmp(a.positions().data(), b.positions().data(), bytes) == 0 && "스레드 수와 무관하다");
        assert(std::memcmp(a.positions().data(), c.positions().data(), bytes) == 0 && "실행마다 같다");
        std::printf("천 결정성 통과\n");
    }

    // ---- 고정 변 선택: 오브젝트를 X 축 180도 돌리면 반대 변이 위다 ----
    {
        scene::Scene scene = makeScene();
        scene.cloths[0].resolution = 16;
        physics::ClothTopology upright;
        assert(physics::buildClothTopology(gridPositions(16), scene.world(0), scene.cloths[0], upright));
        scene.objects[0].transform.rotation =
            glm::angleAxis(glm::radians(180.0F), glm::vec3{1.0F, 0.0F, 0.0F}) * scene.objects[0].transform.rotation;
        scene.refresh();
        physics::ClothTopology flipped;
        assert(physics::buildClothTopology(gridPositions(16), scene.world(0), scene.cloths[0], flipped));
        bool differs = false;
        for (uint32_t i = 0; i < upright.rest.size(); ++i) {
            if ((upright.rest[i].w == 0.0F) != (flipped.rest[i].w == 0.0F)) {
                differs = true;
            }
        }
        assert(differs && "고정 변은 월드 높이로 고른다");
        std::printf("천 고정 변 통과\n");
    }

    std::printf("천 자체 점검 통과\n");
    return 0;
}
