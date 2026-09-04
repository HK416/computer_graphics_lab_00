#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>

#include <glm/geometric.hpp>

#include "asset/primitives.h"
#include "asset/vertex_pack.h"

namespace {

// assert 는 어느 도형에서 깨졌는지 알려 주지 않는다. 도형 이름을 함께 찍고 멈춘다.
void check(bool condition, const char* what, const std::string& label) {
    if (!condition) {
        std::fprintf(stderr, "기본 도형 «%s» 검사 실패: %s\n", label.c_str(), what);
        std::abort();
    }
}

// 삼각형 셋의 위치와 압축을 푼 법선.
struct Triangle {
    glm::vec3 position[3];
    glm::vec3 normal[3];
    glm::vec2 uv[3];
    glm::vec4 tangent[3];
};

Triangle triangleOf(const asset::Mesh& mesh, uint32_t triangle) {
    Triangle out;
    for (int corner = 0; corner < 3; ++corner) {
        const asset::Vertex& vertex = mesh.vertices[mesh.indices[triangle * 3 + corner]];
        out.position[corner] = vertex.position;
        out.normal[corner] = asset::unpackUnitVector(vertex.normal);
        out.uv[corner] = vertex.uv;
        out.tangent[corner] = asset::unpackTangent(vertex.tangent);
    }
    return out;
}

// UV 로부터 v 가 늘어나는 방향(dP/dv). 셰이더의 접선 공간이 이것과 맞아야 노멀 맵이 바로 선다.
// 삼각형의 UV 가 퇴화하면(면적 0) 방향이 없다.
bool surfaceBitangent(const Triangle& triangle, glm::vec3& out) {
    glm::vec3 edge1 = triangle.position[1] - triangle.position[0];
    glm::vec3 edge2 = triangle.position[2] - triangle.position[0];
    glm::vec2 duv1 = triangle.uv[1] - triangle.uv[0];
    glm::vec2 duv2 = triangle.uv[2] - triangle.uv[0];
    float determinant = duv1.x * duv2.y - duv2.x * duv1.y;
    if (std::abs(determinant) < 1e-9F) {
        return false;
    }
    out = (edge2 * duv1.x - edge1 * duv2.x) / determinant;
    return glm::length(out) > 1e-6F;
}

} // namespace

int main() {
    for (uint32_t i = 0; i < static_cast<uint32_t>(asset::Primitive::COUNT); ++i) {
        auto primitive = static_cast<asset::Primitive>(i);
        std::string label = asset::primitiveLabel(primitive);
        asset::Model model = asset::makePrimitive(primitive);

        check(model.meshes.size() == 1 && model.materials.size() == 1, "도형 하나에 메쉬 하나", label);
        const asset::Mesh& mesh = model.meshes[0];
        check(!mesh.vertices.empty(), "정점이 있어야 한다", label);
        check(!mesh.indices.empty() && mesh.indices.size() % 3 == 0, "삼각형 목록", label);

        for (uint32_t index : mesh.indices) {
            check(index < mesh.vertices.size(), "인덱스가 범위 안", label);
        }

        // 경계 구가 모든 정점을 담아야 한다. 컬링과 카메라 프레이밍이 이 값을 믿는다.
        float farthest = 0.0F;
        for (const asset::Vertex& vertex : mesh.vertices) {
            farthest = std::max(farthest, glm::distance(vertex.position, mesh.boundsCenter));
        }
        check(farthest <= mesh.boundsRadius + 1e-4F, "경계 구가 모든 정점을 담는다", label);
        // 헐거우면 컬링이 헛돈다. computeBounds 는 가장 먼 정점을 반지름으로 삼는다.
        check(farthest >= mesh.boundsRadius - 1e-4F, "경계 구가 빡빡하다", label);

        for (const asset::Vertex& vertex : mesh.vertices) {
            glm::vec3 normal = asset::unpackUnitVector(vertex.normal);
            check(std::abs(glm::length(normal) - 1.0F) < 1e-2F, "법선이 단위", label);
            glm::vec4 tangent = asset::unpackTangent(vertex.tangent);
            check(std::abs(glm::length(glm::vec3{tangent}) - 1.0F) < 1e-2F, "탄젠트가 단위", label);
            // 탄젠트가 법선과 나란하면 접선 공간이 무너진다. 압축 오차를 감안해 느슨하게 본다.
            check(std::abs(glm::dot(normal, glm::vec3{tangent})) < 0.2F, "탄젠트가 법선과 다른 방향", label);
        }

        auto triangles = static_cast<uint32_t>(mesh.indices.size() / 3);
        for (uint32_t index = 0; index < triangles; ++index) {
            Triangle triangle = triangleOf(mesh, index);
            glm::vec3 geometric =
                glm::cross(triangle.position[1] - triangle.position[0], triangle.position[2] - triangle.position[0]);
            // 퇴화 삼각형은 그리기 낭비이고 감기 방향도 없다. 극과 꼭짓점의 가드가 살아 있는지 본다.
            check(glm::length(geometric) > 1e-6F, "퇴화하지 않은 삼각형", label);

            glm::vec3 averaged = triangle.normal[0] + triangle.normal[1] + triangle.normal[2];
            check(glm::dot(geometric, averaged) > 0.0F, "바깥을 보는 감기 방향", label);

            // 접선 공간의 손 방향. shaders/material.glsl 이 cross(N, T) * w 를 종법선으로 쓰므로
            // 그것이 UV 에서 나온 dP/dv 와 같은 쪽이어야 한다. 노멀 맵이 없으면 화면에 드러나지
            // 않아 눈으로는 절대 잡히지 않는 종류의 결함이다.
            glm::vec3 bitangent;
            if (!surfaceBitangent(triangle, bitangent)) {
                continue;
            }
            for (int corner = 0; corner < 3; ++corner) {
                glm::vec3 shaded = glm::cross(triangle.normal[corner], glm::vec3{triangle.tangent[corner]}) *
                                   triangle.tangent[corner].w;
                check(glm::dot(shaded, bitangent) > 0.0F, "탄젠트 손 방향이 UV 와 맞는다", label);
            }
        }

        // 이름으로 enum 을 되찾을 수 있어야 편집기와 장면 도구가 도형을 식별한다.
        check(asset::primitiveFromAssetName(asset::primitiveAssetName(primitive)) == primitive, "이름 왕복", label);
    }

    assert(asset::primitiveFromAssetName("<builtin:없음>") == asset::Primitive::COUNT);
    assert(asset::primitiveFromAssetName("Fox.glb") == asset::Primitive::COUNT);

    // 내장 구는 유체 입자가 그대로 쓴다. 반지름 1 이 아니면 입자 크기가 달라진다.
    {
        asset::Model sphere = asset::makePrimitive(asset::Primitive::SPHERE);
        assert(std::abs(sphere.meshes[0].boundsRadius - 1.0F) < 1e-4F && "구는 반지름 1");
        assert(sphere.meshes[0].boundsCenter == glm::vec3{0.0F});
        asset::Model ico = asset::makePrimitive(asset::Primitive::ICO_SPHERE);
        assert(std::abs(ico.meshes[0].boundsRadius - 1.0F) < 1e-4F && "정이십면체 구도 반지름 1");
        // 삼각형 수가 위도·경도 구와 비슷해야 입자 수만 개를 그려도 부담이 같다.
        assert(ico.meshes[0].indices.size() / 3 == 320);
    }

    // 도형들이 나란히 놓였을 때 어울리려면 «경계 구»가 아니라 «축 정렬 크기»가 맞아야 한다. 모두
    // ±1 정육면체 안에 들어가고, 적어도 한 축은 그 벽에 닿는다(구의 지름과 같은 크기라는 뜻이다).
    for (uint32_t i = 0; i < static_cast<uint32_t>(asset::Primitive::COUNT); ++i) {
        auto primitive = static_cast<asset::Primitive>(i);
        std::string label = asset::primitiveLabel(primitive);
        asset::Model model = asset::makePrimitive(primitive);
        float reach = 0.0F;
        for (const asset::Vertex& vertex : model.meshes[0].vertices) {
            for (int axis = 0; axis < 3; ++axis) {
                reach = std::max(reach, std::abs(vertex.position[axis]));
            }
        }
        check(reach <= 1.0F + 1e-4F, "±1 정육면체 안", label);
        check(reach >= 1.0F - 1e-4F, "한 축은 벽에 닿는다", label);
    }

    std::printf("기본 도형 자체 점검 통과\n");
    return 0;
}
