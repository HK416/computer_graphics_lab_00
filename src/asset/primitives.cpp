#include "asset/primitives.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <numbers>
#include <utility>

#include <glm/geometric.hpp>

#include "asset/vertex_pack.h"

namespace asset {

namespace {

constexpr float PI = std::numbers::pi_v<float>;

// 원기둥·콘·캡슐·토러스의 둘레 분할. 모두 같은 값을 써야 나란히 놓았을 때 실루엣이 어울린다.
constexpr uint32_t RADIAL_SEGMENTS = 24;

struct Builder {
    Mesh mesh;

    // 법선과 탄젠트는 이미 정규화된 것을 받는다. 도형마다 해석적으로 낼 수 있어 나중에 다시 재지 않는다.
    //
    // handedness 는 shaders/material.glsl 의 규약을 따른다: cross(normal, tangent) * handedness 가
    // UV 의 v 가 늘어나는 방향과 같아야 한다. 틀려도 노멀 맵이 없으면 화면에 드러나지 않으므로
    // primitives 테스트가 도형마다 검산한다.
    uint32_t vertex(glm::vec3 position, glm::vec3 normal, glm::vec3 tangent, glm::vec2 uv, float handedness = 1.0F) {
        Vertex value;
        value.position = position;
        value.normal = packUnitVector(normal);
        value.tangent = packTangent(glm::vec4{tangent, handedness});
        value.uv = uv;
        mesh.vertices.push_back(value);
        return static_cast<uint32_t>(mesh.vertices.size() - 1);
    }

    // 바깥에서 볼 때 반시계 방향.
    void triangle(uint32_t a, uint32_t b, uint32_t c) {
        mesh.indices.push_back(a);
        mesh.indices.push_back(b);
        mesh.indices.push_back(c);
    }

    void quad(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
        triangle(a, b, c);
        triangle(a, c, d);
    }
};

// 경도 phi 에서 구·원기둥의 둘레 접선. UV 의 u 가 늘어나는 방향이다.
glm::vec3 radialTangent(float phi) {
    return glm::vec3{-std::sin(phi), 0.0F, std::cos(phi)};
}

Material makeMaterial(const char* name, glm::vec3 color, float roughness) {
    Material material;
    material.name = name;
    material.baseColorFactor = glm::vec4{color, 1.0F};
    material.metallicFactor = 0.0F;
    material.roughnessFactor = roughness;
    return material;
}

// 유체 입자가 쓰는 색이기도 하다. 구 계열은 이 색을 지킨다.
const glm::vec3 SPHERE_COLOR{0.35F, 0.55F, 0.9F};
const glm::vec3 NEUTRAL_COLOR{0.78F, 0.78F, 0.78F};

// XZ 평면의 사각형. 법선은 +Y 뿐이라 아래에서 보면 사라진다(Unity 의 Plane 과 같다).
//
// 8x8 로 나누는 이유는 meshlet 한 장(정점 64, 삼각형 124)을 넘겨 meshlet 컬링과 LOD 가 실제로
// 돌게 하기 위해서다. 4x4 는 정점 25 라 meshlet 하나로 끝나 아무것도 나뉘지 않는다.
void buildPlane(Builder& builder) {
    constexpr uint32_t SEGMENTS = 8;
    for (uint32_t z = 0; z <= SEGMENTS; ++z) {
        for (uint32_t x = 0; x <= SEGMENTS; ++x) {
            float u = static_cast<float>(x) / static_cast<float>(SEGMENTS);
            float v = static_cast<float>(z) / static_cast<float>(SEGMENTS);
            // v 는 +Z 로 늘어나는데 cross(+Y, +X) 는 -Z 다.
            builder.vertex(glm::vec3{u * 2.0F - 1.0F, 0.0F, v * 2.0F - 1.0F},
                           glm::vec3{0.0F, 1.0F, 0.0F},
                           glm::vec3{1.0F, 0.0F, 0.0F},
                           glm::vec2{u, v},
                           -1.0F);
        }
    }
    for (uint32_t z = 0; z < SEGMENTS; ++z) {
        for (uint32_t x = 0; x < SEGMENTS; ++x) {
            uint32_t base = z * (SEGMENTS + 1) + x;
            uint32_t next = base + SEGMENTS + 1;
            builder.quad(base, next, next + 1, base + 1);
        }
    }
}

// 반쪽 크기 1 의 정육면체. 면마다 법선이 달라 정점을 나눠 둔다.
void buildBox(Builder& builder) {
    struct Face {
        glm::vec3 normal;
        glm::vec3 tangent;
    };
    const std::array<Face, 6> FACES{Face{{0.0F, 0.0F, 1.0F}, {1.0F, 0.0F, 0.0F}},
                                    Face{{0.0F, 0.0F, -1.0F}, {-1.0F, 0.0F, 0.0F}},
                                    Face{{1.0F, 0.0F, 0.0F}, {0.0F, 0.0F, -1.0F}},
                                    Face{{-1.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 1.0F}},
                                    Face{{0.0F, 1.0F, 0.0F}, {1.0F, 0.0F, 0.0F}},
                                    Face{{0.0F, -1.0F, 0.0F}, {1.0F, 0.0F, 0.0F}}};
    for (const Face& face : FACES) {
        glm::vec3 bitangent = glm::cross(face.normal, face.tangent);
        uint32_t base = builder.vertex(face.normal - face.tangent - bitangent, face.normal, face.tangent, {0.0F, 0.0F});
        builder.vertex(face.normal + face.tangent - bitangent, face.normal, face.tangent, {1.0F, 0.0F});
        builder.vertex(face.normal + face.tangent + bitangent, face.normal, face.tangent, {1.0F, 1.0F});
        builder.vertex(face.normal - face.tangent + bitangent, face.normal, face.tangent, {0.0F, 1.0F});
        builder.quad(base, base + 1, base + 2, base + 3);
    }
}

// 반지름 1 의 위도·경도 구. 유체 입자와 «메쉬 › 구» 가 쓰던 것 그대로다(16x12, 352 삼각형).
void buildUvSphere(Builder& builder) {
    constexpr uint32_t SEGMENTS = 16;
    constexpr uint32_t RINGS = 12;
    for (uint32_t ring = 0; ring <= RINGS; ++ring) {
        float theta = PI * static_cast<float>(ring) / static_cast<float>(RINGS);
        for (uint32_t segment = 0; segment <= SEGMENTS; ++segment) {
            float phi = 2.0F * PI * static_cast<float>(segment) / static_cast<float>(SEGMENTS);
            glm::vec3 normal{std::sin(theta) * std::cos(phi), std::cos(theta), std::sin(theta) * std::sin(phi)};
            builder.vertex(normal,
                           normal,
                           radialTangent(phi),
                           glm::vec2{static_cast<float>(segment) / static_cast<float>(SEGMENTS),
                                     static_cast<float>(ring) / static_cast<float>(RINGS)});
        }
    }
    // 극에서는 삼각형 하나가 퇴화하므로 그 줄은 하나만 넣는다.
    for (uint32_t ring = 0; ring < RINGS; ++ring) {
        for (uint32_t segment = 0; segment < SEGMENTS; ++segment) {
            uint32_t a = ring * (SEGMENTS + 1) + segment;
            uint32_t b = a + SEGMENTS + 1;
            if (ring > 0) {
                builder.triangle(a, a + 1, b);
            }
            if (ring + 1 < RINGS) {
                builder.triangle(a + 1, b + 1, b);
            }
        }
    }
}

// 정이십면체를 두 번 나눈 구(320 삼각형). 위도·경도 구와 달리 극에 삼각형이 몰리지 않아 조명이 고르다.
void buildIcoSphere(Builder& builder) {
    constexpr uint32_t SUBDIVISIONS = 2;
    // 정이십면체의 꼭짓점은 세 개의 황금 직사각형이 서로 직교해 놓인 자리다.
    const float GOLDEN = (1.0F + std::sqrt(5.0F)) * 0.5F;
    std::vector<glm::vec3> points{{-1.0F, GOLDEN, 0.0F},
                                  {1.0F, GOLDEN, 0.0F},
                                  {-1.0F, -GOLDEN, 0.0F},
                                  {1.0F, -GOLDEN, 0.0F},
                                  {0.0F, -1.0F, GOLDEN},
                                  {0.0F, 1.0F, GOLDEN},
                                  {0.0F, -1.0F, -GOLDEN},
                                  {0.0F, 1.0F, -GOLDEN},
                                  {GOLDEN, 0.0F, -1.0F},
                                  {GOLDEN, 0.0F, 1.0F},
                                  {-GOLDEN, 0.0F, -1.0F},
                                  {-GOLDEN, 0.0F, 1.0F}};
    for (glm::vec3& point : points) {
        point = glm::normalize(point);
    }
    std::vector<glm::uvec3> faces{{0, 11, 5},  {0, 5, 1},  {0, 1, 7},  {0, 7, 10}, {0, 10, 11}, {1, 5, 9}, {5, 11, 4},
                                  {11, 10, 2}, {10, 7, 6}, {7, 1, 8},  {3, 9, 4},  {3, 4, 2},   {3, 2, 6}, {3, 6, 8},
                                  {3, 8, 9},   {4, 9, 5},  {2, 4, 11}, {6, 2, 10}, {8, 6, 7},   {9, 8, 1}};

    // 모서리 중점은 두 면이 나눠 쓰므로 한 번만 만든다. 그러지 않으면 정점이 세 배로 늘어난다.
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> midpoints;
    auto midpoint = [&](uint32_t a, uint32_t b) {
        auto key = std::minmax(a, b);
        auto found = midpoints.find(key);
        if (found != midpoints.end()) {
            return found->second;
        }
        points.push_back(glm::normalize((points[a] + points[b]) * 0.5F));
        auto index = static_cast<uint32_t>(points.size() - 1);
        midpoints.emplace(key, index);
        return index;
    };
    for (uint32_t level = 0; level < SUBDIVISIONS; ++level) {
        std::vector<glm::uvec3> split;
        split.reserve(faces.size() * 4);
        for (const glm::uvec3& face : faces) {
            uint32_t ab = midpoint(face.x, face.y);
            uint32_t bc = midpoint(face.y, face.z);
            uint32_t ca = midpoint(face.z, face.x);
            split.push_back({face.x, ab, ca});
            split.push_back({face.y, bc, ab});
            split.push_back({face.z, ca, bc});
            split.push_back({ab, bc, ca});
        }
        faces = std::move(split);
    }

    // 구면 좌표 UV 는 경도 0 에서 u 가 1 에서 0 으로 되감기고, 극에서는 경도 자체가 없다. 그대로 두면
    // 그 띠와 극에서 텍스처가 되감기고 접선 공간도 뒤집힌다. 면마다 u 를 정하고, 같은 점이라도 u 가
    // 다르면 정점을 나눠 둔다. 나뉘는 것은 이음매와 극 근처뿐이라 정점 수는 조금만 는다.
    auto longitude = [](const glm::vec3& point) { return std::atan2(point.z, point.x) / (2.0F * PI) + 0.5F; };
    auto isPole = [](const glm::vec3& point) { return std::abs(point.y) > 1.0F - 1e-4F; };

    std::map<std::pair<uint32_t, int32_t>, uint32_t> emitted;
    auto emit = [&](uint32_t point, float u) {
        // 같은 u 를 소수점 오차 없이 맞추려고 눈금으로 접어 key 를 만든다.
        auto key = std::make_pair(point, static_cast<int32_t>(std::lround(u * 4096.0F)));
        auto found = emitted.find(key);
        if (found != emitted.end()) {
            return found->second;
        }
        const glm::vec3& position = points[point];
        float phi = 2.0F * PI * (u - 0.5F);
        uint32_t index = builder.vertex(
            position, position, radialTangent(phi), glm::vec2{u, std::acos(std::clamp(position.y, -1.0F, 1.0F)) / PI});
        emitted.emplace(key, index);
        return index;
    };

    for (const glm::uvec3& face : faces) {
        std::array<uint32_t, 3> corners{face.x, face.y, face.z};
        std::array<float, 3> u{};
        float sum = 0.0F;
        uint32_t counted = 0;
        for (uint32_t corner = 0; corner < 3; ++corner) {
            if (isPole(points[corners[corner]])) {
                continue;
            }
            u[corner] = longitude(points[corners[corner]]);
            sum += u[corner];
            ++counted;
        }
        // 이음매를 지나면 작은 쪽을 한 바퀴 올려 세 값을 같은 바퀴에 모은다.
        float lowest = 1.0F;
        float highest = 0.0F;
        for (uint32_t corner = 0; corner < 3; ++corner) {
            if (isPole(points[corners[corner]])) {
                continue;
            }
            lowest = std::min(lowest, u[corner]);
            highest = std::max(highest, u[corner]);
        }
        if (counted > 0 && highest - lowest > 0.5F) {
            sum = 0.0F;
            for (uint32_t corner = 0; corner < 3; ++corner) {
                if (isPole(points[corners[corner]])) {
                    continue;
                }
                if (u[corner] < 0.5F) {
                    u[corner] += 1.0F;
                }
                sum += u[corner];
            }
        }
        // 극은 경도가 없으니 나머지 꼭짓점의 평균을 쓴다. 그래야 삼각형 안에서 u 가 단조롭다.
        for (uint32_t corner = 0; corner < 3; ++corner) {
            if (isPole(points[corners[corner]])) {
                u[corner] = counted > 0 ? sum / static_cast<float>(counted) : 0.5F;
            }
        }
        builder.triangle(emit(corners[0], u[0]), emit(corners[1], u[1]), emit(corners[2], u[2]));
    }
}

// 옆면 하나와 위아래 뚜껑. topRadius 가 0 이면 콘이다.
void buildCone(Builder& builder, float topRadius) {
    constexpr float HALF_HEIGHT = 1.0F;
    constexpr float BOTTOM_RADIUS = 1.0F;
    // 옆면의 기울기. 법선은 축 방향 성분을 이만큼 가진다.
    float slope = (BOTTOM_RADIUS - topRadius) / (2.0F * HALF_HEIGHT);
    float normalScale = 1.0F / std::sqrt(1.0F + slope * slope);

    uint32_t sideBase = static_cast<uint32_t>(builder.mesh.vertices.size());
    for (uint32_t segment = 0; segment <= RADIAL_SEGMENTS; ++segment) {
        float u = static_cast<float>(segment) / static_cast<float>(RADIAL_SEGMENTS);
        float phi = 2.0F * PI * u;
        glm::vec3 outward{std::cos(phi), 0.0F, std::sin(phi)};
        glm::vec3 normal = (outward + glm::vec3{0.0F, slope, 0.0F}) * normalScale;
        // v 는 아래에서 위로 늘어나는데 cross(바깥 법선, 둘레 접선) 은 아래를 가리킨다. 구·캡슐은
        // v 가 위에서 아래로 흘러 부호가 반대다.
        builder.vertex(outward * BOTTOM_RADIUS - glm::vec3{0.0F, HALF_HEIGHT, 0.0F},
                       normal,
                       radialTangent(phi),
                       glm::vec2{u, 0.0F},
                       -1.0F);
        builder.vertex(outward * topRadius + glm::vec3{0.0F, HALF_HEIGHT, 0.0F},
                       normal,
                       radialTangent(phi),
                       glm::vec2{u, 1.0F},
                       -1.0F);
    }
    for (uint32_t segment = 0; segment < RADIAL_SEGMENTS; ++segment) {
        uint32_t bottom = sideBase + segment * 2;
        // 꼭짓점이 한 점으로 모이면 위쪽 삼각형이 퇴화한다.
        if (topRadius > 0.0F) {
            builder.quad(bottom, bottom + 1, bottom + 3, bottom + 2);
        } else {
            builder.triangle(bottom, bottom + 1, bottom + 2);
        }
    }

    // 뚜껑. 법선이 옆면과 달라 정점을 따로 둔다.
    auto cap = [&](float y, float radius, bool up) {
        if (radius <= 0.0F) {
            return;
        }
        glm::vec3 normal{0.0F, up ? 1.0F : -1.0F, 0.0F};
        // 두 뚜껑 모두 v 가 +Z 로 늘어난다. cross(+Y, +X) 는 -Z, cross(-Y, +X) 는 +Z 다.
        float handedness = up ? -1.0F : 1.0F;
        uint32_t center = builder.vertex(
            glm::vec3{0.0F, y, 0.0F}, normal, glm::vec3{1.0F, 0.0F, 0.0F}, glm::vec2{0.5F, 0.5F}, handedness);
        for (uint32_t segment = 0; segment <= RADIAL_SEGMENTS; ++segment) {
            float phi = 2.0F * PI * static_cast<float>(segment) / static_cast<float>(RADIAL_SEGMENTS);
            glm::vec3 outward{std::cos(phi), 0.0F, std::sin(phi)};
            builder.vertex(outward * radius + glm::vec3{0.0F, y, 0.0F},
                           normal,
                           glm::vec3{1.0F, 0.0F, 0.0F},
                           glm::vec2{outward.x * 0.5F + 0.5F, outward.z * 0.5F + 0.5F},
                           handedness);
        }
        for (uint32_t segment = 0; segment < RADIAL_SEGMENTS; ++segment) {
            uint32_t first = center + 1 + segment;
            if (up) {
                builder.triangle(center, first + 1, first);
            } else {
                builder.triangle(center, first, first + 1);
            }
        }
    };
    cap(HALF_HEIGHT, topRadius, true);
    cap(-HALF_HEIGHT, BOTTOM_RADIUS, false);
}

// 반지름 0.5 의 반구 둘과 그 사이의 원기둥. 전체 높이가 2 라 다른 도형과 키가 같다.
void buildCapsule(Builder& builder) {
    constexpr float RADIUS = 0.5F;
    constexpr float HALF_HEIGHT = 0.5F; // 원통 부분의 절반.
    constexpr uint32_t CAP_RINGS = 6;
    // 아래 반구 → 원통 → 위 반구를 하나의 세로 격자로 잇는다. 줄 번호가 이어져 있어야 이음매가 없다.
    constexpr uint32_t RINGS = CAP_RINGS * 2 + 1;

    for (uint32_t ring = 0; ring <= RINGS; ++ring) {
        // 위 반구는 0..CAP_RINGS, 아래 반구는 CAP_RINGS+1..RINGS 이고 그 사이가 원통 이음매다.
        bool upper = ring <= CAP_RINGS;
        uint32_t capRing = upper ? ring : ring - CAP_RINGS - 1;
        float theta = 0.5F * PI * static_cast<float>(capRing) / static_cast<float>(CAP_RINGS);
        float ringRadius = upper ? std::sin(theta) : std::cos(theta);
        float ringY = upper ? std::cos(theta) : -std::sin(theta);
        float offset = upper ? HALF_HEIGHT : -HALF_HEIGHT;
        for (uint32_t segment = 0; segment <= RADIAL_SEGMENTS; ++segment) {
            float u = static_cast<float>(segment) / static_cast<float>(RADIAL_SEGMENTS);
            float phi = 2.0F * PI * u;
            glm::vec3 normal{ringRadius * std::cos(phi), ringY, ringRadius * std::sin(phi)};
            builder.vertex(normal * RADIUS + glm::vec3{0.0F, offset, 0.0F},
                           normal,
                           radialTangent(phi),
                           glm::vec2{u, static_cast<float>(ring) / static_cast<float>(RINGS)});
        }
    }
    for (uint32_t ring = 0; ring < RINGS; ++ring) {
        for (uint32_t segment = 0; segment < RADIAL_SEGMENTS; ++segment) {
            uint32_t a = ring * (RADIAL_SEGMENTS + 1) + segment;
            uint32_t b = a + RADIAL_SEGMENTS + 1;
            if (ring > 0) {
                builder.triangle(a, a + 1, b);
            }
            if (ring + 1 < RINGS) {
                builder.triangle(a + 1, b + 1, b);
            }
        }
    }
}

// 바깥 반지름이 1 이 되도록 중심 반지름 0.75, 관 반지름 0.25.
void buildTorus(Builder& builder) {
    constexpr float MAJOR_RADIUS = 0.75F;
    constexpr float MINOR_RADIUS = 0.25F;
    constexpr uint32_t TUBE_SEGMENTS = 12;

    for (uint32_t ring = 0; ring <= RADIAL_SEGMENTS; ++ring) {
        float u = static_cast<float>(ring) / static_cast<float>(RADIAL_SEGMENTS);
        float phi = 2.0F * PI * u;
        glm::vec3 outward{std::cos(phi), 0.0F, std::sin(phi)};
        glm::vec3 center = outward * MAJOR_RADIUS;
        for (uint32_t tube = 0; tube <= TUBE_SEGMENTS; ++tube) {
            float v = static_cast<float>(tube) / static_cast<float>(TUBE_SEGMENTS);
            float theta = 2.0F * PI * v;
            glm::vec3 normal = outward * std::cos(theta) + glm::vec3{0.0F, std::sin(theta), 0.0F};
            // v 는 관을 도는 방향이고 theta=0 에서 +Y 다. cross(법선, 둘레 접선) 은 그 반대다.
            builder.vertex(center + normal * MINOR_RADIUS, normal, radialTangent(phi), glm::vec2{u, v}, -1.0F);
        }
    }
    for (uint32_t ring = 0; ring < RADIAL_SEGMENTS; ++ring) {
        for (uint32_t tube = 0; tube < TUBE_SEGMENTS; ++tube) {
            uint32_t a = ring * (TUBE_SEGMENTS + 1) + tube;
            uint32_t b = a + TUBE_SEGMENTS + 1;
            builder.quad(a, a + 1, b + 1, b);
        }
    }
}

struct Definition {
    const char* label;
    const char* assetName;
    const char* materialName;
    glm::vec3 color;
    float roughness;
};

const std::array<Definition, static_cast<size_t>(Primitive::COUNT)> DEFINITIONS{
    Definition{"평면", "<builtin:plane>", "평면", NEUTRAL_COLOR, 0.6F},
    Definition{"큐브", "<builtin:box>", "큐브", NEUTRAL_COLOR, 0.5F},
    Definition{"구", "<builtin:sphere>", "구", SPHERE_COLOR, 0.2F},
    Definition{"구(정이십면체)", "<builtin:icosphere>", "구", SPHERE_COLOR, 0.2F},
    Definition{"원기둥", "<builtin:cylinder>", "원기둥", NEUTRAL_COLOR, 0.5F},
    Definition{"콘", "<builtin:cone>", "콘", NEUTRAL_COLOR, 0.5F},
    Definition{"캡슐", "<builtin:capsule>", "캡슐", NEUTRAL_COLOR, 0.5F},
    Definition{"토러스", "<builtin:torus>", "토러스", NEUTRAL_COLOR, 0.5F},
};

const Definition& definitionOf(Primitive primitive) {
    // COUNT 는 도형이 아니라 개수다. primitiveFromAssetName 이 실패할 때 그것을 돌려주므로 그대로
    // 넘어올 수 있다. 배열 밖을 읽지 않게 접는다.
    size_t index = std::min(static_cast<size_t>(primitive), DEFINITIONS.size() - 1);
    return DEFINITIONS[index];
}

} // namespace

void computeBounds(Mesh& mesh) {
    if (mesh.vertices.empty()) {
        return;
    }
    glm::vec3 minimum = mesh.vertices.front().position;
    glm::vec3 maximum = minimum;
    for (const Vertex& vertex : mesh.vertices) {
        minimum = glm::min(minimum, vertex.position);
        maximum = glm::max(maximum, vertex.position);
    }
    mesh.boundsCenter = (minimum + maximum) * 0.5F;
    mesh.boundsRadius = 0.0F;
    for (const Vertex& vertex : mesh.vertices) {
        mesh.boundsRadius = std::max(mesh.boundsRadius, glm::distance(vertex.position, mesh.boundsCenter));
    }
}

const char* primitiveLabel(Primitive primitive) {
    return definitionOf(primitive).label;
}

const char* primitiveAssetName(Primitive primitive) {
    return definitionOf(primitive).assetName;
}

Primitive primitiveFromAssetName(std::string_view name) {
    for (uint32_t i = 0; i < static_cast<uint32_t>(Primitive::COUNT); ++i) {
        if (name == DEFINITIONS[i].assetName) {
            return static_cast<Primitive>(i);
        }
    }
    return Primitive::COUNT;
}

Model makePrimitive(Primitive primitive) {
    const Definition& definition = definitionOf(primitive);

    Builder builder;
    builder.mesh.name = definition.label;
    builder.mesh.materialIndex = 0;
    switch (primitive) {
    case Primitive::PLANE:
        buildPlane(builder);
        break;
    case Primitive::BOX:
        buildBox(builder);
        break;
    case Primitive::SPHERE:
        buildUvSphere(builder);
        break;
    case Primitive::ICO_SPHERE:
        buildIcoSphere(builder);
        break;
    case Primitive::CYLINDER:
        buildCone(builder, 1.0F);
        break;
    case Primitive::CONE:
        buildCone(builder, 0.0F);
        break;
    case Primitive::CAPSULE:
        buildCapsule(builder);
        break;
    case Primitive::TORUS:
        buildTorus(builder);
        break;
    case Primitive::COUNT:
        // 도형이 아니다. 빈 메쉬가 나가고 meshLive 가 거짓이라 그려지지 않는다.
        break;
    }
    computeBounds(builder.mesh);

    Model model;
    model.name = definition.assetName;
    model.materials.push_back(makeMaterial(definition.materialName, definition.color, definition.roughness));
    model.meshes.push_back(std::move(builder.mesh));
    return model;
}

} // namespace asset
