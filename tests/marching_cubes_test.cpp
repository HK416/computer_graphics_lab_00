#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <glm/geometric.hpp>

#include "physics/marching_cubes.h"

namespace {

using physics::MC_CORNERS;
using physics::MC_EDGES;
using physics::MC_TABLE;
using physics::MC_TABLE_WIDTH;

// 셀의 여섯 면. 각각 꼭짓점 넷과 그 면에 놓인 모서리 넷이다.
constexpr std::array<std::array<uint8_t, 4>, 6> FACE_CORNERS{
    {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 1, 5, 4}, {3, 2, 6, 7}, {0, 3, 7, 4}, {1, 2, 6, 5}}};
constexpr std::array<std::array<uint8_t, 4>, 6> FACE_EDGES{
    {{0, 1, 2, 3}, {4, 5, 6, 7}, {0, 9, 4, 8}, {2, 10, 6, 11}, {3, 11, 7, 8}, {1, 10, 5, 9}}};

bool inside(uint32_t code, uint8_t corner) {
    return ((code >> corner) & 1U) != 0U;
}

// 모서리가 등치면을 가로지르는지. 두 끝의 부호가 달라야 한다.
bool active(uint32_t code, uint8_t edge) {
    return inside(code, MC_EDGES[edge][0]) != inside(code, MC_EDGES[edge][1]);
}

glm::vec3 edgeMidpoint(uint8_t edge) {
    return (glm::vec3{MC_CORNERS[MC_EDGES[edge][0]]} + glm::vec3{MC_CORNERS[MC_EDGES[edge][1]]}) * 0.5F;
}

// 삼각형이 걸친 세 모서리의 «안쪽 끝 → 바깥쪽 끝» 방향을 더한 것. 안팎 무게중심의 차와 달리 안팎이
// 대칭인 케이스(체커보드 등)에서도 0 이 되지 않는다.
glm::vec3 outwardDirection(uint32_t code, const std::array<uint8_t, 3>& triangle) {
    glm::vec3 total{0.0F};
    for (uint8_t edge : triangle) {
        uint8_t a = MC_EDGES[edge][0];
        uint8_t b = MC_EDGES[edge][1];
        uint8_t from = inside(code, a) ? a : b;
        uint8_t to = inside(code, a) ? b : a;
        total += glm::vec3{MC_CORNERS[to]} - glm::vec3{MC_CORNERS[from]};
    }
    return total;
}

uint32_t entryCount(uint32_t code) {
    uint32_t count = 0;
    while (count < MC_TABLE_WIDTH && MC_TABLE[code][count] >= 0) {
        ++count;
    }
    return count;
}

// 한 면에 놓인 «표면 조각의 테두리». 면을 맞대는 두 셀이 같은 테두리를 내야 표면에 틈이 없다.
//
// 부채꼴로 자르면서 생긴 안쪽 변은 삼각형 둘이 서로 반대 방향으로 나눠 가지므로, 방향 있는 변이
// 한 번만 나온 것이 테두리다. 그렇게 걸러야 우연히 같은 면에 놓인 안쪽 변이 섞이지 않는다.
std::set<std::pair<uint8_t, uint8_t>> faceContour(uint32_t code, size_t face) {
    std::set<uint8_t> onFace(FACE_EDGES[face].begin(), FACE_EDGES[face].end());
    std::set<std::pair<uint8_t, uint8_t>> directed;
    uint32_t count = entryCount(code);
    for (uint32_t i = 0; i < count; i += 3) {
        for (uint32_t k = 0; k < 3; ++k) {
            auto a = static_cast<uint8_t>(MC_TABLE[code][i + k]);
            auto b = static_cast<uint8_t>(MC_TABLE[code][i + (k + 1) % 3]);
            directed.insert({a, b});
        }
    }
    std::set<std::pair<uint8_t, uint8_t>> contour;
    for (const auto& [a, b] : directed) {
        if (directed.count({b, a}) != 0) {
            continue;
        }
        if (onFace.count(a) != 0 && onFace.count(b) != 0) {
            contour.insert({std::min(a, b), std::max(a, b)});
        }
    }
    return contour;
}

// 셰이더 표를 읽어 니블을 푼다. 두 벌이 갈리면 백엔드마다 다른 물 표면이 나온다.
std::vector<std::array<int, MC_TABLE_WIDTH>> readShaderTable() {
    std::ifstream file(std::string{CG_LAB_SHADER_DIR} + "/marching_cubes.glsl");
    assert(file.is_open() && "shaders/marching_cubes.glsl 을 열지 못했다");
    std::vector<uint32_t> words;
    std::string line;
    while (std::getline(file, line)) {
        size_t at = 0;
        while ((at = line.find("0x", at)) != std::string::npos) {
            words.push_back(static_cast<uint32_t>(std::stoul(line.substr(at + 2, 8), nullptr, 16)));
            at += 2;
        }
    }
    assert(words.size() == 512 && "셰이더 표는 케이스마다 uvec2 하나여야 한다");
    std::vector<std::array<int, MC_TABLE_WIDTH>> table(256);
    for (uint32_t code = 0; code < 256; ++code) {
        for (uint32_t i = 0; i < MC_TABLE_WIDTH; ++i) {
            uint32_t word = words[code * 2 + (i < 8 ? 0 : 1)];
            uint32_t nibble = (word >> (4 * (i & 7))) & 15U;
            table[code][i] = nibble == 15U ? -1 : static_cast<int>(nibble);
        }
    }
    return table;
}

} // namespace

int main() {
    // 1) 표의 모양. 삼각형은 셋씩이고 끝 표시 뒤에는 아무것도 없다.
    for (uint32_t code = 0; code < 256; ++code) {
        uint32_t count = entryCount(code);
        assert(count % 3 == 0 && "삼각형은 모서리 셋씩이다");
        assert(count <= physics::MC_MAX_TRIANGLES * 3 && "한 셀이 내는 삼각형이 상한을 넘었다");
        for (uint32_t i = count; i < MC_TABLE_WIDTH; ++i) {
            assert(MC_TABLE[code][i] < 0 && "끝 표시 뒤에는 항목이 없어야 한다");
        }
    }

    // 2) 쓰이는 모서리는 모두 등치면을 가로지르고, 가로지르는 모서리는 모두 쓰인다.
    for (uint32_t code = 0; code < 256; ++code) {
        std::set<uint8_t> used;
        uint32_t count = entryCount(code);
        for (uint32_t i = 0; i < count; ++i) {
            auto edge = static_cast<uint8_t>(MC_TABLE[code][i]);
            assert(edge < 12 && "모서리 번호가 범위를 넘었다");
            assert(active(code, edge) && "부호가 같은 모서리에는 등치면이 지나지 않는다");
            used.insert(edge);
        }
        for (uint8_t edge = 0; edge < 12; ++edge) {
            assert((!active(code, edge) || used.count(edge) != 0) && "가로지르는 모서리가 빠졌다");
        }
        for (uint32_t i = 0; i < count; i += 3) {
            assert(MC_TABLE[code][i] != MC_TABLE[code][i + 1] && MC_TABLE[code][i + 1] != MC_TABLE[code][i + 2] &&
                   MC_TABLE[code][i] != MC_TABLE[code][i + 2] && "삼각형이 찌그러졌다");
        }
    }

    // 3) 감기. 면 법선은 유체 안에서 바깥을 향해야 셰이더가 장의 기울기로 내는 법선과 같은 쪽이다.
    //    셀의 한 면에 납작하게 눕는 조각은 법선이 그 면에 수직이라 안팎 방향과 직교한다. 그런 조각은
    //    방향이 없어도 되고, 케이스마다 적어도 하나는 확실히 바깥을 향해야 한다.
    for (uint32_t code = 1; code < 255; ++code) {
        uint32_t count = entryCount(code);
        bool anyOutward = false;
        for (uint32_t i = 0; i < count; i += 3) {
            std::array<uint8_t, 3> triangle{static_cast<uint8_t>(MC_TABLE[code][i]),
                                            static_cast<uint8_t>(MC_TABLE[code][i + 1]),
                                            static_cast<uint8_t>(MC_TABLE[code][i + 2])};
            glm::vec3 a = edgeMidpoint(triangle[0]);
            glm::vec3 b = edgeMidpoint(triangle[1]);
            glm::vec3 c = edgeMidpoint(triangle[2]);
            glm::vec3 normal = glm::cross(b - a, c - a);
            float facing = glm::dot(normal, outwardDirection(code, triangle));
            assert(facing >= -1.0e-6F && "삼각형 감기가 뒤집혔다");
            anyOutward = anyOutward || facing > 1.0e-6F;
        }
        assert((count == 0 || anyOutward) && "케이스 전체가 방향을 못 정했다");
    }

    // 3-2) 한 케이스의 삼각형들은 서로 같은 방향으로 감겨야 한다. 방향 있는 변이 같은 방향으로 두 번
    //      나오면 이웃한 삼각형 하나가 뒤집힌 것이다.
    for (uint32_t code = 0; code < 256; ++code) {
        std::set<std::pair<uint8_t, uint8_t>> directed;
        uint32_t count = entryCount(code);
        for (uint32_t i = 0; i < count; i += 3) {
            for (uint32_t k = 0; k < 3; ++k) {
                auto from = static_cast<uint8_t>(MC_TABLE[code][i + k]);
                auto to = static_cast<uint8_t>(MC_TABLE[code][i + (k + 1) % 3]);
                assert(directed.insert({from, to}).second && "같은 방향의 변이 두 번 나왔다");
            }
        }
    }

    // 4) 틈 없음. 면의 윤곽은 그 면의 네 꼭짓점 부호만으로 정해져야 한다. 그래야 이웃한 두 셀이
    //    맞닿는 면에서 같은 선분을 내고 표면이 닫힌다.
    for (size_t face = 0; face < 6; ++face) {
        std::array<std::set<std::pair<uint8_t, uint8_t>>, 16> expected;
        std::array<bool, 16> seen{};
        for (uint32_t code = 0; code < 256; ++code) {
            uint32_t pattern = 0;
            for (uint32_t k = 0; k < 4; ++k) {
                if (inside(code, FACE_CORNERS[face][k])) {
                    pattern |= 1U << k;
                }
            }
            std::set<std::pair<uint8_t, uint8_t>> contour = faceContour(code, face);
            if (!seen[pattern]) {
                seen[pattern] = true;
                expected[pattern] = contour;
                continue;
            }
            assert(contour == expected[pattern] && "같은 면 부호인데 윤곽이 다르다. 표면에 틈이 생긴다");
        }
    }

    // 5) 셰이더 표가 C++ 표와 같은지. 두 벌이 갈리면 백엔드마다 다른 물이 나온다.
    std::vector<std::array<int, MC_TABLE_WIDTH>> shader = readShaderTable();
    for (uint32_t code = 0; code < 256; ++code) {
        for (uint32_t i = 0; i < MC_TABLE_WIDTH; ++i) {
            assert(shader[code][i] == static_cast<int>(MC_TABLE[code][i]) && "셰이더 표가 C++ 표와 다르다");
        }
    }

    // 6) 구를 잘라 보면 반지름 근처에 정점이 놓이고 법선이 바깥을 향한다.
    {
        constexpr uint32_t RESOLUTION = 24;
        constexpr float RADIUS = 0.6F;
        uint32_t samples = RESOLUTION + 1;
        std::vector<float> field(static_cast<size_t>(samples) * samples * samples);
        for (uint32_t z = 0; z < samples; ++z) {
            for (uint32_t y = 0; y < samples; ++y) {
                for (uint32_t x = 0; x < samples; ++x) {
                    glm::vec3 p = glm::vec3{x, y, z} / static_cast<float>(RESOLUTION) * 2.0F - 1.0F;
                    field[(static_cast<size_t>(z) * samples + y) * samples + x] = RADIUS - glm::length(p);
                }
            }
        }
        std::vector<physics::SurfaceVertex> vertices(200000);
        uint32_t written = physics::marchFluidField(field,
                                                    RESOLUTION,
                                                    glm::vec3{-1.0F},
                                                    glm::vec3{2.0F / static_cast<float>(RESOLUTION)},
                                                    0.0F,
                                                    vertices.data(),
                                                    static_cast<uint32_t>(vertices.size()),
                                                    nullptr);
        assert(written > 0 && written % 3 == 0 && "구를 자르면 삼각형이 나와야 한다");
        float cell = 2.0F / static_cast<float>(RESOLUTION);
        for (uint32_t i = 0; i < written; ++i) {
            float distance = glm::length(vertices[i].position);
            assert(std::abs(distance - RADIUS) < cell && "정점이 구 표면 근처에 있어야 한다");
        }
        for (uint32_t i = 0; i < written; i += 3) {
            glm::vec3 a = vertices[i].position;
            glm::vec3 b = vertices[i + 1].position;
            glm::vec3 c = vertices[i + 2].position;
            glm::vec3 normal = glm::cross(b - a, c - a);
            glm::vec3 center = (a + b + c) / 3.0F;
            assert(glm::dot(normal, center) > 0.0F && "구 표면 삼각형이 뒤집혔다");
        }
        std::printf("  구 표면 삼각형 %u개\n", written / 3);
    }

    std::printf("마칭 큐브 자체 점검 통과\n");
    return 0;
}
