#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace asset {

// 재질별로 다른 파이프라인 경로를 타므로 렌더러가 이 값으로 그리기 순서를 나눈다.
enum class AlphaMode : uint32_t {
    SOLID = 0,
    CUTOFF = 1,
    TRANSLUCENT = 2,
};

// 셰이더의 scalar 레이아웃 정점과 동일한 배치여야 한다.
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec4 tangent;
    glm::vec2 uv;
};

struct Mesh {
    std::string name;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    glm::vec3 boundsCenter{0.0F};
    float boundsRadius = 0.0F;
    uint32_t materialIndex = 0;
};

struct Material {
    std::string name;
    glm::vec4 baseColorFactor{1.0F};
    glm::vec3 emissiveFactor{0.0F};
    float metallicFactor = 1.0F;
    float roughnessFactor = 1.0F;
    float alphaCutoff = 0.5F;
    AlphaMode alphaMode = AlphaMode::SOLID;
    bool doubleSided = false;
};

struct Instance {
    std::string name;
    uint32_t meshIndex = 0;
    glm::mat4 transform{1.0F};
};

struct Model {
    std::string name;
    std::vector<Mesh> meshes;
    std::vector<Material> materials;
    std::vector<Instance> instances;
};

Model loadGltf(const std::filesystem::path& path);

} // namespace asset
