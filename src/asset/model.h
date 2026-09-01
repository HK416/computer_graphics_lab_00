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

inline constexpr uint32_t INVALID_TEXTURE = 0xFFFFFFFFU;

// glTF 원본 열거값을 그대로 담고, Vulkan 값으로의 변환은 gfx 계층에서 한다.
struct SamplerDesc {
    uint32_t magFilter = 0x2601; // LINEAR
    uint32_t minFilter = 0x2703; // LINEAR_MIPMAP_LINEAR
    uint32_t wrapS = 0x2901;     // REPEAT
    uint32_t wrapT = 0x2901;

    bool operator==(const SamplerDesc&) const = default;
};

struct Texture {
    std::string name;
    uint32_t width = 0;
    uint32_t height = 0;
    // 기저 색과 방사는 sRGB, 나머지는 선형으로 해석한다.
    bool srgb = false;
    SamplerDesc sampler;
    std::vector<uint8_t> pixels; // RGBA8
};

// 셰이더의 scalar 레이아웃 정점과 동일한 배치여야 한다.
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec4 tangent;
    glm::vec2 uv;
};

// meshoptimizer 로 나눈 meshlet 하나. mesh shader 경로와 클러스터 컬링에 함께 쓴다.
struct Meshlet {
    glm::vec4 boundingSphere{0.0F}; // xyz 중심, w 반지름
    glm::vec4 cone{0.0F};           // xyz 법선 원뿔 축, w 컷오프
    uint32_t vertexOffset = 0;      // Mesh::meshletVertices 기준
    uint32_t triangleOffset = 0;    // Mesh::meshletTriangles 기준
    uint32_t vertexCount = 0;
    uint32_t triangleCount = 0;
};

struct Mesh {
    std::string name;
    std::vector<Vertex> vertices;
    // meshlet 단위로 삼각형이 연속되도록 재배열된다.
    std::vector<uint32_t> indices;

    std::vector<Meshlet> meshlets;
    std::vector<uint32_t> meshletVertices;
    std::vector<uint8_t> meshletTriangles;
    // 삼각형마다 속한 meshlet 번호. 고전 경로의 meshlet 시각화에 쓴다.
    std::vector<uint32_t> triangleMeshlets;
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
    float normalScale = 1.0F;
    float occlusionStrength = 1.0F;
    AlphaMode alphaMode = AlphaMode::SOLID;
    bool doubleSided = false;

    // Model::textures 의 인덱스. 없으면 INVALID_TEXTURE.
    uint32_t baseColorTexture = INVALID_TEXTURE;
    uint32_t metallicRoughnessTexture = INVALID_TEXTURE;
    uint32_t normalTexture = INVALID_TEXTURE;
    uint32_t occlusionTexture = INVALID_TEXTURE;
    uint32_t emissiveTexture = INVALID_TEXTURE;
};

struct Instance {
    std::string name;
    uint32_t meshIndex = 0;
    glm::mat4 transform{1.0F};
};

struct Model {
    std::string name;
    std::vector<Texture> textures;
    std::vector<Mesh> meshes;
    std::vector<Material> materials;
    std::vector<Instance> instances;
};

Model loadGltf(const std::filesystem::path& path);

// 정점 캐시와 오버드로를 최적화한 뒤 meshlet 으로 나눈다. 인덱스 버퍼도 meshlet 순서로 재배열된다.
void buildMeshlets(Mesh& mesh);

} // namespace asset
