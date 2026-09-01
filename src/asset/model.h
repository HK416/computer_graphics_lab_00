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

} // namespace asset
