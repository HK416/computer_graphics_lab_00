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

// meshoptimizer 로 나눈 meshlet 하나. mesh shader 경로와 클러스터 컬링, LOD 선정에 함께 쓴다.
//
// LOD 선정은 Nanite 방식을 따른다. 자신이 나온 그룹의 오차(error, errorSphere)가 화면에서 충분히
// 작고, 부모 그룹의 오차(parentError, parentSphere)는 너무 클 때만 이 meshlet 을 그린다. 같은 그룹의
// meshlet 은 같은 판정을 받으므로 경계에 틈이 생기지 않는다.
struct Meshlet {
    glm::vec4 boundingSphere{0.0F}; // xyz 중심, w 반지름
    glm::vec4 cone{0.0F};           // xyz 법선 원뿔 축, w 컷오프
    glm::vec4 errorSphere{0.0F};    // 자신이 나온 그룹의 경계 구
    glm::vec4 parentSphere{0.0F};   // 부모 그룹의 경계 구
    float error = 0.0F;
    float parentError = 0.0F;
    uint32_t vertexOffset = 0;   // Mesh::vertices 기준. meshlet 마다 정점을 따로 소유한다.
    uint32_t triangleOffset = 0; // Mesh::meshletTriangles 기준
    uint32_t vertexCount = 0;
    uint32_t triangleCount = 0;
    uint32_t level = 0;
    uint32_t padding = 0;
};

// LOD 단계 하나가 차지하는 인덱스와 meshlet 구간.
struct MeshLod {
    uint32_t indexOffset = 0;
    uint32_t indexCount = 0;
    uint32_t meshletOffset = 0;
    uint32_t meshletCount = 0;
};

struct Mesh {
    std::string name;
    std::vector<Vertex> vertices;
    // meshlet 단위로 삼각형이 연속되도록 재배열된다.
    std::vector<uint32_t> indices;

    std::vector<Meshlet> meshlets;
    std::vector<MeshLod> lods;
    // meshlet 안에서만 쓰는 지역 정점 인덱스. mesh shader 경로가 그대로 쓴다.
    std::vector<uint8_t> meshletTriangles;
    // 정점마다 속한 meshlet 번호. flat 보간으로 프래그먼트까지 내려 시각화에 쓴다.
    std::vector<uint32_t> vertexMeshlets;
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

// 정점 캐시와 오버드로를 최적화한 뒤 meshlet 으로 나누고, 단계별로 묶어 단순화해 LOD DAG 를 만든다.
// 정점 버퍼는 meshlet 마다 정점을 소유하도록, 인덱스 버퍼는 LOD 단계별로 이어지도록 다시 만들어진다.
void buildLodHierarchy(Mesh& mesh);

} // namespace asset
