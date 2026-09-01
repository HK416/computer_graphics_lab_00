#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include <glm/gtc/quaternion.hpp>
#include <glm/mat4x4.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>

namespace core {
class JobSystem;
} // namespace core

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
    // 적재 직후에는 인코딩된 바이트만 담고, 디코딩은 나중에 병렬로 한다.
    std::vector<uint8_t> encoded;
    std::vector<uint8_t> pixels; // RGBA8
};

// 셰이더의 scalar 레이아웃 정점과 동일한 배치여야 한다.
struct Vertex {
    glm::vec3 position;
    glm::vec3 normal;
    glm::vec4 tangent;
    glm::vec2 uv;
    // 스킨 조인트 넷을 바이트 하나씩, 가중치 넷을 unorm8 로 담는다. 스킨이 없으면 둘 다 0 이다.
    uint32_t joints = 0;
    uint32_t weights = 0;
};

// 스킨 하나가 가질 수 있는 조인트 수. 정점이 조인트 번호를 바이트 하나에 담기 때문에 생기는 한계다.
//
// ponytail: 더 큰 스켈레톤이 필요하면 정점의 joints 를 uvec2 로 넓혀 16비트씩 담으면 된다.
inline constexpr size_t MAX_SKIN_JOINTS = 256;

enum class AnimationPath : uint32_t {
    TRANSLATION = 0,
    ROTATION = 1,
    SCALE = 2,
};

// glTF 노드 하나의 지역 변환. 애니메이션이 이 값을 덮어쓴 뒤 계층을 따라 세계 변환을 만든다.
struct Node {
    int32_t parent = -1;
    glm::vec3 translation{0.0F};
    glm::quat rotation{1.0F, 0.0F, 0.0F, 0.0F};
    glm::vec3 scale{1.0F};
};

struct Skin {
    // Skeleton::nodes 인덱스. 정점의 조인트 번호는 이 배열 기준이다.
    std::vector<uint32_t> joints;
    std::vector<glm::mat4> inverseBind;
};

// 시간 표본과 값. 회전은 xyzw 사원수로, 나머지는 xyz 로 읽는다.
struct AnimationSampler {
    std::vector<float> times;
    std::vector<glm::vec4> values;
    bool step = false;
};

struct AnimationChannel {
    uint32_t sampler = 0;
    uint32_t node = 0;
    AnimationPath path = AnimationPath::TRANSLATION;
};

struct Animation {
    std::string name;
    float duration = 0.0F;
    std::vector<AnimationSampler> samplers;
    std::vector<AnimationChannel> channels;
};

// 한 모델의 노드 계층과 스킨, 애니메이션. 인스턴스는 skin 으로 여기를 가리킨다.
struct Skeleton {
    std::vector<Node> nodes;
    std::vector<Skin> skins;
    std::vector<Animation> animations;
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
    uint32_t indexOffset = 0;    // Mesh::indices 기준. meshlet 의 삼각형이 연속으로 놓인다.
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
    // Skeleton::skins 인덱스. 스킨이 없으면 -1.
    int32_t skin = -1;
};

struct Model {
    std::string name;
    std::vector<Texture> textures;
    std::vector<Mesh> meshes;
    std::vector<Material> materials;
    std::vector<Instance> instances;
    Skeleton skeleton;
};

Model loadGltf(const std::filesystem::path& path);

// clip 을 time 위치에서 표본화해 노드마다 세계 변환을 만든다. clip 이 범위를 벗어나면 바인드 포즈다.
void poseNodes(const Skeleton& skeleton, uint32_t clip, float time, std::vector<glm::mat4>& worlds);

// 세계 변환에서 스킨 하나의 조인트 행렬을 뽑는다. 셰이더가 이 행렬로 정점을 옮긴다.
void skinMatrices(const Skeleton& skeleton,
                  const std::vector<glm::mat4>& worlds,
                  uint32_t skin,
                  std::vector<glm::mat4>& out);

// 정점 캐시와 오버드로를 최적화한 뒤 meshlet 으로 나누고, 단계별로 묶어 단순화해 LOD DAG 를 만든다.
// 정점 버퍼는 meshlet 마다 정점을 소유하도록, 인덱스 버퍼는 LOD 단계별로 이어지도록 다시 만들어진다.
void buildLodHierarchy(Mesh& mesh, core::JobSystem* jobs = nullptr);

// 인코딩된 바이트를 RGBA8 로 푼다. 텍스처마다 독립이라 병렬로 돌릴 수 있다.
void decodeTexture(Texture& texture);

} // namespace asset
