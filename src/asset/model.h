#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
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

// 텍스처 화소 포맷. RGBA8 는 stb 가 푼 것이고 나머지는 KTX2 에 미리 압축된 블록이다. 색 공간(sRGB)은
// 재질 슬롯이 정하므로 여기 넣지 않고 GPU 포맷을 고를 때 붙인다.
enum class TextureFormat : uint32_t { RGBA8, BC1, BC3, BC4, BC5, BC7 };

bool isBlockCompressed(TextureFormat format);
// 밉 단계 하나가 차지하는 바이트 수. 블록 포맷은 4x4 블록 단위로 올림한다.
size_t textureLevelBytes(TextureFormat format, uint32_t width, uint32_t height);

struct Texture {
    std::string name;
    uint32_t width = 0;
    uint32_t height = 0;
    // 기저 색과 방사는 sRGB, 나머지는 선형으로 해석한다.
    bool srgb = false;
    SamplerDesc sampler;
    // 적재 직후에는 인코딩된 바이트만 담고, 디코딩은 나중에 병렬로 한다.
    std::vector<uint8_t> encoded;
    TextureFormat format = TextureFormat::RGBA8;
    // pixels 에 담긴 밉 단계 수. 1 이면 GPU 가 밉을 만든다(RGBA8 만 가능).
    uint32_t mipLevels = 1;
    // 밉 0 부터 차례로 이어 붙인 화소. 단계마다 textureLevelBytes 만큼이다.
    std::vector<uint8_t> pixels;
};

// 셰이더의 scalar 레이아웃 정점과 동일한 배치여야 한다. 28 바이트. 위치는 단순화와 가속 구조가
// float 그대로 읽어야 하고, UV 는 half 로 줄이면 4K 텍스처에서 텍셀 두 개씩 어긋나 float 로 둔다.
struct Vertex {
    glm::vec3 position;
    // 8진법 snorm16x2. asset/vertex_pack.h 로 넣고 셰이더의 decodeUnitVector 로 푼다.
    uint32_t normal = 0;
    // 방향은 8진법 snorm16x2, 손 방향(w 부호)은 y 의 최하위 비트. packTangent/decodeTangent.
    uint32_t tangent = 0;
    glm::vec2 uv{0.0F};
};
static_assert(sizeof(Vertex) == 28, "정점 배치가 셰이더와 어긋난다");

// 스킨 정점 하나의 조인트 넷(바이트씩)과 가중치 넷(unorm8). 스킨 메쉬만 가지며 정점과 순서가 같다.
// 정점에서 떼어 둔 것은 스킨이 없는 대다수 메쉬가 정점마다 8 바이트를 버리지 않게 하려는 것이다.
struct SkinWeight {
    uint32_t joints = 0;
    uint32_t weights = 0;
};

// 스킨 하나가 가질 수 있는 조인트 수. 스킨 가중치가 조인트 번호를 바이트 하나에 담기 때문에 생기는 한계다.
//
// ponytail: 더 큰 스켈레톤이 필요하면 SkinWeight::joints 를 uvec2 로 넓혀 16비트씩 담으면 된다.
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
    uint32_t vertexOffset = 0;   // Mesh::meshletVertices 기준. 이 meshlet 이 쓰는 정점 번호가 연속으로 놓인다.
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
    // 원본 정점 그대로. 모든 LOD 단계와 meshlet 이 이 하나를 공유한다. meshlet 마다 복제하면 경계
    // 정점과 LOD 단계마다 사본이 생겨 원본의 세 배가 되므로 그렇게 하지 않는다.
    std::vector<Vertex> vertices;
    // 스킨 메쉬만 정점 수만큼 갖는다. 스킨 컴퓨트가 정점과 같은 번호로 읽는다.
    std::vector<SkinWeight> skinWeights;
    // meshlet 단위로 삼각형이 연속되도록 재배열된다. 고전 경로와 광선 경로가 쓴다.
    std::vector<uint32_t> indices;

    std::vector<Meshlet> meshlets;
    std::vector<MeshLod> lods;
    // meshlet 마다 쓰는 정점의 번호(vertices 기준). mesh shader 가 이 번호로 정점을 읽는다.
    std::vector<uint32_t> meshletVertices;
    // meshlet 안에서만 쓰는 지역 정점 인덱스. meshletVertices 구간 안의 위치다.
    std::vector<uint8_t> meshletTriangles;
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

struct LoadProgress;

// 파일을 열지 못하거나 지원하지 않는 내용이면 사유를 로그에 남기고 비어 있는 값을 돌려준다. 종료는
// 부르는 쪽이 정한다. 백그라운드 스레드에서도 부르므로 여기서 프로세스를 끝내면 안 된다.
// progress 를 주면 파싱과 정점 변환 단계의 진행을 적는다. jobs 를 주면 프리미티브 변환을 워커에 나눈다.
std::optional<Model>
loadGltf(const std::filesystem::path& path, LoadProgress* progress = nullptr, core::JobSystem* jobs = nullptr);

// clip 을 time 위치에서 표본화해 노드마다 세계 변환을 만든다. clip 이 범위를 벗어나면 바인드 포즈다.
void poseNodes(const Skeleton& skeleton, uint32_t clip, float time, std::vector<glm::mat4>& worlds);

// 세계 변환에서 스킨 하나의 조인트 행렬을 뽑는다. 셰이더가 이 행렬로 정점을 옮긴다.
void skinMatrices(const Skeleton& skeleton,
                  const std::vector<glm::mat4>& worlds,
                  uint32_t skin,
                  std::vector<glm::mat4>& out);

// 정점 캐시와 오버드로를 최적화한 뒤 meshlet 으로 나누고, 단계별로 묶어 단순화해 LOD DAG 를 만든다.
// 정점 버퍼는 그대로 두고, meshlet 마다 쓰는 정점 번호 목록과 LOD 단계별로 이어진 인덱스 버퍼를 만든다.
// progress 를 주면 lodWorkEstimate(mesh) 만큼을 그룹 단위로 나눠 더한다. 총량은 부르는 쪽이 미리 잡는다.
void buildLodHierarchy(Mesh& mesh, core::JobSystem* jobs = nullptr, LoadProgress* progress = nullptr);

// buildLodHierarchy 가 progress 에 더하는 총량. 단계마다 절반씩 줄어드는 인덱스 수의 합을 어림한 값이다.
uint64_t lodWorkEstimate(const Mesh& mesh);

// 인코딩된 바이트를 푼다. PNG/JPEG 는 RGBA8 한 장으로, KTX2 는 담긴 블록과 밉을 그대로 꺼낸다.
// 텍스처마다 독립이라 병렬로 돌릴 수 있다.
void decodeTexture(Texture& texture);

// KTX2 식별자로 시작하는지.
bool isKtx2(const std::vector<uint8_t>& bytes);
// 초압축 없는 KTX2(BC1/3/4/5/7 또는 RGBA8, 2D 한 장)를 푼다. 실패하면 사유를 로그에 남기고 false.
bool loadKtx2(const std::vector<uint8_t>& bytes, Texture& texture);

} // namespace asset
