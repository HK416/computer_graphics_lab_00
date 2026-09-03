#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

#include "asset/model.h"
#include "gfx/resources.h"

namespace gfx {

struct Context;

// 아래 세 구조체는 shaders/scene_data.glsl 의 scalar 레이아웃 정의와 일치해야 한다.
struct GpuMesh {
    glm::vec4 boundingSphere; // xyz 중심, w 반지름
    uint32_t indexOffset;
    uint32_t indexCount;
    int32_t vertexOffset;
    uint32_t materialIndex;
    uint32_t meshletOffset;
    uint32_t meshletCount;
    uint32_t lodOffset;
    uint32_t lodCount;
};

struct GpuMeshLod {
    uint32_t indexOffset;
    uint32_t indexCount;
    uint32_t meshletOffset;
    uint32_t meshletCount;
};

struct GpuMeshlet {
    glm::vec4 boundingSphere; // xyz 중심, w 반지름
    glm::vec4 cone;           // xyz 법선 원뿔 축, w 컷오프
    glm::vec4 errorSphere;    // 자신이 나온 그룹의 경계 구
    glm::vec4 parentSphere;   // 부모 그룹의 경계 구
    float error;
    float parentError;
    uint32_t indexOffset;    // 전역 인덱스 버퍼 기준
    uint32_t vertexOffset;   // 전역 meshlet 정점 목록 기준. 목록의 값이 전역 정점 번호다.
    uint32_t triangleOffset; // 전역 meshlet 삼각형 버퍼 기준
    uint32_t vertexCount;
    uint32_t triangleCount;
    uint32_t level;
    uint32_t padding;
};

struct GpuMaterial {
    glm::vec4 baseColorFactor;
    glm::vec4 emissiveAndCutoff; // xyz 방사, w 알파 컷오프
    float metallicFactor;
    float roughnessFactor;
    float normalScale;
    float occlusionStrength;
    // bindless 슬롯 번호. 없으면 asset::INVALID_TEXTURE.
    uint32_t baseColorTexture;
    uint32_t metallicRoughnessTexture;
    uint32_t normalTexture;
    uint32_t occlusionTexture;
    uint32_t emissiveTexture;
    uint32_t alphaMode;
    uint32_t flags; // bit 0: 양면
    uint32_t padding;
};

// 스킨이 없는 인스턴스의 조인트 오프셋. 셰이더가 이 값으로 스키닝 여부를 가른다.
inline constexpr uint32_t NO_JOINTS = 0xFFFFFFFFU;

// 그리지 않는 오브젝트의 인스턴스 슬롯.
inline constexpr uint32_t INVALID_INSTANCE_SLOT = 0xFFFFFFFFU;

// 스킨 결과를 따로 뽑아 두지 않은 인스턴스의 정점 오프셋.
inline constexpr uint32_t NO_SKINNED_VERTICES = 0xFFFFFFFFU;

// 스킨 가중치가 없는 메쉬의 가중치 오프셋. 스킨 컴퓨트가 이 값이면 정점을 그대로 복사한다.
inline constexpr uint32_t NO_SKIN_WEIGHTS = 0xFFFFFFFFU;

struct GpuInstance {
    glm::mat4 model;
    // 지난 프레임의 세계 변환. 모션 벡터 전용이며, 이전 값이 없으면 model 과 같은 값이 들어간다.
    glm::mat4 previousModel;
    glm::mat4 normalMatrix;
    uint32_t meshIndex;
    // 재질 경로와 면 방향 조합. 컬링 컴퓨트가 이 값으로 그리기 명령 구간을 고른다.
    uint32_t bucket;
    uint32_t bucketBase;
    // 프레임 조인트 버퍼에서 이 인스턴스의 조인트 행렬이 시작하는 위치. 스킨이 없으면 NO_JOINTS.
    uint32_t jointOffset;
    // 스킨 컴퓨트가 이 인스턴스의 변형 정점을 써 둔 위치. 래스터와 광선 경로가 모두 여기서 정점을
    // 읽는다. 없으면 NO_SKINNED_VERTICES.
    uint32_t skinnedVertexOffset;
    // 지난 프레임 포즈의 변형 정점. 지난 포즈가 없으면 skinnedVertexOffset 과 같다.
    uint32_t previousSkinnedVertexOffset;
    // 변형 정점에서 다시 잰 meshlet 경계 구가 시작하는 위치.
    uint32_t skinnedMeshletOffset;
    // 이 인스턴스의 meshlet 가시성 비트가 시작하는 번호. 메쉬의 모든 LOD 단계 meshlet 이 이어진다.
    uint32_t visibilityBase;
};

// 모든 모델의 정점과 인덱스를 하나의 버퍼로 합쳐 간접 그리기 한 번으로 장면 전체를 그릴 수 있게 한다.
class GeometryStore {
public:
    explicit GeometryStore(Context& context);
    ~GeometryStore();
    GeometryStore(const GeometryStore&) = delete;
    GeometryStore& operator=(const GeometryStore&) = delete;

    // 모델을 누적하고 이 모델의 메쉬가 시작되는 전역 인덱스를 돌려준다.
    // textureSlots 는 model.textures 와 같은 순서의 bindless 슬롯 번호다.
    uint32_t addModel(const asset::Model& model, const std::vector<uint32_t>& textureSlots);
    void build();

    const GpuMesh& mesh(uint32_t index) const { return meshes[index]; }
    const asset::Material& material(uint32_t index) const { return sourceMaterials[index]; }
    const std::string& meshName(uint32_t index) const { return meshNames[index]; }
    // 스킨 결과 버퍼를 잡고 가속 구조 범위를 정하는 데 쓴다. GpuMesh 는 정점 수를 담지 않는다.
    uint32_t meshVertexCount(uint32_t index) const { return meshVertexCounts[index]; }
    // 스킨 가중치 버퍼에서 이 메쉬의 구간이 시작하는 위치. 스킨이 없으면 NO_SKIN_WEIGHTS.
    uint32_t meshSkinOffset(uint32_t index) const { return meshSkinOffsets[index]; }
    uint32_t meshCount() const { return static_cast<uint32_t>(meshes.size()); }
    uint32_t meshletCount() const { return static_cast<uint32_t>(meshlets.size()); }
    const GpuMeshLod& lod(uint32_t index) const { return lods[index]; }
    const GpuMeshlet& meshlet(uint32_t index) const { return meshlets[index]; }
    uint32_t maxLodCount() const { return maxLods; }

    Buffer vertexBuffer;
    // 스킨 메쉬의 정점별 조인트와 가중치. 정점과 떼어 두어 스킨 없는 메쉬는 자리를 쓰지 않는다.
    Buffer skinWeightBuffer;
    Buffer indexBuffer;
    Buffer meshBuffer;
    Buffer materialBuffer;
    Buffer meshletBuffer;
    Buffer lodBuffer;
    // meshlet 안의 지역 정점 인덱스. 8비트 저장을 요구하지 않으려고 uint32 로 펼쳐 둔다.
    Buffer meshletTriangleBuffer;
    // meshlet 마다 쓰는 정점의 전역 번호. mesh shader 가 지역 인덱스를 이 목록으로 풀어 정점을 읽는다.
    Buffer meshletVertexBuffer;

private:
    Context& context;
    std::vector<asset::Vertex> vertices;
    std::vector<asset::SkinWeight> skinWeights;
    std::vector<uint32_t> indices;
    std::vector<GpuMesh> meshes;
    std::vector<GpuMeshlet> meshlets;
    std::vector<GpuMeshLod> lods;
    uint32_t maxLods = 1;
    std::vector<uint32_t> meshletTriangles;
    std::vector<uint32_t> meshletVertices;
    std::vector<GpuMaterial> materials;
    std::vector<asset::Material> sourceMaterials;
    std::vector<std::string> meshNames;
    std::vector<uint32_t> meshVertexCounts;
    std::vector<uint32_t> meshSkinOffsets;
};

} // namespace gfx
