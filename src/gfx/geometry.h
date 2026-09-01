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

struct GpuInstance {
    glm::mat4 model;
    glm::mat4 normalMatrix;
    uint32_t meshIndex;
    uint32_t padding[3];
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
    uint32_t meshCount() const { return static_cast<uint32_t>(meshes.size()); }

    Buffer vertexBuffer;
    Buffer indexBuffer;
    Buffer meshBuffer;
    Buffer materialBuffer;

private:
    Context& context;
    std::vector<asset::Vertex> vertices;
    std::vector<uint32_t> indices;
    std::vector<GpuMesh> meshes;
    std::vector<GpuMaterial> materials;
    std::vector<asset::Material> sourceMaterials;
    std::vector<std::string> meshNames;
};

} // namespace gfx
