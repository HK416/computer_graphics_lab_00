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
class Uploader;

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
    uint32_t flags; // MATERIAL_FLAG_*
    uint32_t padding;
};

// GpuMaterial::flags. shaders/scene_types.glsl 의 MATERIAL_FLAG_* 와 같아야 한다.
inline constexpr uint32_t MATERIAL_FLAG_DOUBLE_SIDED = 1U;
// 노멀 맵이 두 채널(BC4/BC5)이라 셰이더가 z 를 복원한다.
inline constexpr uint32_t MATERIAL_FLAG_TWO_CHANNEL_NORMAL = 2U;

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
//
// 정점·인덱스·meshlet 목록 같은 큰 배열은 업로드하고 나면 CPU 에 남기지 않는다. 모델이 더해지면 GPU
// 버퍼를 새 크기로 잡고 옛 내용을 GPU 안에서 옮긴 뒤 새 모델 몫만 올린다. 메쉬·재질·meshlet 표는
// 렌더러가 CPU 에서도 읽으므로 그대로 둔다.
class GeometryStore {
public:
    explicit GeometryStore(Context& context);
    ~GeometryStore();
    GeometryStore(const GeometryStore&) = delete;
    GeometryStore& operator=(const GeometryStore&) = delete;

    // 모델 하나가 표에서 차지하는 연속 구간. removeModel 이 되돌리는 데 쓴다.
    struct ModelRange {
        uint32_t meshBase = 0;
        uint32_t meshCount = 0;
        uint32_t materialBase = 0;
        uint32_t materialCount = 0;
    };

    // 모델을 누적하고 이 모델의 메쉬가 시작되는 전역 인덱스 등을 돌려준다. 해제된 자리가 있으면 거기부터 채운다.
    // textureSlots 는 model.textures 와 같은 순서의 bindless 슬롯 번호다.
    ModelRange addModel(const asset::Model& model, const std::vector<uint32_t>& textureSlots);
    // 모델이 쓰던 구간을 모두 돌려준다. 메쉬 번호는 남은 장면과 되돌리기 기록이 들고 있으므로 옮기지
    // 않고, 항목만 비워 무덤으로 남긴다(meshLive 가 거짓). 큰 배열의 구간은 다음 모델이 재활용하고,
    // 꼬리였다면 다음 build 에서 버퍼가 실제로 줄어든다. 호출 전에 장치가 놀고 있어야 한다.
    void removeModel(const ModelRange& range);
    // 마지막 build 뒤에 더해지거나 해제된 것을 GPU 에 반영한다. 버퍼를 새로 잡으므로 호출 전에 장치가
    // 놀고 있어야 하고, 부르는 쪽이 주소를 다시 읽어야 한다(가속 구조 등).
    void build();

    // 이 모델이 GPU 에서 차지할 바이트. 올리기 전에 예산과 견주는 데 쓴다.
    static VkDeviceSize estimateModelBytes(const asset::Model& model);
    // 지금 GPU 에 올라가 있는 큰 배열의 바이트. build 가 키울 때 옛 버퍼와 새 버퍼가 잠시 함께 있으므로
    // 그 겹침만큼도 예산에 넣어야 한다.
    VkDeviceSize residentBytes() const;
    // 더해졌지만 아직 올리지 않은 꼬리의 바이트. 장면 파일이 모델 여럿을 build 하나로 올릴 때 앞 모델
    // 몫을 다음 모델의 예산 검사에 넣기 위한 것이다.
    VkDeviceSize pendingBytes() const;

    const GpuMesh& mesh(uint32_t index) const { return meshes[index]; }
    const asset::Material& material(uint32_t index) const { return sourceMaterials[index]; }
    const std::string& meshName(uint32_t index) const { return meshNames[index]; }
    // 스킨 결과 버퍼를 잡고 가속 구조 범위를 정하는 데 쓴다. GpuMesh 는 정점 수를 담지 않는다.
    uint32_t meshVertexCount(uint32_t index) const { return meshVertexCounts[index]; }
    // 스킨 가중치 버퍼에서 이 메쉬의 구간이 시작하는 위치. 스킨이 없으면 NO_SKIN_WEIGHTS.
    uint32_t meshSkinOffset(uint32_t index) const { return meshSkinOffsets[index]; }
    uint32_t meshCount() const { return static_cast<uint32_t>(meshes.size()); }
    // 해제된 모델의 메쉬는 번호를 지키려고 빈 항목(무덤)으로 남는다. 그리거나 가속 구조를 세우기 전에 본다.
    bool meshLive(uint32_t index) const { return index < meshes.size() && meshes[index].lodCount > 0; }
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
    // meshlet 안의 지역 정점 인덱스(8비트). uint32 하나에 넷씩 묶여 있고 셰이더가 비트 이동으로 꺼낸다.
    // 8비트 스토리지 기능 없이도 되고 uint32 로 펼치던 때보다 4 배 작다.
    Buffer meshletTriangleBuffer;
    // meshlet 마다 쓰는 정점의 전역 번호. mesh shader 가 지역 인덱스를 이 목록으로 풀어 정점을 읽는다.
    Buffer meshletVertexBuffer;

private:
    struct Range {
        size_t offset = 0;
        size_t count = 0;
    };
    // 빈 구간 목록으로 연속 구간을 나눠 준다. 해제된 자리를 first-fit 으로 재활용하고 없으면 꼬리에 붙인다.
    // 꼬리가 해제되면 total 이 줄어 버퍼도 실제로 작아질 수 있다. 큰 배열과 CPU 표가 함께 쓴다.
    struct RangeAllocator {
        std::vector<Range> freeRanges;
        size_t total = 0;
        size_t allocate(size_t count);
        void release(size_t offset, size_t count);
    };
    // 큰 배열 하나. GPU 에 올라간 것은 CPU 에 남기지 않고, 아직 올리지 않은 조각만 pending 에 둔다.
    // build 가 조각을 제자리에 올리고 비운다.
    template <typename T> struct GrowingArray {
        struct Chunk {
            size_t offset;
            std::vector<T> data;
        };
        RangeAllocator ranges;
        std::vector<Chunk> pending;
        // GPU 버퍼에 유효한 원소 수. 버퍼를 다시 잡을 때 이만큼을 옮긴다.
        size_t uploaded = 0;
        size_t total() const { return ranges.total; }
        size_t allocate(size_t count) { return ranges.allocate(count); }
        void write(size_t offset, const T* data, size_t count) {
            if (count > 0) {
                pending.push_back(Chunk{offset, std::vector<T>(data, data + count)});
            }
        }
        void release(size_t offset, size_t count);
        size_t pendingCount() const;
    };
    // 버퍼를 array.total 크기에 맞추고(키우거나 절반 이하로 줄이고) 조각을 올린다. 옛 버퍼는 flush 뒤에
    // 지우도록 retired 에 넣는다.
    template <typename T>
    void growAndUpload(Uploader& uploader,
                       Buffer& buffer,
                       GrowingArray<T>& array,
                       VkBufferUsageFlags usage,
                       const char* name,
                       std::vector<Buffer>& retired);
    // CPU 표에서 연속 구간을 잡는다. 표는 필요한 만큼 늘리고 해제된 꼬리만큼 줄인다.
    template <typename T> uint32_t allocateTable(std::vector<T>& table, RangeAllocator& ranges, size_t count);
    template <typename T> void releaseTable(std::vector<T>& table, RangeAllocator& ranges, size_t offset, size_t count);

    Context& context;
    GrowingArray<asset::Vertex> vertices;
    GrowingArray<asset::SkinWeight> skinWeights;
    GrowingArray<uint32_t> indices;
    GrowingArray<uint8_t> meshletTriangles;
    GrowingArray<uint32_t> meshletVertices;
    std::vector<GpuMesh> meshes;
    std::vector<GpuMeshlet> meshlets;
    std::vector<GpuMeshLod> lods;
    std::vector<GpuMaterial> materials;
    RangeAllocator meshRanges;
    RangeAllocator meshletRanges;
    RangeAllocator lodRanges;
    RangeAllocator materialRanges;
    uint32_t maxLods = 1;
    // 표가 바뀌어(더하거나 해제) 다시 올려야 한다.
    bool tablesDirty = false;
    std::vector<asset::Material> sourceMaterials;
    std::vector<std::string> meshNames;
    std::vector<uint32_t> meshVertexCounts;
    std::vector<uint32_t> meshSkinOffsets;
    // 해제할 때 돌려줘야 하는 메쉬별 구간. GpuMesh 는 meshlet 삼각형·정점 목록의 구간을 담지 않는다.
    std::vector<Range> meshTriangleRanges;
    std::vector<Range> meshMeshletVertexRanges;
};

} // namespace gfx
