#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <optional>
#include <vector>

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#include <glm/geometric.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/matrix_decompose.hpp>
#include <meshoptimizer.h>
#include <spdlog/spdlog.h>
#include <stb_image.h>

#include "asset/load_progress.h"
#include "asset/model.h"
#include "asset/vertex_pack.h"

namespace asset {
namespace {

const cgltf_accessor* findAttribute(const cgltf_primitive& primitive, cgltf_attribute_type type, int index) {
    for (cgltf_size i = 0; i < primitive.attributes_count; ++i) {
        const cgltf_attribute& attribute = primitive.attributes[i];
        if (attribute.type == type && attribute.index == index) {
            return attribute.data;
        }
    }
    return nullptr;
}

void readFloats(const cgltf_accessor* accessor, cgltf_size index, float* out, cgltf_size count) {
    if (accessor == nullptr || cgltf_accessor_read_float(accessor, index, out, count) == 0) {
        std::fill_n(out, count, 0.0F);
    }
}

// 조인트 번호 넷을 바이트 하나씩 담는다. 스킨 하나의 조인트 수는 MAX_SKIN_JOINTS 로 제한된다.
uint32_t packJoints(const cgltf_uint indices[4]) {
    return (indices[0] & 0xFFU) | ((indices[1] & 0xFFU) << 8U) | ((indices[2] & 0xFFU) << 16U) |
           ((indices[3] & 0xFFU) << 24U);
}

// 가중치 넷을 unorm8 로 담는다. 셰이더에서 다시 정규화하므로 양자화 오차는 상쇄된다.
uint32_t packWeights(const float weights[4]) {
    uint32_t packed = 0;
    for (uint32_t i = 0; i < 4; ++i) {
        auto quantized = static_cast<uint32_t>(std::lround(std::clamp(weights[i], 0.0F, 1.0F) * 255.0F));
        packed |= quantized << (i * 8U);
    }
    return packed;
}

// 적재 중에만 쓰는 풀린 정점. 노멀과 탄젠트를 만들고 나서 Vertex 로 압축한다.
struct LoadVertex {
    glm::vec3 position{0.0F};
    glm::vec3 normal{0.0F};
    glm::vec4 tangent{0.0F};
    glm::vec2 uv{0.0F};
};

// 노멀이 없는 프리미티브는 삼각형 면적 가중 평균으로 채운다.
void generateNormals(std::vector<LoadVertex>& vertices, const std::vector<uint32_t>& indices) {
    for (LoadVertex& vertex : vertices) {
        vertex.normal = glm::vec3{0.0F};
    }
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        LoadVertex& a = vertices[indices[i]];
        LoadVertex& b = vertices[indices[i + 1]];
        LoadVertex& c = vertices[indices[i + 2]];
        glm::vec3 faceNormal = glm::cross(b.position - a.position, c.position - a.position);
        a.normal += faceNormal;
        b.normal += faceNormal;
        c.normal += faceNormal;
    }
    for (LoadVertex& vertex : vertices) {
        float length = glm::length(vertex.normal);
        vertex.normal = length > 0.0F ? vertex.normal / length : glm::vec3{0.0F, 1.0F, 0.0F};
    }
}

// 탄젠트가 없는데 노멀 맵을 쓰는 재질이 있으면 접선 공간이 깨지므로 UV 로부터 만들어 둔다.
void generateTangents(std::vector<LoadVertex>& vertices, const std::vector<uint32_t>& indices) {
    std::vector<glm::vec3> tangents(vertices.size(), glm::vec3{0.0F});
    std::vector<glm::vec3> bitangents(vertices.size(), glm::vec3{0.0F});

    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        const LoadVertex& a = vertices[indices[i]];
        const LoadVertex& b = vertices[indices[i + 1]];
        const LoadVertex& c = vertices[indices[i + 2]];

        glm::vec3 edge1 = b.position - a.position;
        glm::vec3 edge2 = c.position - a.position;
        glm::vec2 deltaUv1 = b.uv - a.uv;
        glm::vec2 deltaUv2 = c.uv - a.uv;

        float determinant = deltaUv1.x * deltaUv2.y - deltaUv2.x * deltaUv1.y;
        if (std::abs(determinant) < 1e-12F) {
            continue;
        }
        float inverse = 1.0F / determinant;
        glm::vec3 tangent = (edge1 * deltaUv2.y - edge2 * deltaUv1.y) * inverse;
        glm::vec3 bitangent = (edge2 * deltaUv1.x - edge1 * deltaUv2.x) * inverse;

        for (size_t corner = 0; corner < 3; ++corner) {
            tangents[indices[i + corner]] += tangent;
            bitangents[indices[i + corner]] += bitangent;
        }
    }

    for (size_t i = 0; i < vertices.size(); ++i) {
        glm::vec3 normal = vertices[i].normal;
        glm::vec3 tangent = tangents[i] - normal * glm::dot(normal, tangents[i]);
        float length = glm::length(tangent);
        tangent = length > 1e-8F ? tangent / length : glm::vec3{1.0F, 0.0F, 0.0F};
        float handedness = glm::dot(glm::cross(normal, tangent), bitangents[i]) < 0.0F ? -1.0F : 1.0F;
        vertices[i].tangent = glm::vec4{tangent, handedness};
    }
}

void computeBounds(Mesh& mesh) {
    if (mesh.vertices.empty()) {
        return;
    }
    glm::vec3 minimum = mesh.vertices.front().position;
    glm::vec3 maximum = minimum;
    for (const Vertex& vertex : mesh.vertices) {
        minimum = glm::min(minimum, vertex.position);
        maximum = glm::max(maximum, vertex.position);
    }
    mesh.boundsCenter = (minimum + maximum) * 0.5F;
    mesh.boundsRadius = 0.0F;
    for (const Vertex& vertex : mesh.vertices) {
        mesh.boundsRadius = std::max(mesh.boundsRadius, glm::distance(vertex.position, mesh.boundsCenter));
    }
}

AlphaMode toAlphaMode(cgltf_alpha_mode mode) {
    switch (mode) {
    case cgltf_alpha_mode_mask:
        return AlphaMode::CUTOFF;
    case cgltf_alpha_mode_blend:
        return AlphaMode::TRANSLUCENT;
    default:
        return AlphaMode::SOLID;
    }
}

uint32_t textureIndex(const cgltf_texture_view& view, const cgltf_data& data) {
    if (view.texture == nullptr) {
        return INVALID_TEXTURE;
    }
    return static_cast<uint32_t>(view.texture - data.textures);
}

SamplerDesc toSamplerDesc(const cgltf_sampler* sampler) {
    SamplerDesc desc;
    if (sampler == nullptr) {
        return desc;
    }
    if (sampler->mag_filter != 0) {
        desc.magFilter = static_cast<uint32_t>(sampler->mag_filter);
    }
    if (sampler->min_filter != 0) {
        desc.minFilter = static_cast<uint32_t>(sampler->min_filter);
    }
    if (sampler->wrap_s != 0) {
        desc.wrapS = static_cast<uint32_t>(sampler->wrap_s);
    }
    if (sampler->wrap_t != 0) {
        desc.wrapT = static_cast<uint32_t>(sampler->wrap_t);
    }
    return desc;
}

// 인코딩된 바이트만 모아 둔다. 실제 디코딩은 decodeTexture 가 나중에 병렬로 처리한다.
bool loadTextureSource(const cgltf_image& image, const std::filesystem::path& baseDirectory, Texture& texture) {
    if (image.buffer_view != nullptr && image.buffer_view->buffer->data != nullptr) {
        const auto* bytes = static_cast<const uint8_t*>(image.buffer_view->buffer->data) + image.buffer_view->offset;
        texture.encoded.assign(bytes, bytes + image.buffer_view->size);
        return true;
    }
    if (image.uri == nullptr) {
        return false;
    }

    std::filesystem::path file = baseDirectory / image.uri;
    std::error_code error;
    auto size = std::filesystem::file_size(file, error);
    if (error || size == 0) {
        return false;
    }
    texture.encoded.resize(size);
    std::FILE* handle = std::fopen(file.string().c_str(), "rb");
    if (handle == nullptr) {
        return false;
    }
    size_t read = std::fread(texture.encoded.data(), 1, size, handle);
    std::fclose(handle);
    return read == size;
}

// 같은 이미지라도 쓰이는 슬롯에 따라 sRGB 여부가 다르므로 재질을 먼저 훑어 색 공간을 정한다.
std::vector<bool> collectSrgbFlags(const cgltf_data& data) {
    std::vector<bool> srgb(data.textures_count, false);
    for (cgltf_size i = 0; i < data.materials_count; ++i) {
        const cgltf_material& material = data.materials[i];
        if (material.has_pbr_metallic_roughness != 0) {
            uint32_t index = textureIndex(material.pbr_metallic_roughness.base_color_texture, data);
            if (index != INVALID_TEXTURE) {
                srgb[index] = true;
            }
        }
        uint32_t emissive = textureIndex(material.emissive_texture, data);
        if (emissive != INVALID_TEXTURE) {
            srgb[emissive] = true;
        }
    }
    return srgb;
}

Material convertMaterial(const cgltf_material& source, const cgltf_data& data) {
    Material material;
    material.name = source.name != nullptr ? source.name : "재질";
    material.alphaMode = toAlphaMode(source.alpha_mode);
    material.alphaCutoff = source.alpha_cutoff;
    material.doubleSided = source.double_sided != 0;
    material.emissiveFactor = glm::make_vec3(source.emissive_factor);
    material.normalScale = source.normal_texture.scale;
    material.occlusionStrength = source.occlusion_texture.scale;
    material.normalTexture = textureIndex(source.normal_texture, data);
    material.occlusionTexture = textureIndex(source.occlusion_texture, data);
    material.emissiveTexture = textureIndex(source.emissive_texture, data);
    if (source.has_pbr_metallic_roughness != 0) {
        material.baseColorFactor = glm::make_vec4(source.pbr_metallic_roughness.base_color_factor);
        material.metallicFactor = source.pbr_metallic_roughness.metallic_factor;
        material.roughnessFactor = source.pbr_metallic_roughness.roughness_factor;
        material.baseColorTexture = textureIndex(source.pbr_metallic_roughness.base_color_texture, data);
        material.metallicRoughnessTexture =
            textureIndex(source.pbr_metallic_roughness.metallic_roughness_texture, data);
    }
    return material;
}

struct PrimitiveRange {
    uint32_t first = 0;
    uint32_t count = 0;
};

// 정점 합치기 전후의 수. 모델 단위로 합쳐 로그에 남긴다.
struct WeldStats {
    size_t before = 0;
    size_t after = 0;
};

// 파일이 준 속성(위치·노멀·탄젠트·UV, 스킨이면 가중치까지)의 바이트가 완전히 같은 정점을 하나로 합친다.
// 스캔 데이터는 삼각형마다 정점을 따로 두는 일이 많은데, 그대로 두면 meshlet 이 삼각형 124 개가 아니라
// 정점 64 개 한도에 먼저 걸려 삼각형 21 개짜리로 잘게 쪼개지고 단순화는 모든 변을 경계로 보고 잠근다.
// 노멀·탄젠트를 만들기 전에 해야 한다. 만든 뒤에 하면 사본마다 인접 삼각형이 하나뿐이라 값이 달라져
// 합쳐지지 않고, 합친 뒤에 만들어야 인덱스 메쉬와 같은 평균값이 나온다. 속성이 조금이라도 다르면
// 합치지 않으므로 화면은 그대로다. 아무 삼각형도 쓰지 않는 정점도 여기서 빠진다.
void weldVertices(std::vector<LoadVertex>& vertices,
                  std::vector<SkinWeight>& skinWeights,
                  std::vector<uint32_t>& indices,
                  WeldStats& stats) {
    size_t before = vertices.size();
    stats.before += before;
    if (before == 0 || indices.empty()) {
        stats.after += before;
        return;
    }
    struct Key {
        LoadVertex vertex;
        SkinWeight weight;
    };
    static_assert(sizeof(LoadVertex) == 48 && sizeof(Key) == sizeof(LoadVertex) + sizeof(SkinWeight),
                  "채움 바이트가 있으면 비교가 틀어진다");
    std::vector<Key> keys(before);
    for (size_t i = 0; i < before; ++i) {
        keys[i] = Key{vertices[i], skinWeights.empty() ? SkinWeight{} : skinWeights[i]};
    }
    std::vector<uint32_t> remap(before);
    size_t unique =
        meshopt_generateVertexRemap(remap.data(), indices.data(), indices.size(), keys.data(), before, sizeof(Key));
    stats.after += unique;
    if (unique == before) {
        return;
    }

    std::vector<LoadVertex> welded(unique);
    meshopt_remapVertexBuffer(welded.data(), vertices.data(), before, sizeof(LoadVertex), remap.data());
    vertices = std::move(welded);
    if (!skinWeights.empty()) {
        std::vector<SkinWeight> weldedWeights(unique);
        meshopt_remapVertexBuffer(weldedWeights.data(), skinWeights.data(), before, sizeof(SkinWeight), remap.data());
        skinWeights = std::move(weldedWeights);
    }
    // 원소마다 destination[i] = remap[indices[i]] 이라 같은 버퍼를 제자리에서 바꿔도 된다.
    meshopt_remapIndexBuffer(indices.data(), indices.data(), indices.size(), remap.data());
}

void appendPrimitive(Model& model, const cgltf_data& data, const cgltf_primitive& primitive, WeldStats& weld) {
    const cgltf_accessor* positions = findAttribute(primitive, cgltf_attribute_type_position, 0);
    if (positions == nullptr || primitive.type != cgltf_primitive_type_triangles) {
        return;
    }
    const cgltf_accessor* normals = findAttribute(primitive, cgltf_attribute_type_normal, 0);
    const cgltf_accessor* tangents = findAttribute(primitive, cgltf_attribute_type_tangent, 0);
    const cgltf_accessor* uvs = findAttribute(primitive, cgltf_attribute_type_texcoord, 0);
    const cgltf_accessor* joints = findAttribute(primitive, cgltf_attribute_type_joints, 0);
    const cgltf_accessor* weights = findAttribute(primitive, cgltf_attribute_type_weights, 0);

    Mesh mesh;
    mesh.name =
        primitive.material != nullptr && primitive.material->name != nullptr ? primitive.material->name : "프리미티브";
    std::vector<LoadVertex> loaded(positions->count);
    bool skinned = joints != nullptr && weights != nullptr;
    if (skinned) {
        mesh.skinWeights.resize(positions->count);
    }
    for (cgltf_size i = 0; i < positions->count; ++i) {
        LoadVertex& vertex = loaded[i];
        readFloats(positions, i, glm::value_ptr(vertex.position), 3);
        readFloats(normals, i, glm::value_ptr(vertex.normal), 3);
        readFloats(uvs, i, glm::value_ptr(vertex.uv), 2);
        if (tangents != nullptr) {
            readFloats(tangents, i, glm::value_ptr(vertex.tangent), 4);
        }
        if (skinned) {
            cgltf_uint jointIndices[4] = {0, 0, 0, 0};
            float jointWeights[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            cgltf_accessor_read_uint(joints, i, jointIndices, 4);
            readFloats(weights, i, jointWeights, 4);
            mesh.skinWeights[i].joints = packJoints(jointIndices);
            mesh.skinWeights[i].weights = packWeights(jointWeights);
        }
    }

    if (primitive.indices != nullptr) {
        mesh.indices.resize(primitive.indices->count);
        for (cgltf_size i = 0; i < primitive.indices->count; ++i) {
            mesh.indices[i] = static_cast<uint32_t>(cgltf_accessor_read_index(primitive.indices, i));
        }
    } else {
        mesh.indices.resize(positions->count);
        for (uint32_t i = 0; i < mesh.indices.size(); ++i) {
            mesh.indices[i] = i;
        }
    }

    weldVertices(loaded, mesh.skinWeights, mesh.indices, weld);
    if (normals == nullptr) {
        generateNormals(loaded, mesh.indices);
    }
    if (tangents == nullptr) {
        generateTangents(loaded, mesh.indices);
    }
    mesh.vertices.resize(loaded.size());
    for (size_t i = 0; i < loaded.size(); ++i) {
        mesh.vertices[i].position = loaded[i].position;
        mesh.vertices[i].normal = packUnitVector(loaded[i].normal);
        mesh.vertices[i].tangent = packTangent(loaded[i].tangent);
        mesh.vertices[i].uv = loaded[i].uv;
    }
    computeBounds(mesh);

    mesh.materialIndex = static_cast<uint32_t>(model.materials.size() - 1);
    if (primitive.material != nullptr) {
        mesh.materialIndex = static_cast<uint32_t>(primitive.material - data.materials);
    }
    model.meshes.push_back(std::move(mesh));
}

AnimationPath toAnimationPath(cgltf_animation_path_type path) {
    switch (path) {
    case cgltf_animation_path_type_rotation:
        return AnimationPath::ROTATION;
    case cgltf_animation_path_type_scale:
        return AnimationPath::SCALE;
    default:
        return AnimationPath::TRANSLATION;
    }
}

// 노드 계층과 스킨, 애니메이션을 그대로 옮겨 둔다. 포즈 계산은 매 프레임 poseNodes 가 한다.
bool loadSkeleton(const cgltf_data& data, Skeleton& skeleton) {
    skeleton.nodes.resize(data.nodes_count);
    for (cgltf_size i = 0; i < data.nodes_count; ++i) {
        const cgltf_node& source = data.nodes[i];
        Node& node = skeleton.nodes[i];
        node.parent = source.parent != nullptr ? static_cast<int32_t>(source.parent - data.nodes) : -1;
        if (source.has_matrix != 0) {
            // 행렬로만 주어진 노드는 채널이 덮어쓸 수 있도록 TRS 로 분해해 둔다.
            glm::mat4 matrix = glm::make_mat4(source.matrix);
            glm::vec3 skew;
            glm::vec4 perspective;
            glm::decompose(matrix, node.scale, node.rotation, node.translation, skew, perspective);
        } else {
            node.translation = glm::make_vec3(source.translation);
            node.rotation = glm::quat{source.rotation[3], source.rotation[0], source.rotation[1], source.rotation[2]};
            node.scale = glm::make_vec3(source.scale);
        }
    }

    skeleton.skins.resize(data.skins_count);
    for (cgltf_size i = 0; i < data.skins_count; ++i) {
        const cgltf_skin& source = data.skins[i];
        Skin& skin = skeleton.skins[i];
        if (source.joints_count > MAX_SKIN_JOINTS) {
            spdlog::error("조인트가 {}개를 넘는 스킨은 지원하지 않습니다: {}", MAX_SKIN_JOINTS, source.joints_count);
            return false;
        }
        skin.joints.reserve(source.joints_count);
        skin.inverseBind.assign(source.joints_count, glm::mat4{1.0F});
        for (cgltf_size j = 0; j < source.joints_count; ++j) {
            skin.joints.push_back(static_cast<uint32_t>(source.joints[j] - data.nodes));
            if (source.inverse_bind_matrices != nullptr) {
                readFloats(source.inverse_bind_matrices, j, glm::value_ptr(skin.inverseBind[j]), 16);
            }
        }
    }

    skeleton.animations.resize(data.animations_count);
    for (cgltf_size i = 0; i < data.animations_count; ++i) {
        const cgltf_animation& source = data.animations[i];
        Animation& animation = skeleton.animations[i];
        animation.name = source.name != nullptr ? source.name : "클립";
        animation.samplers.resize(source.samplers_count);
        for (cgltf_size j = 0; j < source.samplers_count; ++j) {
            const cgltf_animation_sampler& sampler = source.samplers[j];
            AnimationSampler& target = animation.samplers[j];
            target.step = sampler.interpolation == cgltf_interpolation_type_step;
            // ponytail: CUBICSPLINE 은 접선을 버리고 값만 선형 보간한다. 필요하면 에르미트 보간을 넣으면 된다.
            bool cubic = sampler.interpolation == cgltf_interpolation_type_cubic_spline;
            cgltf_size stride = cubic ? 3 : 1;
            cgltf_size offset = cubic ? 1 : 0;
            cgltf_size count = sampler.input->count;
            target.times.resize(count);
            target.values.assign(count, glm::vec4{0.0F});
            for (cgltf_size k = 0; k < count; ++k) {
                readFloats(sampler.input, k, &target.times[k], 1);
                readFloats(sampler.output, k * stride + offset, glm::value_ptr(target.values[k]), 4);
            }
            if (count > 0) {
                animation.duration = std::max(animation.duration, target.times.back());
            }
        }
        animation.channels.reserve(source.channels_count);
        for (cgltf_size j = 0; j < source.channels_count; ++j) {
            const cgltf_animation_channel& sourceChannel = source.channels[j];
            if (sourceChannel.target_node == nullptr || sourceChannel.sampler == nullptr ||
                sourceChannel.target_path == cgltf_animation_path_type_weights) {
                continue;
            }
            AnimationChannel channel;
            channel.sampler = static_cast<uint32_t>(sourceChannel.sampler - source.samplers);
            channel.node = static_cast<uint32_t>(sourceChannel.target_node - data.nodes);
            channel.path = toAnimationPath(sourceChannel.target_path);
            animation.channels.push_back(channel);
        }
    }
    return true;
}

void appendNode(Model& model,
                const cgltf_node& node,
                const std::vector<PrimitiveRange>& meshRanges,
                const cgltf_data& data) {
    if (node.mesh != nullptr) {
        auto meshIndex = static_cast<size_t>(node.mesh - data.meshes);
        const PrimitiveRange& range = meshRanges[meshIndex];

        // glTF 는 스킨 메쉬 노드의 변환을 무시한다. 조인트 행렬이 이미 정점을 장면 공간으로 보내므로
        // 인스턴스 변환은 단위 행렬로 두고, 사용자가 기즈모로 옮긴 값만 그 위에 곱해진다.
        int32_t skin = node.skin != nullptr ? static_cast<int32_t>(node.skin - data.skins) : -1;
        glm::mat4 transform{1.0F};
        if (skin < 0) {
            cgltf_float world[16];
            cgltf_node_transform_world(&node, world);
            transform = glm::make_mat4(world);
        }

        for (uint32_t i = 0; i < range.count; ++i) {
            Instance instance;
            instance.name = node.name != nullptr ? node.name : "오브젝트";
            instance.meshIndex = range.first + i;
            instance.transform = transform;
            instance.skin = skin;
            model.instances.push_back(std::move(instance));
        }
    }
    for (cgltf_size i = 0; i < node.children_count; ++i) {
        appendNode(model, *node.children[i], meshRanges, data);
    }
}

} // namespace

std::optional<Model> loadGltf(const std::filesystem::path& path, LoadProgress* progress) {
    using Clock = std::chrono::steady_clock;
    auto elapsedMs = [](Clock::time_point since) {
        return std::chrono::duration<double, std::milli>(Clock::now() - since).count();
    };
    if (progress != nullptr) {
        progress->begin(LoadProgress::Stage::PARSE);
    }
    Clock::time_point parseStart = Clock::now();
    cgltf_options options{};
    cgltf_data* data = nullptr;
    if (cgltf_parse_file(&options, path.string().c_str(), &data) != cgltf_result_success) {
        spdlog::error("glTF 파일을 열 수 없습니다: {}", path.string());
        return std::nullopt;
    }
    if (cgltf_load_buffers(&options, data, path.string().c_str()) != cgltf_result_success) {
        cgltf_free(data);
        spdlog::error("glTF 버퍼를 읽을 수 없습니다: {}", path.string());
        return std::nullopt;
    }
    if (cgltf_validate(data) != cgltf_result_success) {
        cgltf_free(data);
        spdlog::error("glTF 파일이 유효하지 않습니다: {}", path.string());
        return std::nullopt;
    }

    Model model;
    model.name = path.stem().string();

    std::filesystem::path baseDirectory = path.parent_path();
    std::vector<bool> srgbFlags = collectSrgbFlags(*data);
    model.textures.reserve(data->textures_count);
    for (cgltf_size i = 0; i < data->textures_count; ++i) {
        const cgltf_texture& source = data->textures[i];
        Texture texture;
        texture.name = source.name != nullptr ? source.name : "텍스처";
        texture.srgb = srgbFlags[i];
        texture.sampler = toSamplerDesc(source.sampler);
        // KHR_texture_basisu 가 가리키는 KTX2 를 우선한다. 미리 압축된 에셋은 그쪽에만 이미지가 있다.
        const cgltf_image* image =
            source.has_basisu && source.basisu_image != nullptr ? source.basisu_image : source.image;
        if (image == nullptr || !loadTextureSource(*image, baseDirectory, texture)) {
            spdlog::warn("텍스처를 해석하지 못했습니다: {} ({})", texture.name, model.name);
        }
        model.textures.push_back(std::move(texture));
    }

    model.materials.reserve(data->materials_count + 1);
    for (cgltf_size i = 0; i < data->materials_count; ++i) {
        model.materials.push_back(convertMaterial(data->materials[i], *data));
    }
    // 재질이 지정되지 않은 프리미티브가 쓸 기본 재질을 마지막에 둔다.
    Material fallback;
    fallback.name = "기본 재질";
    fallback.metallicFactor = 0.0F;
    fallback.roughnessFactor = 1.0F;
    model.materials.push_back(fallback);

    if (!loadSkeleton(*data, model.skeleton)) {
        cgltf_free(data);
        return std::nullopt;
    }

    double parseMs = elapsedMs(parseStart);
    Clock::time_point convertStart = Clock::now();
    if (progress != nullptr) {
        uint64_t primitiveCount = 0;
        for (cgltf_size i = 0; i < data->meshes_count; ++i) {
            primitiveCount += data->meshes[i].primitives_count;
        }
        progress->begin(LoadProgress::Stage::CONVERT, primitiveCount);
    }
    std::vector<PrimitiveRange> meshRanges(data->meshes_count);
    WeldStats weld;
    for (cgltf_size i = 0; i < data->meshes_count; ++i) {
        meshRanges[i].first = static_cast<uint32_t>(model.meshes.size());
        for (cgltf_size j = 0; j < data->meshes[i].primitives_count; ++j) {
            appendPrimitive(model, *data, data->meshes[i].primitives[j], weld);
            if (progress != nullptr) {
                progress->advance();
            }
        }
        meshRanges[i].count = static_cast<uint32_t>(model.meshes.size()) - meshRanges[i].first;
    }

    const cgltf_scene* scene = data->scene != nullptr ? data->scene : (data->scenes_count > 0 ? data->scenes : nullptr);
    if (scene != nullptr) {
        for (cgltf_size i = 0; i < scene->nodes_count; ++i) {
            appendNode(model, *scene->nodes[i], meshRanges, *data);
        }
    } else {
        for (cgltf_size i = 0; i < data->nodes_count; ++i) {
            if (data->nodes[i].parent == nullptr) {
                appendNode(model, data->nodes[i], meshRanges, *data);
            }
        }
    }

    cgltf_free(data);

    size_t triangleCount = 0;
    for (const Mesh& mesh : model.meshes) {
        triangleCount += mesh.indices.size() / 3;
    }
    spdlog::info("glTF 적재: {} (메쉬 {}, 재질 {}, 텍스처 {}, 인스턴스 {}, 삼각형 {}, 정점 {} → 합친 뒤 {}, 스킨 {}, "
                 "애니메이션 {}; 파싱 {:.1f} ms, 정점 변환 {:.1f} ms)",
                 model.name,
                 model.meshes.size(),
                 model.materials.size(),
                 model.textures.size(),
                 model.instances.size(),
                 triangleCount,
                 weld.before,
                 weld.after,
                 model.skeleton.skins.size(),
                 model.skeleton.animations.size(),
                 parseMs,
                 elapsedMs(convertStart));
    return model;
}

void decodeTexture(Texture& texture) {
    if (texture.encoded.empty()) {
        return;
    }
    if (isKtx2(texture.encoded)) {
        if (!loadKtx2(texture.encoded, texture)) {
            texture.pixels.clear();
        }
        texture.encoded.clear();
        texture.encoded.shrink_to_fit();
        return;
    }
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* decoded = stbi_load_from_memory(
        texture.encoded.data(), static_cast<int>(texture.encoded.size()), &width, &height, &channels, 4);
    texture.encoded.clear();
    texture.encoded.shrink_to_fit();
    if (decoded == nullptr) {
        spdlog::warn("텍스처를 해석하지 못했습니다: {}", texture.name);
        return;
    }
    texture.width = static_cast<uint32_t>(width);
    texture.height = static_cast<uint32_t>(height);
    texture.format = TextureFormat::RGBA8;
    texture.mipLevels = 1;
    texture.pixels.assign(decoded, decoded + static_cast<size_t>(width) * height * 4);
    stbi_image_free(decoded);
}

} // namespace asset
