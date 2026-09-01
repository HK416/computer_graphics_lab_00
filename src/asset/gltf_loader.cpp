#include <algorithm>
#include <cmath>
#include <vector>

#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#include <glm/geometric.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <spdlog/spdlog.h>
#include <stb_image.h>

#include "asset/model.h"
#include "core/error.h"

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

// 노멀이 없는 프리미티브는 삼각형 면적 가중 평균으로 채운다.
void generateNormals(std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices) {
    for (Vertex& vertex : vertices) {
        vertex.normal = glm::vec3{0.0F};
    }
    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        Vertex& a = vertices[indices[i]];
        Vertex& b = vertices[indices[i + 1]];
        Vertex& c = vertices[indices[i + 2]];
        glm::vec3 faceNormal = glm::cross(b.position - a.position, c.position - a.position);
        a.normal += faceNormal;
        b.normal += faceNormal;
        c.normal += faceNormal;
    }
    for (Vertex& vertex : vertices) {
        float length = glm::length(vertex.normal);
        vertex.normal = length > 0.0F ? vertex.normal / length : glm::vec3{0.0F, 1.0F, 0.0F};
    }
}

// 탄젠트가 없는데 노멀 맵을 쓰는 재질이 있으면 접선 공간이 깨지므로 UV 로부터 만들어 둔다.
void generateTangents(std::vector<Vertex>& vertices, const std::vector<uint32_t>& indices) {
    std::vector<glm::vec3> tangents(vertices.size(), glm::vec3{0.0F});
    std::vector<glm::vec3> bitangents(vertices.size(), glm::vec3{0.0F});

    for (size_t i = 0; i + 2 < indices.size(); i += 3) {
        const Vertex& a = vertices[indices[i]];
        const Vertex& b = vertices[indices[i + 1]];
        const Vertex& c = vertices[indices[i + 2]];

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

bool decodePixels(const uint8_t* bytes, size_t size, Texture& texture) {
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* decoded = stbi_load_from_memory(bytes, static_cast<int>(size), &width, &height, &channels, 4);
    if (decoded == nullptr) {
        return false;
    }
    texture.width = static_cast<uint32_t>(width);
    texture.height = static_cast<uint32_t>(height);
    texture.pixels.assign(decoded, decoded + static_cast<size_t>(width) * height * 4);
    stbi_image_free(decoded);
    return true;
}

bool loadTexturePixels(const cgltf_image& image, const std::filesystem::path& baseDirectory, Texture& texture) {
    if (image.buffer_view != nullptr && image.buffer_view->buffer->data != nullptr) {
        const auto* bytes = static_cast<const uint8_t*>(image.buffer_view->buffer->data) + image.buffer_view->offset;
        return decodePixels(bytes, image.buffer_view->size, texture);
    }
    if (image.uri == nullptr) {
        return false;
    }

    std::filesystem::path file = baseDirectory / image.uri;
    int width = 0;
    int height = 0;
    int channels = 0;
    stbi_uc* decoded = stbi_load(file.string().c_str(), &width, &height, &channels, 4);
    if (decoded == nullptr) {
        return false;
    }
    texture.width = static_cast<uint32_t>(width);
    texture.height = static_cast<uint32_t>(height);
    texture.pixels.assign(decoded, decoded + static_cast<size_t>(width) * height * 4);
    stbi_image_free(decoded);
    return true;
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

void appendPrimitive(Model& model, const cgltf_data& data, const cgltf_primitive& primitive) {
    const cgltf_accessor* positions = findAttribute(primitive, cgltf_attribute_type_position, 0);
    if (positions == nullptr || primitive.type != cgltf_primitive_type_triangles) {
        return;
    }
    const cgltf_accessor* normals = findAttribute(primitive, cgltf_attribute_type_normal, 0);
    const cgltf_accessor* tangents = findAttribute(primitive, cgltf_attribute_type_tangent, 0);
    const cgltf_accessor* uvs = findAttribute(primitive, cgltf_attribute_type_texcoord, 0);

    Mesh mesh;
    mesh.name =
        primitive.material != nullptr && primitive.material->name != nullptr ? primitive.material->name : "프리미티브";
    mesh.vertices.resize(positions->count);
    for (cgltf_size i = 0; i < positions->count; ++i) {
        Vertex& vertex = mesh.vertices[i];
        readFloats(positions, i, glm::value_ptr(vertex.position), 3);
        readFloats(normals, i, glm::value_ptr(vertex.normal), 3);
        readFloats(uvs, i, glm::value_ptr(vertex.uv), 2);
        if (tangents != nullptr) {
            readFloats(tangents, i, glm::value_ptr(vertex.tangent), 4);
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

    if (normals == nullptr) {
        generateNormals(mesh.vertices, mesh.indices);
    }
    if (tangents == nullptr) {
        generateTangents(mesh.vertices, mesh.indices);
    }
    computeBounds(mesh);

    mesh.materialIndex = static_cast<uint32_t>(model.materials.size() - 1);
    if (primitive.material != nullptr) {
        mesh.materialIndex = static_cast<uint32_t>(primitive.material - data.materials);
    }
    model.meshes.push_back(std::move(mesh));
}

void appendNode(Model& model,
                const cgltf_node& node,
                const std::vector<PrimitiveRange>& meshRanges,
                const cgltf_data& data) {
    if (node.mesh != nullptr) {
        auto meshIndex = static_cast<size_t>(node.mesh - data.meshes);
        const PrimitiveRange& range = meshRanges[meshIndex];

        cgltf_float world[16];
        cgltf_node_transform_world(&node, world);
        glm::mat4 transform = glm::make_mat4(world);

        for (uint32_t i = 0; i < range.count; ++i) {
            Instance instance;
            instance.name = node.name != nullptr ? node.name : "오브젝트";
            instance.meshIndex = range.first + i;
            instance.transform = transform;
            model.instances.push_back(std::move(instance));
        }
    }
    for (cgltf_size i = 0; i < node.children_count; ++i) {
        appendNode(model, *node.children[i], meshRanges, data);
    }
}

} // namespace

Model loadGltf(const std::filesystem::path& path) {
    cgltf_options options{};
    cgltf_data* data = nullptr;
    if (cgltf_parse_file(&options, path.string().c_str(), &data) != cgltf_result_success) {
        core::fatal("glTF 파일을 열 수 없습니다: {}", path.string());
    }
    if (cgltf_load_buffers(&options, data, path.string().c_str()) != cgltf_result_success) {
        cgltf_free(data);
        core::fatal("glTF 버퍼를 읽을 수 없습니다: {}", path.string());
    }
    if (cgltf_validate(data) != cgltf_result_success) {
        cgltf_free(data);
        core::fatal("glTF 파일이 유효하지 않습니다: {}", path.string());
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
        if (source.image == nullptr || !loadTexturePixels(*source.image, baseDirectory, texture)) {
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

    std::vector<PrimitiveRange> meshRanges(data->meshes_count);
    for (cgltf_size i = 0; i < data->meshes_count; ++i) {
        meshRanges[i].first = static_cast<uint32_t>(model.meshes.size());
        for (cgltf_size j = 0; j < data->meshes[i].primitives_count; ++j) {
            appendPrimitive(model, *data, data->meshes[i].primitives[j]);
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
    spdlog::info("glTF 적재: {} (메쉬 {}, 재질 {}, 텍스처 {}, 인스턴스 {}, 삼각형 {})",
                 model.name,
                 model.meshes.size(),
                 model.materials.size(),
                 model.textures.size(),
                 model.instances.size(),
                 triangleCount);
    return model;
}

} // namespace asset
