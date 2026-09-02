#ifndef MATERIAL_GLSL
#define MATERIAL_GLSL

#include "bindless.glsl"
#include "scene_types.glsl"

// 재질 텍스처를 읽어 셰이딩 입력으로 푼다. 래스터 프래그먼트, 경로 추적 적중 셰이더, 반사 컴퓨트가
// 같은 함수를 써서 세 경로가 같은 표면을 본다.
struct MaterialSample {
    vec3 albedo;
    float alpha;
    float metallic;
    float roughness;
    float occlusion;
    vec3 emissive;
};

MaterialSample sampleMaterial(Material material, vec2 uv) {
    MaterialSample result;

    vec4 baseColor = material.baseColorFactor;
    if (material.baseColorTexture != INVALID_TEXTURE) {
        baseColor *= sampleBindless(material.baseColorTexture, uv);
    }
    result.albedo = baseColor.rgb;
    result.alpha = baseColor.a;

    float metallic = material.metallicFactor;
    float roughness = material.roughnessFactor;
    if (material.metallicRoughnessTexture != INVALID_TEXTURE) {
        vec4 sampled = sampleBindless(material.metallicRoughnessTexture, uv);
        roughness *= sampled.g;
        metallic *= sampled.b;
    }
    result.metallic = clamp(metallic, 0.0, 1.0);
    // 완전한 거울은 GGX 표본의 분모를 0 으로 만든다. 모든 경로가 같은 하한을 쓴다.
    result.roughness = clamp(roughness, 0.03, 1.0);

    result.occlusion = 1.0;
    if (material.occlusionTexture != INVALID_TEXTURE) {
        float sampled = sampleBindless(material.occlusionTexture, uv).r;
        result.occlusion = mix(1.0, sampled, material.occlusionStrength);
    }

    result.emissive = material.emissiveAndCutoff.rgb;
    if (material.emissiveTexture != INVALID_TEXTURE) {
        result.emissive *= sampleBindless(material.emissiveTexture, uv).rgb;
    }
    return result;
}

// 알파만 필요한 곳(교차 판정)용. 기저 색 전체를 읽지 않는다.
float materialAlpha(Material material, vec2 uv) {
    float alpha = material.baseColorFactor.a;
    if (material.baseColorTexture != INVALID_TEXTURE) {
        alpha *= sampleBindless(material.baseColorTexture, uv).a;
    }
    return alpha;
}

// 접선 공간 노멀 맵. normal 은 정규화된 기하 노멀이고 tangent 는 정규화 전이어도 된다.
vec3 perturbNormal(Material material, vec3 normal, vec3 tangent, float tangentSign, vec2 uv) {
    if (material.normalTexture == INVALID_TEXTURE) {
        return normal;
    }
    vec3 orthogonalTangent = normalize(tangent - normal * dot(normal, tangent));
    vec3 bitangent = cross(normal, orthogonalTangent) * tangentSign;
    vec3 sampled = sampleBindless(material.normalTexture, uv).xyz * 2.0 - 1.0;
    sampled.xy *= material.normalScale;
    return normalize(mat3(orthogonalTangent, bitangent, normal) * sampled);
}

// 광선이 맞힌 삼각형을 보간한 표면. 노멀 맵과 뒷면 뒤집기는 부르는 쪽이 한다.
struct HitSurface {
    vec3 position;
    // 지난 프레임의 세계 위치. 모션 벡터에만 쓴다.
    vec3 previousPosition;
    vec3 normal;
    vec3 tangent;
    float tangentSign;
    vec2 uv;
    uint materialIndex;
};

// 하위 가속 구조는 LOD 0 으로만 세우고, 인덱스는 메쉬 지역 번호라 스킨 정점 구간에서도 그대로 쓴다.
HitSurface interpolateHit(InstanceBuffer instances,
                          MeshBuffer meshes,
                          MeshLodBuffer lods,
                          VertexBuffer vertices,
                          VertexBuffer skinnedVertices,
                          IndexBuffer indices,
                          uint instanceIndex,
                          uint primitiveIndex,
                          vec2 barycentrics) {
    Instance instance = instances.items[instanceIndex];
    Mesh mesh = meshes.items[instance.meshIndex];
    MeshLod lod = lods.items[mesh.lodOffset];

    uint base = lod.indexOffset + primitiveIndex * 3;
    bool skinned = instance.skinnedVertexOffset != NO_SKINNED_VERTICES;
    VertexBuffer source = skinned ? skinnedVertices : vertices;
    uint vertexBase = skinned ? instance.skinnedVertexOffset : uint(mesh.vertexOffset);
    uint i0 = indices.items[base];
    uint i1 = indices.items[base + 1];
    uint i2 = indices.items[base + 2];
    Vertex v0 = source.items[vertexBase + i0];
    Vertex v1 = source.items[vertexBase + i1];
    Vertex v2 = source.items[vertexBase + i2];

    vec3 weights = vec3(1.0 - barycentrics.x - barycentrics.y, barycentrics.x, barycentrics.y);
    vec3 localPosition = v0.position * weights.x + v1.position * weights.y + v2.position * weights.z;
    vec3 localNormal = v0.normal * weights.x + v1.normal * weights.y + v2.normal * weights.z;
    vec4 localTangent = v0.tangent * weights.x + v1.tangent * weights.y + v2.tangent * weights.z;

    // 스킨 인스턴스는 지난 포즈의 변형 정점이 따로 있어 모션 벡터가 관절 움직임까지 담는다.
    vec3 previousLocalPosition = localPosition;
    if (skinned) {
        uint previousBase = instance.previousSkinnedVertexOffset;
        previousLocalPosition = source.items[previousBase + i0].position * weights.x +
                                source.items[previousBase + i1].position * weights.y +
                                source.items[previousBase + i2].position * weights.z;
    }

    mat3 normalMatrix = mat3(instance.normalMatrix);
    HitSurface hit;
    hit.position = (instance.model * vec4(localPosition, 1.0)).xyz;
    hit.previousPosition = (instance.previousModel * vec4(previousLocalPosition, 1.0)).xyz;
    hit.normal = normalize(normalMatrix * localNormal);
    hit.tangent = normalize(normalMatrix * localTangent.xyz);
    hit.tangentSign = localTangent.w;
    hit.uv = v0.uv * weights.x + v1.uv * weights.y + v2.uv * weights.z;
    hit.materialIndex = mesh.materialIndex;
    return hit;
}

// 교차 판정용 UV 만 보간한다. 위치와 노멀은 읽지 않는다.
vec2 interpolateHitUv(InstanceBuffer instances,
                      MeshBuffer meshes,
                      MeshLodBuffer lods,
                      VertexBuffer vertices,
                      VertexBuffer skinnedVertices,
                      IndexBuffer indices,
                      uint instanceIndex,
                      uint primitiveIndex,
                      vec2 barycentrics) {
    Instance instance = instances.items[instanceIndex];
    Mesh mesh = meshes.items[instance.meshIndex];
    MeshLod lod = lods.items[mesh.lodOffset];
    uint base = lod.indexOffset + primitiveIndex * 3;
    bool skinned = instance.skinnedVertexOffset != NO_SKINNED_VERTICES;
    VertexBuffer source = skinned ? skinnedVertices : vertices;
    uint vertexBase = skinned ? instance.skinnedVertexOffset : uint(mesh.vertexOffset);
    vec2 uv0 = source.items[vertexBase + indices.items[base]].uv;
    vec2 uv1 = source.items[vertexBase + indices.items[base + 1]].uv;
    vec2 uv2 = source.items[vertexBase + indices.items[base + 2]].uv;
    return uv0 * (1.0 - barycentrics.x - barycentrics.y) + uv1 * barycentrics.x + uv2 * barycentrics.y;
}

#endif
