// 프레임 버퍼 예약과 장면 → GPU 배치(인스턴스·드로우·조명·LOD 네트워크·가속 구조).
// Renderer 의 멤버 함수 정의만 나눠 담은 번역 단위다. 선언은 renderer.h 하나에 있다.

#include "gfx/renderer_internal.h"

namespace gfx {

void Renderer::reserveInstances(Frame& frame, uint32_t instanceCount) {
    if (instanceCount <= frame.instanceCapacity) {
        return;
    }
    uint32_t capacity = std::max(instanceCount, std::max(frame.instanceCapacity * 2, MINIMUM_INSTANCE_CAPACITY));
    destroyBuffer(context, frame.instanceBuffer);
    destroyBuffer(context, frame.drawBuffer);
    destroyBuffer(context, frame.drawMeshletBuffer);
    frame.instanceBuffer = createBuffer(context,
                                        static_cast<VkDeviceSize>(capacity) * sizeof(GpuInstance),
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                        MemoryLocation::HOST_WRITE,
                                        "인스턴스");
    frame.drawBuffer = createBuffer(context,
                                    static_cast<VkDeviceSize>(capacity) * sizeof(VkDrawIndexedIndirectCommand),
                                    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                    MemoryLocation::HOST_WRITE,
                                    "간접 그리기 명령");
    frame.drawMeshletBuffer = createBuffer(context,
                                           static_cast<VkDeviceSize>(capacity) * sizeof(uint32_t),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           MemoryLocation::HOST_WRITE,
                                           "명령별 meshlet");
    frame.instanceCapacity = capacity;
}

void Renderer::reserveMeshletGroups(Frame& frame, uint32_t groupCount) {
    if (frame.meshTaskIndirectBuffer.handle == VK_NULL_HANDLE) {
        frame.meshTaskIndirectBuffer =
            createBuffer(context,
                         sizeof(VkDrawMeshTasksIndirectCommandEXT) * ALPHA_MODE_COUNT * 2,
                         VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         MemoryLocation::HOST_WRITE,
                         "mesh task 간접 명령");
    }
    if (groupCount <= frame.groupCapacity) {
        return;
    }
    uint32_t capacity = std::max(groupCount, std::max(frame.groupCapacity * 2, MINIMUM_INSTANCE_CAPACITY));
    destroyBuffer(context, frame.meshletGroupBuffer);
    frame.meshletGroupBuffer = createBuffer(context,
                                            static_cast<VkDeviceSize>(capacity) * sizeof(GpuMeshletGroup),
                                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                            MemoryLocation::HOST_WRITE,
                                            "meshlet 그룹");
    frame.groupCapacity = capacity;
}

void Renderer::reserveMeshletVisibility(uint32_t meshletCount) {
    uint32_t needed = std::max(meshletCount, 32U);
    if (needed <= meshletVisibilityCapacity) {
        return;
    }
    // 지난 프레임이 아직 읽고 있을 수 있어 장치를 세운다. meshlet 수가 늘어나는 순간에만 일어난다.
    waitIdle();
    destroyBuffer(context, meshletVisibilityBuffer);
    meshletVisibilityCapacity = std::max(needed, meshletVisibilityCapacity * 2);
    meshletVisibilityBuffer =
        createBuffer(context,
                     static_cast<VkDeviceSize>((meshletVisibilityCapacity + 31) / 32) * sizeof(uint32_t),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     MemoryLocation::DEVICE,
                     "meshlet 가시성");
    visibilityNeedsClear = true;
}

void Renderer::reserveJoints(Frame& frame, uint32_t jointCount) {
    // 스킨이 없는 장면에서도 셰이더가 주소를 읽으므로 최소 하나는 잡아 둔다.
    uint32_t needed = std::max(jointCount, 1U);
    if (needed <= frame.jointCapacity) {
        return;
    }
    uint32_t capacity = std::max(needed, frame.jointCapacity * 2);
    destroyBuffer(context, frame.jointBuffer);
    VkDeviceSize bytes = static_cast<VkDeviceSize>(capacity) * sizeof(glm::mat4);
    frame.jointBuffer =
        createBuffer(context, bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryLocation::HOST_WRITE, "조인트 행렬");
    frame.jointCapacity = capacity;
}

void Renderer::reserveLights(Frame& frame, uint32_t lightCount) {
    if (frame.shadowMatrixBuffer.handle == VK_NULL_HANDLE) {
        frame.shadowMatrixBuffer = createBuffer(context,
                                                sizeof(glm::mat4) * MAX_SHADOW_VIEWS,
                                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                MemoryLocation::HOST_WRITE,
                                                "그림자 시점 행렬");
    }
    // 조명이 없는 장면에서도 셰이더가 주소를 읽으므로 최소 하나는 잡아 둔다.
    uint32_t needed = std::max(lightCount, 1U);
    if (needed <= frame.lightCapacity) {
        return;
    }
    uint32_t capacity = std::max(needed, frame.lightCapacity * 2);
    destroyBuffer(context, frame.lightBuffer);
    frame.lightBuffer = createBuffer(context,
                                     static_cast<VkDeviceSize>(capacity) * sizeof(GpuLight),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                     MemoryLocation::HOST_WRITE,
                                     "조명");
    frame.lightCapacity = capacity;
}

void Renderer::reserveShadowDraws(Frame& frame, uint32_t drawCount) {
    uint32_t needed = std::max(drawCount, 1U);
    if (needed <= frame.shadowDrawCapacity) {
        return;
    }
    uint32_t capacity = std::max(needed, frame.shadowDrawCapacity * 2);
    destroyBuffer(context, frame.shadowDrawBuffer);
    frame.shadowDrawBuffer = createBuffer(context,
                                          static_cast<VkDeviceSize>(capacity) * sizeof(VkDrawIndexedIndirectCommand),
                                          VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                                          MemoryLocation::HOST_WRITE,
                                          "그림자 그리기 명령");
    frame.shadowDrawCapacity = capacity;
}

void Renderer::reserveMeshletDraws(Frame& frame, uint32_t drawCount) {
    if (frame.lodNetworkBuffer.handle == VK_NULL_HANDLE) {
        frame.lodNetworkBuffer = createBuffer(context,
                                              sizeof(GpuLodNetwork),
                                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                              MemoryLocation::HOST_WRITE,
                                              "LOD 신경망 가중치");
    }
    if (frame.drawCountBuffer.handle == VK_NULL_HANDLE) {
        frame.drawCountBuffer = createBuffer(context,
                                             sizeof(uint32_t) * BUCKET_COUNT,
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                             MemoryLocation::DEVICE,
                                             "그리기 개수");
    }
    if (drawCount <= frame.meshletDrawCapacity) {
        return;
    }
    uint32_t capacity = std::max(drawCount, std::max(frame.meshletDrawCapacity * 2, MINIMUM_INSTANCE_CAPACITY));
    destroyBuffer(context, frame.meshletDrawBuffer);
    destroyBuffer(context, frame.meshletDrawMeshletBuffer);
    frame.meshletDrawBuffer = createBuffer(context,
                                           static_cast<VkDeviceSize>(capacity) * sizeof(VkDrawIndexedIndirectCommand),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                                               VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                           MemoryLocation::DEVICE,
                                           "meshlet 그리기 명령");
    frame.meshletDrawMeshletBuffer = createBuffer(context,
                                                  static_cast<VkDeviceSize>(capacity) * sizeof(uint32_t),
                                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                  MemoryLocation::DEVICE,
                                                  "meshlet 명령별 meshlet");
    frame.meshletDrawCapacity = capacity;
}

namespace {
// 광원에서 direction 을 바라보는 시점 행렬. 방향이 위쪽과 나란하면 기준 축을 바꾼다.
glm::mat4 lookAlong(glm::vec3 eye, glm::vec3 direction) {
    glm::vec3 up = std::abs(direction.y) > 0.99F ? glm::vec3{0.0F, 0.0F, 1.0F} : glm::vec3{0.0F, 1.0F, 0.0F};
    return glm::lookAt(eye, eye + direction, up);
}

// shaders/shadow.glsl 의 cubeFaceIndex 와 같은 순서여야 한다.
constexpr std::array<glm::vec3, 6> CUBE_FACE_DIRECTIONS{glm::vec3{1.0F, 0.0F, 0.0F},
                                                        glm::vec3{-1.0F, 0.0F, 0.0F},
                                                        glm::vec3{0.0F, 1.0F, 0.0F},
                                                        glm::vec3{0.0F, -1.0F, 0.0F},
                                                        glm::vec3{0.0F, 0.0F, 1.0F},
                                                        glm::vec3{0.0F, 0.0F, -1.0F}};
} // namespace

void Renderer::buildLights(Frame& frame, const scene::Scene& scene) {
    frameLights.clear();
    shadowViews.clear();

    // 방향광의 그림자 절두체를 맞추려면 보이는 메쉬 전체의 세계 경계가 필요하다.
    glm::vec3 minimum{std::numeric_limits<float>::max()};
    glm::vec3 maximum{std::numeric_limits<float>::lowest()};
    bool hasBounds = false;
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        if (!geometry.meshLive(scene.meshOf(index)) || !scene.visibleCached(index)) {
            continue;
        }
        const glm::mat4& world = scene.world(index);
        glm::vec4 sphere = geometry.mesh(scene.meshOf(index)).boundingSphere;
        glm::vec3 center = glm::vec3(world * glm::vec4{glm::vec3(sphere), 1.0F});
        float scale = std::sqrt(std::max({glm::dot(glm::vec3(world[0]), glm::vec3(world[0])),
                                          glm::dot(glm::vec3(world[1]), glm::vec3(world[1])),
                                          glm::dot(glm::vec3(world[2]), glm::vec3(world[2]))}));
        minimum = glm::min(minimum, center - sphere.w * scale);
        maximum = glm::max(maximum, center + sphere.w * scale);
        hasBounds = true;
    }
    glm::vec3 sceneCenter = hasBounds ? (minimum + maximum) * 0.5F : glm::vec3{0.0F};
    // 멤버에 담아 SSAO 반지름을 장면 크기에 맞추는 데도 쓴다.
    sceneRadius = hasBounds ? std::max(glm::length(maximum - minimum) * 0.5F, 1.0F) : 1.0F;

    bool sunAssigned = false;
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        const scene::Object& object = scene.objects[index];
        if (object.light < 0 || static_cast<size_t>(object.light) >= scene.lights.size() ||
            !scene.visibleCached(index)) {
            continue;
        }
        const scene::Light& source = scene.lights[static_cast<size_t>(object.light)];
        const glm::mat4& world = scene.world(index);
        glm::vec3 position = glm::vec3(world[3]);
        // glTF 와 Unity 처럼 -Z 를 앞으로 본다.
        glm::vec3 direction = glm::normalize(-glm::vec3(world[2]));

        GpuLight light{};
        light.positionRange = glm::vec4{position, source.range};
        light.directionIntensity = glm::vec4{direction, source.intensity};
        light.colorType = glm::vec4{source.color, static_cast<float>(source.type)};
        light.coneSize = glm::vec4{std::cos(glm::radians(source.innerConeDegrees)),
                                   std::cos(glm::radians(source.outerConeDegrees)),
                                   source.size.x * 0.5F,
                                   source.size.y * 0.5F};
        light.rightShadow = glm::vec4{glm::normalize(glm::vec3(world[0])), -1.0F};
        light.up = glm::vec4{glm::normalize(glm::vec3(world[1])), 0.0F};

        // 하늘의 태양과 그림자 방향이 어긋나면 곧바로 눈에 띈다. 첫 방향광을 따라간다.
        if (source.type == scene::LightType::DIRECTIONAL && !sunAssigned) {
            sunDirection = direction;
            sunAssigned = true;
        }

        // 영역광은 반구 전체로 빛을 내보내 시점 하나로 담을 수 없어 그림자를 만들지 않는다.
        uint32_t viewsNeeded = 0;
        if (settings.shadowsEnabled && source.castsShadow) {
            switch (source.type) {
            case scene::LightType::DIRECTIONAL:
            case scene::LightType::SPOT:
                viewsNeeded = 1;
                break;
            case scene::LightType::POINT:
                viewsNeeded = 6;
                break;
            default:
                break;
            }
        }
        // 방향광은 캐스케이드 수만큼 층을 쓴다. 층이 모자라면 버리지 말고 캐스케이드를 줄인다.
        if (source.type == scene::LightType::DIRECTIONAL && viewsNeeded > 0) {
            viewsNeeded = std::min(std::clamp(settings.shadowCascades, 1U, MAX_SHADOW_CASCADES),
                                   static_cast<uint32_t>(MAX_SHADOW_VIEWS - shadowViews.size()));
        }

        if (viewsNeeded > 0 && shadowViews.size() + viewsNeeded <= MAX_SHADOW_VIEWS) {
            light.rightShadow.w = static_cast<float>(shadowViews.size());
            light.up.w = static_cast<float>(viewsNeeded);
            ShadowView view{};
            view.origin = position;
            view.sweepDirection = direction;
            if (source.type == scene::LightType::DIRECTIONAL) {
                view.directional = true;
                // 평행이동 없는 회전만의 광 시점. 광원을 카메라 쪽으로 옮겨 가며 스냅하면
                // 격자가 카메라를 따라다녀 스냅이 무의미해진다.
                glm::mat4 lightRotation = lookAlong(glm::vec3{0.0F}, direction);
                glm::vec3 sceneInLight = glm::vec3(lightRotation * glm::vec4{sceneCenter, 1.0F});
                float depthNear = -(sceneInLight.z + sceneRadius);
                float depthFar = -(sceneInLight.z - sceneRadius);

                float farDistance =
                    settings.shadowDistance > 0.0F ? settings.shadowDistance : std::min(4.0F * sceneRadius, 500.0F);
                std::array<float, MAX_SHADOW_CASCADES> splits{};
                cascadeSplits(scene.camera.nearPlane, farDistance, viewsNeeded, settings.shadowSplitLambda, splits);

                float aspect =
                    static_cast<float>(currentRenderExtent.width) / static_cast<float>(currentRenderExtent.height);
                float fov = glm::radians(scene.camera.fovYDegrees);
                glm::vec3 forward = scene.camera.forward();
                float previous = scene.camera.nearPlane;
                for (uint32_t cascade = 0; cascade < viewsNeeded; ++cascade) {
                    CascadeSphere sphere = fitCascadeSphere(previous, splits[cascade], fov, aspect);
                    glm::vec3 center = scene.camera.position + forward * sphere.distance;
                    view.viewProjection =
                        snapCascadeMatrix(lightRotation, center, sphere.radius, depthNear, depthFar, SHADOW_MAP_SIZE);
                    shadowViews.push_back(view);

                    light.cascadeSplits[cascade] = splits[cascade];
                    light.cascadeTexelSizes[cascade] = 2.0F * sphere.radius / static_cast<float>(SHADOW_MAP_SIZE);
                    previous = splits[cascade];
                }
            } else if (source.type == scene::LightType::SPOT) {
                float fov = std::min(glm::radians(source.outerConeDegrees) * 2.0F, glm::radians(170.0F));
                view.viewProjection =
                    glm::perspectiveRH_ZO(fov, 1.0F, SHADOW_NEAR_PLANE, source.range) * lookAlong(position, direction);
                shadowViews.push_back(view);
            } else {
                glm::mat4 projection =
                    glm::perspectiveRH_ZO(glm::radians(90.0F), 1.0F, SHADOW_NEAR_PLANE, source.range);
                for (const glm::vec3& face : CUBE_FACE_DIRECTIONS) {
                    view.viewProjection = projection * lookAlong(position, face);
                    shadowViews.push_back(view);
                }
            }
        }
        frameLights.push_back(light);
    }

    reserveLights(frame, static_cast<uint32_t>(frameLights.size()));
    if (!frameLights.empty()) {
        std::ranges::copy(frameLights, static_cast<GpuLight*>(frame.lightBuffer.mapped));
    }
    auto* shadowMatrices = static_cast<glm::mat4*>(frame.shadowMatrixBuffer.mapped);
    for (size_t view = 0; view < shadowViews.size(); ++view) {
        shadowMatrices[view] = shadowViews[view].viewProjection;
    }
}

FrameBatches Renderer::buildDrawCommands(Frame& frame, const scene::Scene& scene) {
    // 유체 입자는 오브젝트 인스턴스 뒤에 이어 붙으므로 그만큼 더 잡는다. 내장 구가 없으면 그리지 않는다.
    fluid->setParticleLimit(settings.fluidParticleLimit);
    bool fluidActive = fluid->prepare(scene, &scene != lastScene);
    uint32_t particleTotal = geometry.meshLive(fluidSphereMesh) ? fluid->totalParticles() : 0;
    reserveInstances(frame, static_cast<uint32_t>(scene.objects.size()) + particleTotal);

    // 장면이 통째로 바뀌면(장면 전환) 프레임 캐시가 다른 장면 것이다. 유체가 움직이는 프레임도 장면이
    // 바뀐 것으로 친다. 가속 구조 재구축, 그림자 캐시 무효화, 경로 추적 누적 초기화가 한꺼번에 맞는다.
    sceneChangedThisFrame =
        &scene != lastScene || scene.revision() != lastSceneRevision || (fluidActive && particleTotal > 0);
    // 오브젝트 번호는 추가/삭제로 밀리므로 구성이 바뀐 프레임에는 지난 값을 버린다. 그 한 프레임만
    // 변위가 0 이고 다음 프레임부터 다시 맞는다. 장면 자체가 바뀐 경우도 같다.
    bool temporalReset = &scene != lastScene || scene.topologyRevision() != lastTopologyRevision ||
                         previousWorld.size() != scene.objects.size();
    lastScene = &scene;
    lastSceneRevision = scene.revision();
    lastTopologyRevision = scene.topologyRevision();
    previousWorld.resize(scene.objects.size());

    objectInstanceSlots.assign(scene.objects.size(), INVALID_INSTANCE_SLOT);
    objectSkinnedBlas.assign(scene.objects.size(), RayTracer::NO_SKINNED_BLAS);
    instanceBounds.assign(scene.objects.size(), glm::vec4{0.0F});
    skinDispatches.clear();
    skinnedInstances.clear();

    // 경로 추적 프레임은 래스터 패스를 건너뛰므로 meshlet 그룹을 아무도 읽지 않는다. 자동 LOD 는
    // 메쉬의 모든 단계를 후보로 올리기 때문에 그냥 두면 헛일이 적지 않다. 인스턴스는 상위 가속
    // 구조가 쓰므로 그대로 채운다.
    bool needMeshletGroups = !(settings.usePathTracing && rayTracer != nullptr);
    uint32_t skinnedVertexCursor = 0;
    uint32_t skinnedMeshletCursor = 0;
    uint32_t visibilityCursor = 0;
    // 스킨 인스턴스 슬롯과 디스패치 번호. 버퍼 용량이 정해진 뒤 절대 위치를 채운다.
    std::vector<std::pair<uint32_t, uint32_t>> skinnedSlots;

    uint32_t instanceZone = frameProfiler.begin("인스턴스 구성");
    // 재질 경로와 면 방향 조합마다 명령이 연속 구간을 이루도록 두 번 순회한다.
    auto bucketOf = [this, &scene](uint32_t index) {
        const asset::Material& material = geometry.material(geometry.mesh(scene.meshOf(index)).materialIndex);
        return std::pair<size_t, size_t>{static_cast<size_t>(material.alphaMode), material.doubleSided ? 1U : 0U};
    };
    auto lodFor = [this, &scene](uint32_t index) -> const GpuMeshLod& {
        const GpuMesh& mesh = geometry.mesh(scene.meshOf(index));
        return geometry.lod(mesh.lodOffset + std::min(settings.lodLevel, mesh.lodCount - 1));
    };
    // 자동 선정은 GPU 가 DAG 전체에서 고르므로 모든 단계의 meshlet 을 후보로 올려야 한다. 한
    // 단계만 올리면 오차가 "부모를 그려라"로 판정될 때 그 부모가 후보에 없어 아무것도 그려지지
    // 않고 구멍이 남는다. 렌더 배율을 낮추면 투영 오차가 함께 줄어 이 판정이 쉽게 나온다.
    // 고정 단계일 때는 GPU 가 그 단계만 통과시키므로 그 범위만 올린다.
    auto meshletRangeFor = [this, &scene](uint32_t index) {
        const GpuMesh& mesh = geometry.mesh(scene.meshOf(index));
        if (settings.automaticLod) {
            return std::pair<uint32_t, uint32_t>{mesh.meshletOffset, mesh.meshletCount};
        }
        const GpuMeshLod& fixed = geometry.lod(mesh.lodOffset + std::min(settings.lodLevel, mesh.lodCount - 1));
        return std::pair<uint32_t, uint32_t>{fixed.meshletOffset, fixed.meshletCount};
    };
    auto groupsFor = [&meshletRangeFor](uint32_t index) {
        return (meshletRangeFor(index).second + MESHLET_GROUP_SIZE - 1) / MESHLET_GROUP_SIZE;
    };
    // 조상이 숨겨져 있으면 자식도 그리지 않는다. 변환만 담는 노드는 메쉬가 없어 걸러진다.
    auto drawable = [this, &scene](uint32_t index) {
        return scene.visibleCached(index) && geometry.meshLive(scene.meshOf(index));
    };

    // 오브젝트마다의 판단을 워커에 나눠 한 번만 뽑는다. 아래 두 직렬 패스는 이 배열만 훑는다.
    // 그리지 않는 첨자에 지난 프레임 값이 남지 않게 통째로 비운다. 아래 두 패스는 drawable 로 먼저
    // 거르지만, 나중에 누가 bucket 을 무조건 읽으면 조용히 틀리기 때문이다.
    objectPlan.assign(scene.objects.size(), ObjectPlan{});
    if (!scene.objects.empty()) {
        jobs.parallelFor(static_cast<uint32_t>(scene.objects.size()), 512, [&](uint32_t begin, uint32_t end) {
            for (uint32_t index = begin; index < end; ++index) {
                ObjectPlan& plan = objectPlan[index];
                plan.drawable = drawable(index);
                if (!plan.drawable) {
                    continue;
                }
                auto [mode, sided] = bucketOf(index);
                plan.bucket = static_cast<uint8_t>(mode * 2 + sided);
                plan.meshletCount = geometry.mesh(scene.meshOf(index)).meshletCount;
                plan.groupCount = needMeshletGroups ? groupsFor(index) : 0;
            }
        });
    }

    FrameBatches batches{};
    uint32_t totalGroups = 0;
    uint32_t totalMeshletDraws = 0;
    uint32_t totalVisibilityBits = 0;
    for (const ObjectPlan& plan : objectPlan) {
        if (!plan.drawable) {
            continue;
        }
        size_t mode = plan.bucket / 2;
        size_t sided = plan.bucket % 2;
        ++batches.draws[mode][sided].count;
        totalVisibilityBits += plan.meshletCount;
        if (needMeshletGroups) {
            batches.groups[mode][sided].count += plan.groupCount;
            totalGroups += plan.groupCount;
            // 컴퓨트 컬링은 모든 단계의 meshlet 을 후보로 보므로 상한도 전체 개수로 잡는다.
            batches.meshletDraws[mode][sided].count += plan.meshletCount;
            totalMeshletDraws += plan.meshletCount;
        }
        ++batches.instanceCount;
    }
    reserveMeshletGroups(frame, totalGroups);
    reserveMeshletDraws(frame, totalMeshletDraws);
    reserveMeshletVisibility(totalVisibilityBits);

    // 조인트 행렬은 (애니메이터, 스킨) 마다 한 번만 올리고 인스턴스는 그 구간의 시작점만 가리킨다.
    std::vector<std::vector<uint32_t>> skinOffsets(scene.animators.size());
    uint32_t totalJoints = 0;
    for (size_t animator = 0; animator < scene.animators.size(); ++animator) {
        const std::vector<std::vector<glm::mat4>>& matrices = scene.animators[animator].jointMatrices;
        skinOffsets[animator].assign(matrices.size(), NO_JOINTS);
        for (size_t skin = 0; skin < matrices.size(); ++skin) {
            if (matrices[skin].empty()) {
                continue;
            }
            skinOffsets[animator][skin] = totalJoints;
            totalJoints += static_cast<uint32_t>(matrices[skin].size());
        }
    }
    reserveJoints(frame, totalJoints);
    jointMatrices.assign(totalJoints, glm::mat4{1.0F});
    for (size_t animator = 0; animator < skinOffsets.size(); ++animator) {
        for (size_t skin = 0; skin < skinOffsets[animator].size(); ++skin) {
            if (skinOffsets[animator][skin] != NO_JOINTS) {
                std::ranges::copy(scene.animators[animator].jointMatrices[skin],
                                  jointMatrices.begin() + skinOffsets[animator][skin]);
            }
        }
    }
    std::ranges::copy(jointMatrices, static_cast<glm::mat4*>(frame.jointBuffer.mapped));
    auto jointOffsetFor = [&skinOffsets, &scene](uint32_t index) {
        int32_t animator = scene.objects[index].animator;
        int32_t skin = scene.skinOf(index);
        // 부품이 오브젝트와 함께 사라질 수 있으므로 첨자를 그대로 믿지 않는다.
        if (animator < 0 || skin < 0 || static_cast<size_t>(animator) >= skinOffsets.size()) {
            return NO_JOINTS;
        }
        const std::vector<uint32_t>& offsets = skinOffsets[static_cast<size_t>(animator)];
        return static_cast<size_t>(skin) < offsets.size() ? offsets[static_cast<size_t>(skin)] : NO_JOINTS;
    };

    uint32_t drawOffset = 0;
    uint32_t groupOffset = 0;
    uint32_t meshletDrawOffset = 0;
    for (size_t mode = 0; mode < ALPHA_MODE_COUNT; ++mode) {
        for (size_t sided = 0; sided < 2; ++sided) {
            batches.draws[mode][sided].first = drawOffset;
            drawOffset += batches.draws[mode][sided].count;
            batches.groups[mode][sided].first = groupOffset;
            groupOffset += batches.groups[mode][sided].count;
            batches.meshletDraws[mode][sided].first = meshletDrawOffset;
            meshletDrawOffset += batches.meshletDraws[mode][sided].count;
        }
    }

    instanceData.resize(scene.objects.size());
    auto* instances = instanceData.data();
    // 유체마다 드로우 하나가 오브젝트 명령 뒤에 온다.
    uint32_t fluidDrawCount = particleTotal > 0 ? fluid->fluidCount() : 0;
    batches.fluidDraws = DrawBatch{static_cast<uint32_t>(scene.objects.size()), fluidDrawCount};
    batches.fluidInstanceBase = static_cast<uint32_t>(scene.objects.size());
    // 그리기 명령은 CPU 사본에 먼저 쓴다. 그림자 시점마다 이 명령을 골라 복사하는데, 매핑된 쓰기 결합
    // 메모리를 읽으면 캐시가 없어 오브젝트 만 개에서 프레임당 100 ms 를 넘긴다.
    drawCommands.resize(scene.objects.size() + fluidDrawCount);
    drawMeshletData.resize(scene.objects.size() + fluidDrawCount);
    auto* draws = drawCommands.data();
    std::vector<GpuMeshletGroup> groupData(totalGroups);
    auto* groups = groupData.data();

    std::array<std::array<uint32_t, 2>, ALPHA_MODE_COUNT> drawCursors{};
    std::array<std::array<uint32_t, 2>, ALPHA_MODE_COUNT> groupCursors{};
    for (size_t mode = 0; mode < ALPHA_MODE_COUNT; ++mode) {
        for (size_t sided = 0; sided < 2; ++sided) {
            drawCursors[mode][sided] = batches.draws[mode][sided].first;
            groupCursors[mode][sided] = batches.groups[mode][sided].first;
        }
    }

    // 슬롯 배정은 버킷 커서를 순서대로 밀어야 해서 직렬이다. 채우기는 오브젝트마다 독립이라 워커에 나눈다.
    // 각 오브젝트가 쓸 그룹·가시성 비트 시작도 여기서 미리 정한다.
    objectGroupBase.assign(scene.objects.size(), 0);
    objectVisibilityBase.assign(scene.objects.size(), 0);
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        const ObjectPlan& plan = objectPlan[index];
        if (!plan.drawable) {
            continue;
        }
        size_t mode = plan.bucket / 2;
        size_t sided = plan.bucket % 2;
        objectInstanceSlots[index] = drawCursors[mode][sided]++;
        objectGroupBase[index] = groupCursors[mode][sided];
        groupCursors[mode][sided] += plan.groupCount;
        objectVisibilityBase[index] = visibilityCursor;
        visibilityCursor += plan.meshletCount;
    }

    auto fillObject = [&](uint32_t index) {
        uint32_t slot = objectInstanceSlots[index];
        if (slot == INVALID_INSTANCE_SLOT) {
            return;
        }
        auto [mode, sided] = bucketOf(index);
        const GpuMesh& mesh = geometry.mesh(scene.meshOf(index));
        const GpuMeshLod& lod = lodFor(index);
        const glm::mat4& model = scene.world(index);

        instances[slot].model = model;
        instances[slot].previousModel = temporalReset ? model : previousWorld[index];
        instances[slot].normalMatrix = glm::mat4(glm::inverseTranspose(glm::mat3(model)));
        instanceBounds[slot] = transformBoundingSphere(model, mesh.boundingSphere);
        instances[slot].meshIndex = scene.meshOf(index);
        instances[slot].bucket = static_cast<uint32_t>(mode * 2 + sided);
        instances[slot].bucketBase = batches.meshletDraws[mode][sided].first;
        instances[slot].jointOffset = jointOffsetFor(index);
        // 스킨 인스턴스의 변형 정점 위치는 버퍼 용량이 정해진 뒤 채운다.
        instances[slot].skinnedVertexOffset = NO_SKINNED_VERTICES;
        instances[slot].previousSkinnedVertexOffset = NO_SKINNED_VERTICES;
        instances[slot].skinnedMeshletOffset = 0;
        instances[slot].visibilityBase = objectVisibilityBase[index];

        draws[slot].indexCount = lod.indexCount;
        draws[slot].instanceCount = 1;
        draws[slot].firstIndex = lod.indexOffset;
        draws[slot].vertexOffset = mesh.vertexOffset;
        // 셰이더는 gl_InstanceIndex 로 인스턴스 배열을 참조한다.
        draws[slot].firstInstance = slot;
        // 명령 하나가 LOD 단계 하나를 통째로 그리므로 meshlet 은 그 단계의 첫 것으로 대표한다.
        drawMeshletData[slot] = lod.meshletOffset;

        if (needMeshletGroups) {
            auto [meshletBase, meshletTotal] = meshletRangeFor(index);
            uint32_t groupCount = groupsFor(index);
            for (uint32_t group = 0; group < groupCount; ++group) {
                uint32_t groupSlot = objectGroupBase[index] + group;
                uint32_t first = group * MESHLET_GROUP_SIZE;
                groups[groupSlot].instanceIndex = slot;
                groups[groupSlot].firstMeshlet = meshletBase + first;
                groups[groupSlot].meshletCount = std::min(MESHLET_GROUP_SIZE, meshletTotal - first);
                groups[groupSlot].padding = 0;
            }
        }
    };
    if (!scene.objects.empty()) {
        jobs.parallelFor(static_cast<uint32_t>(scene.objects.size()), 256, [&fillObject](uint32_t begin, uint32_t end) {
            for (uint32_t index = begin; index < end; ++index) {
                fillObject(index);
            }
        });
    }

    // 스킨 인스턴스는 변형 정점을 따로 뽑아 두고 래스터와 광선 경로가 모두 그 구간을 읽는다. 같은 메쉬를
    // 여러 오브젝트가 서로 다른 포즈로 쓸 수 있어 오브젝트마다 하나씩 잡는다. 디스패치 순서가 곧 하위 가속
    // 구조 번호라 직렬로 모은다.
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        uint32_t slot = objectInstanceSlots[index];
        if (slot == INVALID_INSTANCE_SLOT || instances[slot].jointOffset == NO_JOINTS) {
            continue;
        }
        const GpuMesh& mesh = geometry.mesh(scene.meshOf(index));
        uint32_t vertexCount = geometry.meshVertexCount(scene.meshOf(index));
        objectSkinnedBlas[index] = static_cast<uint32_t>(skinDispatches.size());
        skinDispatches.push_back(SkinDispatch{static_cast<uint32_t>(mesh.vertexOffset),
                                              skinnedVertexCursor,
                                              instances[slot].jointOffset,
                                              vertexCount,
                                              geometry.meshSkinOffset(scene.meshOf(index)),
                                              mesh.meshletOffset,
                                              mesh.meshletCount,
                                              skinnedMeshletCursor});
        skinnedSlots.emplace_back(slot, objectSkinnedBlas[index]);
        skinnedVertexCursor += vertexCount;
        skinnedMeshletCursor += mesh.meshletCount;
    }

    // 유체 드로우. 입자 인스턴스는 유체 패스가 GPU 에서 채우고, 여기서는 구의 0단계 LOD 를 입자 수만큼 그리는
    // 명령만 둔다. 경계는 용기 구로 두어 그림자 컬링이 보수적으로 남긴다.
    fluidBounds.assign(fluidDrawCount, glm::vec4{0.0F});
    if (fluidDrawCount > 0) {
        const GpuMesh& sphere = geometry.mesh(fluidSphereMesh);
        const GpuMeshLod& lod = geometry.lod(sphere.lodOffset);
        uint32_t particleBase = batches.fluidInstanceBase;
        for (uint32_t f = 0; f < fluidDrawCount; ++f) {
            uint32_t slot = batches.fluidDraws.first + f;
            uint32_t count = fluid->particleCount(f);
            draws[slot].indexCount = lod.indexCount;
            // 표면으로 그리는 유체는 입자를 그리지 않는다. 인스턴스는 그대로 써 두어 광선 경로가
            // 입자를 보게 한다(표면은 아직 하위 가속 구조를 세우지 않는다).
            draws[slot].instanceCount = fluid->surfaceActive(f) ? 0 : count;
            draws[slot].firstIndex = lod.indexOffset;
            draws[slot].vertexOffset = sphere.vertexOffset;
            draws[slot].firstInstance = particleBase;
            drawMeshletData[slot] = lod.meshletOffset;
            fluidBounds[f] = fluid->bounds(f);
            particleBase += count;
        }
    }

    // 매핑된 쓰기 결합 메모리로 붓는 복사다. 한 스레드로는 대역폭을 다 못 쓴다. 오브젝트 만 개 규모에서
    // 인스턴스만 3 MB 가 넘어가므로 워커에 나눈다.
    parallelCopy(jobs,
                 drawCommands.data(),
                 static_cast<VkDrawIndexedIndirectCommand*>(frame.drawBuffer.mapped),
                 drawCommands.size());
    parallelCopy(
        jobs, drawMeshletData.data(), static_cast<uint32_t*>(frame.drawMeshletBuffer.mapped), drawMeshletData.size());
    parallelCopy(
        jobs, groupData.data(), static_cast<GpuMeshletGroup*>(frame.meshletGroupBuffer.mapped), groupData.size());

    // 그리지 않은 오브젝트도 지난 변환을 갱신해 둔다. 숨겼다 다시 보인 오브젝트가 오래된 자리에서
    // 날아오는 것처럼 보이지 않게 한다. 오브젝트마다 완전히 독립이라 워커에 나눈다.
    if (!scene.objects.empty()) {
        jobs.parallelFor(static_cast<uint32_t>(scene.objects.size()), 1024, [&](uint32_t begin, uint32_t end) {
            for (uint32_t index = begin; index < end; ++index) {
                previousWorld[index] = scene.world(index);
            }
        });
    }

    // 커질 때만 다시 잡는다. 지난 프레임이 아직 읽고 있을 수 있어 그때는 장치를 세운다. 스킨
    // 오브젝트가 늘어나는 순간에만 일어나므로 프레임마다 드는 비용은 아니다. 반쪽 둘이라 두 배다.
    bool skinBuffersReset = false;
    if (skinnedVertexCursor > skinnedVertexCapacity) {
        waitIdle();
        destroyBuffer(context, skinnedVertexBuffer);
        skinnedVertexCapacity = skinnedVertexCursor * 2;
        skinnedVertexBuffer = createBuffer(context,
                                           static_cast<VkDeviceSize>(skinnedVertexCapacity) * 2 * sizeof(asset::Vertex),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                               VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                                           MemoryLocation::DEVICE,
                                           "스킨 정점");
        skinBuffersReset = true;
    }
    if (skinnedMeshletCursor > skinnedBoundsCapacity) {
        waitIdle();
        destroyBuffer(context, skinnedBoundsBuffer);
        skinnedBoundsCapacity = skinnedMeshletCursor * 2;
        skinnedBoundsBuffer = createBuffer(context,
                                           static_cast<VkDeviceSize>(skinnedBoundsCapacity) * sizeof(glm::vec4),
                                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                           MemoryLocation::DEVICE,
                                           "스킨 meshlet 경계");
    }
    // 반쪽을 번갈아 쓴다. 지난 프레임과 스킨 목록이 같으면 다른 반쪽이 그대로 지난 포즈다.
    skinnedHalf ^= 1U;
    uint32_t currentBase = skinnedHalf * skinnedVertexCapacity;
    uint32_t previousBase = (skinnedHalf ^ 1U) * skinnedVertexCapacity;
    bool previousPoseValid = !temporalReset && !skinBuffersReset && skinDispatches == previousSkinDispatches;
    previousSkinDispatches = skinDispatches;
    // 디스패치 순서가 곧 하위 가속 구조 번호다. objectSkinnedBlas 가 같은 번호를 가리킨다.
    skinnedInstances.resize(skinDispatches.size());
    for (auto [slot, dispatchIndex] : skinnedSlots) {
        const SkinDispatch& dispatch = skinDispatches[dispatchIndex];
        uint32_t offset = currentBase + dispatch.destinationOffset;
        instances[slot].skinnedVertexOffset = offset;
        instances[slot].previousSkinnedVertexOffset =
            (previousPoseValid ? previousBase : currentBase) + dispatch.destinationOffset;
        instances[slot].skinnedMeshletOffset = dispatch.boundsOffset;
        skinnedInstances[dispatchIndex] = SkinnedInstance{instances[slot].meshIndex, offset};
    }
    parallelCopy(
        jobs, instanceData.data(), static_cast<GpuInstance*>(frame.instanceBuffer.mapped), instanceData.size());
    frameProfiler.end(instanceZone);

    auto* meshTasks = static_cast<VkDrawMeshTasksIndirectCommandEXT*>(frame.meshTaskIndirectBuffer.mapped);
    for (size_t mode = 0; mode < ALPHA_MODE_COUNT; ++mode) {
        for (size_t sided = 0; sided < 2; ++sided) {
            size_t bucket = mode * 2 + sided;
            meshTasks[bucket].groupCountX = batches.groups[mode][sided].count;
            meshTasks[bucket].groupCountY = 1;
            meshTasks[bucket].groupCountZ = 1;
        }
    }

    // 카메라 변환을 먼저 구한다. 그림자 캐스터 컬링이 이번 프레임의 카메라 절두체를 봐야 한다.
    float aspect = static_cast<float>(currentRenderExtent.width) / static_cast<float>(currentRenderExtent.height);
    glm::mat4 unjitteredViewProjection = scene.camera.projectionMatrix(aspect) * scene.camera.viewMatrix();

    // 시간축 업스케일러가 붙어 있으면 화소 안에서 표본 위치를 프레임마다 흩는다. 지터는 이번
    // 프레임 투영에만 넣고 previousViewProjection 에는 넣지 않는다. 두 프레임이 같은 격자 위에
    // 있어야 모션 벡터가 실제 화면 이동만 담는다. NDC 평행이동을 앞에 곱하면 클립 좌표의 xy 가
    // w 에 비례해 밀려, 깊이와 무관하게 정확히 지터 픽셀만큼 옮겨진다.
    //
    // 경로 추적은 흔들지 않는다. 카메라가 바뀐 것으로 보여 누적을 매 프레임 버리게 된다.
    // 다만 Ray Reconstruction 은 누적 자체를 하지 않고 지터를 요구하므로 그때는 넣는다.
    glm::vec2 jitterNdc{0.0F};
    currentJitter = glm::vec2{0.0F};
    if (temporalReady() && (!(settings.usePathTracing && rayTracer != nullptr) || rayReconstructionActive())) {
        uint32_t phases = jitterPhaseCount(currentRenderExtent.width, currentDisplayExtent.width);
        currentJitter = haltonJitter(jitterIndex % phases + 1);
        ++jitterIndex;
        jitterNdc =
            2.0F * currentJitter /
            glm::vec2{static_cast<float>(currentRenderExtent.width), static_cast<float>(currentRenderExtent.height)};
    }
    glm::mat4 cameraViewProjection =
        glm::translate(glm::mat4{1.0F}, glm::vec3{jitterNdc, 0.0F}) * unjitteredViewProjection;

    uint32_t lightZone = frameProfiler.begin("조명 구성");
    buildLights(frame, scene);
    frameProfiler.end(lightZone);
    if (settings.usePathTracing && rayTracer != nullptr) {
        // 경로 추적은 그림자 아틀라스를 읽지 않는다. 시점마다 캐스터를 걸러 명령을 짜는 CPU 비용을
        // 통째로 아끼고, 편집기 표시도 실제로 그리는 양과 어긋나지 않게 0 으로 둔다.
        shadowBatches.clear();
        shadowLayerDirty.assign(MAX_SHADOW_VIEWS, 0);
        shadowDrawsIssued = 0;
        shadowDrawsTotal = 0;
    } else {
        uint32_t shadowDrawZone = frameProfiler.begin("그림자 드로우 구성");
        buildShadowDraws(frame, batches, cameraViewProjection);
        frameProfiler.end(shadowDrawZone);
    }

    auto* camera = static_cast<GpuCamera*>(frame.cameraBuffer.mapped);
    camera->viewProjection = cameraViewProjection;
    camera->position = glm::vec4{scene.camera.position, 1.0F};
    // 화면 공간 오차 = 월드 오차 * projectionScale / 거리.
    float projectionScale = static_cast<float>(currentRenderExtent.height) /
                            (2.0F * std::tan(glm::radians(scene.camera.fovYDegrees) * 0.5F));
    camera->parameters = glm::vec4{scene.camera.nearPlane,
                                   projectionScale,
                                   settings.lodErrorThreshold,
                                   settings.automaticLod ? -1.0F : static_cast<float>(settings.lodLevel)};
    camera->shading = glm::uvec4{static_cast<uint32_t>(frameLights.size()),
                                 shadowViews.empty() ? asset::INVALID_TEXTURE : targets.shadowAtlasSlot,
                                 settings.useSsao ? targets.ssaoSlot : asset::INVALID_TEXTURE,
                                 environment->environmentSlot()};
    bool iblReady = settings.useIbl && environment->ready();
    camera->environment = glm::uvec4{environment->irradianceSlot(),
                                     environment->prefilterSlot(),
                                     environment->brdfSlot(),
                                     iblReady ? environment->prefilterMipCount() : 0U};
    bool rayShadows = settings.useRayQueryShadows && rayQueryShadowsAvailable();
    camera->ambient =
        glm::vec4{scene.ambientColor * scene.ambientIntensity, rayShadows ? settings.rayShadowDistance : 0.0F};
    camera->viewport = glm::vec4{static_cast<float>(currentRenderExtent.width),
                                 static_cast<float>(currentRenderExtent.height),
                                 1.0F / static_cast<float>(currentRenderExtent.width),
                                 1.0F / static_cast<float>(currentRenderExtent.height)};
    camera->inverseViewProjection = glm::inverse(camera->viewProjection);
    camera->previousViewProjection = temporalReset ? unjitteredViewProjection : previousViewProjection;
    camera->hzb = glm::uvec4{targets.hzbSampledSlot,
                             static_cast<uint32_t>(targets.hzbStorageSlots.size() - 1),
                             targets.hzbExtent.width,
                             targets.hzbExtent.height};
    camera->jitter = glm::vec4{
        jitterNdc, reflectionsActive() ? settings.reflectionRoughnessCutoff : 0.0F, settings.reflectionIntensity};
    camera->fog = glm::vec4{scene.post.fogColor, scene.post.fogDensity};
    camera->fogParameters = glm::vec4{scene.post.fogHeight, scene.post.fogFalloff, 0.0F, 0.0F};
    camera->flags = glm::uvec4{settings.debugMode, 0U, 0U, 0U};
    previousViewProjection = unjitteredViewProjection;
    temporalResetThisFrame = temporalReset;

    // 카메라나 화면, 장면, 경로 추적 설정이 바뀌면 누적을 처음부터 다시 쌓는다. 설정 변경을 빼면
    // 수백 표본이 쌓인 뒤에는 새 표본이 1/N 밖에 못 섞여 화면이 멈춘 것처럼 보인다.
    // 경로 추적이 그릴 수 있는 디버그 뷰만 넘긴다. 편집기가 고른 값은 그대로 두어 래스터로
    // 돌아갔을 때 이어지게 한다. 이 대입이 traceInputsChanged 를 통해 누적을 초기화한다.
    settings.pathTrace.debugMode = pathTraceSupportsDebugMode(settings.debugMode) ? settings.debugMode : 0U;
    bool traceInputsChanged = settings.pathTrace != lastPathTrace || settings.useIbl != lastUseIbl ||
                              camera->fog != lastFog || camera->fogParameters != lastFogParameters;
    lastPathTrace = settings.pathTrace;
    lastUseIbl = settings.useIbl;
    lastFog = camera->fog;
    lastFogParameters = camera->fogParameters;
    if (camera->viewProjection != lastViewProjection || sceneChangedThisFrame || traceInputsChanged) {
        lastViewProjection = camera->viewProjection;
        pathSampleCount = 0;
    }

    updateLodNetwork(scene, frame, projectionScale);

    return batches;
}

void Renderer::updateLodNetwork(const scene::Scene& scene, Frame& frame, float projectionScale) {
    // 가중치는 매 프레임 올린다. 학습을 꺼도 GPU 가 마지막 가중치를 그대로 쓰게 된다.
    auto uploadWeights = [&frame, this]() {
        std::memcpy(frame.lodNetworkBuffer.mapped, &lodNetwork.weights(), sizeof(GpuLodNetwork));
    };
    if (!settings.useNeuralLod || !settings.trainLodNetwork) {
        uploadWeights();
        return;
    }

    uint32_t candidateCount = 0;
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        if (scene.objects[index].visible && geometry.meshLive(scene.meshOf(index))) {
            candidateCount += geometry.mesh(scene.meshOf(index)).meshletCount;
        }
    }
    if (candidateCount == 0) {
        uploadWeights();
        return;
    }

    uint32_t stride = std::max(candidateCount / MAX_LOD_SAMPLES, 1U);
    std::vector<LodSample> samples;
    samples.reserve(std::min(candidateCount, MAX_LOD_SAMPLES) + 1);

    glm::vec3 cameraPosition = scene.camera.position;
    uint32_t candidateIndex = 0;
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        const scene::Object& object = scene.objects[index];
        if (!object.visible || !geometry.meshLive(scene.meshOf(index))) {
            continue;
        }
        const GpuMesh& mesh = geometry.mesh(scene.meshOf(index));
        glm::mat4 model = object.transform.matrix();
        float modelScale = std::sqrt(std::max({glm::dot(glm::vec3(model[0]), glm::vec3(model[0])),
                                               glm::dot(glm::vec3(model[1]), glm::vec3(model[1])),
                                               glm::dot(glm::vec3(model[2]), glm::vec3(model[2]))}));

        for (uint32_t i = 0; i < mesh.meshletCount; ++i, ++candidateIndex) {
            if (candidateIndex % stride != 0) {
                continue;
            }
            const GpuMeshlet& meshlet = geometry.meshlet(mesh.meshletOffset + i);
            // 최상위 meshlet 은 오차가 무한대라 학습 표본에서 뺀다.
            if (!std::isfinite(meshlet.error) || meshlet.error <= 0.0F) {
                continue;
            }

            glm::vec3 center = glm::vec3(model * glm::vec4{glm::vec3(meshlet.errorSphere), 1.0F});
            float radius = meshlet.errorSphere.w * modelScale;
            float viewDistance = std::max(glm::distance(center, cameraPosition) - radius, 1e-3F);
            float projected = meshlet.error * projectionScale / viewDistance;

            LodSample sample;
            sample.features[0] = std::log2(std::max(viewDistance, 1e-3F)) / 8.0F;
            sample.features[1] = std::log2(std::max(radius, 1e-6F)) / 8.0F;
            sample.features[2] = std::log2(std::max(meshlet.error, 1e-6F)) / 8.0F;
            sample.features[3] = std::log2(std::max(projected, 1e-6F)) / 8.0F;
            sample.projectedError = projected;
            sample.triangleCount = static_cast<float>(meshlet.triangleCount);
            samples.push_back(sample);
        }
    }

    // 표본은 전체의 일부이므로 예산도 같은 비율로 줄인다.
    float sampleRatio = static_cast<float>(samples.size() * stride) / static_cast<float>(candidateCount);
    lodNetwork.train(samples, settings.lodErrorThreshold, settings.triangleBudget * std::max(sampleRatio, 1e-3F));
    uploadWeights();
}

void Renderer::updateAccelerationStructures(VkCommandBuffer commandBuffer, const scene::Scene& scene) {
    // 장면이 그대로면 가속 구조도 그대로다. 포즈가 바뀌면 애니메이터가 장면 리비전을 올리므로
    // 같은 조건으로 걸러진다. 스킨 인스턴스가 있으면 변형 정점 반쪽이 프레임마다 번갈아 바뀌어
    // 지난 구조가 가리키던 자리가 덮어써지므로 포즈가 그대로여도 다시 세운다.
    // ponytail: 포즈가 그대로인 스킨 오브젝트까지 매 프레임 다시 세운다. 반쪽을 번갈지 않고 복사로
    // 지난 포즈를 남기면 건너뛸 수 있다.
    if (!ensureBottomLevel()) {
        return;
    }
    if (!sceneChangedThisFrame && skinnedInstances.empty() && rayTracer->ready()) {
        return;
    }
    rayTracer->updateSkinnedBottomLevel(commandBuffer, skinnedVertexBuffer, skinnedInstances);
    rayTracer->updateTopLevel(commandBuffer,
                              scene,
                              objectInstanceSlots,
                              objectSkinnedBlas,
                              static_cast<uint32_t>(frameIndex % FRAMES_IN_FLIGHT),
                              fluidTlasPrepended);
}

} // namespace gfx
