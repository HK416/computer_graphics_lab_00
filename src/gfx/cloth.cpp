#include "gfx/cloth.h"

#include <algorithm>
#include <cstring>

#include <spdlog/spdlog.h>

#include "asset/model.h"
#include "asset/vertex_pack.h"
#include "core/job_system.h"
#include "gfx/bindless.h"
#include "gfx/context.h"
#include "gfx/geometry.h"
#include "gfx/vk_check.h"

namespace gfx {
namespace {

// 만들지 못하면 VK_NULL_HANDLE 을 돌려준다. 천은 CPU 백엔드로 내려갈 수 있어 중단하지 않는다.
VkPipeline createComputePipeline(Context& context, VkPipelineLayout layout, const char* shaderName) {
    VkShaderModule module = tryCreateShaderModule(context.device, shaderName);
    if (module == VK_NULL_HANDLE) {
        return VK_NULL_HANDLE;
    }
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = stage;
    info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
        spdlog::warn("천 컴퓨트 파이프라인을 만들지 못했습니다: {}", shaderName);
        pipeline = VK_NULL_HANDLE;
    }
    vkDestroyShaderModule(context.device, module, nullptr);
    return pipeline;
}

uint32_t groupsFor(uint32_t count) {
    return (count + CLOTH_GROUP_SIZE - 1) / CLOTH_GROUP_SIZE;
}

Buffer hostBuffer(Context& context, VkDeviceSize bytes, const char* name, VkBufferUsageFlags extra = 0) {
    return createBuffer(context,
                        std::max<VkDeviceSize>(bytes, 16),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extra,
                        MemoryLocation::HOST_WRITE,
                        name);
}

Buffer deviceBuffer(Context& context, VkDeviceSize bytes, const char* name, VkBufferUsageFlags extra = 0) {
    return createBuffer(context,
                        std::max<VkDeviceSize>(bytes, 16),
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | extra,
                        MemoryLocation::DEVICE,
                        name);
}

} // namespace

ClothSimulator::ClothSimulator(Context& context,
                               BindlessTextures& bindless,
                               core::JobSystem& jobs,
                               VkDescriptorSetLayout accelerationLayout)
    : context(context), bindless(bindless), jobs(jobs) {
    createPipelines(accelerationLayout);
}

ClothSimulator::~ClothSimulator() {
    for (State& state : states) {
        destroyState(state);
    }
    for (VkPipeline pipeline :
         {predictPipeline, constraintPipeline, finishPipeline, finishRayQueryPipeline, writePipeline}) {
        vkDestroyPipeline(context.device, pipeline, nullptr);
    }
    vkDestroyPipelineLayout(context.device, rayQueryLayout, nullptr);
    vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
}

void ClothSimulator::createPipelines(VkDescriptorSetLayout accelerationLayout) {
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.size = sizeof(ClothPushConstants);
    std::array<VkDescriptorSetLayout, 2> sets{bindless.layout(), accelerationLayout};
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = sets.data();
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &range;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &pipelineLayout));
    predictPipeline = createComputePipeline(context, pipelineLayout, "cloth_predict.comp.spv");
    constraintPipeline = createComputePipeline(context, pipelineLayout, "cloth_constraint.comp.spv");
    finishPipeline = createComputePipeline(context, pipelineLayout, "cloth_finish.comp.spv");
    writePipeline = createComputePipeline(context, pipelineLayout, "cloth_write.comp.spv");
    gpuReady = predictPipeline != VK_NULL_HANDLE && constraintPipeline != VK_NULL_HANDLE &&
               finishPipeline != VK_NULL_HANDLE && writePipeline != VK_NULL_HANDLE;
    if (!gpuReady) {
        spdlog::warn("천 GPU 백엔드를 만들지 못해 CPU 로 돈다");
        return;
    }
    if (accelerationLayout == VK_NULL_HANDLE) {
        return;
    }
    layoutInfo.setLayoutCount = 2;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &rayQueryLayout));
    finishRayQueryPipeline = createComputePipeline(context, rayQueryLayout, "cloth_finish_rq.comp.spv");
    if (finishRayQueryPipeline == VK_NULL_HANDLE) {
        spdlog::warn("천 광선 질의 변종을 만들지 못해 GPU 천이 메쉬 콜라이더를 지나간다");
    }
}

void ClothSimulator::destroyState(State& state) {
    // 진행 중인 프레임의 명령 버퍼에 주소가 실려 있을 수 있다. 그 프레임이 끝난 뒤에 지운다.
    for (Buffer* buffer : {&state.positions,
                           &state.predicted,
                           &state.velocities,
                           &state.lambdas,
                           &state.rest,
                           &state.info,
                           &state.grid,
                           &state.constraints}) {
        context.retireBuffer(*buffer);
    }
    for (Buffer& buffer : state.params) {
        context.retireBuffer(buffer);
    }
    for (Buffer& buffer : state.staging) {
        context.retireBuffer(buffer);
    }
    state = State{};
}

void ClothSimulator::ensureBuffers(State& state) {
    auto vertexCount = static_cast<uint32_t>(state.topology.rest.size());
    auto constraintCount = static_cast<uint32_t>(state.topology.constraints.size());
    if (vertexCount <= state.vertexCapacity && constraintCount <= state.constraintCapacity) {
        return;
    }
    // 크기가 바뀌면 통째로 새로 잡는다. 판단에 쓰는 값은 살린다.
    State keep;
    keep.cpu = state.cpu;
    keep.topology = std::move(state.topology);
    keep.solver = std::move(state.solver);
    keep.objectIndex = state.objectIndex;
    keep.vertexCount = state.vertexCount;
    keep.lastSettings = state.lastSettings;
    keep.lastWorld = state.lastWorld;
    keep.valid = state.valid;
    destroyState(state);
    state = std::move(keep);
    state.needsReset = true;
    state.vertexCapacity = vertexCount;
    state.constraintCapacity = constraintCount;
    VkDeviceSize vec4Bytes = static_cast<VkDeviceSize>(vertexCount) * sizeof(glm::vec4);
    if (!state.cpu) {
        state.positions = deviceBuffer(context, vec4Bytes, "천 위치");
        state.predicted = deviceBuffer(context, vec4Bytes, "천 예측 위치");
        state.velocities = deviceBuffer(context, vec4Bytes, "천 속도");
        state.lambdas = deviceBuffer(context,
                                     static_cast<VkDeviceSize>(constraintCount) * sizeof(float),
                                     "천 람다",
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        state.rest = hostBuffer(context, vec4Bytes, "천 정지 자세");
        state.info = hostBuffer(
            context, static_cast<VkDeviceSize>(vertexCount) * sizeof(physics::ClothVertexInfo), "천 정점 정보");
        state.grid = hostBuffer(
            context, static_cast<VkDeviceSize>(state.topology.gridToVertex.size()) * sizeof(uint32_t), "천 격자");
        state.constraints = hostBuffer(
            context, static_cast<VkDeviceSize>(constraintCount) * sizeof(physics::ClothConstraint), "천 제약");
        for (Buffer& buffer : state.params) {
            buffer = hostBuffer(context, sizeof(GpuClothParams), "천 설정");
        }
    } else {
        for (Buffer& buffer : state.staging) {
            buffer = hostBuffer(context,
                                static_cast<VkDeviceSize>(vertexCount) * sizeof(asset::Vertex),
                                "천 정점 스테이징",
                                VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        }
    }
}

void ClothSimulator::uploadStatic(State& state) {
    if (state.cpu) {
        return;
    }
    const physics::ClothTopology& topology = state.topology;
    std::memcpy(state.rest.mapped, topology.rest.data(), topology.rest.size() * sizeof(glm::vec4));
    std::memcpy(
        state.info.mapped, topology.vertices.data(), topology.vertices.size() * sizeof(physics::ClothVertexInfo));
    std::memcpy(state.grid.mapped, topology.gridToVertex.data(), topology.gridToVertex.size() * sizeof(uint32_t));
    std::memcpy(state.constraints.mapped,
                topology.constraints.data(),
                topology.constraints.size() * sizeof(physics::ClothConstraint));
}

void ClothSimulator::packCpuVertices(State& state, uint32_t frameSlot) {
    const physics::ClothTopology& topology = state.topology;
    auto* vertices = static_cast<asset::Vertex*>(state.staging[frameSlot % CLOTH_FRAMES].mapped);
    const auto& positions = state.solver.positions();
    const auto& normals = state.solver.normals();
    const auto& tangents = state.solver.tangents();
    auto n = static_cast<float>(topology.resolution);
    for (uint32_t i = 0; i < positions.size(); ++i) {
        asset::Vertex vertex;
        vertex.position = positions[i];
        vertex.normal = asset::packUnitVector(normals[i]);
        // 내장 평면과 같은 손 방향(-1).
        vertex.tangent = asset::packTangent(glm::vec4{tangents[i], -1.0F});
        vertex.uv =
            glm::vec2{static_cast<float>(topology.vertices[i].gridX), static_cast<float>(topology.vertices[i].gridZ)} /
            n;
        vertices[i] = vertex;
    }
}

bool ClothSimulator::prepare(const scene::Scene& scene,
                             bool sceneSwitched,
                             float deltaSeconds,
                             const GeometryStore& geometry) {
    if (sceneSwitched || scene.id != lastSceneId) {
        for (State& state : states) {
            destroyState(state);
        }
        states.clear();
        lastSceneId = scene.id;
    }
    for (size_t i = scene.cloths.size(); i < states.size(); ++i) {
        destroyState(states[i]);
    }
    states.resize(scene.cloths.size());

    // 부품 배열이 압축되면 첨자가 밀린다. 그런 프레임에는 전부 다시 짓는다.
    if (scene.componentRevision() != lastComponentRevision) {
        lastComponentRevision = scene.componentRevision();
        for (State& state : states) {
            state.needsReset = true;
            state.lastSettings = scene::Cloth{};
            state.lastWorld = glm::mat4{0.0F};
        }
    }

    std::vector<uint32_t> owners(scene.cloths.size(), UINT32_MAX);
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        int32_t slot = scene.objects[index].cloth;
        if (slot >= 0 && static_cast<size_t>(slot) < owners.size()) {
            owners[static_cast<size_t>(slot)] = index;
        }
    }
    bool playEdge = scene.simulating != wasSimulating;
    wasSimulating = scene.simulating;

    bool anyStepped = false;
    for (size_t i = 0; i < states.size(); ++i) {
        State& state = states[i];
        const scene::Cloth& settings = scene.cloths[i];
        state.objectIndex = owners[i];
        state.steppedThisFrame = false;
        state.valid = false;
        if (state.objectIndex == UINT32_MAX) {
            continue;
        }
        // 메쉬가 천 격자여야 한다. 정점 수와 콜라이더 메쉬 사본(정점 순서의 근거)으로 가린다.
        uint32_t mesh = scene.meshOf(state.objectIndex);
        const scene::ColliderMesh* colliderMesh = scene.colliderMesh(state.objectIndex);
        uint32_t side = settings.resolution + 1;
        if (mesh == scene::INVALID_MESH || !geometry.meshLive(mesh) || colliderMesh == nullptr ||
            geometry.meshVertexCount(mesh) != side * side || colliderMesh->positions.size() != side * side) {
            continue;
        }
        bool wantsCpu = settings.backend == scene::SimulationBackend::CPU || !gpuReady;
        if (wantsCpu != state.cpu) {
            destroyState(state);
            state.objectIndex = owners[i];
            state.cpu = wantsCpu;
        }
        glm::mat4 world = scene.visibleCached(state.objectIndex) ? scene.world(state.objectIndex) : glm::mat4{0.0F};
        if (world == glm::mat4{0.0F}) {
            continue;
        }
        // 설정·변환이 바뀌면 위상과 정지 자세를 다시 짓는다. 재생 에지도 처음으로 돌린다.
        if (!(settings == state.lastSettings) || world != state.lastWorld || playEdge) {
            if (!physics::buildClothTopology(colliderMesh->positions, world, settings, state.topology)) {
                continue;
            }
            state.lastSettings = settings;
            state.lastWorld = world;
            state.needsReset = true;
        }
        state.vertexCount = static_cast<uint32_t>(state.topology.rest.size());
        state.valid = true;
        ensureBuffers(state);
        if (state.needsReset) {
            uploadStatic(state);
            if (state.cpu) {
                state.solver.reset(state.topology);
            }
        }
        state.frameParams = physics::deriveClothParams(settings, scene, scene.simulating ? deltaSeconds : 0.0F);
        bool advance = scene.simulating && state.frameParams.frameStep > 0.0F;
        state.steppedThisFrame = advance || state.needsReset;
        if (state.cpu && advance) {
            // CPU 백엔드는 여기서 돈다. 메쉬 삼각형은 천 자신을 뺀 보이는 MESH 강체다.
            physics::collectMeshTriangles(scene, state.objectIndex, state.triangles);
            state.solver.step(state.topology, state.frameParams, state.triangles, &jobs);
        }
        anyStepped = anyStepped || state.steppedThisFrame;
    }
    return anyStepped;
}

bool ClothSimulator::active(uint32_t index) const {
    return index < states.size() && states[index].valid;
}

bool ClothSimulator::stepped(uint32_t index) const {
    return index < states.size() && states[index].valid && states[index].steppedThisFrame;
}

bool ClothSimulator::onCpu(uint32_t index) const {
    return index < states.size() && states[index].cpu;
}

glm::vec4 ClothSimulator::bounds(uint32_t index) const {
    if (!active(index)) {
        return glm::vec4{0.0F};
    }
    const State& state = states[index];
    if (state.cpu) {
        return state.solver.bounds();
    }
    // ponytail: GPU 천의 자세는 CPU 가 모른다. 정지 자세 대각의 세 배면 매달린 천은 담기지만 멀리 떨어진 천은 그림자
    // 컬링에서 빠진다. GPU AABB 되읽기가 해법이다.
    return glm::vec4{state.topology.restCenter, state.topology.restRadius * 3.0F + 1.0F};
}

void ClothSimulator::record(VkCommandBuffer commandBuffer,
                            uint32_t frameSlot,
                            uint32_t index,
                            VkBuffer skinnedVertexBuffer,
                            VkDeviceAddress skinnedVertexAddress,
                            uint32_t destinationVertexOffset,
                            VkDescriptorSet accelerationSet,
                            bool rayQuery) {
    State& state = states[index];
    if (!state.valid) {
        return;
    }
    frameSlot %= CLOTH_FRAMES;
    if (state.cpu) {
        // 스테이징에 이번 프레임 정점을 채우고 변형 정점 버퍼로 복사한다. 배리어는 스킨 패스가 앞뒤로 친다.
        packCpuVertices(state, frameSlot);
        VkBufferCopy copy{};
        copy.dstOffset = static_cast<VkDeviceSize>(destinationVertexOffset) * sizeof(asset::Vertex);
        copy.size = static_cast<VkDeviceSize>(state.vertexCount) * sizeof(asset::Vertex);
        vkCmdCopyBuffer(commandBuffer, state.staging[frameSlot].handle, skinnedVertexBuffer, 1, &copy);
        state.needsReset = false;
        return;
    }

    const physics::ClothParams& params = state.frameParams;
    GpuClothParams gpuParams;
    float h = params.frameStep / static_cast<float>(params.substeps);
    gpuParams.accelerationDamping = glm::vec4{params.gravity + params.wind, params.damping};
    gpuParams.thicknessFriction = glm::vec4{params.thickness, params.friction, h, 0.0F};
    gpuParams.vertexCount = state.vertexCount;
    gpuParams.constraintCount = static_cast<uint32_t>(state.topology.constraints.size());
    gpuParams.resolution = state.topology.resolution;
    gpuParams.colliderCount = params.colliderCount;
    for (uint32_t i = 0; i < params.colliderCount; ++i) {
        const physics::FluidCollider& source = params.colliders[i];
        GpuFluidCollider& collider = gpuParams.colliders[i];
        collider.type = static_cast<uint32_t>(source.shape);
        collider.data0 = glm::vec4{source.halfExtents, source.radius};
        collider.world = source.world;
        collider.inverseWorld = source.inverseWorld;
    }
    std::memcpy(state.params[frameSlot].mapped, &gpuParams, sizeof(gpuParams));

    rayQuery = rayQuery && finishRayQueryPipeline != VK_NULL_HANDLE;
    ClothPushConstants push;
    push.params = state.params[frameSlot].address;
    push.positions = state.positions.address;
    push.predicted = state.predicted.address;
    push.velocities = state.velocities.address;
    push.constraints = state.constraints.address;
    push.lambdas = state.lambdas.address;
    push.rest = state.rest.address;
    push.info = state.info.address;
    push.grid = state.grid.address;
    push.vertices = skinnedVertexAddress;
    push.destinationOffset = destinationVertexOffset;
    push.vertexCount = state.vertexCount;
    push.dt = h;

    VkDescriptorSet bindlessSet = bindless.set();
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    auto dispatch = [&](VkPipeline pipeline, VkPipelineLayout layout, uint32_t count) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdPushConstants(commandBuffer, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(commandBuffer, groupsFor(count), 1, 1);
    };
    auto computeToCompute = [&]() {
        memoryBarrier(commandBuffer,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    };

    // 지난 프레임의 마무리·쓰기 컴퓨트가 자체 버퍼(위치·속도·람다)에 쓴 것을 이번 예측이 읽는다. 스킨 패스의
    // 배리어는 변형 정점 버퍼만 보므로 여기서 따로 친다.
    memoryBarrier(commandBuffer,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    if (state.needsReset) {
        push.flags = CLOTH_FLAG_RESET;
        dispatch(predictPipeline, pipelineLayout, state.vertexCount);
        computeToCompute();
        push.flags = 0;
        state.needsReset = false;
    }

    uint32_t substeps = params.frameStep > 0.0F ? params.substeps : 0U;
    for (uint32_t substep = 0; substep < substeps; ++substep) {
        // XPBD 람다는 서브스텝마다 0 에서 시작한다.
        vkCmdFillBuffer(commandBuffer, state.lambdas.handle, 0, VK_WHOLE_SIZE, 0);
        memoryBarrier(commandBuffer,
                      VK_PIPELINE_STAGE_2_CLEAR_BIT,
                      VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        dispatch(predictPipeline, pipelineLayout, state.vertexCount);
        computeToCompute();
        for (uint32_t iteration = 0; iteration < params.iterations; ++iteration) {
            for (uint32_t color = 0; color < physics::CLOTH_COLORS; ++color) {
                push.constraintFirst = state.topology.colorBegin[color];
                push.constraintCount = state.topology.colorBegin[color + 1] - push.constraintFirst;
                if (push.constraintCount == 0) {
                    continue;
                }
                dispatch(constraintPipeline, pipelineLayout, push.constraintCount);
                computeToCompute();
            }
        }
        if (rayQuery) {
            std::array<VkDescriptorSet, 2> sets{bindlessSet, accelerationSet};
            vkCmdBindDescriptorSets(commandBuffer,
                                    VK_PIPELINE_BIND_POINT_COMPUTE,
                                    rayQueryLayout,
                                    0,
                                    static_cast<uint32_t>(sets.size()),
                                    sets.data(),
                                    0,
                                    nullptr);
            push.flags = CLOTH_FLAG_RAY_QUERY;
            dispatch(finishRayQueryPipeline, rayQueryLayout, state.vertexCount);
            push.flags = 0;
            vkCmdBindDescriptorSets(
                commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
        } else {
            dispatch(finishPipeline, pipelineLayout, state.vertexCount);
        }
        computeToCompute();
    }
    // 변형 정점 버퍼에 쓴다. 뒤의 경계 구 컴퓨트·그림자·장면 패스는 스킨 패스의 배리어가 맡는다.
    dispatch(writePipeline, pipelineLayout, state.vertexCount);
}

} // namespace gfx
