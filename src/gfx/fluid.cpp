#include "gfx/fluid.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include "gfx/bindless.h"
#include "gfx/context.h"
#include "gfx/vk_check.h"

namespace gfx {
namespace {

VkPipeline createComputePipeline(Context& context, VkPipelineLayout layout, const char* shaderName) {
    VkShaderModule module = createShaderModule(context.device, shaderName);
    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = stage;
    info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline));
    vkDestroyShaderModule(context.device, module, nullptr);
    return pipeline;
}

void memoryBarrier(VkCommandBuffer commandBuffer,
                   VkPipelineStageFlags2 sourceStage,
                   VkAccessFlags2 sourceAccess,
                   VkPipelineStageFlags2 destinationStage,
                   VkAccessFlags2 destinationAccess) {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = sourceStage;
    barrier.srcAccessMask = sourceAccess;
    barrier.dstStageMask = destinationStage;
    barrier.dstAccessMask = destinationAccess;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

// 인스턴스와 가속 구조 입력을 읽는 단계 전부. 스킨 패스와 같은 집합이다.
constexpr VkPipelineStageFlags2 READER_STAGES =
    VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
    VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
    VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
    VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;

uint32_t groupsFor(uint32_t count) {
    return (count + FLUID_GROUP_SIZE - 1) / FLUID_GROUP_SIZE;
}

} // namespace

FluidSimulator::FluidSimulator(Context& context, BindlessTextures& bindless) : context(context), bindless(bindless) {
    createPipelines();
}

FluidSimulator::~FluidSimulator() {
    for (State& state : states) {
        destroyState(state);
    }
    for (VkPipeline pipeline : {emitPipeline, gridPipeline, densityPipeline, integratePipeline, instancesPipeline}) {
        vkDestroyPipeline(context.device, pipeline, nullptr);
    }
    vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
}

void FluidSimulator::createPipelines() {
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.size = sizeof(FluidPushConstants);
    VkDescriptorSetLayout bindlessLayout = bindless.layout();
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &bindlessLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &range;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &pipelineLayout));

    emitPipeline = createComputePipeline(context, pipelineLayout, "fluid_emit.comp.spv");
    gridPipeline = createComputePipeline(context, pipelineLayout, "fluid_grid.comp.spv");
    densityPipeline = createComputePipeline(context, pipelineLayout, "fluid_density.comp.spv");
    integratePipeline = createComputePipeline(context, pipelineLayout, "fluid_integrate.comp.spv");
    instancesPipeline = createComputePipeline(context, pipelineLayout, "fluid_instances.comp.spv");
}

void FluidSimulator::destroyState(State& state) {
    // 진행 중인 프레임의 명령 버퍼에 이 버퍼들의 주소가 이미 실려 있을 수 있다. 바로 지우면 GPU 가
    // 사라진 메모리를 읽는다. 맡겨 두고 그 프레임이 끝난 뒤에 지운다.
    for (Buffer& buffer : state.positions) {
        context.retireBuffer(buffer);
    }
    for (Buffer& buffer : state.velocities) {
        context.retireBuffer(buffer);
    }
    context.retireBuffer(state.previousRendered);
    context.retireBuffer(state.cellCounts);
    context.retireBuffer(state.cellParticles);
    for (Buffer& buffer : state.params) {
        context.retireBuffer(buffer);
    }
    // 슬롯이 다른 유체에게 넘어갈 수 있으므로 판단에 쓰는 값도 함께 지운다. 남겨 두면 설정이 같은
    // 다른 유체가 «변한 것이 없다»로 보여 옛 입자를 그대로 이어받는다.
    state = State{};
}

void FluidSimulator::ensureCapacity(State& state, uint32_t count) {
    if (count <= state.capacity && state.capacity > 0) {
        return;
    }
    // destroyState 가 옛 버퍼를 맡겨 두므로 진행 중인 프레임을 기다릴 필요가 없다. 이번 프레임 prepare
    // 가 이미 정해 둔 두 값은 살려 둔다.
    uint32_t objectIndex = state.objectIndex;
    uint32_t particleCount = state.count;
    destroyState(state);
    state.objectIndex = objectIndex;
    state.count = particleCount;
    state.capacity = count;
    // 셀은 입자의 두 배쯤을 2 의 거듭제곱으로. 해시가 마스크로 접히므로 거듭제곱이어야 한다.
    state.cellCount = std::clamp<uint32_t>(std::bit_ceil(count * 2), 1024, 65536);
    VkDeviceSize particleBytes = static_cast<VkDeviceSize>(count) * sizeof(glm::vec4);
    for (Buffer& buffer : state.positions) {
        buffer = createBuffer(
            context, particleBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryLocation::DEVICE, "유체 위치");
    }
    for (Buffer& buffer : state.velocities) {
        buffer = createBuffer(
            context, particleBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryLocation::DEVICE, "유체 속도");
    }
    state.previousRendered = createBuffer(
        context, particleBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryLocation::DEVICE, "유체 지난 위치");
    state.cellCounts = createBuffer(context,
                                    static_cast<VkDeviceSize>(state.cellCount) * sizeof(uint32_t),
                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                    MemoryLocation::DEVICE,
                                    "유체 격자 개수");
    state.cellParticles =
        createBuffer(context,
                     static_cast<VkDeviceSize>(state.cellCount) * FLUID_CELL_CAPACITY * sizeof(uint32_t),
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     MemoryLocation::DEVICE,
                     "유체 격자 입자");
    for (Buffer& buffer : state.params) {
        buffer = createBuffer(context,
                              sizeof(GpuFluidParams),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              MemoryLocation::HOST_WRITE,
                              "유체 설정");
    }
    state.needsEmit = true;
}

bool FluidSimulator::prepare(const scene::Scene& scene, bool sceneSwitched) {
    // 장면이 바뀌면 상태를 통째로 버린다. 유체 번호가 다른 장면 것과 겹치기 때문이다.
    if (sceneSwitched || &scene != lastScene) {
        for (State& state : states) {
            destroyState(state);
        }
        states.clear();
        lastScene = &scene;
    }
    if (states.size() > scene.fluids.size()) {
        for (size_t i = scene.fluids.size(); i < states.size(); ++i) {
            destroyState(states[i]);
        }
    }
    states.resize(scene.fluids.size());

    // 부품 배열이 압축되면 첨자가 밀려 states[k] 가 다른 유체를 맡게 된다. 설정과 변환이 우연히
    // 같으면 «변한 것이 없다»로 보여 남의 입자를 그대로 이어받으므로, 그런 프레임에는 전부 버린다.
    if (scene.componentRevision() != lastComponentRevision) {
        lastComponentRevision = scene.componentRevision();
        for (State& state : states) {
            state.needsEmit = true;
            state.lastSettings = scene::Fluid{};
            state.lastWorld = glm::mat4{0.0F};
        }
    }

    // 부품 번호 -> 오브젝트. 오브젝트가 없는 부품은 그리지 않는다.
    std::vector<uint32_t> owners(scene.fluids.size(), UINT32_MAX);
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        int32_t slot = scene.objects[index].fluid;
        if (slot >= 0 && static_cast<size_t>(slot) < owners.size()) {
            owners[static_cast<size_t>(slot)] = index;
        }
    }

    // 재생을 시작하거나 멈추면 처음 자리로 돌린다.
    bool playEdge = scene.simulating != wasSimulating;
    wasSimulating = scene.simulating;

    bool active = false;
    for (size_t i = 0; i < states.size(); ++i) {
        State& state = states[i];
        const scene::Fluid& settings = scene.fluids[i];
        state.objectIndex = owners[i];
        state.count = state.objectIndex == UINT32_MAX ? 0 : std::min(settings.particleCount, particleLimit);
        if (state.count == 0) {
            continue;
        }
        ensureCapacity(state, state.count);
        glm::mat4 world = scene.visibleCached(state.objectIndex) ? scene.world(state.objectIndex) : glm::mat4{0.0F};
        if (!(settings == state.lastSettings) || world != state.lastWorld || playEdge) {
            state.needsEmit = true;
            state.lastSettings = settings;
            state.lastWorld = world;
        }
        active = active || state.needsEmit || scene.simulating;
    }
    return active;
}

uint32_t FluidSimulator::particleCount(uint32_t index) const {
    return index < states.size() ? states[index].count : 0;
}

uint32_t FluidSimulator::totalParticles() const {
    uint32_t total = 0;
    for (const State& state : states) {
        total += state.count;
    }
    return total;
}

glm::vec4 FluidSimulator::bounds(uint32_t index) const {
    if (index >= states.size()) {
        return glm::vec4{0.0F};
    }
    const scene::Fluid& settings = states[index].lastSettings;
    glm::vec3 center = (settings.containerMin + settings.containerMax) * 0.5F;
    float radius = glm::length(settings.containerMax - settings.containerMin) * 0.5F + settings.particleRadius;
    return glm::vec4{center, radius};
}

void FluidSimulator::fillParams(GpuFluidParams& params, const State& state, const scene::Scene& scene) const {
    const scene::Fluid& settings = state.lastSettings;
    float spacing = settings.particleRadius * 2.0F;
    params.emitterWorld = state.lastWorld;
    params.emitterHalfExtents = glm::vec4{settings.emitterHalfExtents, spacing};
    params.containerMin = glm::vec4{settings.containerMin, settings.particleRadius};
    params.containerMax = glm::vec4{settings.containerMax, 0.3F};
    // 질량은 입자 하나가 간격 세제곱의 물을 대신하도록 잡는다. 그래야 밀도가 기준 밀도 언저리에서 시작한다.
    params.gravity = glm::vec4{settings.gravity, settings.restDensity * spacing * spacing * spacing};
    params.smoothingRadius = std::max(settings.smoothingRadius, spacing);
    params.restDensity = settings.restDensity;
    params.stiffness = settings.stiffness;
    params.viscosity = settings.viscosity;
    // 방출 격자. 상자 안에 x, y 로 채우고 남는 입자는 z 로 이어 쌓는다.
    auto along = [spacing](float half) {
        return std::max(1U, static_cast<uint32_t>(std::floor(half * 2.0F / spacing)));
    };
    uint32_t nx = along(settings.emitterHalfExtents.x);
    uint32_t ny = along(settings.emitterHalfExtents.y);
    uint32_t nz = std::max(1U, (state.count + nx * ny - 1) / (nx * ny));
    params.lattice = glm::uvec4{nx, ny, nz, state.cellCount};

    // 강체를 콜라이더로 넘긴다. 일방향 결합이라 입자가 강체를 밀지는 못한다.
    params.colliderCount = 0;
    for (uint32_t index = 0; index < scene.objects.size() && params.colliderCount < FLUID_MAX_COLLIDERS; ++index) {
        int32_t slot = scene.objects[index].rigidBody;
        if (slot < 0 || static_cast<size_t>(slot) >= scene.rigidBodies.size() || !scene.visibleCached(index)) {
            continue;
        }
        const scene::RigidBody& body = scene.rigidBodies[static_cast<size_t>(slot)];
        scene::Transform world = scene::Transform::fromMatrix(scene.world(index));
        GpuFluidCollider& collider = params.colliders[params.colliderCount++];
        collider.type = static_cast<uint32_t>(body.shape);
        glm::mat4 rigid = glm::translate(glm::mat4{1.0F}, world.position) * glm::mat4_cast(world.rotation);
        switch (body.shape) {
        case scene::ColliderShape::SPHERE:
            collider.data0 =
                glm::vec4{world.position, body.radius * std::max({world.scale.x, world.scale.y, world.scale.z})};
            break;
        case scene::ColliderShape::BOX:
            collider.data0 = glm::vec4{body.halfExtents * world.scale, 0.0F};
            collider.world = rigid;
            collider.inverseWorld = glm::inverse(rigid);
            break;
        case scene::ColliderShape::PLANE: {
            glm::vec3 normal = world.rotation * glm::vec3{0.0F, 1.0F, 0.0F};
            collider.data0 = glm::vec4{normal, glm::dot(normal, world.position)};
            break;
        }
        }
    }
}

void FluidSimulator::record(VkCommandBuffer commandBuffer,
                            uint32_t frameSlot,
                            uint32_t index,
                            const scene::Scene& scene,
                            float deltaSeconds,
                            VkDeviceAddress instances,
                            uint32_t instanceBase,
                            VkDeviceAddress tlasInstances,
                            uint32_t tlasBase,
                            VkDeviceAddress sphereBlas,
                            uint32_t sphereMesh,
                            bool resetHistory) {
    State& state = states[index];
    if (state.count == 0) {
        return;
    }
    frameSlot %= FLUID_FRAMES;

    GpuFluidParams params;
    fillParams(params, state, scene);
    std::memcpy(state.params[frameSlot].mapped, &params, sizeof(params));

    // 서브스텝. 압력파 속도(√강성)와 낙하 속도로 안정 간격을 어림한다.
    float h = params.smoothingRadius;
    float gravity = std::max(glm::length(glm::vec3{params.gravity}), 0.1F);
    float stableStep = std::min(0.4F * h / std::sqrt(std::max(params.stiffness, 1.0F)), 0.25F * std::sqrt(h / gravity));
    float frameStep = std::min(deltaSeconds, 1.0F / 30.0F);
    auto substeps =
        scene.simulating ? std::clamp(static_cast<uint32_t>(std::ceil(frameStep / stableStep)), 1U, 8U) : 0U;
    float dt = substeps > 0 ? frameStep / static_cast<float>(substeps) : 0.0F;

    VkDescriptorSet bindlessSet = bindless.set();
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &bindlessSet, 0, nullptr);

    FluidPushConstants push;
    push.params = state.params[frameSlot].address;
    push.cellCounts = state.cellCounts.address;
    push.cellParticles = state.cellParticles.address;
    push.previousRendered = state.previousRendered.address;
    push.instances = instances;
    push.tlasInstances = tlasInstances;
    push.particleCount = state.count;
    push.instanceBase = instanceBase;
    push.tlasBase = tlasBase;
    push.sphereMesh = sphereMesh;
    push.dt = dt;
    push.blasLow = static_cast<uint32_t>(sphereBlas & 0xFFFFFFFFULL);
    push.blasHigh = static_cast<uint32_t>(sphereBlas >> 32U);
    auto setBuffers = [&](uint32_t in, uint32_t out) {
        push.positionsIn = state.positions[in].address;
        push.positionsOut = state.positions[out].address;
        push.velocitiesIn = state.velocities[in].address;
        push.velocitiesOut = state.velocities[out].address;
    };
    auto dispatch = [&](VkPipeline pipeline) {
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(commandBuffer, groupsFor(state.count), 1, 1);
    };
    auto computeToCompute = [&]() {
        memoryBarrier(commandBuffer,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    };

    // 지난 프레임의 인스턴스 쓰기와 그것을 읽던 단계가 끝나야 위치를 덮어쓸 수 있다.
    memoryBarrier(commandBuffer,
                  READER_STAGES,
                  VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    bool emitted = false;
    if (state.needsEmit) {
        setBuffers(state.current ^ 1U, state.current);
        dispatch(emitPipeline);
        computeToCompute();
        state.needsEmit = false;
        emitted = true;
    }

    for (uint32_t step = 0; step < substeps; ++step) {
        uint32_t in = state.current;
        uint32_t out = state.current ^ 1U;
        setBuffers(in, out);
        vkCmdFillBuffer(commandBuffer, state.cellCounts.handle, 0, VK_WHOLE_SIZE, 0);
        memoryBarrier(commandBuffer,
                      VK_PIPELINE_STAGE_2_CLEAR_BIT,
                      VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        dispatch(gridPipeline);
        computeToCompute();
        dispatch(densityPipeline);
        computeToCompute();
        dispatch(integratePipeline);
        computeToCompute();
        state.current = out;
    }

    // 프레임 끝: 입자마다 인스턴스를 쓴다. 다시 뿌린 프레임은 지난 위치가 없다.
    setBuffers(state.current, state.current ^ 1U);
    push.flags = (resetHistory || emitted ? FLUID_FLAG_RESET_HISTORY : 0U) |
                 (tlasInstances != 0 && sphereBlas != 0 ? FLUID_FLAG_WRITE_TLAS : 0U);
    dispatch(instancesPipeline);

    // 인스턴스는 그림자·장면·컬링·광선 경로가, TLAS 인스턴스는 가속 구조 구축이 읽는다.
    memoryBarrier(commandBuffer,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  READER_STAGES,
                  VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR);
}

} // namespace gfx
