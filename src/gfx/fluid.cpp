#include "gfx/fluid.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <spdlog/spdlog.h>

#include "core/job_system.h"
#include "gfx/bindless.h"
#include "gfx/context.h"
#include "gfx/geometry.h"
#include "gfx/vk_check.h"

namespace gfx {
namespace {

// 만들지 못하면 VK_NULL_HANDLE 을 돌려준다. 유체는 CPU 백엔드로 내려갈 수 있어 중단하지 않는다.
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
        spdlog::warn("유체 컴퓨트 파이프라인을 만들지 못했습니다: {}", shaderName);
        pipeline = VK_NULL_HANDLE;
    }
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

// 해시 격자의 셀 수. 입자의 두 배쯤을 2 의 거듭제곱으로 잡는다. 해시가 마스크로 접히므로 거듭제곱이
// 아니면 안 된다. 두 백엔드가 같은 값을 써야 격자 규칙이 갈리지 않는다.
uint32_t cellCountFor(uint32_t particles) {
    return std::clamp<uint32_t>(std::bit_ceil(std::max(particles, 1U) * 2), 1024, 65536);
}

} // namespace

FluidSimulator::FluidSimulator(Context& context, BindlessTextures& bindless, core::JobSystem& jobs)
    : context(context), bindless(bindless), jobs(jobs) {
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
    // 하나라도 없으면 GPU 백엔드를 쓸 수 없다. 그때는 모든 유체가 CPU 로 내려간다.
    gpuReady = emitPipeline != VK_NULL_HANDLE && gridPipeline != VK_NULL_HANDLE && densityPipeline != VK_NULL_HANDLE &&
               integratePipeline != VK_NULL_HANDLE && instancesPipeline != VK_NULL_HANDLE;
    if (!gpuReady) {
        spdlog::warn("유체 GPU 백엔드를 만들지 못해 CPU 로 돈다");
    }
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
    state.cellCount = cellCountFor(count);
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
        // AUTO 는 GPU 를 쓴다. 컴퓨트 파이프라인을 만들지 못한 장치에서는 그것도 CPU 로 내려간다.
        // GPU 를 명시해도 파이프라인이 없으면 CPU 로 내려간다. 편집기는 그때 GPU 항목 자체를 잠근다.
        bool wantsCpu = settings.backend == scene::SimulationBackend::CPU || !gpuReady;
        if (wantsCpu != state.cpu) {
            // 백엔드를 바꾸면 상태를 새로 만든다. 옛 입자는 다른 쪽 저장소에 있다.
            destroyState(state);
            state.objectIndex = owners[i];
            state.count = std::min(settings.particleCount, particleLimit);
            state.cpu = wantsCpu;
        }
        if (state.cpu) {
            // GPU 버퍼는 만들지 않지만 격자 규칙은 같아야 한다. 셀 수를 «용량» 에서 내는 것까지
            // 같아야 입자 수를 줄였을 때 두 백엔드의 해시 충돌이 갈리지 않는다.
            if (state.count > state.capacity || state.capacity == 0) {
                state.capacity = state.count;
                state.cellCount = cellCountFor(state.count);
            }
        } else {
            ensureCapacity(state, state.count);
        }
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

bool FluidSimulator::onCpu(uint32_t index) const {
    return index < states.size() && states[index].cpu;
}

void FluidSimulator::writeCpuInstances(uint32_t index,
                                       const scene::Scene& scene,
                                       float deltaSeconds,
                                       void* instances,
                                       uint32_t instanceBase,
                                       void* tlasInstances,
                                       uint32_t tlasBase,
                                       VkDeviceAddress sphereBlas,
                                       uint32_t sphereMesh,
                                       bool resetHistory) {
    State& state = states[index];
    if (!state.cpu || state.count == 0 || instances == nullptr) {
        return;
    }

    physics::FluidParams params = deriveParams(state, scene);
    bool emitted = false;
    if (state.needsEmit || state.solver.particleCount() != state.count) {
        state.solver.emit(params, state.count);
        state.needsEmit = false;
        emitted = true;
    }
    if (scene.simulating) {
        state.solver.step(params, deltaSeconds, &jobs);
    }

    // 인스턴스 배치는 shaders/fluid_instances.comp 와 같아야 한다. 래스터와 광선 경로가 오브젝트
    // 인스턴스와 같은 배열을 읽으므로 규칙이 어긋나면 입자만 엉뚱하게 그려진다.
    const std::vector<glm::vec4>& current = state.solver.particles();
    const std::vector<glm::vec4>& previous = state.solver.previousParticles();
    bool history = !(resetHistory || emitted);
    float radius = params.particleRadius;
    auto* target = static_cast<GpuInstance*>(instances);
    auto* tlas = static_cast<VkAccelerationStructureInstanceKHR*>(tlasInstances);
    bool writeTlas = tlas != nullptr && sphereBlas != 0;

    jobs.parallelFor(state.count, 256, [&](uint32_t begin, uint32_t end) {
        for (uint32_t i = begin; i < end; ++i) {
            glm::vec3 position{current[i]};
            glm::vec3 before = history ? glm::vec3{previous[i]} : position;
            glm::mat4 model = glm::scale(glm::translate(glm::mat4{1.0F}, position), glm::vec3{radius});

            uint32_t slot = instanceBase + i;
            GpuInstance& instance = target[slot];
            instance.model = model;
            instance.previousModel = glm::scale(glm::translate(glm::mat4{1.0F}, before), glm::vec3{radius});
            // 균등 배율이라 노멀 행렬은 단위면 된다.
            instance.normalMatrix = glm::mat4{1.0F};
            instance.meshIndex = sphereMesh;
            instance.bucket = 0;
            instance.bucketBase = 0;
            instance.jointOffset = NO_JOINTS;
            instance.skinnedVertexOffset = NO_SKINNED_VERTICES;
            instance.previousSkinnedVertexOffset = NO_SKINNED_VERTICES;
            instance.skinnedMeshletOffset = 0;
            instance.visibilityBase = 0;

            if (writeTlas) {
                // 매핑 버퍼는 쓰기 결합 메모리다. 비트필드에 바로 대입하면 담는 워드를 읽어 들여
                // 고치는데, 그 읽기는 캐시가 없어 입자마다 네 번씩 비싸게 든다. 지역에서 채운 뒤
                // 한 번에 옮긴다(updateTopLevel 도 같은 이유로 그렇게 한다).
                VkAccelerationStructureInstanceKHR entry{};
                // 가속 구조는 행 우선 3x4 다. glm 은 열 우선이라 옮겨 담는다.
                for (uint32_t row = 0; row < 3; ++row) {
                    for (uint32_t column = 0; column < 4; ++column) {
                        entry.transform.matrix[row][column] = model[column][row];
                    }
                }
                entry.instanceCustomIndex = slot & 0xFFFFFFU;
                entry.mask = 0xFF;
                entry.instanceShaderBindingTableRecordOffset = 0;
                entry.flags = 0;
                entry.accelerationStructureReference = sphereBlas;
                tlas[tlasBase + i] = entry;
            }
        }
    });

    state.solver.keepRendered();
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

physics::FluidParams FluidSimulator::deriveParams(const State& state, const scene::Scene& scene) const {
    return physics::deriveFluidParams(state.lastSettings, state.lastWorld, state.count, state.cellCount, scene);
}

void FluidSimulator::fillParams(GpuFluidParams& params, const State& state, const scene::Scene& scene) const {
    // 상수는 CPU 백엔드와 «같은 함수»로 낸다. 두 벌로 두면 백엔드를 바꿀 때마다 물이 달리 흐른다.
    physics::FluidParams shared = deriveParams(state, scene);
    params.emitterWorld = shared.emitterWorld;
    params.emitterHalfExtents = glm::vec4{shared.emitterHalfExtents, shared.spacing};
    params.containerMin = glm::vec4{shared.containerMin, shared.particleRadius};
    params.containerMax = glm::vec4{shared.containerMax, shared.wallRestitution};
    params.gravity = glm::vec4{shared.gravity, shared.particleMass};
    params.smoothingRadius = shared.smoothingRadius;
    params.restDensity = shared.restDensity;
    params.stiffness = shared.stiffness;
    params.viscosity = shared.viscosity;
    params.lattice = glm::uvec4{shared.lattice, shared.cellCount};

    params.colliderCount = shared.colliderCount;
    for (uint32_t i = 0; i < shared.colliderCount; ++i) {
        const physics::FluidCollider& source = shared.colliders[i];
        GpuFluidCollider& collider = params.colliders[i];
        collider.type = static_cast<uint32_t>(source.shape);
        switch (source.shape) {
        case scene::ColliderShape::SPHERE:
            collider.data0 = glm::vec4{source.center, source.radius};
            break;
        case scene::ColliderShape::BOX:
            collider.data0 = glm::vec4{source.halfExtents, 0.0F};
            collider.world = source.world;
            collider.inverseWorld = source.inverseWorld;
            break;
        case scene::ColliderShape::PLANE:
            collider.data0 = glm::vec4{source.normal, source.offset};
            break;
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
    // CPU 백엔드는 writeCpuInstances 가 이미 인스턴스를 써 두었다.
    if (state.count == 0 || state.cpu) {
        return;
    }
    frameSlot %= FLUID_FRAMES;

    GpuFluidParams params;
    fillParams(params, state, scene);
    std::memcpy(state.params[frameSlot].mapped, &params, sizeof(params));

    // 서브스텝 규칙은 CPU 백엔드와 «같은 함수» 를 쓴다. 두 벌로 두면 백엔드를 바꿀 때 물이 달리 흐른다.
    physics::FluidParams shared = deriveParams(state, scene);
    uint32_t substeps = scene.simulating ? physics::fluidSubsteps(shared, deltaSeconds) : 0U;
    float frameStep = std::min(deltaSeconds, physics::FLUID_MAX_FRAME_STEP);
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
