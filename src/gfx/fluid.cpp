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
#include "physics/marching_cubes.h"

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
    for (VkPipeline pipeline : {emitPipeline,
                                gridPipeline,
                                densityPipeline,
                                integratePipeline,
                                instancesPipeline,
                                fieldPipeline,
                                marchingPipeline}) {
        vkDestroyPipeline(context.device, pipeline, nullptr);
    }
    vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
    vkDestroyPipelineLayout(context.device, surfaceLayout, nullptr);
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

    // 표면 패스는 푸시 상수가 달라 배치를 따로 만든다. 한 블록에 다 넣으면 128 바이트를 넘는다.
    VkPushConstantRange surfaceRange{};
    surfaceRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    surfaceRange.size = sizeof(FluidSurfacePushConstants);
    VkPipelineLayoutCreateInfo surfaceInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    surfaceInfo.setLayoutCount = 1;
    surfaceInfo.pSetLayouts = &bindlessLayout;
    surfaceInfo.pushConstantRangeCount = 1;
    surfaceInfo.pPushConstantRanges = &surfaceRange;
    VK_CHECK(vkCreatePipelineLayout(context.device, &surfaceInfo, nullptr, &surfaceLayout));
    fieldPipeline = createComputePipeline(context, surfaceLayout, "fluid_field.comp.spv");
    marchingPipeline = createComputePipeline(context, surfaceLayout, "fluid_marching.comp.spv");
    surfaceReady = fieldPipeline != VK_NULL_HANDLE && marchingPipeline != VK_NULL_HANDLE;
    if (!surfaceReady) {
        spdlog::warn("물 표면 컴퓨트를 만들지 못했습니다. GPU 백엔드는 입자로만 그린다");
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
    context.retireBuffer(state.surfaceField);
    for (Buffer& buffer : state.surfaceVertices) {
        context.retireBuffer(buffer);
    }
    for (Buffer& buffer : state.surfaceDrawArgs) {
        context.retireBuffer(buffer);
    }
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

void FluidSimulator::ensureSurface(State& state, const scene::Fluid& settings) {
    bool wanted = settings.display == scene::FluidDisplay::SURFACE && (state.cpu || surfaceReady);
    // CPU 백엔드는 표본마다 입자 전부를 훑으므로 해상도를 낮게 묶는다. 128³ 을 그대로 두면 명령 기록
    // 도중 렌더 스레드가 수십 초를 먹는다.
    uint32_t ceiling = state.cpu ? FLUID_MAX_CPU_SURFACE_RESOLUTION : FLUID_MAX_SURFACE_RESOLUTION;
    uint32_t resolution = std::clamp(settings.surfaceResolution, 8U, ceiling);
    if (!wanted) {
        // 입자로 돌아갔으면 3 MB 짜리 버퍼를 붙들고 있을 이유가 없다.
        if (state.surfaceResolution != 0) {
            context.retireBuffer(state.surfaceField);
            for (Buffer& buffer : state.surfaceVertices) {
                context.retireBuffer(buffer);
            }
            for (Buffer& buffer : state.surfaceDrawArgs) {
                context.retireBuffer(buffer);
            }
            state.surfaceResolution = 0;
            state.surfaceCapacity = 0;
        }
        state.surfaceReady = false;
        return;
    }
    state.surfaceReady = true;
    if (state.surfaceResolution == resolution && state.surfaceCapacity != 0) {
        return;
    }
    context.retireBuffer(state.surfaceField);
    for (Buffer& buffer : state.surfaceVertices) {
        context.retireBuffer(buffer);
    }
    for (Buffer& buffer : state.surfaceDrawArgs) {
        context.retireBuffer(buffer);
    }
    state.surfaceResolution = resolution;
    state.surfaceCapacity = FLUID_MAX_SURFACE_VERTICES;

    uint32_t samples = resolution + 1;
    VkDeviceSize fieldBytes = static_cast<VkDeviceSize>(samples) * samples * samples * sizeof(float);
    VkDeviceSize vertexBytes = static_cast<VkDeviceSize>(state.surfaceCapacity) * sizeof(physics::SurfaceVertex);
    // CPU 백엔드는 정점을 호스트에서 직접 쓴다. 장은 CPU 쪽에서 std::vector 로 들고 있으므로 GPU
    // 버퍼를 만들지 않는다.
    if (!state.cpu) {
        state.surfaceField =
            createBuffer(context, fieldBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, MemoryLocation::DEVICE, "물 표면 장");
    }
    // 마칭의 원자 카운터가 VkDrawIndirectCommand 의 vertexCount 자리를 직접 쓴다. GPU 백엔드는 원자
    // 연산을 거는 자리라 장치 메모리여야 한다(호스트에서 보이는 메모리에 걸면 매우 느리다).
    // CPU 백엔드는 호스트가 직접 채운다.
    for (uint32_t slot = 0; slot < FLUID_FRAMES; ++slot) {
        state.surfaceVertices[slot] = createBuffer(context,
                                                   vertexBytes,
                                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                                   state.cpu ? MemoryLocation::HOST_WRITE : MemoryLocation::DEVICE,
                                                   "물 표면 정점");
        state.surfaceDrawArgs[slot] = createBuffer(
            context,
            sizeof(VkDrawIndirectCommand),
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            state.cpu ? MemoryLocation::HOST_WRITE : MemoryLocation::DEVICE,
            "물 표면 그리기 인자");
        if (state.cpu) {
            // GPU 경로는 recordSurface 가 프레임마다 채운다.
            *static_cast<VkDrawIndirectCommand*>(state.surfaceDrawArgs[slot].mapped) =
                VkDrawIndirectCommand{0, 1, 0, 0};
        }
    }
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
            // 오브젝트가 사라졌어도 표면 버퍼는 놓아준다. 3 MB 짜리 두 벌이라 붙들 이유가 없다.
            ensureSurface(state, scene::Fluid{});
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
        ensureSurface(state, settings);
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

bool FluidSimulator::surfaceActive(uint32_t index) const {
    if (index >= states.size()) {
        return false;
    }
    const State& state = states[index];
    return state.surfaceReady && state.count > 0 && state.surfaceVertices[0].handle != VK_NULL_HANDLE;
}

VkDeviceAddress FluidSimulator::surfaceVertexAddress(uint32_t frameSlot, uint32_t index) const {
    return index < states.size() ? states[index].surfaceVertices[frameSlot % FLUID_FRAMES].address : 0;
}

VkBuffer FluidSimulator::surfaceDrawBuffer(uint32_t frameSlot, uint32_t index) const {
    return index < states.size() ? states[index].surfaceDrawArgs[frameSlot % FLUID_FRAMES].handle : VK_NULL_HANDLE;
}

void FluidSimulator::recordSurface(VkCommandBuffer commandBuffer,
                                   uint32_t frameSlot,
                                   uint32_t index,
                                   const scene::Scene& scene) {
    State& state = states[index];
    if (!surfaceActive(index) || state.cpu) {
        return;
    }
    // 설정 버퍼는 record 가 이번 프레임 슬롯에 채워 두었다. 다른 슬롯을 읽으면 지난 프레임 값이라
    // 격자 셀 수가 0 일 수 있고, 그러면 해시가 버퍼 밖을 짚어 장치를 잃는다.
    frameSlot %= FLUID_FRAMES;
    // 마지막 서브스텝이 세운 격자는 그 스텝의 «입력» 위치로 만든 것이라 지금 위치와 한 스텝 어긋난다.
    // 장을 만들기 전에 지금 위치로 다시 세운다. 디스패치 하나 값이다.
    VkDescriptorSet bindlessSet = bindless.set();
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &bindlessSet, 0, nullptr);
    FluidPushConstants gridPush;
    gridPush.params = state.params[frameSlot].address;
    gridPush.positionsIn = state.positions[state.current].address;
    gridPush.positionsOut = state.positions[state.current].address;
    gridPush.velocitiesIn = state.velocities[state.current].address;
    gridPush.velocitiesOut = state.velocities[state.current].address;
    gridPush.cellCounts = state.cellCounts.address;
    gridPush.cellParticles = state.cellParticles.address;
    gridPush.previousRendered = state.previousRendered.address;
    gridPush.particleCount = state.count;
    // 지난 프레임이 이 슬롯의 인자를 읽고 있었을 수 있다. 채우기 전에 그것이 끝났음을 못 박는다.
    memoryBarrier(commandBuffer,
                  VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                  VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                  VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    vkCmdFillBuffer(commandBuffer, state.cellCounts.handle, 0, VK_WHOLE_SIZE, 0);
    // 그리기 인자를 처음부터 채운다. 첫 칸(정점 수)은 마칭이 원자 덧셈으로 늘리고, 나머지는 고정이다.
    VkBuffer drawArgs = state.surfaceDrawArgs[frameSlot].handle;
    vkCmdFillBuffer(commandBuffer, drawArgs, 0, sizeof(uint32_t), 0);
    vkCmdFillBuffer(commandBuffer, drawArgs, sizeof(uint32_t), sizeof(uint32_t), 1);
    vkCmdFillBuffer(commandBuffer, drawArgs, 2 * sizeof(uint32_t), 2 * sizeof(uint32_t), 0);
    memoryBarrier(commandBuffer,
                  VK_PIPELINE_STAGE_2_CLEAR_BIT,
                  VK_ACCESS_2_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, gridPipeline);
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(gridPush), &gridPush);
    vkCmdDispatch(commandBuffer, groupsFor(state.count), 1, 1);
    memoryBarrier(commandBuffer,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    const scene::Fluid& settings = scene.fluids[index];
    FluidSurfacePushConstants push;
    push.params = state.params[frameSlot].address;
    push.positionsIn = state.positions[state.current].address;
    push.cellCounts = state.cellCounts.address;
    push.cellParticles = state.cellParticles.address;
    push.surfaceField = state.surfaceField.address;
    push.surfaceVertices = state.surfaceVertices[frameSlot].address;
    push.surfaceCounter = state.surfaceDrawArgs[frameSlot].address;
    push.surfaceResolution = state.surfaceResolution;
    push.surfaceCapacity = state.surfaceCapacity;
    push.surfaceIso = settings.surfaceIso;

    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, surfaceLayout, 0, 1, &bindlessSet, 0, nullptr);
    vkCmdPushConstants(commandBuffer, surfaceLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);

    uint32_t samples = state.surfaceResolution + 1;
    uint32_t fieldGroups = (samples + FLUID_SURFACE_GROUP_SIZE - 1) / FLUID_SURFACE_GROUP_SIZE;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, fieldPipeline);
    vkCmdDispatch(commandBuffer, fieldGroups, fieldGroups, fieldGroups);
    memoryBarrier(commandBuffer,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    uint32_t cellGroups = (state.surfaceResolution + FLUID_SURFACE_GROUP_SIZE - 1) / FLUID_SURFACE_GROUP_SIZE;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, marchingPipeline);
    vkCmdDispatch(commandBuffer, cellGroups, cellGroups, cellGroups);
    // 정점은 정점 셰이더가 주소로 읽고, 개수는 간접 그리기가 읽는다.
    memoryBarrier(commandBuffer,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT);
}

void FluidSimulator::buildCpuSurface(uint32_t frameSlot, uint32_t index, const scene::Scene& scene) {
    State& state = states[index];
    if (!surfaceActive(index) || !state.cpu) {
        return;
    }
    const scene::Fluid& settings = scene.fluids[index];
    physics::FluidParams shared = deriveParams(state, scene);
    // GPU 경로와 «같은 함수» 로 장을 만들고 «같은 표» 로 자른다. 두 벌로 두면 백엔드마다 물이 달라진다.
    physics::buildFluidField(state.solver.particles(), shared, state.surfaceResolution, state.cpuField, &jobs);
    glm::vec3 cell = (shared.containerMax - shared.containerMin) / static_cast<float>(state.surfaceResolution);
    frameSlot %= FLUID_FRAMES;
    auto* vertices = static_cast<physics::SurfaceVertex*>(state.surfaceVertices[frameSlot].mapped);
    if (vertices == nullptr) {
        return;
    }
    uint32_t written = physics::marchFluidField(state.cpuField,
                                                state.surfaceResolution,
                                                shared.containerMin,
                                                cell,
                                                settings.surfaceIso,
                                                vertices,
                                                state.surfaceCapacity,
                                                &jobs);
    *static_cast<VkDrawIndirectCommand*>(state.surfaceDrawArgs[frameSlot].mapped) =
        VkDrawIndirectCommand{written, 1, 0, 0};
}

} // namespace gfx
