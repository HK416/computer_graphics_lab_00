#include "gfx/particles.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>

#include <glm/trigonometric.hpp>
#include <spdlog/spdlog.h>

#include "gfx/bindless.h"
#include "gfx/context.h"
#include "gfx/vk_check.h"

namespace gfx {
namespace {

// 만들지 못하면 VK_NULL_HANDLE 을 돌려준다. 입자는 선택 기능이라 중단하지 않는다.
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
        spdlog::warn("입자 컴퓨트 파이프라인을 만들지 못했습니다: {}", shaderName);
        pipeline = VK_NULL_HANDLE;
    }
    vkDestroyShaderModule(context.device, module, nullptr);
    return pipeline;
}

} // namespace

ParticleSimulator::ParticleSimulator(Context& context,
                                     BindlessTextures& bindless,
                                     VkDescriptorSetLayout accelerationLayout)
    : context(context), bindless(bindless) {
    createPipelines(accelerationLayout);
}

ParticleSimulator::~ParticleSimulator() {
    for (State& state : states) {
        destroyState(state);
    }
    vkDestroyPipeline(context.device, rayQueryPipeline, nullptr);
    vkDestroyPipeline(context.device, pipeline, nullptr);
    vkDestroyPipelineLayout(context.device, rayQueryLayout, nullptr);
    vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
}

void ParticleSimulator::createPipelines(VkDescriptorSetLayout accelerationLayout) {
    // 스프라이트 정점·프래그먼트도 같은 블록을 읽으므로 세 스테이지를 다 적는다. 렌더러의 그리기 배치도 같은
    // 범위로 만들어야 푸시 상수를 나눠 쓸 수 있다.
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    range.size = sizeof(ParticlePushConstants);
    std::array<VkDescriptorSetLayout, 2> sets{bindless.layout(), accelerationLayout};
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = sets.data();
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &range;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &pipelineLayout));
    pipeline = createComputePipeline(context, pipelineLayout, "particles.comp.spv");
    if (pipeline == VK_NULL_HANDLE) {
        spdlog::warn("입자 컴퓨트를 만들지 못해 입자를 그리지 않는다");
        return;
    }
    sortPipeline = createComputePipeline(context, pipelineLayout, "particle_sort.comp.spv");
    if (sortPipeline == VK_NULL_HANDLE) {
        spdlog::warn("입자 정렬 컴퓨트를 만들지 못해 입자를 번호 순으로 그린다");
    }
    if (accelerationLayout == VK_NULL_HANDLE) {
        return;
    }
    layoutInfo.setLayoutCount = 2;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &rayQueryLayout));
    rayQueryPipeline = createComputePipeline(context, rayQueryLayout, "particles_rq.comp.spv");
    if (rayQueryPipeline == VK_NULL_HANDLE) {
        spdlog::warn("입자 광선 질의 변종을 만들지 못해 입자가 장면과 부딪히지 않는다");
    }
}

void ParticleSimulator::destroyState(State& state) {
    // 진행 중인 프레임의 명령 버퍼에 주소가 실려 있을 수 있다. 그 프레임이 끝난 뒤에 지운다.
    context.retireBuffer(state.particles);
    context.retireBuffer(state.sorted);
    for (Buffer& buffer : state.params) {
        context.retireBuffer(buffer);
    }
    // 슬롯이 다른 시스템에게 넘어갈 수 있으므로 판단에 쓰는 값도 함께 지운다.
    state = State{};
}

void ParticleSimulator::ensureCapacity(State& state, uint32_t count) {
    if (count <= state.capacity && state.capacity > 0) {
        return;
    }
    uint32_t objectIndex = state.objectIndex;
    uint32_t particleCount = state.count;
    destroyState(state);
    state.objectIndex = objectIndex;
    state.count = particleCount;
    state.capacity = count;
    state.particles = createBuffer(context,
                                   static_cast<VkDeviceSize>(count) * sizeof(GpuParticle),
                                   VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                   MemoryLocation::DEVICE,
                                   "입자");
    for (Buffer& buffer : state.params) {
        buffer = createBuffer(context,
                              sizeof(GpuParticleParams),
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              MemoryLocation::HOST_WRITE,
                              "입자 설정");
    }
    state.sortCapacity = std::bit_ceil(count);
    state.sorted = createBuffer(context,
                                static_cast<VkDeviceSize>(state.sortCapacity) * sizeof(glm::uvec2),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                MemoryLocation::DEVICE,
                                "입자 정렬");
    state.needsReset = true;
}

bool ParticleSimulator::prepare(const scene::Scene& scene, bool sceneSwitched, float deltaSeconds) {
    if (pipeline == VK_NULL_HANDLE) {
        return false;
    }
    // 장면이 바뀌면 상태를 통째로 버린다. 부품 번호가 다른 장면 것과 겹치기 때문이다.
    if (sceneSwitched || scene.id != lastSceneId) {
        for (State& state : states) {
            destroyState(state);
        }
        states.clear();
        lastSceneId = scene.id;
    }
    for (size_t i = scene.particleSystems.size(); i < states.size(); ++i) {
        destroyState(states[i]);
    }
    states.resize(scene.particleSystems.size());

    // 부품 배열이 압축되면 첨자가 밀려 states[k] 가 다른 시스템을 맡게 된다. 그런 프레임에는 전부 버린다.
    if (scene.componentRevision() != lastComponentRevision) {
        lastComponentRevision = scene.componentRevision();
        for (State& state : states) {
            state.needsReset = true;
            state.lastSettings = scene::ParticleSystem{};
            state.lastWorld = glm::mat4{0.0F};
        }
    }

    std::vector<uint32_t> owners(scene.particleSystems.size(), UINT32_MAX);
    for (uint32_t index = 0; index < scene.objects.size(); ++index) {
        int32_t slot = scene.objects[index].particleSystem;
        if (slot >= 0 && static_cast<size_t>(slot) < owners.size()) {
            owners[static_cast<size_t>(slot)] = index;
        }
    }

    // 재생을 시작하거나 멈추면 처음으로 돌린다.
    bool playEdge = scene.simulating != wasSimulating;
    wasSimulating = scene.simulating;

    bool active = false;
    for (size_t i = 0; i < states.size(); ++i) {
        State& state = states[i];
        const scene::ParticleSystem& settings = scene.particleSystems[i];
        state.objectIndex = owners[i];
        state.count =
            state.objectIndex == UINT32_MAX ? 0 : std::clamp(settings.maxParticles, 1U, PARTICLE_MAX_PARTICLES);
        state.spawnCount = 0;
        state.dt = 0.0F;
        if (state.count == 0) {
            continue;
        }
        ensureCapacity(state, state.count);
        glm::mat4 world = scene.visibleCached(state.objectIndex) ? scene.world(state.objectIndex) : glm::mat4{0.0F};
        // 방출 자세는 프레임마다 바뀌어도 된다(움직이는 방출기). 설정·재생 에지·보이기 변화만 리셋이다.
        bool visibleChanged = (world == glm::mat4{0.0F}) != (state.lastWorld == glm::mat4{0.0F});
        if (!(settings == state.lastSettings) || playEdge || visibleChanged) {
            state.needsReset = true;
            state.lastSettings = settings;
        }
        state.lastWorld = world;
        if (state.needsReset) {
            state.cursor = 0;
            state.emitAccumulator = 0.0F;
        }
        // 방출은 CPU 가 링 버퍼 구간으로 정한다. 원자 연산 없이 결정적이고, 가장 오래된 입자를 덮는다.
        if (scene.simulating && world != glm::mat4{0.0F}) {
            state.dt = std::min(deltaSeconds, 0.1F);
            state.emitAccumulator += std::max(settings.emitRate, 0.0F) * state.dt;
            auto spawn = static_cast<uint32_t>(std::floor(state.emitAccumulator));
            spawn = std::min(spawn, state.count);
            state.emitAccumulator -= static_cast<float>(spawn);
            state.spawnFirst = state.cursor;
            state.spawnCount = spawn;
            state.cursor = (state.cursor + spawn) % state.count;
        }
        active = true;
    }
    return active;
}

uint32_t ParticleSimulator::count(uint32_t index) const {
    return index < states.size() ? states[index].count : 0;
}

bool ParticleSimulator::wantsCollision() const {
    for (const State& state : states) {
        if (state.count > 0 && state.lastSettings.collide) {
            return true;
        }
    }
    return false;
}

VkDeviceAddress ParticleSimulator::particleAddress(uint32_t index) const {
    return index < states.size() ? states[index].particles.address : 0;
}

VkDeviceAddress ParticleSimulator::paramsAddress(uint32_t frameSlot, uint32_t index) const {
    return index < states.size() ? states[index].params[frameSlot % PARTICLE_FRAMES].address : 0;
}

void ParticleSimulator::record(VkCommandBuffer commandBuffer,
                               uint32_t frameSlot,
                               uint32_t index,
                               const scene::Scene& scene,
                               uint64_t frameIndex,
                               bool rayQuery,
                               VkDescriptorSet accelerationSet,
                               const SceneBuffers& buffers,
                               VkDeviceAddress camera) {
    State& state = states[index];
    if (state.count == 0) {
        return;
    }
    frameSlot %= PARTICLE_FRAMES;
    const scene::ParticleSystem& settings = scene.particleSystems[index];
    rayQuery = rayQuery && settings.collide && rayQueryPipeline != VK_NULL_HANDLE;

    GpuParticleParams params;
    params.emitterWorld = state.lastWorld;
    params.color = settings.color;
    params.emissive = glm::vec4{settings.emissive, settings.restitution};
    params.sizeSpeedSpread = glm::vec4{settings.sizeStart,
                                       settings.sizeEnd,
                                       settings.initialSpeed,
                                       std::cos(glm::radians(std::clamp(settings.spreadAngleDegrees, 0.0F, 180.0F)))};
    params.gravityDrag = glm::vec4{0.0F, -9.81F * settings.gravityScale, 0.0F, std::max(settings.drag, 0.0F)};
    params.particleCount = state.count;
    params.spawnFirst = state.spawnFirst;
    params.spawnCount = state.spawnCount;
    params.frameIndex = static_cast<uint32_t>(frameIndex);
    params.dt = state.dt;
    params.lifetime = std::max(settings.lifetime, 1e-3F);
    params.flags = rayQuery ? PARTICLE_FLAG_COLLIDE : 0U;
    params.vertices = buffers.vertices;
    params.skinnedVertices = buffers.skinnedVertices;
    params.indices = buffers.indices;
    params.meshes = buffers.meshes;
    params.lods = buffers.lods;
    params.instances = buffers.instances;
    std::array<physics::ForceFieldSample, physics::MAX_FORCE_FIELDS> fields{};
    params.fieldCount = physics::collectForceFields(scene, fields);
    fillForceFields(fields, params.fieldCount, params.fields);
    std::memcpy(state.params[frameSlot].mapped, &params, sizeof(params));

    // 지난 프레임의 스프라이트 정점 읽기가 끝나야 입자를 덮어쓸 수 있다.
    memoryBarrier(commandBuffer,
                  READER_STAGES,
                  VK_ACCESS_2_SHADER_READ_BIT,
                  VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                      VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    if (state.needsReset) {
        // 0 으로 채우면 lifetime 0 이라 전부 죽은 상태다.
        vkCmdFillBuffer(commandBuffer, state.particles.handle, 0, VK_WHOLE_SIZE, 0);
        memoryBarrier(commandBuffer,
                      VK_PIPELINE_STAGE_2_CLEAR_BIT,
                      VK_ACCESS_2_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                      VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        state.needsReset = false;
    }

    VkPipelineLayout layout = rayQuery ? rayQueryLayout : pipelineLayout;
    std::array<VkDescriptorSet, 2> sets{bindless.set(), accelerationSet};
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0, rayQuery ? 2U : 1U, sets.data(), 0, nullptr);
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, rayQuery ? rayQueryPipeline : pipeline);
    ParticlePushConstants push;
    push.particles = state.particles.address;
    push.params = state.params[frameSlot].address;
    push.camera = camera;
    push.particleCount = state.count;
    push.sorted = state.sorted.address;
    auto pushAll = [&](VkPipelineLayout target) {
        vkCmdPushConstants(commandBuffer,
                           target,
                           VK_SHADER_STAGE_COMPUTE_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           0,
                           sizeof(push),
                           &push);
    };
    pushAll(layout);
    vkCmdDispatch(commandBuffer, (state.count + PARTICLE_GROUP_SIZE - 1) / PARTICLE_GROUP_SIZE, 1, 1);

    // 카메라 거리로 정렬한다. 바이토닉 망은 고정이라 같은 입력에 같은 순서가 나와 프레임이 결정적이다.
    // 단계는 공유 메모리 블록 안에서 먼저 풀고(sortK = 0), 블록을 넘는 단계만 전역으로 돈 뒤(sortJ ≥ 블록)
    // 나머지를 다시 블록 안에서 푼다(sortJ = 0). 65536 개면 디스패치 28 번이다.
    if (sortPipeline != VK_NULL_HANDLE) {
        auto computeToCompute = [&] {
            memoryBarrier(commandBuffer,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                          VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        };
        uint32_t n = state.sortCapacity;
        uint32_t blocks = std::max(n / PARTICLE_SORT_BLOCK, 1U);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, sortPipeline);
        vkCmdBindDescriptorSets(
            commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, sets.data(), 0, nullptr);
        computeToCompute();
        push.sortK = 0;
        push.sortJ = 0;
        pushAll(pipelineLayout);
        vkCmdDispatch(commandBuffer, blocks, 1, 1);
        for (uint32_t k = PARTICLE_SORT_BLOCK * 2; k <= n; k <<= 1U) {
            for (uint32_t j = k / 2; j >= PARTICLE_SORT_BLOCK; j >>= 1U) {
                computeToCompute();
                push.sortK = k;
                push.sortJ = j;
                pushAll(pipelineLayout);
                vkCmdDispatch(commandBuffer, n / PARTICLE_SORT_BLOCK, 1, 1);
            }
            computeToCompute();
            push.sortK = k;
            push.sortJ = 0;
            pushAll(pipelineLayout);
            vkCmdDispatch(commandBuffer, blocks, 1, 1);
        }
    }

    // 스프라이트 정점·프래그먼트가 입자와 정렬 목록을 읽는다.
    memoryBarrier(commandBuffer,
                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                  VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                  VK_ACCESS_2_SHADER_READ_BIT);
}

VkDeviceAddress ParticleSimulator::sortedAddress(uint32_t index) const {
    return states[index].sorted.address;
}

} // namespace gfx
