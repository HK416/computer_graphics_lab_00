#include "gfx/rigid_body_gpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <initializer_list>

#include <glm/geometric.hpp>
#include <spdlog/spdlog.h>

#include "gfx/bindless.h"
#include "gfx/context.h"
#include "gfx/renderer.h"
#include "gfx/vk_check.h"

namespace gfx {

static_assert(RIGID_READBACK_SLOTS > FRAMES_IN_FLIGHT, "되읽기 슬롯이 모자라면 다 읽기 전에 이번 프레임이 덮어쓴다");

namespace {

// 편집기가 손댔는지 볼 때 쓰는 허용 오차. 되쓰기는 세계 → 지역 → 세계 로 한 바퀴 돌아 정확히 같은
// 값이 나오지 않는다. 콜라이더 크기도 세계 행렬을 분해해 얻으므로 마지막 자리가 흔들린다.
// 사람이 인스펙터에서 바꾸는 폭은 이보다 훨씬 크다.
constexpr float MOVE_EPSILON = 1.0e-3F;

bool nearlyEqual(float a, float b) {
    return std::abs(a - b) <= MOVE_EPSILON * std::max({1.0F, std::abs(a), std::abs(b)});
}

bool nearlyEqual(const glm::vec3& a, const glm::vec3& b) {
    return nearlyEqual(a.x, b.x) && nearlyEqual(a.y, b.y) && nearlyEqual(a.z, b.z);
}

bool nearlyEqual(const glm::vec4& a, const glm::vec4& b) {
    return nearlyEqual(glm::vec3{a}, glm::vec3{b}) && nearlyEqual(a.w, b.w);
}

// 만들지 못하면 VK_NULL_HANDLE 을 돌려준다. 강체는 CPU 백엔드가 있어 중단하지 않는다.
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
        spdlog::warn("강체 컴퓨트 파이프라인을 만들지 못했습니다: {}", shaderName);
        pipeline = VK_NULL_HANDLE;
    }
    vkDestroyShaderModule(context.device, module, nullptr);
    return pipeline;
}

void computeBarrier(VkCommandBuffer commandBuffer) {
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

} // namespace

RigidBodySimulator::RigidBodySimulator(Context& context, BindlessTextures& bindless)
    : context(context), bindless(bindless) {
    readbackFrame.fill(UINT64_MAX);
    createPipelines();
}

RigidBodySimulator::~RigidBodySimulator() {
    for (Buffer& buffer : bodyBuffers) {
        destroyBuffer(context, buffer);
    }
    for (Buffer& buffer : stagings) {
        destroyBuffer(context, buffer);
    }
    for (Buffer& buffer : readbacks) {
        destroyBuffer(context, buffer);
    }
    for (VkPipeline pipeline : {integratePipeline, solvePipeline, finishPipeline}) {
        vkDestroyPipeline(context.device, pipeline, nullptr);
    }
    vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
}

void RigidBodySimulator::createPipelines() {
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.size = sizeof(RigidPushConstants);
    VkDescriptorSetLayout bindlessLayout = bindless.layout();
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &bindlessLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &range;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &pipelineLayout));

    integratePipeline = createComputePipeline(context, pipelineLayout, "rigid_integrate.comp.spv");
    solvePipeline = createComputePipeline(context, pipelineLayout, "rigid_solve.comp.spv");
    finishPipeline = createComputePipeline(context, pipelineLayout, "rigid_finish.comp.spv");
    ready = integratePipeline != VK_NULL_HANDLE && solvePipeline != VK_NULL_HANDLE && finishPipeline != VK_NULL_HANDLE;
    if (!ready) {
        spdlog::warn("강체 GPU 솔버를 만들지 못했습니다. CPU 백엔드만 돕니다");
    }
}

void RigidBodySimulator::reserveBuffers(uint32_t count) {
    if (count <= capacity) {
        return;
    }
    uint32_t wanted = std::max(count, capacity * 2);
    VkDeviceSize bytes = static_cast<VkDeviceSize>(wanted) * sizeof(GpuRigidBody);
    for (Buffer& buffer : bodyBuffers) {
        context.retireBuffer(buffer);
        buffer = createBuffer(context,
                              bytes,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              MemoryLocation::DEVICE,
                              "강체 상태");
    }
    for (size_t slot = 0; slot < stagings.size(); ++slot) {
        context.retireBuffer(stagings[slot]);
        stagings[slot] =
            createBuffer(context, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, MemoryLocation::HOST_WRITE, "강체 업로드");
        context.retireBuffer(readbacks[slot]);
        readbacks[slot] =
            createBuffer(context, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, MemoryLocation::HOST_READ, "강체 되읽기");
        // 옛 버퍼를 버렸으니 담겨 있던 결과도 버린다.
        readbackFrame[slot] = UINT64_MAX;
        readbackCount[slot] = 0;
    }
    capacity = wanted;
}

bool RigidBodySimulator::applyReadback(scene::Scene& scene, uint64_t completedFrames) {
    // 되읽기는 몇 프레임 늦다. 그 사이에 장면이 바뀌거나 부품이 붙었다 떨어졌으면 이 결과는 남의
    // 것이라 오브젝트 번호부터 맞지 않는다.
    if (&scene != readbackScene || scene.componentRevision() != readbackComponents) {
        return false;
    }
    // 끝난 슬롯 가운데 가장 최근 것을 고른다. 이미 적용한 것보다 오래된 슬롯은 «지나간 상태»라
    // 그대로 버린다. 그러지 않으면 스텝이 0 인 프레임에 덮이지 않고 남은 낡은 슬롯이 나중에 뽑혀
    // 화면이 뒤로 되감긴다.
    size_t best = readbacks.size();
    uint64_t bestFrame = 0;
    for (size_t slot = 0; slot < readbacks.size(); ++slot) {
        if (readbackFrame[slot] == UINT64_MAX || readbackFrame[slot] >= completedFrames) {
            continue;
        }
        if (readbackFrame[slot] <= appliedFrame) {
            readbackFrame[slot] = UINT64_MAX;
            continue;
        }
        if (best == readbacks.size() || readbackFrame[slot] > bestFrame) {
            best = slot;
            bestFrame = readbackFrame[slot];
        }
    }
    if (best == readbacks.size() || bodies.empty() || readbackCount[best] != bodies.size() ||
        resident.size() != bodies.size() || readbacks[best].mapped == nullptr) {
        return false;
    }
    // 고른 것보다 오래된 나머지도 함께 버린다.
    for (size_t slot = 0; slot < readbacks.size(); ++slot) {
        if (readbackFrame[slot] != UINT64_MAX && readbackFrame[slot] < bestFrame) {
            readbackFrame[slot] = UINT64_MAX;
        }
    }
    // 되읽기 버퍼가 캐시된 메모리일 수 있다. 세마포어만으로는 장치가 쓴 것이 호스트에 보이지 않는다.
    vmaInvalidateAllocation(context.allocator, readbacks[best].allocation, 0, VK_WHOLE_SIZE);
    const auto* source = static_cast<const GpuRigidBody*>(readbacks[best].mapped);

    for (size_t i = 0; i < bodies.size(); ++i) {
        physics::RigidBodyState& body = bodies[i];
        const GpuRigidBody& result = source[i];
        body.position = glm::vec3{result.position};
        body.rotation =
            glm::normalize(glm::quat{result.rotation.w, result.rotation.x, result.rotation.y, result.rotation.z});
        body.velocity = glm::vec3{result.velocity};
        body.angularVelocity = glm::vec3{result.angularVelocity};
        // 적분이 바꾸는 값만 옮긴다. 나머지는 우리가 올린 그대로 남는다.
        resident[i].position = result.position;
        resident[i].rotation = result.rotation;
        resident[i].velocity = result.velocity;
        resident[i].angularVelocity = result.angularVelocity;
    }
    physics::writeBackRigidBodies(scene, bodies);
    // 한 번 읽은 슬롯은 다시 읽지 않는다.
    readbackFrame[best] = UINT64_MAX;
    appliedFrame = bestFrame;
    return true;
}

void RigidBodySimulator::invalidate() {
    resident.clear();
    readbackFrame.fill(UINT64_MAX);
    readbackScene = nullptr;
    appliedFrame = 0;
}

void RigidBodySimulator::buildUpload() {
    upload.resize(bodies.size());
    for (size_t i = 0; i < bodies.size(); ++i) {
        const physics::RigidBodyState& body = bodies[i];
        GpuRigidBody& target = upload[i];
        target.position = glm::vec4{body.position, body.inverseMass};
        target.rotation = glm::vec4{body.rotation.x, body.rotation.y, body.rotation.z, body.rotation.w};
        target.velocity = glm::vec4{body.velocity, body.radius};
        target.angularVelocity = glm::vec4{body.angularVelocity, body.boundingRadius};
        target.preVelocity = target.velocity;
        target.preAngularVelocity = target.angularVelocity;
        target.inverseInertia = glm::vec4{body.inverseInertia, body.restitution};
        target.halfExtents = glm::vec4{body.halfExtents, body.friction};
        target.shape = static_cast<uint32_t>(body.shape);
        target.flags = body.useGravity ? 1U : 0U;
    }
}

// 값 하나라도 어긋나면 참이다. **모든 비교에 오차를 둔다.** 위치·회전만이 아니라 콜라이더 반지름과
// 관성 역수도 세계 행렬을 분해해 얻은 값이라, 되쓰기 → 다시 모으기를 한 바퀴 돌면 마지막 자리가
// 흔들린다. 그것을 «편집기가 손댔다»로 읽으면 매 프레임 GPU 상태를 버리고 몇 프레임 전 장면 값으로
// 다시 시작해 화면에서 공이 앞뒤로 덜덜거린다.
bool RigidBodySimulator::sceneEdited() const {
    if (resident.size() != upload.size()) {
        return true;
    }
    for (size_t i = 0; i < upload.size(); ++i) {
        const GpuRigidBody& want = upload[i];
        const GpuRigidBody& have = resident[i];
        // 적분이 바꾸는 값. w 에는 질량 역수·콜라이더 반지름·경계 반지름이 실려 있다.
        if (!nearlyEqual(want.position, have.position) || !nearlyEqual(want.velocity, have.velocity) ||
            !nearlyEqual(want.angularVelocity, have.angularVelocity)) {
            return true;
        }
        if (std::abs(glm::dot(want.rotation, have.rotation)) < 1.0F - MOVE_EPSILON) {
            return true;
        }
        // 나머지는 인스펙터가 정하는 값이다. 모양·질량·반발을 고친 것이 GPU 에 가려면 여기서 걸려야 한다.
        if (!nearlyEqual(want.inverseInertia, have.inverseInertia) ||
            !nearlyEqual(want.halfExtents, have.halfExtents) || want.shape != have.shape || want.flags != have.flags) {
            return true;
        }
    }
    return false;
}

void RigidBodySimulator::prepare(const scene::Scene& scene, uint32_t steps, float seconds) {
    stepCount = 0;
    stepSeconds = seconds;
    uploadPending = false;
    if (!ready) {
        bodies.clear();
        return;
    }

    // 장면이 바뀌었거나 부품이 붙었다 떨어졌으면 첨자가 밀렸을 수 있다. GPU 에 남은 것은 남의 상태다.
    if (&scene != readbackScene || scene.componentRevision() != readbackComponents) {
        invalidate();
    }
    physics::collectRigidBodies(scene, scene::SimulationBackend::GPU, bodies);
    readbackScene = &scene;
    readbackComponents = scene.componentRevision();
    if (bodies.empty()) {
        resident.clear();
        return;
    }
    reserveBuffers(static_cast<uint32_t>(bodies.size()));

    buildUpload();
    // 구성이 바뀌었거나 편집기가 손댔으면 GPU 상태를 버리고 장면 값으로 다시 시작한다.
    if (sceneEdited()) {
        resident = upload;
        uploadPending = true;
    }
    stepCount = steps;
}
void RigidBodySimulator::record(VkCommandBuffer commandBuffer, uint64_t frameIndex) {
    if (bodies.empty()) {
        return;
    }
    if (!uploadPending && stepCount == 0) {
        // 올릴 것도 풀 것도 없다. 지난 되읽기가 그대로 유효하다.
        return;
    }
    auto count = static_cast<uint32_t>(bodies.size());
    VkDeviceSize bytes = static_cast<VkDeviceSize>(count) * sizeof(GpuRigidBody);
    auto slot = static_cast<size_t>(frameIndex % RIGID_READBACK_SLOTS);
    uint32_t source = 0;

    // 지난 프레임의 마지막 명령은 bodyBuffers[0] → 되읽기 복사다. 아래 두 쓰기(업로드 복사, 적분
    // 디스패치)가 그것을 앞지르지 않게 막는다. 프레임을 넘는 WAR 이라 제출 순서만으로는 부족하다.
    VkMemoryBarrier2 entryBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    entryBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    entryBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    entryBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    entryBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo entryDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    entryDependency.memoryBarrierCount = 1;
    entryDependency.pMemoryBarriers = &entryBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &entryDependency);

    if (uploadPending) {
        std::memcpy(stagings[slot].mapped, upload.data(), static_cast<size_t>(bytes));
        vmaFlushAllocation(context.allocator, stagings[slot].allocation, 0, VK_WHOLE_SIZE);
        VkBufferCopy region{0, 0, bytes};
        vkCmdCopyBuffer(commandBuffer, stagings[slot].handle, bodyBuffers[0].handle, 1, &region);
        VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependency);
        uploadPending = false;
    }

    VkDescriptorSet bindlessSet = bindless.set();
    vkCmdBindDescriptorSets(
        commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &bindlessSet, 0, nullptr);

    // 상수는 CPU 솔버와 같은 것을 쓴다(physics/rigid_body.h). 한쪽만 고치면 거동이 갈린다.
    RigidPushConstants push{};
    push.bodyCount = count;
    push.dt = stepSeconds;
    push.gravity = physics::GRAVITY;
    push.positionCorrection = physics::POSITION_CORRECTION;
    push.penetrationSlop = physics::PENETRATION_SLOP;
    push.restitutionThreshold = physics::RESTITUTION_THRESHOLD;
    auto dispatch = [&](VkPipeline pipeline) {
        push.bodiesIn = bodyBuffers[source].address;
        push.bodiesOut = bodyBuffers[source ^ 1U].address;
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
        vkCmdDispatch(commandBuffer, (count + RIGID_GROUP_SIZE - 1) / RIGID_GROUP_SIZE, 1, 1);
        computeBarrier(commandBuffer);
        source ^= 1U;
    };

    for (uint32_t step = 0; step < stepCount; ++step) {
        dispatch(integratePipeline);
        for (uint32_t iteration = 0; iteration < RIGID_SOLVER_ITERATIONS; ++iteration) {
            dispatch(solvePipeline);
        }
        for (uint32_t iteration = 0; iteration < RIGID_POSITION_ITERATIONS; ++iteration) {
            dispatch(finishPipeline);
        }
    }
    // 다음 프레임이 이어서 풀 수 있도록 결과를 0번으로 모은다. 반복 수가 짝수라도 적분·마무리 때문에
    // 어느 쪽에 남을지 달라진다.
    if (source != 0) {
        VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
        dependency.memoryBarrierCount = 1;
        dependency.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(commandBuffer, &dependency);
        VkBufferCopy region{0, 0, bytes};
        vkCmdCopyBuffer(commandBuffer, bodyBuffers[1].handle, bodyBuffers[0].handle, 1, &region);
        source = 0;
    }

    // 결과를 되읽기 버퍼로 옮긴다. 이 프레임이 끝나야 CPU 가 읽을 수 있다.
    VkMemoryBarrier2 barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);

    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(commandBuffer, bodyBuffers[0].handle, readbacks[slot].handle, 1, &region);

    // 호스트가 매핑으로 읽으므로 가시성 배리어가 필요하다. 세마포어 신호만으로는 보장되지 않는다.
    VkMemoryBarrier2 hostBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    hostBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    hostBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    hostBarrier.dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT;
    hostBarrier.dstAccessMask = VK_ACCESS_2_HOST_READ_BIT;
    VkDependencyInfo hostDependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    hostDependency.memoryBarrierCount = 1;
    hostDependency.pMemoryBarriers = &hostBarrier;
    vkCmdPipelineBarrier2(commandBuffer, &hostDependency);

    readbackFrame[slot] = frameIndex;
    readbackCount[slot] = count;
}

} // namespace gfx
