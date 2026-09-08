#include "gfx/neural.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <spdlog/spdlog.h>

#include "core/error.h"
#include "gfx/context.h"
#include "gfx/headless_compute.h"
#include "gfx/vk_check.h"

namespace gfx {

namespace {

// 만들지 못하면 VK_NULL_HANDLE 을 돌려준다. CPU 기준이 있으므로 중단하지 않는다.
VkPipeline createComputePipeline(Context& context,
                                 VkPipelineLayout layout,
                                 const char* shaderName,
                                 uint32_t groupSizeOverride = 0,
                                 uint32_t requiredSubgroupSize = 0) {
    VkShaderModule module = tryCreateShaderModule(context.device, shaderName);
    if (module == VK_NULL_HANDLE) {
        return VK_NULL_HANDLE;
    }
    // 작업 그룹 크기를 특수화 상수 0 으로 실어 보낸다. 셰이더에 같은 수를 적어 두지 않으므로 두 벌이
    // 어긋날 자리가 없다.
    VkSpecializationMapEntry entry{};
    entry.constantID = 0;
    entry.offset = 0;
    entry.size = sizeof(uint32_t);
    uint32_t groupSize = groupSizeOverride != 0 ? groupSizeOverride : NEURAL_GROUP_SIZE;
    VkSpecializationInfo specialization{};
    specialization.mapEntryCount = 1;
    specialization.pMapEntries = &entry;
    specialization.dataSize = sizeof(groupSize);
    specialization.pData = &groupSize;

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = module;
    stage.pName = "main";
    stage.pSpecializationInfo = &specialization;
    // 협력 행렬은 서브그룹 전체가 함께 도는 연산이라 작업 그룹이 정확히 한 서브그룹이어야 한다. 크기를
    // 못 박지 않으면 드라이버가 [min, max] 안에서 고를 수 있고, 그러면 작업 그룹이 서브그룹의 일부만
    // 채워 연산 자체가 규정 밖이 된다.
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO};
    if (requiredSubgroupSize != 0) {
        subgroupSize.requiredSubgroupSize = requiredSubgroupSize;
        stage.pNext = &subgroupSize;
    }
    VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    info.stage = stage;
    info.layout = layout;
    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(context.device, VK_NULL_HANDLE, 1, &info, nullptr, &pipeline) != VK_SUCCESS) {
        spdlog::warn("신경망 컴퓨트 파이프라인을 만들지 못했습니다: {}", shaderName);
        pipeline = VK_NULL_HANDLE;
    }
    vkDestroyShaderModule(context.device, module, nullptr);
    return pipeline;
}

Buffer createDataBuffer(Context& context, size_t floats, const char* name) {
    // 크기가 0 인 버퍼는 만들 수 없다. 파라미터가 없는 그래프도 있으므로 한 칸은 잡아 둔다.
    VkDeviceSize bytes = static_cast<VkDeviceSize>(std::max<size_t>(floats, 1)) * sizeof(float);
    return createBuffer(context,
                        bytes,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        MemoryLocation::DEVICE,
                        name);
}

void copyRegion(VkCommandBuffer commandBuffer,
                const Buffer& from,
                VkDeviceSize fromOffset,
                const Buffer& to,
                VkDeviceSize toOffset,
                VkDeviceSize bytes) {
    if (bytes == 0) {
        return;
    }
    VkBufferCopy region{};
    region.srcOffset = fromOffset;
    region.dstOffset = toOffset;
    region.size = bytes;
    vkCmdCopyBuffer(commandBuffer, from.handle, to.handle, 1, &region);
}

} // namespace

NeuralExecutor::NeuralExecutor(Context& context) : context(context) {
    createPipelines();
}

NeuralExecutor::~NeuralExecutor() {
    for (Buffer* buffer : {&tensorBuffer,
                           &opBuffer,
                           &parameterBuffer,
                           &parameterGradientBuffer,
                           &activationBuffer,
                           &activationGradientBuffer,
                           &staging,
                           &readback,
                           &moments[0],
                           &moments[1]}) {
        destroyBuffer(context, *buffer);
    }
    for (VkPipeline pipeline : {elementwisePipeline,
                                linearPipeline,
                                linearCoopPipeline,
                                linearDxPipeline,
                                linearDwPipeline,
                                biasGradPipeline,
                                convPipeline,
                                convDxPipeline,
                                convDwPipeline,
                                convDbPipeline,
                                layerNormPipeline,
                                layerNormDxPipeline,
                                layerNormDparamPipeline,
                                reducePipeline,
                                reduceDxPipeline,
                                adamPipeline,
                                polyakPipeline}) {
        vkDestroyPipeline(context.device, pipeline, nullptr);
    }
    vkDestroyPipelineLayout(context.device, pipelineLayout, nullptr);
}

void NeuralExecutor::createPipelines() {
    VkPushConstantRange range{};
    range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    range.size = sizeof(NeuralPushConstants);
    // 디스크립터 집합이 없다. 이 커널들이 만지는 것은 버퍼뿐이고 전부 주소로 온다.
    //
    // ponytail: 이 저장소의 다른 컴퓨트는 전부 집합 0(bindless)을 묶는다. 지금은 자기 검사가 명령 버퍼를
    // 통째로 쓰므로 문제가 없지만, 이 디스패치를 렌더 그래프 노드 안에 끼우면 **호환되지 않는 레이아웃이
    // 묶여 있던 집합 0 을 흩뜨린다.** 12단계에서 플러그인에 넣을 때 bindless 레이아웃을 함께 준다.
    VkPipelineLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &range;
    VK_CHECK(vkCreatePipelineLayout(context.device, &layoutInfo, nullptr, &pipelineLayout));

    elementwisePipeline = createComputePipeline(context, pipelineLayout, "neural_elementwise.comp.spv");
    linearPipeline = createComputePipeline(context, pipelineLayout, "neural_linear.comp.spv");
    if (context.caps.cooperativeMatrix && context.caps.coopShader != nullptr) {
        // 모양도 셰이더 이름도 이미 골라져 있다(COOP_CANDIDATES 가 둘을 한 줄에 담는다). 여기서 이름을
        // 다시 조합하지 않는 것이 요점이다 — 후보를 더할 때 모양과 셰이더가 갈라지는 것을 막는다.
        linearCoopPipeline = createComputePipeline(context,
                                                   pipelineLayout,
                                                   context.caps.coopShader,
                                                   context.caps.coopSubgroupSize,
                                                   context.caps.coopSubgroupSize);
    }
    linearDxPipeline = createComputePipeline(context, pipelineLayout, "neural_linear_dx.comp.spv");
    linearDwPipeline = createComputePipeline(context, pipelineLayout, "neural_linear_dw.comp.spv");
    biasGradPipeline = createComputePipeline(context, pipelineLayout, "neural_bias_grad.comp.spv");
    convPipeline = createComputePipeline(context, pipelineLayout, "neural_conv.comp.spv");
    convDxPipeline = createComputePipeline(context, pipelineLayout, "neural_conv_dx.comp.spv");
    convDwPipeline = createComputePipeline(context, pipelineLayout, "neural_conv_dw.comp.spv");
    convDbPipeline = createComputePipeline(context, pipelineLayout, "neural_conv_db.comp.spv");
    layerNormPipeline = createComputePipeline(context, pipelineLayout, "neural_layernorm.comp.spv");
    layerNormDxPipeline = createComputePipeline(context, pipelineLayout, "neural_layernorm_dx.comp.spv");
    layerNormDparamPipeline = createComputePipeline(context, pipelineLayout, "neural_layernorm_dparam.comp.spv");
    reducePipeline = createComputePipeline(context, pipelineLayout, "neural_reduce.comp.spv");
    reduceDxPipeline = createComputePipeline(context, pipelineLayout, "neural_reduce_dx.comp.spv");
    adamPipeline = createComputePipeline(context, pipelineLayout, "neural_adam.comp.spv");
    polyakPipeline = createComputePipeline(context, pipelineLayout, "neural_polyak.comp.spv");
    ready = elementwisePipeline != VK_NULL_HANDLE && linearPipeline != VK_NULL_HANDLE &&
            linearDxPipeline != VK_NULL_HANDLE && linearDwPipeline != VK_NULL_HANDLE &&
            biasGradPipeline != VK_NULL_HANDLE && convPipeline != VK_NULL_HANDLE && convDxPipeline != VK_NULL_HANDLE &&
            convDwPipeline != VK_NULL_HANDLE && convDbPipeline != VK_NULL_HANDLE &&
            layerNormPipeline != VK_NULL_HANDLE && layerNormDxPipeline != VK_NULL_HANDLE &&
            layerNormDparamPipeline != VK_NULL_HANDLE && reducePipeline != VK_NULL_HANDLE &&
            reduceDxPipeline != VK_NULL_HANDLE && adamPipeline != VK_NULL_HANDLE && polyakPipeline != VK_NULL_HANDLE;
    if (!ready) {
        spdlog::warn("신경망 GPU 실행기를 만들지 못했습니다. CPU 기준만 돕니다");
    }
}

bool NeuralExecutor::supported(OpKind kind) {
    switch (kind) {
    // 값을 밖에서 채우는 자리라 도는 것이 없다.
    case OpKind::INPUT:
    case OpKind::RELU:
    case OpKind::TANH:
    case OpKind::ADD:
    case OpKind::MUL:
    case OpKind::SCALE:
    case OpKind::MIN2:
    case OpKind::CONCAT:
    case OpKind::LINEAR:
    case OpKind::CONV2D:
    case OpKind::LAYERNORM:
    case OpKind::MEAN:
    case OpKind::MSE:
    case OpKind::HUBER:
        return true;
    default:
        return false;
    }
}

bool NeuralExecutor::build(const Graph& source) {
    // 파이프라인이 없으면 지어 봐야 돌릴 수가 없다. 여기서 막지 않으면 dispatch 가 null 파이프라인을
    // 묶는다.
    if (!ready || !validateForward(source)) {
        return false;
    }
    graph = source;
    parameterCount = graph.parameterCount;
    activationCount = graph.activationCount;
    missing = 0;
    for (const Op& op : graph.ops) {
        missing += supported(op.kind) ? 0U : 1U;
    }

    momentFloats[0] = 0;
    momentFloats[1] = 0;
    for (Buffer* buffer : {&tensorBuffer,
                           &opBuffer,
                           &parameterBuffer,
                           &parameterGradientBuffer,
                           &activationBuffer,
                           &activationGradientBuffer,
                           &staging,
                           &readback,
                           &moments[0],
                           &moments[1]}) {
        // 모멘트도 함께 버린다. 그러지 않으면 다른 표로 다시 지었을 때 옛 크기의 버퍼가 남아
        // recordAdam 이 그것을 넘겨 읽는다.
        context.retireBuffer(*buffer);
    }

    tensorBuffer = createBuffer(context,
                                graph.tensors.size() * sizeof(Tensor),
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                                MemoryLocation::HOST_WRITE,
                                "신경망 텐서 표");
    opBuffer = createBuffer(context,
                            graph.ops.size() * sizeof(Op),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                            MemoryLocation::HOST_WRITE,
                            "신경망 연산 표");
    std::memcpy(tensorBuffer.mapped, graph.tensors.data(), graph.tensors.size() * sizeof(Tensor));
    std::memcpy(opBuffer.mapped, graph.ops.data(), graph.ops.size() * sizeof(Op));
    // 일관성 없는 메모리 타입에 잡혔을 수 있다. 이 저장소의 다른 HOST_WRITE 버퍼와 같은 방어다.
    vmaFlushAllocation(context.allocator, tensorBuffer.allocation, 0, VK_WHOLE_SIZE);
    vmaFlushAllocation(context.allocator, opBuffer.allocation, 0, VK_WHOLE_SIZE);

    parameterBuffer = createDataBuffer(context, parameterCount, "신경망 파라미터");
    parameterGradientBuffer = createDataBuffer(context, parameterCount, "신경망 파라미터 경사");
    activationBuffer = createDataBuffer(context, activationCount, "신경망 활성");
    activationGradientBuffer = createDataBuffer(context, activationCount, "신경망 활성 경사");

    size_t stagingFloatCount = parameterCount + activationCount * 2;
    size_t readbackFloatCount = activationCount * 2 + parameterCount * 2;
    staging = createBuffer(context,
                           static_cast<VkDeviceSize>(std::max<size_t>(stagingFloatCount, 1)) * sizeof(float),
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                           MemoryLocation::HOST_WRITE,
                           "신경망 업로드");
    readback = createBuffer(context,
                            static_cast<VkDeviceSize>(std::max<size_t>(readbackFloatCount, 1)) * sizeof(float),
                            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            MemoryLocation::HOST_READ,
                            "신경망 되읽기");
    stagingFloats = static_cast<float*>(staging.mapped);
    readbackFloats = static_cast<float*>(readback.mapped);
    std::fill_n(stagingFloats, stagingFloatCount, 0.0F);
    return true;
}

void NeuralExecutor::barrier(VkCommandBuffer commandBuffer) {
    VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memory.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    memory.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memory.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

// 버퍼를 vkCmdFillBuffer 로 지우기 **앞**에 놓는 배리어. 지우기는 전송 단계(CLEAR)라 컴퓨트끼리 거는
// barrier() 가 줄 세우지 못한다 — 그것 없이는 걸음 k 의 «경사를 0 으로» 가 걸음 k-1 의 역전파 쓰기나
// 씨앗 복사를 앞질러, 명령 버퍼 하나에 여러 걸음을 담는 순간(자기 검사의 열 걸음이 그렇다) 결과가
// 비결정적으로 갈린다. 검증 레이어의 동기화 검사가 이것을 WRITE_AFTER_WRITE 로 짚는다.
void NeuralExecutor::clearBarrier(VkCommandBuffer commandBuffer) {
    VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    memory.srcStageMask =
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT;
    memory.srcAccessMask =
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    memory.dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    memory.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

void NeuralExecutor::recordUpload(VkCommandBuffer commandBuffer) {
    vmaFlushAllocation(context.allocator, staging.allocation, 0, VK_WHOLE_SIZE);
    VkDeviceSize parameterBytes = static_cast<VkDeviceSize>(parameterCount) * sizeof(float);
    VkDeviceSize activationBytes = static_cast<VkDeviceSize>(activationCount) * sizeof(float);
    copyRegion(commandBuffer, staging, 0, parameterBuffer, 0, parameterBytes);
    copyRegion(commandBuffer, staging, parameterBytes, activationBuffer, 0, activationBytes);
    uploadBarrier(commandBuffer);
}

void NeuralExecutor::recordUploadGradientSeed(VkCommandBuffer commandBuffer) {
    VkDeviceSize parameterBytes = static_cast<VkDeviceSize>(parameterCount) * sizeof(float);
    VkDeviceSize activationBytes = static_cast<VkDeviceSize>(activationCount) * sizeof(float);
    copyRegion(commandBuffer, staging, parameterBytes + activationBytes, activationGradientBuffer, 0, activationBytes);
    uploadBarrier(commandBuffer);
}

void NeuralExecutor::uploadBarrier(VkCommandBuffer commandBuffer) {
    VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    memory.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    memory.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    memory.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memory.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

void NeuralExecutor::recordClearGradients(VkCommandBuffer commandBuffer) {
    clearBarrier(commandBuffer);
    vkCmdFillBuffer(commandBuffer, parameterGradientBuffer.handle, 0, VK_WHOLE_SIZE, 0);
    // ponytail: 활성 경사 쪽은 **지금은 지워도 티가 안 난다.** 유일한 부르는 쪽(자기 검사)이 바로 뒤에
    // recordUploadGradientSeed 로 배열 전체를 덮어쓰기 때문이다. 손실 커널이 생겨(8단계) 씨앗을 GPU 가
    // 스스로 심으면 그때부터 이 줄이 일한다. 그전까지는 돌연변이를 넣어도 자기 검사가 잡지 못한다.
    vkCmdFillBuffer(commandBuffer, activationGradientBuffer.handle, 0, VK_WHOLE_SIZE, 0);
    VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    memory.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    memory.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    // CLEAR 를 함께 둔다. 바로 뒤에 오는 것이 손실 씨앗의 vkCmdFillBuffer 이고, 그것은 COPY 가 아니라
    // CLEAR 단계다.
    memory.dstStageMask =
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT | VK_PIPELINE_STAGE_2_CLEAR_BIT;
    memory.dstAccessMask =
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

NeuralPushConstants NeuralExecutor::basePush() const {
    NeuralPushConstants push;
    push.tensors = tensorBuffer.address;
    push.ops = opBuffer.address;
    push.parameters = parameterBuffer.address;
    push.activations = activationBuffer.address;
    push.parameterGradients = parameterGradientBuffer.address;
    push.activationGradients = activationGradientBuffer.address;
    return push;
}

void NeuralExecutor::dispatchKernel(
    VkCommandBuffer commandBuffer, VkPipeline pipeline, uint32_t opIndex, uint32_t flags, uint32_t threads) {
    if (threads == 0) {
        return;
    }
    NeuralPushConstants push = basePush();
    push.op = opIndex;
    push.flags = flags;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(commandBuffer, (threads + NEURAL_GROUP_SIZE - 1) / NEURAL_GROUP_SIZE, 1, 1);
}

// 작업 그룹 **수**를 그대로 넘기는 갈래. 협력 행렬 커널은 작업 그룹 하나가 타일 하나를 맡으므로
// 스레드 수에서 그룹 수를 되짚는 위 규칙이 맞지 않는다.
void NeuralExecutor::dispatchGroups(
    VkCommandBuffer commandBuffer, VkPipeline pipeline, uint32_t opIndex, uint32_t flags, uint32_t groups) {
    if (groups == 0) {
        return;
    }
    NeuralPushConstants push = basePush();
    push.op = opIndex;
    push.flags = flags;
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(commandBuffer, groups, 1, 1);
}

void NeuralExecutor::dispatch(VkCommandBuffer commandBuffer, uint32_t opIndex, uint32_t flags) {
    const Op& op = graph.ops[opIndex];
    if (op.kind == OpKind::INPUT || !supported(op.kind)) {
        return;
    }
    bool backward = (flags & NEURAL_FLAG_BACKWARD) != 0;
    uint32_t outputCount = graph.tensors[op.output].count();

    if (op.kind == OpKind::LINEAR) {
        const Tensor& input = graph.tensors[op.inputs[0]];
        uint32_t batch = input.dims[0];
        uint32_t inputs = input.dims[1];
        uint32_t outputs = graph.tensors[op.output].dims[1];
        if (!backward && cooperative) {
            // 작업 그룹 하나가 출력 타일 하나를 맡는다. 스레드 수가 아니라 **타일 수**를 넘긴다.
            uint32_t rows = context.caps.coopM;
            uint32_t columns = context.caps.coopN;
            uint32_t tiles = ((batch + rows - 1) / rows) * ((outputs + columns - 1) / columns);
            dispatchGroups(commandBuffer, linearCoopPipeline, opIndex, flags, tiles);
        } else if (!backward) {
            dispatchKernel(commandBuffer, linearPipeline, opIndex, flags, batch * outputs);
        } else {
            // 셋이 서로 다른 자리에 쓰므로 사이에 배리어를 두지 않는다 — 다음 연산으로 넘어가기
            // 전에 아래에서 한 번만 건다. 겹치지 않는 근거는 **입력·가중치·편향이 서로 다른 텐서**이고
            // 연산의 출력은 입력보다 뒤에 잡히기 때문이다. 같은 텐서를 두 자리에 준 표(addLinear(t, t, b))
            // 는 그 근거를 깨지만 빌더가 낼 수 있는 모양이 아니고, 우리 그래프에도 없다.
            // ponytail: validateForward 가 그것을 거절하지는 않는다. 표를 파일에서 읽게 되면 막아야 한다.
            dispatchKernel(commandBuffer, linearDxPipeline, opIndex, flags, batch * inputs);
            dispatchKernel(commandBuffer, linearDwPipeline, opIndex, flags, outputs * inputs);
            dispatchKernel(commandBuffer, biasGradPipeline, opIndex, flags, outputs);
        }
    } else if (op.kind == OpKind::CONV2D) {
        const Tensor& source = graph.tensors[op.inputs[0]];
        const Tensor& weight = graph.tensors[op.inputs[1]];
        if (!backward) {
            dispatchKernel(commandBuffer, convPipeline, opIndex, flags, outputCount);
        } else {
            // 선형과 같은 이유로 셋 사이에 배리어를 두지 않는다. 입력 경사·가중치 경사·편향 경사가
            // 서로 다른 텐서에 쓰기 때문이고, 그 «서로 다르다» 는 addConv2d 와 validateForward 가 세
            // 입력의 저장소가 겹치는 표를 거절해서 성립한다.
            // ponytail: 편향의 원소 수가 출력 채널 수보다 적은 표는 아직 아무도 막지 않는다. 빌더는
            // 낼 수 없지만 파일에서 읽게 되면 db 커널이 편향 밖으로 쓴다.
            dispatchKernel(commandBuffer, convDxPipeline, opIndex, flags, source.count());
            dispatchKernel(commandBuffer, convDwPipeline, opIndex, flags, weight.count());
            dispatchKernel(commandBuffer, convDbPipeline, opIndex, flags, graph.tensors[op.output].dims[1]);
        }
    } else if (op.kind == OpKind::LAYERNORM) {
        const Tensor& source = graph.tensors[op.inputs[0]];
        if (!backward) {
            dispatchKernel(commandBuffer, layerNormPipeline, opIndex, flags, source.dims[0]);
        } else {
            // dx 는 행마다, 이득·편향은 특징마다다. 쓰는 자리가 갈라져 사이에 배리어가 필요 없다.
            dispatchKernel(commandBuffer, layerNormDxPipeline, opIndex, flags, source.dims[0]);
            dispatchKernel(commandBuffer, layerNormDparamPipeline, opIndex, flags, source.dims[1]);
        }
    } else if (op.kind == OpKind::MEAN || op.kind == OpKind::MSE || op.kind == OpKind::HUBER) {
        if (!backward) {
            // 스레드 하나가 통째로 접는다. 덧셈 순서를 CPU 기준과 맞추려는 것이다.
            dispatchKernel(commandBuffer, reducePipeline, opIndex, flags, 1);
        } else {
            dispatchKernel(commandBuffer, reduceDxPipeline, opIndex, flags, graph.tensors[op.inputs[0]].count());
        }
    } else {
        dispatchKernel(commandBuffer, elementwisePipeline, opIndex, flags, outputCount);
    }
    // ponytail: 연산마다 배리어를 하나씩 건다. 서로 닿지 않는 연산끼리도 줄을 세우는 셈이라(갈래가 갈린
    // 구간이 그렇다) 손해지만, 표만 보고 «겹쳐도 되는 구간» 을 가리려면 의존 그래프를 따로 세워야 한다.
    barrier(commandBuffer);
}

void NeuralExecutor::recordForward(VkCommandBuffer commandBuffer) {
    recordForward(commandBuffer, 0, static_cast<uint32_t>(graph.ops.size()));
}

void NeuralExecutor::recordBackward(VkCommandBuffer commandBuffer) {
    recordBackward(commandBuffer, 0, static_cast<uint32_t>(graph.ops.size()));
}

// 연산 **구간**만 돈다. 에이전트는 표가 셋인데 셋이 파라미터 하나를 나눠 써서 한 표로 이어 붙였고
// (gfx::mergeGraphs), 갱신 한 번은 그중 한 구간만 밟는다.
//
// 끝을 자르는 것(min)은 자기 검사가 지켜 주지 못한다. 자르지 않으면 표 밖의 연산을 첨자하는데, 릴리스
// 빌드의 std::vector 는 그것을 진단하지 않고 읽은 쓰레기의 kind 가 supported() 를 통과하지 못해 조용히
// 넘어간다(돌연변이로 확인했다). 값이 맞는지가 아니라 **규격을 읽어서** 지키는 자리다.
void NeuralExecutor::recordForward(VkCommandBuffer commandBuffer, uint32_t begin, uint32_t count) {
    uint32_t total = static_cast<uint32_t>(graph.ops.size());
    uint32_t end = begin >= total ? begin : std::min(begin + count, total);
    for (uint32_t i = begin; i < end; ++i) {
        dispatch(commandBuffer, i, 0);
    }
}

void NeuralExecutor::recordBackward(VkCommandBuffer commandBuffer, uint32_t begin, uint32_t count) {
    uint32_t total = static_cast<uint32_t>(graph.ops.size());
    uint32_t end = begin >= total ? begin : std::min(begin + count, total);
    for (uint32_t i = end; i-- > begin;) {
        dispatch(commandBuffer, i, NEURAL_FLAG_BACKWARD);
    }
}

bool NeuralExecutor::reserveMoments(uint32_t slot, size_t floats) {
    if (slot >= 2) {
        return false;
    }
    context.retireBuffer(moments[slot]);
    moments[slot] = createBuffer(context,
                                 static_cast<VkDeviceSize>(std::max<size_t>(floats, 1)) * sizeof(float),
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                     VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 MemoryLocation::DEVICE,
                                 "신경망 Adam 모멘트");
    momentFloats[slot] = floats;
    return true;
}

void NeuralExecutor::recordClearMoments(VkCommandBuffer commandBuffer, uint32_t slot, float value) {
    if (slot >= 2 || moments[slot].handle == VK_NULL_HANDLE) {
        return;
    }
    uint32_t pattern = 0;
    std::memcpy(&pattern, &value, sizeof(pattern));
    // 앞의 Adam 이 이 버퍼를 읽고 썼을 수도 있고, 자기 검사처럼 **같은 버퍼를 두 번 채우기**도 한다.
    clearBarrier(commandBuffer);
    vkCmdFillBuffer(commandBuffer, moments[slot].handle, 0, VK_WHOLE_SIZE, pattern);
    VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    memory.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    memory.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    memory.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memory.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

void NeuralExecutor::recordAdam(VkCommandBuffer commandBuffer,
                                uint32_t momentSlot,
                                uint32_t begin,
                                uint32_t count,
                                const AdamSettings& settings,
                                uint32_t step) {
    if (momentSlot >= 2 || count == 0 || step == 0) {
        return;
    }
    // **구간이 버퍼 안에 드는지 여기서 본다.** 셰이더는 자기 몫이 어디까지인지 모르고 rangeBegin + i 를
    // 그대로 쓰므로, 어긋난 구간을 주면 가중치 배열 밖을 조용히 덮어쓴다. 12단계에서 에이전트의 크리틱·
    // 액터 구간을 따로 밟을 때 정확히 여기가 틀릴 자리다. 걸리면 치명 오류로 세운다 — 조용히 넘기면
    // «학습이 안 되는데 이유를 모르는» 자리가 된다.
    if (moments[momentSlot].handle == VK_NULL_HANDLE) {
        core::fatal("신경망 Adam: 모멘트 슬롯 {} 을 잡지 않았습니다 (reserveMoments 를 먼저 부릅니다)", momentSlot);
    }
    if (static_cast<size_t>(begin) + count > parameterCount) {
        core::fatal("신경망 Adam: 구간 [{}, {}) 이 가중치 {} 개를 넘습니다", begin, begin + count, parameterCount);
    }
    if (adamMomentCount(count) > momentFloats[momentSlot]) {
        core::fatal("신경망 Adam: 모멘트 슬롯 {} 이 float {} 개인데 {} 개가 필요합니다",
                    momentSlot,
                    momentFloats[momentSlot],
                    adamMomentCount(count));
    }
    // 편향 보정을 호스트에서 낸다. GLSL 의 pow 는 std::pow 와 근사가 달라, GPU 에서 계산하면 첫 걸음부터
    // 두 엔진이 갈린다. **CPU 기준과 같은 함수를 부른다** — 두 벌로 두면 한쪽만 고쳐진다.
    float firstCorrection = 0.0F;
    float secondCorrection = 0.0F;
    if (!adamCorrections(settings, step, firstCorrection, secondCorrection)) {
        return;
    }
    NeuralPushConstants push = basePush();
    push.moments = moments[momentSlot].address;
    push.rangeBegin = begin;
    push.rangeCount = count;
    push.learningRate = settings.learningRate;
    push.beta1 = settings.beta1;
    push.beta2 = settings.beta2;
    push.epsilon = settings.epsilon;
    push.firstCorrection = firstCorrection;
    push.secondCorrection = secondCorrection;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, adamPipeline);
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(commandBuffer, (count + NEURAL_GROUP_SIZE - 1) / NEURAL_GROUP_SIZE, 1, 1);
    barrier(commandBuffer);
}

void NeuralExecutor::recordPolyak(
    VkCommandBuffer commandBuffer, uint32_t onlineBegin, uint32_t targetBegin, uint32_t count, float tau) {
    if (count == 0) {
        return;
    }
    // recordAdam 과 같은 이유로 구간을 확인한다. 이쪽은 구간이 둘이다.
    if (static_cast<size_t>(onlineBegin) + count > parameterCount ||
        static_cast<size_t>(targetBegin) + count > parameterCount) {
        core::fatal("신경망 polyak: 구간 [{}, {}) 또는 [{}, {}) 이 가중치 {} 개를 넘습니다",
                    onlineBegin,
                    onlineBegin + count,
                    targetBegin,
                    targetBegin + count,
                    parameterCount);
    }
    NeuralPushConstants push = basePush();
    push.rangeBegin = onlineBegin;
    push.targetBegin = targetBegin;
    push.rangeCount = count;
    push.tau = tau;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, polyakPipeline);
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(commandBuffer, (count + NEURAL_GROUP_SIZE - 1) / NEURAL_GROUP_SIZE, 1, 1);
    barrier(commandBuffer);
}

void NeuralExecutor::recordUploadActivationRange(
    VkCommandBuffer commandBuffer, VkBuffer source, VkDeviceSize sourceOffset, uint32_t floatOffset, uint32_t floats) {
    if (floats == 0 || source == VK_NULL_HANDLE) {
        return;
    }
    // **복사 앞에도 배리어를 건다.** 앞의 컴퓨트가 이 배열을 쓰고 있었을 수도 있고(WAR), 앞의 업로드
    // 복사와 겹칠 수도 있다(WAW) — uploadBarrier 의 dst 가 COMPUTE 뿐이라 복사끼리는 줄 세우지 못한다.
    // 검증 레이어의 동기화 검사가 vkCmdCopyBuffer 의 WRITE_AFTER_WRITE 로 짚는다.
    VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
    memory.srcAccessMask =
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    memory.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    memory.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);

    VkBufferCopy region{};
    region.srcOffset = sourceOffset;
    region.dstOffset = static_cast<VkDeviceSize>(floatOffset) * sizeof(float);
    region.size = static_cast<VkDeviceSize>(floats) * sizeof(float);
    vkCmdCopyBuffer(commandBuffer, source, activationBuffer.handle, 1, &region);
    uploadBarrier(commandBuffer);
}

void NeuralExecutor::recordSeedLossGradient(VkCommandBuffer commandBuffer, uint32_t tensor) {
    if (tensor >= graph.tensors.size()) {
        return;
    }
    const Tensor& item = graph.tensors[tensor];
    if (item.arena != Arena::ACTIVATION) {
        return;
    }
    // **채우기 앞에 배리어를 건다.** recordClearGradients 가 같은 배열을 통째로 지우는데 그 꼬리
    // 배리어의 dst 에 CLEAR 가 없어(COPY 는 CLEAR 를 포함하지 않는다) 두 채우기가 동기화 없는 WAW 다.
    // 전체 지우기가 뒤로 밀리면 씨앗이 날아가 그 갱신의 경사가 통째로 0 이 된다.
    //
    // 이 배리어와 recordUploadActivationRange 의 uploadBarrier 는 자기 검사가 지켜 주지 못한다 — 빼도
    // 이 드라이버에서는 답이 같다(돌연변이로 확인했다). 검증 레이어의 동기화 검사가 잡는 자리이고,
    // 그래서 규격을 읽어서 지킨다.
    clearBarrier(commandBuffer);
    // 1.0F 의 비트를 채운다. vkCmdFillBuffer 는 uint32 단위라 값을 그대로 넘긴다.
    uint32_t one = 0;
    float value = 1.0F;
    std::memcpy(&one, &value, sizeof(one));
    vkCmdFillBuffer(commandBuffer,
                    activationGradientBuffer.handle,
                    static_cast<VkDeviceSize>(item.offset) * sizeof(float),
                    static_cast<VkDeviceSize>(item.count()) * sizeof(float),
                    one);
    VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    memory.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    memory.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    memory.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memory.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

void NeuralExecutor::recordDownloadTensor(VkCommandBuffer commandBuffer, uint32_t tensor) {
    if (tensor >= graph.tensors.size()) {
        return;
    }
    const Tensor& item = graph.tensors[tensor];
    if (item.arena != Arena::ACTIVATION) {
        return;
    }
    VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    memory.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    memory.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    memory.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);

    // 되읽기 버퍼의 배치는 recordDownload 와 같다(활성이 맨 앞이다). 그래서 activationResult() 가 그대로
    // 이 텐서를 가리킨다.
    VkBufferCopy region{};
    region.srcOffset = static_cast<VkDeviceSize>(item.offset) * sizeof(float);
    region.dstOffset = region.srcOffset;
    region.size = static_cast<VkDeviceSize>(item.count()) * sizeof(float);
    vkCmdCopyBuffer(commandBuffer, activationBuffer.handle, readback.handle, 1, &region);
}

void NeuralExecutor::recordDownload(VkCommandBuffer commandBuffer) {
    VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    // 컴퓨트가 쓴 것뿐 아니라 **업로드가 쓴 것도** 덮어야 한다. 입력 텐서의 활성은 어느 디스패치도 건드리지
    // 않고 복사로만 들어오는데, 그것도 되읽기가 도로 가져간다.
    memory.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
    memory.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    memory.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    memory.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);

    VkDeviceSize parameterBytes = static_cast<VkDeviceSize>(parameterCount) * sizeof(float);
    VkDeviceSize activationBytes = static_cast<VkDeviceSize>(activationCount) * sizeof(float);
    copyRegion(commandBuffer, activationBuffer, 0, readback, 0, activationBytes);
    copyRegion(commandBuffer, parameterGradientBuffer, 0, readback, activationBytes, parameterBytes);
    copyRegion(commandBuffer, activationGradientBuffer, 0, readback, activationBytes + parameterBytes, activationBytes);
    copyRegion(commandBuffer, parameterBuffer, 0, readback, activationBytes * 2 + parameterBytes, parameterBytes);
}

void NeuralExecutor::invalidateReadback() {
    vmaInvalidateAllocation(context.allocator, readback.allocation, 0, VK_WHOLE_SIZE);
}

} // namespace gfx
