#include "gfx/neural.h"

#include <algorithm>
#include <cstring>

#include <spdlog/spdlog.h>

#include "gfx/context.h"
#include "gfx/headless_compute.h"
#include "gfx/vk_check.h"

namespace gfx {

namespace {

// 만들지 못하면 VK_NULL_HANDLE 을 돌려준다. CPU 기준이 있으므로 중단하지 않는다.
VkPipeline createComputePipeline(Context& context, VkPipelineLayout layout, const char* shaderName) {
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
    uint32_t groupSize = NEURAL_GROUP_SIZE;
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
                           &readback}) {
        destroyBuffer(context, *buffer);
    }
    for (VkPipeline pipeline :
         {elementwisePipeline, linearPipeline, linearDxPipeline, linearDwPipeline, biasGradPipeline}) {
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
    linearDxPipeline = createComputePipeline(context, pipelineLayout, "neural_linear_dx.comp.spv");
    linearDwPipeline = createComputePipeline(context, pipelineLayout, "neural_linear_dw.comp.spv");
    biasGradPipeline = createComputePipeline(context, pipelineLayout, "neural_bias_grad.comp.spv");
    ready = elementwisePipeline != VK_NULL_HANDLE && linearPipeline != VK_NULL_HANDLE &&
            linearDxPipeline != VK_NULL_HANDLE && linearDwPipeline != VK_NULL_HANDLE &&
            biasGradPipeline != VK_NULL_HANDLE;
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

    for (Buffer* buffer : {&tensorBuffer,
                           &opBuffer,
                           &parameterBuffer,
                           &parameterGradientBuffer,
                           &activationBuffer,
                           &activationGradientBuffer,
                           &staging,
                           &readback}) {
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
    size_t readbackFloatCount = activationCount * 2 + parameterCount;
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
    vkCmdFillBuffer(commandBuffer, parameterGradientBuffer.handle, 0, VK_WHOLE_SIZE, 0);
    // ponytail: 활성 경사 쪽은 **지금은 지워도 티가 안 난다.** 유일한 부르는 쪽(자기 검사)이 바로 뒤에
    // recordUploadGradientSeed 로 배열 전체를 덮어쓰기 때문이다. 손실 커널이 생겨(8단계) 씨앗을 GPU 가
    // 스스로 심으면 그때부터 이 줄이 일한다. 그전까지는 돌연변이를 넣어도 자기 검사가 잡지 못한다.
    vkCmdFillBuffer(commandBuffer, activationGradientBuffer.handle, 0, VK_WHOLE_SIZE, 0);
    VkMemoryBarrier2 memory{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
    memory.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    memory.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    memory.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
    memory.dstAccessMask =
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    VkDependencyInfo dependency{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dependency.memoryBarrierCount = 1;
    dependency.pMemoryBarriers = &memory;
    vkCmdPipelineBarrier2(commandBuffer, &dependency);
}

void NeuralExecutor::dispatchKernel(
    VkCommandBuffer commandBuffer, VkPipeline pipeline, uint32_t opIndex, uint32_t flags, uint32_t threads) {
    if (threads == 0) {
        return;
    }
    NeuralPushConstants push;
    push.tensors = tensorBuffer.address;
    push.ops = opBuffer.address;
    push.parameters = parameterBuffer.address;
    push.activations = activationBuffer.address;
    push.parameterGradients = parameterGradientBuffer.address;
    push.activationGradients = activationGradientBuffer.address;
    push.op = opIndex;
    push.flags = flags;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push), &push);
    vkCmdDispatch(commandBuffer, (threads + NEURAL_GROUP_SIZE - 1) / NEURAL_GROUP_SIZE, 1, 1);
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
        if (!backward) {
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
    } else {
        dispatchKernel(commandBuffer, elementwisePipeline, opIndex, flags, outputCount);
    }
    // ponytail: 연산마다 배리어를 하나씩 건다. 서로 닿지 않는 연산끼리도 줄을 세우는 셈이라(갈래가 갈린
    // 구간이 그렇다) 손해지만, 표만 보고 «겹쳐도 되는 구간» 을 가리려면 의존 그래프를 따로 세워야 한다.
    barrier(commandBuffer);
}

void NeuralExecutor::recordForward(VkCommandBuffer commandBuffer) {
    for (uint32_t i = 0; i < graph.ops.size(); ++i) {
        dispatch(commandBuffer, i, 0);
    }
}

void NeuralExecutor::recordBackward(VkCommandBuffer commandBuffer) {
    for (size_t i = graph.ops.size(); i-- > 0;) {
        dispatch(commandBuffer, static_cast<uint32_t>(i), NEURAL_FLAG_BACKWARD);
    }
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
}

void NeuralExecutor::invalidateReadback() {
    vmaInvalidateAllocation(context.allocator, readback.allocation, 0, VK_WHOLE_SIZE);
}

} // namespace gfx
