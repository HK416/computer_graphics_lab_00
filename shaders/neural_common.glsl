#ifndef NEURAL_COMMON_GLSL
#define NEURAL_COMMON_GLSL

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require

// 신경망 GPU 실행기. **src/gfx/neural_math.h 의 Tensor / Op 와 배치가 같아야 한다.**
//
// 요점은 CPU 기준과 GPU 가 **문자 그대로 같은 표를 읽는다**는 것이다. 그래서 검증이 «같은 표를 두 엔진에
// 먹여 견준다» 로 떨어진다. 텐서가 buffer device address 대신 arena 와 offset 을 들고 다니는 것도 그
// 때문이다 — 주소를 담으면 버퍼를 다시 잡을 때마다 표를 새로 지어야 하고, 그러면 두 표가 «같은 것» 이
// 아니게 된다. arena 의 시작 주소 넷은 푸시 상수로 실어 보낸다.

#define NEURAL_ARENA_PARAMETER 0u
#define NEURAL_ARENA_ACTIVATION 1u

#define NEURAL_TENSOR_GRAD 1u

// gfx::OpKind 와 번호가 같아야 한다.
#define NEURAL_OP_INPUT 0u
#define NEURAL_OP_CONV2D 1u
#define NEURAL_OP_LINEAR 2u
#define NEURAL_OP_RELU 3u
#define NEURAL_OP_TANH 4u
#define NEURAL_OP_LAYERNORM 5u
#define NEURAL_OP_ADD 6u
#define NEURAL_OP_MUL 7u
#define NEURAL_OP_CONCAT 8u
#define NEURAL_OP_SCALE 9u
#define NEURAL_OP_MEAN 10u
#define NEURAL_OP_MIN2 11u
#define NEURAL_OP_MSE 12u
#define NEURAL_OP_HUBER 13u

// 역전파를 도는 중인지. 같은 파이프라인이 두 방향을 다 돈다.
#define NEURAL_FLAG_BACKWARD 1u

struct Tensor {
    uint arena;
    // arena 안의 **float 첨자**(바이트가 아니다).
    uint offset;
    uint dims[4];
    uint flags;
};

struct Op {
    uint kind;
    uint inputs[3];
    // C++ 쪽 이름은 output 이지만 GLSL 의 예약어라 여기서만 다르게 부른다. 배치는 같다.
    uint result;
    int iparams[4];
    float fparams[2];
    uint pad0;
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer NeuralTensorBuffer {
    Tensor items[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer NeuralOpBuffer {
    Op items[];
};

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer NeuralFloatBuffer {
    float items[];
};

layout(push_constant, scalar) uniform NeuralPushConstants {
    NeuralTensorBuffer tensors;
    NeuralOpBuffer ops;
    NeuralFloatBuffer parameters;
    NeuralFloatBuffer activations;
    NeuralFloatBuffer parameterGradients;
    NeuralFloatBuffer activationGradients;
    // 이번 디스패치가 도는 연산 번호.
    uint op;
    uint flags;
}
push;

uint neuralCount(Tensor tensor) {
    return tensor.dims[0] * tensor.dims[1] * tensor.dims[2] * tensor.dims[3];
}

bool neuralHasGrad(Tensor tensor) {
    return (tensor.flags & NEURAL_TENSOR_GRAD) != 0u;
}

float neuralRead(Tensor tensor, uint index) {
    if (tensor.arena == NEURAL_ARENA_PARAMETER) {
        return push.parameters.items[tensor.offset + index];
    }
    return push.activations.items[tensor.offset + index];
}

// 연산의 **출력은 늘 활성**이다(validateForward 가 파라미터에 쓰는 표를 거절한다). 그래서 값 쓰기는
// 활성만 본다 — 파라미터 갈래를 두면 한 번도 밟지 않는 죽은 코드가 된다.
void neuralWrite(Tensor tensor, uint index, float value) {
    push.activations.items[tensor.offset + index] = value;
}

// 경사는 두 arena 를 다 본다. 가중치·편향의 경사가 파라미터 쪽에 쌓이기 때문이다.
float neuralGradAt(Tensor tensor, uint index) {
    if (tensor.arena == NEURAL_ARENA_PARAMETER) {
        return push.parameterGradients.items[tensor.offset + index];
    }
    return push.activationGradients.items[tensor.offset + index];
}

void neuralSetGrad(Tensor tensor, uint index, float value) {
    if (tensor.arena == NEURAL_ARENA_PARAMETER) {
        push.parameterGradients.items[tensor.offset + index] = value;
        return;
    }
    push.activationGradients.items[tensor.offset + index] = value;
}

// 경사를 **더한다.** 갈래가 여럿이어도 되도록 누적이다.
//
// 원자 연산이 필요 없는 근거는 «갈래가 연산 사이에서만 갈린다» 가 아니라 **별칭이 오프셋을 공유한다**
// 는 것이다. addReshape 가 offset 을 그대로 물려주므로 같은 저장소를 가리키는 텐서들은 첨자까지 같고,
// 스레드 i 는 어느 별칭으로 보든 offset+i 만 만진다. 두 입력이 같은 텐서인 연산(add(t, t))도 한 스레드
// 안에서 두 번 더하므로 2*dy 가 되어 CPU 기준과 같다.
//
// **이음만 예외다.** 첨자를 옮겨 쓰는 유일한 연산이라 두 조각이 겹치면 서로 다른 스레드가 같은 칸을
// 고친다. 그래서 addConcat 과 validateForward 가 겹치는 조각을 아예 거절한다. addReshape 가 오프셋을
// 옮기게 바뀌거나 첨자를 옮기는 연산이 더 생기면 이 근거를 다시 세워야 한다.
void neuralAddGrad(Tensor tensor, uint index, float value) {
    if (!neuralHasGrad(tensor)) {
        return;
    }
    // **precise 가 없으면 컴파일러가 a + b*c 를 FMA 하나로 합친다.** 그러면 곱의 중간 반올림이 사라져
    // CPU 기준과 1 ULP 씩 갈리고, «원소별은 정확히 0» 이라는 자기 검사의 근거가 무너진다. 성능을 조금
    // 내주고 두 엔진이 같은 답을 내는 쪽을 골랐다.
    precise float sum = neuralGradAt(tensor, index) + value;
    neuralSetGrad(tensor, index, sum);
}

#endif
