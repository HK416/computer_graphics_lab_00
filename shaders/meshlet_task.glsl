#ifndef MESHLET_TASK_GLSL
#define MESHLET_TASK_GLSL

// 태스크 워크그룹 하나가 다루는 meshlet 수. mesh shader 디스패치 수의 상한이기도 하다.
#define MESHLET_GROUP_SIZE 32

struct TaskPayload {
    uint instanceIndex;
    uint meshletIndices[MESHLET_GROUP_SIZE];
};

#endif
